#!/usr/bin/env bash
# Deploy the translator to Google Cloud Run. README.md, step 4, explains the
# settings and the free tier.
#
#   ./cloudrun.sh setup       once per project: APIs, service account, image cleanup, token
#   ./cloudrun.sh token       store a new Hugging Face token, e.g. after rotating it
#   ./cloudrun.sh deploy      build the last commit on Cloud Build and deploy it
#   ./cloudrun.sh publish     let anyone open it (a new service starts private)
#   ./cloudrun.sh unpublish   make it private again
#   ./cloudrun.sh stage DIR   write the files that deploy uploads to DIR
#
# Commands act on gcloud's current project (gcloud config set project ID).
set -euo pipefail

REGION=us-central1                  # Tier 1 prices, where the free tier goes furthest
SERVICE=qom-translator
ACCOUNT=qom-translator              # the service's identity; it can only read the token
SECRET=qom-hf-token
REPOSITORY=cloud-run-source-deploy  # where gcloud run deploy --source pushes images

# What the image is built from, taken from the last commit. Only tracked files
# can be in it, so nothing ignored (corpus/, private/, .env) reaches Cloud Build.
FILES=(
  pyproject.toml
  README.md
  src
  translator/app.py
  translator/requirements.txt
  translator/Dockerfile
)

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)

usage() {
  echo "usage: $0 setup | token | deploy | publish | unpublish | stage DIR" >&2
  exit 2
}

project() {
  local id
  if ! command -v gcloud >/dev/null; then
    echo "Install the gcloud CLI first: brew install --cask gcloud-cli" >&2
    exit 1
  fi
  id=$(gcloud config get-value project 2>/dev/null)
  if [[ -z $id ]]; then
    echo "No gcloud project set. Run: gcloud config set project PROJECT_ID" >&2
    exit 1
  fi
  echo "$id"
}

# New projects and service accounts can take a minute to show up in IAM.
retry() {
  for _ in 1 2 3 4 5 6; do
    "$@" && return
    sleep 10
  done
  return 1
}

stage() {
  local dir=$1
  if [[ -n $(git -C "$ROOT" status --porcelain -- "${FILES[@]}") ]]; then
    echo "Commit first. The image is built from the last commit, and these differ:" >&2
    git -C "$ROOT" status --short -- "${FILES[@]}" >&2
    exit 1
  fi
  git -C "$ROOT" archive HEAD -- "${FILES[@]}" | tar -x -C "$dir"
  mv "$dir/translator/Dockerfile" "$dir/Dockerfile"
}

grant_token_access() {
  retry gcloud secrets add-iam-policy-binding "$SECRET" --project "$PROJECT" \
    --condition=None --member "serviceAccount:$ACCOUNT@$PROJECT.iam.gserviceaccount.com" \
    --role roles/secretmanager.secretAccessor >/dev/null
}

store_token() {
  local token
  if [[ ! -t 0 ]]; then
    echo "The token prompt needs a terminal. Run: $0 token" >&2
    exit 1
  fi
  read -rsp "Hugging Face token (fine-grained, read-only, qom-nlp/qom-mt-v2 only): " token
  echo
  if [[ $token != hf_* ]]; then
    echo "That isn't a Hugging Face token; they start with hf_." >&2
    exit 1
  fi
  if gcloud secrets describe "$SECRET" --project "$PROJECT" >/dev/null 2>&1; then
    printf %s "$token" | gcloud secrets versions add "$SECRET" --project "$PROJECT" --data-file=-
  else
    printf %s "$token" | gcloud secrets create "$SECRET" --project "$PROJECT" \
      --replication-policy automatic --data-file=-
  fi
  grant_token_access
  echo "Token stored."
}

setup() {
  local number policy
  number=$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')

  gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com secretmanager.googleapis.com \
    iam.googleapis.com storage.googleapis.com --project "$PROJECT"

  # Source deploys build as the Compute Engine default service account, which
  # appears shortly after the Cloud Run API is turned on.
  retry gcloud projects add-iam-policy-binding "$PROJECT" --condition=None --quiet \
    --member "serviceAccount:$number-compute@developer.gserviceaccount.com" \
    --role roles/run.builder >/dev/null

  if ! gcloud iam service-accounts describe "$ACCOUNT@$PROJECT.iam.gserviceaccount.com" \
    --project "$PROJECT" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$ACCOUNT" --project "$PROJECT" \
      --display-name "Qom translator on Cloud Run"
  fi

  # Every deploy pushes a new image of about 0.4 GB, and 0.5 GB of storage is
  # free. Keep the two newest: the live one and the one before it.
  if ! gcloud artifacts repositories describe "$REPOSITORY" --project "$PROJECT" \
    --location "$REGION" >/dev/null 2>&1; then
    gcloud artifacts repositories create "$REPOSITORY" --project "$PROJECT" \
      --location "$REGION" --repository-format docker
  fi
  policy=$(mktemp)
  cat >"$policy" <<'JSON'
[
  {"name": "keep-two-newest", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 2}},
  {"name": "delete-older", "action": {"type": "Delete"}, "condition": {"tagState": "any"}}
]
JSON
  gcloud artifacts repositories set-cleanup-policies "$REPOSITORY" --project "$PROJECT" \
    --location "$REGION" --policy "$policy" --no-dry-run >/dev/null
  rm -f "$policy"

  if gcloud secrets describe "$SECRET" --project "$PROJECT" >/dev/null 2>&1; then
    grant_token_access
  else
    store_token
  fi
  echo "Setup done. Next: $0 deploy"
}

deploy() {
  local commit access=""
  STAGED=$(mktemp -d)
  trap 'rm -rf -- "$STAGED"' EXIT
  stage "$STAGED"
  commit=$(git -C "$ROOT" rev-parse --short HEAD)
  echo "Deploying commit $commit. Uploading:"
  (cd "$STAGED" && find . -type f | sort | sed 's|^\./|  |')

  # A new service starts private, until `publish`. A redeploy keeps its access.
  if ! gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
    >/dev/null 2>&1; then
    access=--no-allow-unauthenticated
  fi

  # README.md, step 4, explains each setting. In short: the server stops when
  # idle, at most one runs, it's billed only while a request is open, and
  # 16 GiB (which needs 4 vCPU) holds one float32 model plus both downloads.
  gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" \
    --source "$STAGED" \
    --service-account "$ACCOUNT@$PROJECT.iam.gserviceaccount.com" \
    --set-secrets "HF_TOKEN=$SECRET:latest" \
    --execution-environment gen2 --cpu 4 --memory 16Gi \
    --cpu-throttling --cpu-boost \
    --min-instances 0 --max-instances 1 \
    --timeout 10m \
    --labels "commit=$commit" \
    ${access:+"$access"}
}

publish() {
  gcloud run services add-iam-policy-binding "$SERVICE" --project "$PROJECT" \
    --region "$REGION" --member allUsers --role roles/run.invoker >/dev/null
  echo "Public at $(gcloud run services describe "$SERVICE" --project "$PROJECT" \
    --region "$REGION" --format 'value(status.url)')"
}

unpublish() {
  gcloud run services remove-iam-policy-binding "$SERVICE" --project "$PROJECT" \
    --region "$REGION" --member allUsers --role roles/run.invoker >/dev/null
  echo "Private again. To open it yourself: gcloud run services proxy $SERVICE --region $REGION"
}

case ${1:-} in
  setup) PROJECT=$(project); setup ;;
  token) PROJECT=$(project); store_token ;;
  deploy) PROJECT=$(project); deploy ;;
  publish) PROJECT=$(project); publish ;;
  unpublish) PROJECT=$(project); unpublish ;;
  stage)
    [[ $# -eq 2 ]] || usage
    mkdir -p "$2"
    stage "$2"
    ;;
  *) usage ;;
esac
