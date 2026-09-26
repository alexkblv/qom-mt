---
title: Traductor Qom - Español
emoji: 🗣️
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 6.26.0
python_version: "3.10"
app_file: app.py
pinned: false
short_description: Traductor automático experimental qom-español (versión de prueba)
---

# Traductor Qom – Español (beta)

Traducción automática experimental entre **qom** (toba, ISO 639-3 `tob`) y
español, servida por los modelos `qom-nlp/qom-mt-v2`: fine-tunes de
`facebook/nllb-200-distilled-600M`.

**Los resultados pueden contener errores y no deben usarse como traducción de
referencia.** El qom es una lengua de bajos recursos y ausente de los
benchmarks multilingües principales; este traductor es una demostración de
investigación, no un producto.

**La primera traducción tarda.** El servidor se apaga cuando nadie lo usa. Si está
apagado, la página tarda un poco en abrir, y la primera traducción en cada dirección
puede tardar uno o dos minutos mientras se carga el modelo. Después, cada traducción
tarda unos segundos.

## Cómo funciona

NLLB no tiene un código de lengua para el qom, así que se lo representa con la
etiqueta **`grn_Latn`** (guaraní) como proxy. No se inventa ninguna etiqueta qom.

| Dirección | Origen | Destino |
|---|---|---|
| Español → Qom | `spa_Latn` | `grn_Latn` |
| Qom → Español | `grn_Latn` | `spa_Latn` |

El texto de entrada se normaliza igual que el corpus de entrenamiento: NFC,
unificación del apóstrofo (que escribe la oclusiva glotal) a U+0027, y colapso
de espacios. La variación ortográfica del qom (`ỹ`/`ȳ`, `ñ`, d/r, h/j, e/i) **no**
se modifica.

No hay glosario, ni modelo de respaldo, ni post-edición: la salida es
exactamente lo que genera el checkpoint.

---

## Development

The models are **private**: the app reads them with a read-only Hugging Face token in
`HF_TOKEN`, and only this UI is public. It imports `qomlaq` from this repository for text
normalization, language tags and decoding, so it translates exactly the way the pipeline
evaluates.

### Pins are load-bearing

The published checkpoints declare `tie_word_embeddings: true` but store a
trained `lm_head` that differs from `model.shared` in all 256206 rows. `app.py`
passes `tie_word_embeddings=False` at load time to force the correct untied
load, and `_assert_untied()` re-checks it afterwards. Whether a given
`transformers` release honors that override is version-specific, and getting it
wrong yields fluent-looking garbage rather than an error.

**Re-run `smoke_test.py` after changing any pin in `requirements.txt`.**

### Float32 on CPU

The checkpoints are float16, but `app.py` loads them in float32. On x86 CPUs without
AVX512-FP16, PyTorch's float16 matrix math takes a fallback path 40–500x slower than
float32 ([pytorch/pytorch#146508](https://github.com/pytorch/pytorch/issues/146508),
still open in September 2026), and Cloud Run doesn't say which CPU it uses. On the
smoke-test pairs, float32 gives the same translations as float16. It doubles a model's
memory, to about 5.6 GB per direction, and makes local translations slower on Apple
silicon: about 4 seconds instead of 1 on an M2.

### 1. Read-only token

On huggingface.co: Settings → Access Tokens → **Create new token** →
*Fine-grained*, scoped to the `qom-nlp/qom-mt-v2` repository with
"Read access to contents of selected repos". Nothing wider — this token ends up on a
public-facing server.

### 2. Set up and smoke test (from this folder)

    python -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt -e ..
    printf 'HF_TOKEN=hf_...\n' > .env && chmod 600 .env
    set -a && source .env && set +a
    .venv/bin/python smoke_test.py

Stage 1 resolves the language tags against the fine-tuned tokenizer (~30 MB) and
fails loudly if either maps to `<unk>`. Stage 2 translates a few pairs from the
baseline experiment's test split (~5 GB of checkpoints). The pairs are read from the
split artifacts, never written into this folder: corpus text is not ours to
redistribute and this folder is public.

### 3. Run locally

    ./run.sh                       # http://localhost:7860

`QOM_VARIANT=nostrat` serves the other split strategy. `QOM_PRELOAD=0` skips the
startup model load, as the container does, so the model loads on the first translation.

### 4. Deploy to Cloud Run

The public translator runs on Google Cloud Run's free tier. Cloud Run starts the server
when someone opens the page and stops it when nobody has used it for a while, so a quiet
month costs nothing. The price is a wait on the first visit (see "The initial wait" below).

You need the gcloud CLI (`brew install --cask gcloud-cli`), a Google Cloud project with
billing turned on (the free tier needs a billing account), and a token made as in step 1
just for this server.

    gcloud auth login
    gcloud config set project PROJECT_ID
    ./cloudrun.sh setup      # once; copy the token first
    ./cloudrun.sh deploy
    ./cloudrun.sh publish    # when you want anyone to open it

`setup` turns on the APIs, creates a service account that can only read the token,
stores the token in Secret Manager, and sets Artifact Registry to keep only the two
newest images. It's safe to run again. On a Mac, the token comes from the clipboard, so
it never passes through the terminal, and the clipboard is cleared afterward. `deploy`
builds the last commit on Cloud Build and deploys it, and it stops if a file it uploads
has uncommitted changes.

A new service starts private. To try it before anyone else can, run `gcloud run services
proxy qom-translator --region us-central1` and open http://localhost:8080. `publish` lets
anyone open it and prints the URL; `unpublish` makes it private again. A redeploy keeps
whichever access the service has.

The upload is a `git archive` of `pyproject.toml`, `README.md`, `src/` and the
translator's `app.py`, `requirements.txt` and `Dockerfile`. Untracked files, like the
corpus, `private/` and `.env`, can't get into it. `./cloudrun.sh stage DIR` writes the
same files to `DIR` so you can check them. After rotating the token, copy the new one
and run `./cloudrun.sh token`. New instances pick it up without a redeploy.

The settings in `cloudrun.sh` and the `Dockerfile`:

- **At most one instance, and none when idle.** One instance caps the cost. Gradio's
  queue also needs all of a visitor's requests to reach the same server.
- **Billed only while a request is open** (request-based billing). The app holds no
  connection open per tab, which is why the direction sits in a hidden textbox and the
  character counter runs in the browser. An open connection would keep the server
  running, and billed, for as long as the tab stays open.
- **4 vCPU and 16 GiB.** One direction takes about 5.6 GB in float32. Cloud Run keeps
  downloaded files in memory, so once both directions have been used, the server also
  holds both 2.8 GB checkpoints. More than 8 GiB needs 4 vCPU.
- **us-central1 (Iowa).** The free tier is a discount at Tier 1 prices, so it covers the
  most in a Tier 1 region. São Paulo and Santiago are Tier 2.
- **The model loads on the first translation** (`QOM_PRELOAD=0`). Cloud Run holds a
  visitor's first request until the app is listening, so loading at startup would leave
  them on a blank page. This way, bots and Cloud Run's startup check on each deploy don't
  set off a 2.8 GB download either.

Free tier, checked September 26, 2026: 180,000 vCPU-seconds, 360,000 GiB-seconds and 2
million requests a month. At 16 GiB, memory runs out first, after about 6 hours of
active server time a month. A visit that starts the server and translates both ways
should use about 3 minutes of that, so the free tier covers roughly 120 such visits.
Visitors who come while the server is still up share its startup. Past the free tier,
active time costs up to about $0.50 an hour. With one instance, even a server kept busy
all month would cost at most about $350, so set a budget alert (Billing → Budgets &
alerts in the console) to hear about any charge by email. Cloud Build, Secret Manager
and the source upload stay within their free tiers. The two kept images take about
0.7 GB against 0.5 GB of free storage, which comes to roughly 2 cents a month.

**The initial wait.** When the server is off, the page takes a little while to open as
it starts. The first translation in each direction then downloads that direction's model
from Hugging Face (2.8 GB) and loads it. That's the "uno o dos minutos" the app mentions
under its title, and a "Cargando el modelo" notice shows while it happens. Later
translations take a few seconds. These times are estimates until the first deploy, so
check them then and adjust `WAIT_NOTE` in `app.py` if they're off. Wherever you link to
the translator, mention the wait too, for example: «El traductor puede tardar uno o dos
minutos en responder la primera vez, porque el servidor se apaga cuando nadie lo usa.»

## Later (out of scope for the beta)

Migrating to serverless GPU inference reuses `app.py`'s `translate()` logic; only the
serving wrapper changes.
