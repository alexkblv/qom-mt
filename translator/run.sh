#!/usr/bin/env bash
# Run the translator locally at http://localhost:7860 with the token from .env.
# Same as README step 3.
set -euo pipefail
cd "$(dirname "$0")"
set -a
source .env
set +a
exec .venv/bin/python app.py
