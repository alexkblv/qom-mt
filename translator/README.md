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
startup model load.

### 4. Deploy

A deployment installs `qomlaq` from this repository alongside `requirements.txt` (for
example, a container built from the repository root), sets `HF_TOKEN` as a secret, and
runs `app.py`. `.env` and `.venv/` are gitignored and must never be committed or
deployed.

## Later (out of scope for the beta)

Migrating to serverless GPU inference reuses `app.py`'s `translate()` logic; only the
serving wrapper changes.
