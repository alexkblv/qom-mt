"""
Qom <-> Spanish translator (beta).

Serves the qom-nlp/qom-mt-v2 NLLB-200-distilled-600M fine-tunes from a PRIVATE
org, read with a read-only HF token in the HF_TOKEN environment variable. The
models never become public; only this translator UI is public.

Text normalization, language tags and decoding come from the qomlaq package in
this repository, so the demo normalizes and decodes exactly the way evaluation
does. Nothing here augments the model: there is no glossary, no fallback model
and no post-editing, so what the UI shows is what the model does.

--- Why this file overrides the checkpoint config -------------------------------

The published checkpoints declare ``tie_word_embeddings: true`` but store
``lm_head.weight`` and ``model.shared.weight`` as different tensors -- they
differ in all 256206 rows (mean |diff| 0.16). The trained output projection is
``lm_head``. A loader that honored the config would overwrite it with
``model.shared`` and silently emit fluent-looking garbage.

The model repo is read-only to us, so the override lives here instead:
``tie_word_embeddings=False`` is passed at load time, and ``_assert_untied``
re-checks the loaded weights. Both are deliberate; do not remove either without
re-running smoke_test.py.
"""

import os

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from qomlaq.config import DIRECTIONS as LANGUAGES
from qomlaq.config import GENERATION
from qomlaq.generate import LoadedModel
from qomlaq.generate import translate as translate_batch
from qomlaq.normalize import normalize_text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VARIANT = os.environ.get("QOM_VARIANT", "strat")  # "strat" or "nostrat"
REPO = "qom-nlp/qom-mt-v2"
HF_TOKEN = os.environ.get("HF_TOKEN")  # required, no silent fallback
PRELOAD = os.environ.get("QOM_PRELOAD", "1") == "1"

#: UI label -> pipeline direction. Tags and columns come from qomlaq.config.
DIRECTIONS = {
    "Español -> Qom": "es2qom",
    "Qom -> Español": "qom2es",
}

DEFAULT_DIRECTION = "Qom -> Español"


def subfolder(direction: str) -> str:
    return f"qom-mt-v2-{VARIANT}/qom-mt-v2-{VARIANT}-{direction}"


# ---------------------------------------------------------------------------
# Loading
#
# One-slot cache: hold only the most recently used model so a small machine
# (2 vCPU / 16 GB RAM) never tries to load two ~2.8 GB checkpoints at once.
# ---------------------------------------------------------------------------

_cache = {"key": None, "tok": None, "model": None}


def _require_token() -> str:
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. The models are private, and without an explicit "
            "token huggingface_hub falls back to any cached login on the machine, "
            "which would make this work locally and fail when deployed. Export it "
            "from .env (run.sh does this)."
        )
    return HF_TOKEN


def _assert_untied(model) -> None:
    """The trained output projection must survive loading. See module docstring."""
    lm_head = model.lm_head.weight
    shared = model.model.shared.weight
    if lm_head.data_ptr() == shared.data_ptr() or torch.equal(lm_head, shared):
        raise RuntimeError(
            "lm_head.weight was tied to model.shared.weight during loading. The "
            "trained output projection has been overwritten and this model would "
            "emit fluent-looking garbage. This means the installed transformers "
            f"({__import__('transformers').__version__}) honors the checkpoint's "
            "tie_word_embeddings flag despite the tie_word_embeddings=False "
            "override. Do not serve translations. Pin transformers to the version "
            "in requirements.txt."
        )


def _load(direction_key):
    if _cache["key"] == direction_key:
        return _cache["tok"], _cache["model"]

    token = _require_token()
    direction = DIRECTIONS[direction_key]

    # Drop the previous model before allocating the next one, so peak RSS stays
    # at one checkpoint rather than two.
    _cache.update({"key": None, "tok": None, "model": None})

    tok = AutoTokenizer.from_pretrained(
        REPO,
        subfolder=subfolder(direction),
        src_lang=LANGUAGES[direction].src_lang,
        token=token,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        REPO,
        subfolder=subfolder(direction),
        token=token,
        tie_word_embeddings=False,  # see module docstring; not optional
    )
    _assert_untied(model)
    model.eval()

    _cache.update({"key": direction_key, "tok": tok, "model": model})
    return tok, model


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def translate(text, direction_key):
    """Translate one string."""
    normalized = normalize_text(text or "")
    if not normalized:
        return ""

    language = LANGUAGES[DIRECTIONS[direction_key]]
    tok, model = _load(direction_key)
    loaded = LoadedModel(model=model, tokenizer=tok, device=str(model.device))

    tok.src_lang = language.src_lang
    n_tokens = len(tok(normalized)["input_ids"])
    translation = translate_batch(
        [normalized], loaded,
        src_lang=language.src_lang, tgt_lang=language.tgt_lang, progress=False,
    )[0]

    limit = GENERATION.max_source_length
    if n_tokens > limit:
        gr.Warning(
            f"El texto tiene {n_tokens} tokens y se recortó a {limit}. "
            "Solo se tradujo la primera parte."
        )

    return translation


# ---------------------------------------------------------------------------
# UI
#
# A language bar with a swap control sitting above two text panels, rather than
# a direction radio: the pair is what you manipulate, so it belongs where you
# read it.
#
# Everything is targeted by elem_id. Gradio 6 renders elem_classes on Column
# without emitting the class, and its own layout CSS overrides height on
# stretched flex children, so ids are the only stable hook here.
# ---------------------------------------------------------------------------

LANG_LABELS = {
    "Qom -> Español": ("Qom", "Español"),
    "Español -> Qom": ("Español", "Qom"),
}

PLACEHOLDERS = {
    "Qom": "Escribí un texto en qom",
    "Español": "Escribí un texto en español",
}

THEME = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
    radius_size="lg",
    font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
)

CSS = """
#app-wrap { max-width: 1040px; margin: 0 auto; }
#title h2 { margin-bottom: 2px !important; }
#subtitle p {
    color: var(--body-text-color-subdued) !important;
    font-size: 0.9rem !important;
    margin-top: 0 !important;
}
#lang-bar { margin-bottom: 6px; align-items: center !important; }
#src-lang p, #tgt-lang p {
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    margin: 0 0 0 18px !important;
}
#swap-btn {
    border-radius: 999px !important;
    min-width: 44px !important;
    max-width: 44px !important;
    max-height: 44px !important;
    font-size: 1.15rem !important;
    padding: 0 !important;
    line-height: 1 !important;
    margin: 0 auto !important;
}
#src-box, #tgt-box {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 16px !important;
    background: var(--background-fill-primary) !important;
    padding: 10px 18px 14px !important;
    min-height: 210px;
}
#tgt-box { background: var(--background-fill-secondary) !important; }
#src-box textarea, #tgt-box textarea {
    font-size: 1.2rem !important;
    line-height: 1.55 !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background: transparent !important;
    padding: 0 !important;
    resize: none !important;
}
#counter p {
    font-size: 0.78rem !important;
    color: var(--body-text-color-subdued) !important;
    text-align: right !important;
    margin: 6px 4px 0 0 !important;
}
#go-btn { margin-top: 16px; }
footer { display: none !important; }
"""


def _other(direction):
    return "Español -> Qom" if direction == "Qom -> Español" else "Qom -> Español"


def swap(direction, tgt_text):
    """Flip the pair and carry the translation up into the source box, the way
    a translator UI is expected to behave."""
    new_dir = _other(direction)
    src_lbl, tgt_lbl = LANG_LABELS[new_dir]
    return (
        new_dir,
        src_lbl,
        tgt_lbl,
        gr.update(value=tgt_text or "", placeholder=PLACEHOLDERS[src_lbl]),
        "",
    )


def count_chars(text):
    return f"{len(text or '')} caracteres"


_src0, _tgt0 = LANG_LABELS[DEFAULT_DIRECTION]

with gr.Blocks(
    title="Traductor Qom - Español (beta)", analytics_enabled=False
) as demo:
    direction = gr.State(DEFAULT_DIRECTION)

    with gr.Column(elem_id="app-wrap"):
        gr.Markdown("## Traductor Qom – Español", elem_id="title")
        gr.Markdown(
            "Versión de prueba. Traducción automática experimental: los resultados "
            "pueden contener errores y no deben usarse como traducción de referencia.",
            elem_id="subtitle",
        )

        with gr.Row(elem_id="lang-bar"):
            with gr.Column(scale=10, min_width=200):
                src_label = gr.Markdown(_src0, elem_id="src-lang")
            with gr.Column(scale=1, min_width=64):
                swap_btn = gr.Button("⇄", elem_id="swap-btn", variant="secondary")
            with gr.Column(scale=10, min_width=200):
                tgt_label = gr.Markdown(_tgt0, elem_id="tgt-lang")

        with gr.Row():
            with gr.Column(scale=10, min_width=200):
                src = gr.Textbox(
                    lines=8,
                    max_lines=8,
                    show_label=False,
                    container=False,
                    placeholder=PLACEHOLDERS[_src0],
                    elem_id="src-box",
                )
                counter = gr.Markdown("0 caracteres", elem_id="counter")
            with gr.Column(scale=1, min_width=64):
                gr.HTML("")
            with gr.Column(scale=10, min_width=200):
                tgt = gr.Textbox(
                    lines=8,
                    max_lines=8,
                    show_label=False,
                    container=False,
                    interactive=False,
                    elem_id="tgt-box",
                )

        btn = gr.Button("Traducir", variant="primary", elem_id="go-btn")

    src.change(count_chars, inputs=src, outputs=counter, show_progress="hidden")
    swap_btn.click(
        swap,
        inputs=[direction, tgt],
        outputs=[direction, src_label, tgt_label, src, tgt],
    )
    btn.click(
        translate,
        inputs=[src, direction],
        outputs=[tgt],
        concurrency_limit=1,
    )

demo.queue(max_size=20, default_concurrency_limit=1)

if __name__ == "__main__":
    if PRELOAD:
        # Pay the load once at boot, where the Space shows a starting state,
        # rather than on the first visitor's request, where it looks hung.
        print(f"Preloading {DEFAULT_DIRECTION} ...", flush=True)
        _load(DEFAULT_DIRECTION)
        print("Ready.", flush=True)
    # Gradio 6 takes theme and css at launch(), not on Blocks().
    demo.launch(theme=THEME, css=CSS)
