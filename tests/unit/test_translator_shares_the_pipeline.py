"""The translator must use the pipeline's text handling and decoding, not copies.

The app needs gradio and a pinned transformers, which the test environment doesn't
install, so these checks read its source.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TRANSLATOR = Path(__file__).resolve().parents[2] / "translator"
SOURCES = sorted(TRANSLATOR.glob("*.py"))


def test_translator_sources_exist():
    assert {p.name for p in SOURCES} >= {"app.py", "smoke_test.py"}


def test_app_imports_the_shared_pieces():
    source = (TRANSLATOR / "app.py").read_text(encoding="utf-8")
    for name in ("normalize_text", "GENERATION", "DIRECTIONS", "translate"):
        assert re.search(rf"from qomlaq\.\w+ import .*\b{name}\b", source), name


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_local_copy_of_a_pipeline_setting(path):
    source = path.read_text(encoding="utf-8")
    copies = {
        "language tag": r"""["'](grn_Latn|spa_Latn)["']""",
        "apostrophe table": r"APOSTROPHE_VARIANTS\s*=",
        "decoding setting": r"\b(num_beams|no_repeat_ngram_size|max_new_tokens)\s*=",
        "normalizer": r"def normalize_text\b",
    }
    found = [what for what, pattern in copies.items() if re.search(pattern, source)]
    assert not found, f"{path.name} defines its own {found}; import it from qomlaq"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_corpus_text_or_tokens_in_the_translator(path):
    source = path.read_text(encoding="utf-8")
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", source), "a Hugging Face token is in the source"
