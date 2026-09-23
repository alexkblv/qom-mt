"""Notebooks must stay thin.

Pipeline logic pasted into notebooks drifts apart from copy to copy. Keeping it in the
package is a test here, not a convention.

``legacy/`` is exempt: those notebooks are the historical record.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"

#: Doing any of these inside a notebook means a second implementation of something the
#: package already owns.
BANNED_TOKENS = (
    "read_excel",
    "read_csv",
    "GroupShuffleSplit",
    "train_test_split",
    "drop_duplicates",
    "corpus_chrf",
    "corpus_bleu",
    "sacrebleu.",
    "random.Random",
    "Seq2SeqTrainingArguments",
    "AutoModelForSeq2SeqLM",
    "drop_na",
    "load_split(",  # splits are loaded through qomlaq.pipeline, which asserts the pins
)


def notebooks() -> list[Path]:
    return sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def code_of(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def test_notebooks_exist():
    assert notebooks(), "no v2 notebooks found"


@pytest.mark.parametrize("path", notebooks(), ids=lambda p: p.name)
def test_notebook_is_valid_json_with_cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]


@pytest.mark.parametrize("path", notebooks(), ids=lambda p: p.name)
def test_notebook_has_no_pipeline_logic(path):
    source = code_of(path)
    offenders = [token for token in BANNED_TOKENS if token in source]
    assert not offenders, (
        f"{path.name} reimplements pipeline logic ({offenders}). Call the qomlaq package "
        "instead of keeping a second implementation."
    )


@pytest.mark.parametrize("path", notebooks(), ids=lambda p: p.name)
def test_notebook_starts_with_a_title(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    first = notebook["cells"][0]
    assert first["cell_type"] == "markdown"
    assert "".join(first["source"]).lstrip().startswith("# ")


@pytest.mark.parametrize("path", notebooks(), ids=lambda p: p.name)
def test_notebook_takes_splits_and_runs_from_an_experiment(path):
    """Hashes and run matrices live in experiments/; a notebook only names an experiment."""
    source = code_of(path)
    assert "EXPERIMENT" in source, f"{path.name} does not name an experiment"
    assert not re.search(r"\b[0-9a-f]{64}\b", source), f"{path.name} hard-codes a hash"


@pytest.mark.parametrize("path", notebooks(), ids=lambda p: p.name)
def test_no_hardcoded_absolute_paths(path):
    """Paths come from qomlaq.paths, so a notebook runs unchanged on any machine."""
    offenders = re.findall(r"""["'](/[A-Za-z][^"']*)["']""", code_of(path))
    assert not offenders, f"{path.name} hard-codes {offenders}"


def test_training_notebook_computes_no_metrics():
    """Training and scoring are separate stages, so a re-score never needs a retrain."""
    source = code_of(NOTEBOOK_DIR / "10_train.ipynb")
    for token in ("evaluate(", "score(", "chrf"):
        assert token not in source, f"10_train should not score ({token})"
