from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "corpus"


def pytest_collection_modifyitems(config, items):
    """Skip corpus-backed tests when the raw sources are not present.

    The corpus is gitignored (copyright), so a fresh clone runs the unit tests only.
    """
    if CORPUS_DIR.exists():
        return
    skip = pytest.mark.skip(reason="raw corpus/ not present")
    for item in items:
        if "corpus" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def corpus_dir() -> Path:
    if not CORPUS_DIR.exists():
        pytest.skip("raw corpus/ not present")
    return CORPUS_DIR


@pytest.fixture(scope="session")
def built_corpus(corpus_dir: Path):
    """The corpus the baseline experiment uses."""
    from qomlaq.corpus import build_corpus
    from qomlaq.experiments import load_experiment

    return build_corpus(corpus_dir, load_experiment("baseline").corpus)


@pytest.fixture(scope="session")
def splits(built_corpus):
    """Every split the baseline experiment declares, built once."""
    from qomlaq.experiments import load_experiment
    from qomlaq.splits import make_split

    out = {}
    for spec in load_experiment("baseline").splits:
        parts, manifest, report = make_split(
            built_corpus,
            config=spec.config,
            strategy=spec.strategy,
            bible_grouping=spec.bible_grouping,
            purpose=spec.purpose,
        )
        out[manifest["split_id"]] = (parts, manifest, report)
    return out


@pytest.fixture(scope="session")
def headline_splits(splits):
    return {k: v for k, v in splits.items() if v[1]["purpose"] == "headline"}


@pytest.fixture
def toy_frame() -> pd.DataFrame:
    """Minimal canonical-shaped frame for unit tests."""
    rows = [
        ("u1", "qom one", "es one", "S", "S__frag__a", "fragmento", "line"),
        ("u2", "qom two", "es two", "S", "S__frag__a", "fragmento", "line"),
        ("u3", "qom three", "es three", "S", "S__frag__b", "fragmento", "line"),
        ("u4", "qom four", "es four", "S", "S__frag__b", "fragmento", "line"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["pair_uid", "qom", "es", "source_doc", "group_id", "unit_level", "record_kind"],
    )
    frame["norm_key"] = frame["qom"] + "\x1f" + frame["es"]
    return frame
