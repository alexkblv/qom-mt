"""Where the data, experiments and artifacts live.

Every location the pipeline reads or writes is defined here and nowhere else. The data,
the experiment files and the artifacts can each be moved with an environment variable,
which is all a different machine needs.

    artifacts/corpus/<corpus>/                     corpus table + manifest
    artifacts/splits/<corpus sha8>/<split id>/     partitions + manifest
    artifacts/models/<experiment>/<run>/model/     checkpoint + model card
    artifacts/results/<experiment>/<run>/<eval>/   eval record + hypotheses
    artifacts/tables/<experiment>/                 generated tables
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _located(explicit: str | Path | None, env: str, default: Path) -> Path:
    if explicit:
        return Path(explicit)
    if value := os.environ.get(env):
        return Path(value)
    return default


def for_manifest(path: str | Path) -> str:
    """A path as recorded in a manifest: relative to the repo when inside it."""
    path = Path(path).resolve()
    try:
        return path.relative_to(repo_root()).as_posix()
    except ValueError:
        return str(path)


def data_dir(explicit: str | Path | None = None) -> Path:
    """Directory holding the raw corpus sources."""
    return _located(explicit, "QOMLAQ_DATA_DIR", repo_root() / "corpus")


def experiments_dir(explicit: str | Path | None = None) -> Path:
    """Directory holding the experiment files."""
    return _located(explicit, "QOMLAQ_EXPERIMENTS_DIR", repo_root() / "experiments")


def artifacts_dir(explicit: str | Path | None = None) -> Path:
    """Directory for generated artifacts (corpus table, splits, results, tables)."""
    return _located(explicit, "QOMLAQ_ARTIFACTS_DIR", repo_root() / "artifacts")


def corpus_dir(corpus: str, root: str | Path | None = None) -> Path:
    return artifacts_dir(root) / "corpus" / corpus


def splits_root(root: str | Path | None = None) -> Path:
    return artifacts_dir(root) / "splits"


def split_dir(corpus_sha8: str, split_id: str, root: str | Path | None = None) -> Path:
    return splits_root(root) / corpus_sha8 / split_id


def model_dir(experiment: str, run_id: str, root: str | Path | None = None) -> Path:
    """Output directory for one training run. Unique per run by construction."""
    return artifacts_dir(root) / "models" / experiment / run_id


def results_dir(experiment: str, run_id: str, split_id: str, partition: str,
                split_sha256: str, root: str | Path | None = None) -> Path:
    """Output directory for one evaluation: which model, on exactly which data.

    The split hash is part of the name, so two splits that share an id (built from
    different corpus versions) can never overwrite each other's results.
    """
    name = f"{split_id.replace('/', '__')}__{partition}__{split_sha256[:8]}"
    return artifacts_dir(root) / "results" / experiment / run_id / name


def tables_dir(experiment: str, root: str | Path | None = None) -> Path:
    return artifacts_dir(root) / "tables" / experiment


def find_model(experiment: str, run_id: str, root: str | Path | None = None) -> Path:
    """The trained checkpoint of one run, or a clear error if it isn't there."""
    path = model_dir(experiment, run_id, root) / "model"
    if not (path / "train_manifest.json").exists():
        raise FileNotFoundError(
            f"no checkpoint for {experiment}/{run_id} at {path}; train it first, or copy it "
            "there from the machine that trained it."
        )
    return path
