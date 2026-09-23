"""The pipeline's operations against the real corpus, in a scratch artifacts directory.

Skipped when ``corpus/`` is absent.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from qomlaq.cli import main
from qomlaq.experiments import load_experiment
from qomlaq.pipeline import PinError, build, load_pinned_split, report, run_status

pytestmark = pytest.mark.corpus


@pytest.fixture(scope="module")
def built(corpus_dir, tmp_path_factory):
    artifacts = tmp_path_factory.mktemp("artifacts")
    return build("baseline", data_dir=corpus_dir, artifacts_dir=artifacts, verbose=False), artifacts


def test_baseline_builds_exactly_the_pinned_splits(built):
    """Building must reproduce every pinned hash; a mismatch would raise."""
    result, _ = built
    for spec in result.experiment.splits:
        assert result.split_sha256(spec.split_id) == spec.sha256


def test_a_changed_split_fails_the_build(corpus_dir, built):
    _, artifacts = built
    exp = load_experiment("baseline")
    wrong = dataclasses.replace(exp.splits[0], sha256="0" * 64)
    tampered = dataclasses.replace(exp, splits=(wrong, *exp.splits[1:]))
    with pytest.raises(PinError, match="differ from their pinned hashes"):
        build(tampered, data_dir=corpus_dir, artifacts_dir=artifacts, verbose=False)


def test_an_unpinned_split_cannot_be_loaded_for_training(built):
    result, artifacts = built
    unpinned = dataclasses.replace(result.experiment.splits[0], sha256=None)
    exp = dataclasses.replace(result.experiment, splits=(unpinned, *result.experiment.splits[1:]))
    with pytest.raises(PinError, match="isn't pinned"):
        load_pinned_split(dataclasses.replace(result, experiment=exp), unpinned.split_id,
                          artifacts_dir=artifacts)


def test_a_pinned_split_loads(built):
    result, artifacts = built
    split = load_pinned_split(result, "base/stratified", artifacts_dir=artifacts)
    assert split.sha256 == result.experiment.split("base/stratified").sha256


def test_report_writes_the_corpus_and_split_tables(built):
    _, artifacts = built
    written = report("baseline", artifacts_dir=artifacts)
    assert {"table1_corpus", "table2_splits", "macros", "numbers"} <= set(written)
    numbers = json.loads(written["numbers"].read_text(encoding="utf-8"))
    assert numbers["experiment"] == "baseline"
    assert numbers["splits"]["base/stratified"] == load_experiment("baseline").split(
        "base/stratified").sha256
    table1 = (written["table1_corpus"].with_suffix(".csv")).read_text(encoding="utf-8")
    assert "Total (base)" in table1 and "Total (base_bible)" in table1


def test_run_status_lists_every_run_untrained(built):
    _, artifacts = built
    status = run_status("baseline", artifacts_dir=artifacts)
    assert len(status) == 12
    assert set(status["trained"]) == {""}


def test_cli_splits_shows_pinned_splits(built, capsys):
    _, artifacts = built
    assert main(["splits", "baseline", "--artifacts-dir", str(artifacts)]) == 0
    out = capsys.readouterr().out
    assert "base/stratified" in out and "yes" in out


def test_cli_train_refuses_without_run_ids(capsys):
    assert main(["train", "baseline"]) == 2


def test_rebuilding_leaves_unchanged_splits_untouched(corpus_dir, built):
    """Only provenance would change, so tracked manifests are not rewritten."""
    _, artifacts = built
    manifests = sorted((artifacts / "splits").rglob("split_manifest.json"))
    before = {p: p.read_bytes() for p in manifests}
    build("baseline", data_dir=corpus_dir, artifacts_dir=artifacts, verbose=False)
    assert {p: p.read_bytes() for p in manifests} == before


def test_a_rebuild_from_data_outside_the_repo_leaves_the_corpus_manifest(corpus_dir, built,
                                                                         tmp_path):
    """Another machine keeps its data elsewhere; the tracked manifest must not change."""
    import shutil

    _, artifacts = built
    manifest = artifacts / "corpus" / "text-v1" / "corpus_manifest.json"
    before = manifest.read_bytes()
    elsewhere = tmp_path / "data"
    shutil.copytree(corpus_dir, elsewhere)
    build("baseline", data_dir=elsewhere, artifacts_dir=artifacts, rebuild=True, verbose=False)
    assert manifest.read_bytes() == before
