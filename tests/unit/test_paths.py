"""Where runs write their outputs, and how a checkpoint is found again."""

from __future__ import annotations

import pytest

from qomlaq import paths
from qomlaq.experiments import load_experiment


def test_model_and_results_paths_are_unique_per_run(tmp_path):
    exp = load_experiment("baseline")
    seen = set()
    for run in exp.runs:
        model = paths.model_dir(exp.name, run.run_id, root=tmp_path)
        results = paths.results_dir(exp.name, run.run_id, run.split_id, "test", "f" * 64,
                                    root=tmp_path)
        assert model not in seen and results not in seen
        seen.update({model, results})


def test_the_same_run_id_in_two_experiments_never_collides(tmp_path):
    assert paths.model_dir("a", "r", tmp_path) != paths.model_dir("b", "r", tmp_path)


def test_results_for_two_splits_with_one_id_never_collide(tmp_path):
    """Two corpus versions can each have a base/stratified split; their results differ."""
    one = paths.results_dir("e", "r", "base/stratified", "test", "1" * 64, tmp_path)
    two = paths.results_dir("e", "r", "base/stratified", "test", "2" * 64, tmp_path)
    assert one != two


def test_find_model_returns_the_run_checkpoint(tmp_path):
    model = paths.model_dir("baseline", "base__stratified__es2qom", tmp_path) / "model"
    model.mkdir(parents=True)
    (model / "train_manifest.json").write_text("{}")
    assert paths.find_model("baseline", "base__stratified__es2qom", tmp_path) == model


def test_find_model_explains_a_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="train it first"):
        paths.find_model("baseline", "base__stratified__es2qom", tmp_path)


def test_locations_can_be_moved_with_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("QOMLAQ_ARTIFACTS_DIR", str(tmp_path / "a"))
    monkeypatch.setenv("QOMLAQ_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("QOMLAQ_EXPERIMENTS_DIR", str(tmp_path / "e"))
    assert paths.artifacts_dir() == tmp_path / "a"
    assert paths.data_dir() == tmp_path / "d"
    assert paths.experiments_dir() == tmp_path / "e"
    assert paths.model_dir("x", "y") == tmp_path / "a" / "models" / "x" / "y"
