"""Experiment files: parsing, validation, and everything derived from them."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from qomlaq.config import DEFAULT_SPLITS, DIRECTIONS, TRAINING, config_has_verses
from qomlaq.experiments import (
    ExperimentError,
    RunRef,
    list_experiments,
    load_experiment,
    parse_experiment,
)
from qomlaq.splits import allocate_groups

MINIMAL = {
    "corpus": "text-v1",
    "model": "nllb-600m",
    "directions": ["es2qom", "qom2es"],
    "splits": [
        {"config": "base", "strategy": "stratified"},
        {"config": "base_bible", "strategy": "stratified", "bible_grouping": "chapter"},
    ],
}


def _with(**changes) -> dict:
    data = copy.deepcopy(MINIMAL)
    data.update(changes)
    return data


# ------------------------------------------------------------------------ baseline


def test_baseline_is_listed_and_loads():
    assert "baseline" in list_experiments()
    exp = load_experiment("baseline")
    assert exp.corpus == "text-v1" and exp.model == "nllb-600m"


def test_baseline_pins_every_split():
    exp = load_experiment("baseline")
    assert all(spec.sha256 for spec in exp.splits)


def test_baseline_runs_are_every_split_in_every_direction():
    exp = load_experiment("baseline")
    assert len(exp.runs) == len(exp.splits) * len(exp.directions) == 12
    assert len(exp.runs_by_id) == len(exp.runs), "run ids must be unique"
    assert {r.split_id for r in exp.runs} == {s.split_id for s in exp.splits}


def test_baseline_run_ids():
    ids = set(load_experiment("baseline").runs_by_id)
    assert {"base__stratified__es2qom", "bible_only__random__qom2es",
            "base_bible__stratified__es2qom__verse"} <= ids


def test_evaluation_references_resolve():
    exp = load_experiment("baseline")
    evaluation = exp.evaluation("domain-shift")
    assert evaluation.split_id == "base/stratified"
    assert all(ref.experiment == "baseline" and ref.run_id in exp.runs_by_id
               for ref in evaluation.runs)


# ---------------------------------------------------------------------- derivation


def test_split_ids_name_bible_grouping_only_where_the_bible_is():
    exp = parse_experiment("demo", MINIMAL)
    assert [s.split_id for s in exp.splits] == ["base/stratified", "base_bible/stratified__chapter"]
    assert not config_has_verses("base") and config_has_verses("base_bible")


def test_run_directions_come_from_one_table():
    exp = parse_experiment("demo", MINIMAL)
    for run in exp.runs:
        direction = DIRECTIONS[run.direction]
        assert (run.src_lang, run.tgt_lang) == (direction.src_lang, direction.tgt_lang)
        assert (run.src_col, run.tgt_col) == (direction.src_col, direction.tgt_col)


def test_training_defaults_and_overrides():
    assert parse_experiment("demo", MINIMAL).training == TRAINING
    exp = parse_experiment("demo", _with(training={"learning_rate": 3e-4, "num_train_epochs": 4}))
    assert exp.training.learning_rate == 3e-4 and exp.training.num_train_epochs == 4
    assert exp.training.seed == TRAINING.seed


def test_integer_learning_rate_is_accepted_as_float():
    exp = parse_experiment("demo", _with(training={"learning_rate": 1}))
    assert exp.training.learning_rate == 1.0 and isinstance(exp.training.learning_rate, float)


def test_init_from_a_model_id():
    exp = parse_experiment("demo", _with(init_from="facebook/nllb-200-1.3B"))
    assert exp.init_from.describe("es2qom") == "facebook/nllb-200-1.3B"


def test_init_from_another_run_fills_in_the_direction():
    exp = parse_experiment(
        "demo", _with(init_from={"experiment": "baseline", "run": "base__stratified__{direction}"})
    )
    assert exp.init_from.run == RunRef("baseline", "base__stratified__{direction}")
    assert exp.init_from.describe("qom2es") == "baseline/base__stratified__qom2es"


def test_evaluation_can_name_a_run_from_another_experiment():
    exp = parse_experiment("demo", _with(evaluations=[{
        "name": "compare", "split": "base/stratified",
        "runs": ["base__stratified__es2qom", "baseline/base__stratified__es2qom"],
    }]))
    assert exp.evaluation("compare").runs == (
        RunRef("demo", "base__stratified__es2qom"),
        RunRef("baseline", "base__stratified__es2qom"),
    )


# ---------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "data, fragment",
    [
        (_with(corpus="text-v9"), "corpus"),
        (_with(model="nllb-9b"), "model"),
        (_with(directions=["es2qom", "es2qom"]), "directions"),
        (_with(directions=["es2gn"]), "directions"),
        (_with(splits=[]), "splits"),
        (_with(splits=[{"config": "speech", "strategy": "stratified"}]), "config"),
        (_with(splits=[{"config": "base", "strategy": "shuffled"}]), "strategy"),
        (_with(splits=[{"config": "base", "strategy": "stratified", "bible_grouping": "verse"}]),
         "bible_grouping"),
        (_with(splits=[{"config": "base", "strategy": "stratified", "sha256": "abc"}]), "sha256"),
        (_with(splits=[{"config": "base", "strategy": "stratified", "purpose": "main"}]), "purpose"),
        (_with(splits=[{"config": "base", "strategy": "stratified"}] * 2), "more than once"),
        (_with(splits=[{"config": "base", "strategy": "stratified", "stratgy": "x"}]), "unknown key"),
        (_with(training={"learning_rat": 1e-4}), "unknown key"),
        (_with(training={"num_train_epochs": "ten"}), "num_train_epochs"),
        (_with(init_from=3), "init_from"),
        (_with(evaluations=[{"name": "x", "split": "base/random", "runs": []}]), "split"),
        (_with(evaluations=[{"name": "x", "split": "base/stratified", "runs": ["nope"]}]),
         "no such run"),
        (_with(colour="red"), "unknown key"),
    ],
)
def test_mistakes_are_reported_with_their_key(data, fragment):
    with pytest.raises(ExperimentError, match=fragment):
        parse_experiment("demo", data)


def test_unknown_experiment_names_the_available_ones():
    with pytest.raises(ExperimentError, match="baseline"):
        load_experiment("no-such-experiment")


def test_unknown_run_points_to_the_run_list():
    with pytest.raises(ExperimentError, match="qomlaq runs"):
        load_experiment("baseline").run("base__stratified__es2gn")


# ------------------------------------------------------------ corpus growth


def test_adding_a_source_moves_no_unit_of_a_stratified_split():
    """Each source is packed on its own, so a new source leaves every old unit in place.

    Random splits do not have this property. Stratified splits are what keep old results
    comparable when new data arrives (as long as the new source repeats no existing pair).
    """
    groups = pd.DataFrame({
        "group_id": [f"a{i}" for i in range(30)] + [f"b{i}" for i in range(12)],
        "stratum": ["A"] * 30 + ["B"] * 12,
        "n_rows": [1 + (i * 7) % 40 for i in range(30)] + [2 + (i * 5) % 17 for i in range(12)],
    })
    new_source = pd.DataFrame({
        "group_id": [f"c{i}" for i in range(20)],
        "stratum": ["C"] * 20,
        "n_rows": [3 + (i * 11) % 25 for i in range(20)],
    })
    grown = pd.concat([groups, new_source], ignore_index=True)
    fractions = DEFAULT_SPLITS.as_dict()

    before = allocate_groups(groups, fractions, stratify=True)
    after = allocate_groups(grown, fractions, stratify=True)
    assert all(after[g] == p for g, p in before.items())

    moved = allocate_groups(grown, fractions, stratify=False)
    unstratified = allocate_groups(groups, fractions, stratify=False)
    assert any(moved[g] != p for g, p in unstratified.items()), "random packing reshuffles"
