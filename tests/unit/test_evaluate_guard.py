"""The contamination guard.

These tests pin the guard to a precondition that cannot be bypassed by accident.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from qomlaq.experiments import RunSpec
from qomlaq.evaluate import (
    ContaminatedEvaluation,
    EvalRecord,
    EvaluationError,
    ModelCard,
    evaluate,
    rescore,
)
from qomlaq.metrics import PRIMARY
from qomlaq.splits import Split


def _split(train_rows, test_rows) -> Split:
    def frame(rows):
        df = pd.DataFrame(
            rows, columns=["pair_uid", "qom", "es", "source_doc", "group_id", "source_ref"]
        )
        df["unit_level"] = "fragmento"
        df["record_kind"] = "line"
        return df

    manifest = {
        "split_id": "base/stratified",
        "split_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "corpus_sha8": "b" * 8,
        "target_fractions": {"train": 0.8, "dev": 0.1, "test": 0.1},
        "purpose": "headline",
    }
    return Split(
        train=frame(train_rows), dev=frame([]), test=frame(test_rows), manifest=manifest
    )


def _card(split_id: str = "base/stratified") -> ModelCard:
    return ModelCard(
        run_id="base__stratified__es2qom",
        config="base",
        strategy="stratified",
        bible_grouping="chapter",
        direction="es2qom",
        src_lang="spa_Latn",
        tgt_lang="grn_Latn",
        split_id=split_id,
        split_sha256="a" * 64,
        corpus_sha256="b" * 64,
        corpus_sha8="b" * 8,
        base_model="facebook/nllb-200-distilled-600M",
    )


TRAIN = [("t1", "qom uno", "es uno", "S", "g1", "r1")]
TEST = [
    ("e1", "qom dos", "es dos", "S", "g2", "r2"),
    ("e2", "qom tres", "es tres", "S", "g3", "r3"),
]


def test_clean_evaluation_produces_a_record():
    split = _split(TRAIN, TEST)
    record, dump = evaluate(
        ["qom dos", "qom tres"],
        split=split,
        partition="test",
        model_card=_card(),
        train_uids={"t1"},
    )
    assert record.contaminated is False
    assert record.n_pairs == 2
    assert record.primary_metric == PRIMARY.display_name
    assert record.primary == pytest.approx(100.0, abs=1e-6)
    assert list(dump["pair_uid"]) == ["e1", "e2"]


def test_evaluation_on_training_data_is_refused():
    split = _split(TRAIN, TEST)
    with pytest.raises(ContaminatedEvaluation) as excinfo:
        evaluate(
            ["a", "b"],
            split=split,
            partition="test",
            model_card=_card(),
            train_uids={"t1", "e1"},  # e1 is in the test partition
        )
    assert "training data" in str(excinfo.value)


def test_contamination_can_be_recorded_but_must_be_explicit():
    split = _split(TRAIN, TEST)
    record, _ = evaluate(
        ["a", "b"],
        split=split,
        partition="test",
        model_card=_card(),
        train_uids={"e1"},
        allow_overlap=True,
    )
    assert record.contaminated is True
    assert record.overlap["overlap"] == 1


def test_guard_runs_before_scoring():
    """A refusal must not depend on the scorer succeeding first."""
    split = _split(TRAIN, TEST)
    with pytest.raises(ContaminatedEvaluation):
        evaluate(
            ["only-one-hypothesis-would-also-be-wrong", "x"],
            split=split,
            partition="test",
            model_card=_card(),
            train_uids={"e1", "e2"},
        )


def test_hypothesis_count_mismatch_is_rejected():
    """An off-by-one between generation and the split must not score silently."""
    split = _split(TRAIN, TEST)
    with pytest.raises(EvaluationError):
        evaluate(["only one"], split=split, partition="test", model_card=_card(), train_uids=set())


def test_cross_split_evaluation_is_flagged():
    """Ablation evaluates a model against another split; the record says so."""
    split = _split(TRAIN, TEST)
    record, _ = evaluate(
        ["a", "b"],
        split=split,
        partition="test",
        model_card=_card(split_id="bible_only/random__chapter"),
        train_uids=set(),
    )
    assert record.cross_split is True


def test_record_round_trips_and_rescores_from_the_dump(tmp_path):
    """Every future metric question is a CPU rescore, not a retrain."""
    split = _split(TRAIN, TEST)
    record, dump = evaluate(
        ["qom dos", "casi tres"],
        split=split,
        partition="test",
        model_card=_card(),
        train_uids={"t1"},
    )
    record.save(tmp_path, dump)

    reloaded = EvalRecord.load(tmp_path)
    assert reloaded.scores == record.scores

    recomputed = rescore(tmp_path)
    assert recomputed["scores"][PRIMARY.display_name] == pytest.approx(
        record.scores[PRIMARY.display_name], abs=1e-6
    )


def test_per_source_breakdown_is_recorded():
    """Base+Bible test sets are ~90% Bible; a single corpus score hides that."""
    rows = [
        ("e1", "q1", "e1es", "La Biblia", "g2", "GEN 1:1"),
        ("e2", "q2", "e2es", "Arte verbal qom", "g3", "arte_1"),
    ]
    split = _split(TRAIN, rows)
    record, _ = evaluate(
        ["e1es", "wrong"], split=split, partition="test", model_card=_card(), train_uids={"t1"}
    )
    assert set(record.per_source) == {"La Biblia", "Arte verbal qom"}
    assert record.per_source["La Biblia"] > record.per_source["Arte verbal qom"]


def test_model_card_round_trips(tmp_path):
    card = _card()
    card.save(tmp_path)
    assert ModelCard.load(tmp_path).run_id == card.run_id


def test_missing_model_card_is_an_error(tmp_path):
    """A checkpoint whose training data is unknown cannot be evaluated safely."""
    with pytest.raises(EvaluationError):
        ModelCard.load(tmp_path)


def test_model_card_written_before_experiments_still_loads(tmp_path):
    """Cards from before experiments existed lack the experiment fields; they default."""
    card = _card()
    record = json.loads(json.dumps(card.__dict__))
    for key in ("experiment", "model", "init_from"):
        record.pop(key)
    (tmp_path / "train_manifest.json").write_text(json.dumps(record), encoding="utf-8")
    loaded = ModelCard.load(tmp_path)
    assert loaded.experiment == "" and loaded.run_id == card.run_id


def test_run_spec_direction_wiring():
    """grn_Latn is the Qom proxy tag; spa_Latn is Spanish. Both directions, both ways."""
    es2qom = RunSpec("demo", "base", "stratified", "es2qom")
    qom2es = RunSpec("demo", "base", "stratified", "qom2es")
    assert (es2qom.src_lang, es2qom.tgt_lang) == ("spa_Latn", "grn_Latn")
    assert (es2qom.src_col, es2qom.tgt_col) == ("es", "qom")
    assert (qom2es.src_lang, qom2es.tgt_lang) == ("grn_Latn", "spa_Latn")
    assert (qom2es.src_col, qom2es.tgt_col) == ("qom", "es")
