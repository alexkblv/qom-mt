"""Scoring a model against a split, and the record that results.

The guard that matters lives here. :func:`evaluate` refuses to score an evaluation set
that overlaps the model's own training data unless the caller explicitly asks for it,
and it does so as a *precondition* -- before any score is computed, inside the function
that produces the record, where no caller can skip it.

Every record also carries a per-row hypothesis dump. One GPU pass then answers every
future metric question on CPU: per-source breakdowns, a different chrF parameterisation,
word-frequency analyses, a reviewer asking for BLEU with a different tokenizer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from . import __version__, code_fingerprint, runtime_versions
from .checks import CheckResult, check_overlap_with_train
from .config import DIRECTIONS, GENERATION
from .experiments import RunSpec
from .metrics import DEFAULT_METRICS, PRIMARY, MetricSpec, score, score_by_group
from .splits import Split

RECORD_NAME = "eval_record.json"
HYPOTHESES_NAME = "hypotheses.csv"
CARD_NAME = "train_manifest.json"
SCHEMA_VERSION = "2.0.0"


class EvaluationError(RuntimeError):
    pass


class ContaminatedEvaluation(EvaluationError):
    """Raised when an evaluation set overlaps the evaluated model's training data."""


@dataclass(frozen=True)
class ModelCard:
    """What a trained checkpoint records about its own provenance.

    Written next to the weights at training time. Without it there is no way to ask
    "did this model see these sentences?"
    """

    run_id: str
    config: str
    strategy: str
    bible_grouping: str
    direction: str
    src_lang: str
    tgt_lang: str
    split_id: str
    split_sha256: str
    corpus_sha256: str
    corpus_sha8: str
    base_model: str
    training: dict = field(default_factory=dict)
    selection: dict = field(default_factory=dict)
    package_version: str = __version__
    code_fingerprint: str = ""
    runtime: dict = field(default_factory=dict)
    experiment: str = ""
    #: Model profile name, e.g. "nllb-600m".
    model: str = ""
    #: The checkpoint training started from, when it wasn't the base model.
    init_from: str = ""

    @classmethod
    def for_run(cls, run: RunSpec, split: Split, *, base_model: str, training: dict,
                selection: dict | None = None, model: str = "",
                init_from: str | None = None) -> "ModelCard":
        return cls(
            run_id=run.run_id,
            config=run.config,
            strategy=run.strategy,
            bible_grouping=run.bible_grouping,
            direction=run.direction,
            src_lang=run.src_lang,
            tgt_lang=run.tgt_lang,
            split_id=split.split_id,
            split_sha256=split.sha256,
            corpus_sha256=split.corpus_sha256,
            corpus_sha8=split.manifest["corpus_sha8"],
            base_model=base_model,
            training=training,
            selection=selection or {},
            code_fingerprint=code_fingerprint(),
            runtime=runtime_versions(),
            experiment=run.experiment,
            model=model,
            init_from=init_from or "",
        )

    def save(self, model_dir: str | Path) -> Path:
        path = Path(model_dir) / CARD_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, model_dir: str | Path) -> "ModelCard":
        path = Path(model_dir) / CARD_NAME
        if not path.exists():
            raise EvaluationError(
                f"no {CARD_NAME} in {model_dir}. A checkpoint without a model card cannot "
                "be evaluated safely, because there is no way to tell what it trained on."
            )
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class EvalRecord:
    schema_version: str
    evaluated_utc: str
    run_id: str
    model_card: dict
    split_id: str
    split_sha256: str
    partition: str
    cross_split: bool
    n_pairs: int
    overlap: dict
    contaminated: bool
    scores: dict
    signatures: dict
    per_source: dict
    generation: dict
    primary_metric: str
    runtime: dict
    experiment: str = ""

    @property
    def primary(self) -> float:
        return self.scores[self.primary_metric]

    def save(self, results_dir: str | Path, hypotheses: pd.DataFrame | None = None) -> Path:
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / RECORD_NAME).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if hypotheses is not None:
            hypotheses.to_csv(out / HYPOTHESES_NAME, index=False)
        return out

    @classmethod
    def load(cls, results_dir: str | Path) -> "EvalRecord":
        path = Path(results_dir) / RECORD_NAME
        if not path.exists():
            raise EvaluationError(f"no {RECORD_NAME} in {results_dir}")
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def summary_row(self) -> dict:
        return {
            "experiment": self.experiment,
            "run_id": self.run_id,
            "config": self.model_card.get("config"),
            "strategy": self.model_card.get("strategy"),
            "bible_grouping": self.model_card.get("bible_grouping"),
            "direction": self.model_card.get("direction"),
            "split_id": self.split_id,
            "partition": self.partition,
            "n_pairs": self.n_pairs,
            "contaminated": self.contaminated,
            **self.scores,
        }


def check_evaluation_is_held_out(
    evaluated: pd.DataFrame, train_uids: set[str], *, label: str
) -> CheckResult:
    return check_overlap_with_train(evaluated, train_uids, label=label)


def evaluate(
    hypotheses: Sequence[str],
    *,
    split: Split,
    partition: str,
    model_card: ModelCard,
    train_uids: set[str],
    metrics: Sequence[MetricSpec] = DEFAULT_METRICS,
    allow_overlap: bool = False,
    generation: dict | None = None,
) -> tuple[EvalRecord, pd.DataFrame]:
    """Score ``hypotheses`` against a partition and build the record.

    ``train_uids`` are the pair ids the evaluated model trained on -- normally
    ``load_split(...).train_uids`` for the model's own split, which the caller resolves
    from :attr:`ModelCard.split_id`. Passing an empty set is not a way around the guard;
    it just means the caller asserted the model trained on nothing.
    """
    frame = split.partition(partition)
    if len(hypotheses) != len(frame):
        raise EvaluationError(
            f"{len(hypotheses)} hypotheses for {len(frame)} rows in {split.split_id}/"
            f"{partition}. Generation must preserve row order and count."
        )

    run = model_card
    direction = DIRECTIONS[run.direction]
    src_col, tgt_col = direction.src_col, direction.tgt_col

    dump = pd.DataFrame(
        {
            "pair_uid": frame["pair_uid"].to_numpy(),
            "source_doc": frame["source_doc"].to_numpy(),
            "source_ref": frame["source_ref"].to_numpy(),
            "src": frame[src_col].to_numpy(),
            "ref": frame[tgt_col].to_numpy(),
            "hyp": list(hypotheses),
        }
    )

    guard = check_evaluation_is_held_out(
        frame, train_uids, label=f"{run.run_id} on {split.split_id}/{partition}"
    )
    if not guard.passed and not allow_overlap:
        raise ContaminatedEvaluation(
            f"{guard.detail}\n"
            "Scoring a model on its own training data measures memorisation, not "
            "translation. Evaluate on a partition the model did not train on, or pass "
            "allow_overlap=True to record the number as explicitly contaminated."
        )

    scored = score(dump["hyp"].tolist(), dump["ref"].tolist(), metrics=metrics)
    per_source = score_by_group(
        dump["hyp"].tolist(), dump["ref"].tolist(), dump["source_doc"].tolist(), metric=PRIMARY
    )

    record = EvalRecord(
        schema_version=SCHEMA_VERSION,
        evaluated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        run_id=run.run_id,
        model_card=asdict(run),
        split_id=split.split_id,
        split_sha256=split.sha256,
        partition=partition,
        cross_split=split.split_id != run.split_id,
        n_pairs=len(dump),
        overlap=guard.value,
        contaminated=not guard.passed,
        scores=scored.scores,
        signatures=scored.signatures,
        per_source=per_source,
        generation=generation or GENERATION.as_dict(),
        primary_metric=PRIMARY.display_name,
        runtime=runtime_versions(),
        experiment=run.experiment,
    )
    return record, dump


def rescore(results_dir: str | Path, *, metrics: Sequence[MetricSpec] = DEFAULT_METRICS) -> dict:
    """Recompute scores from a stored hypothesis dump, on CPU.

    This is what makes a metric question cheap forever: the corrected chrF++ for any
    past run is a rescore, not a retrain.
    """
    results_dir = Path(results_dir)
    dump_path = results_dir / HYPOTHESES_NAME
    if not dump_path.exists():
        raise EvaluationError(f"no {HYPOTHESES_NAME} in {results_dir}")
    dump = pd.read_csv(dump_path, keep_default_na=False)
    recomputed = score(dump["hyp"].tolist(), dump["ref"].tolist(), metrics=metrics)
    return recomputed.as_dict()
