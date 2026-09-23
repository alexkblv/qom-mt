"""The pipeline's operations, one function each.

The CLI and the notebooks call these and nothing else, so running a stage from a
terminal and from a notebook does exactly the same thing.

    build            the experiment's corpus and splits, checked against their pins
    train            one run (needs the [train] extra and a GPU)
    evaluate_run     one run on its own split's test set
    run_evaluation   one of the experiment's named evaluations
    report           every table for one experiment

Building is cheap (seconds) and deterministic, so training and evaluation rebuild first:
a run can never be handed a split built from different sources.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import paths
from .corpus import MANIFEST_NAME as CORPUS_MANIFEST
from .corpus import CorpusArtifact, build_corpus, load_corpus, save_corpus
from .experiments import Experiment, RunSpec, load_experiment
from .splits import MANIFEST_NAME as SPLIT_MANIFEST
from .splits import Split, load_split, write_split


class PinError(RuntimeError):
    """A split differs from the hash its experiment pins, or has no pin yet."""


@dataclass(frozen=True)
class Built:
    experiment: Experiment
    corpus: CorpusArtifact
    manifests: dict[str, dict]

    def split_sha256(self, split_id: str) -> str:
        return self.manifests[split_id]["split_sha256"]

    @property
    def unpinned(self) -> list[str]:
        return [s.split_id for s in self.experiment.splits if not s.sha256]


def _experiment(experiment: str | Experiment) -> Experiment:
    return experiment if isinstance(experiment, Experiment) else load_experiment(experiment)


def _pin_help(experiment: str, split_id: str, sha256: str) -> str:
    return f'{split_id}: sha256 = "{sha256}"   (in experiments/{experiment}.toml)'


# --------------------------------------------------------------------------- build


def build(
    experiment: str | Experiment,
    *,
    data_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    rebuild: bool = False,
    verbose: bool = True,
) -> Built:
    """Build (or load) the experiment's corpus and write every split it declares.

    A split whose hash differs from its pin fails the build: the data or the split method
    changed, and results on that split would no longer be comparable.
    """
    exp = _experiment(experiment)
    data_dir = paths.data_dir(data_dir)
    artifacts_dir = paths.artifacts_dir(artifacts_dir)

    corpus: CorpusArtifact | None = None
    if not rebuild:
        try:
            corpus = load_corpus(artifacts_dir, exp.corpus)
            if verbose:
                print(f"corpus: loaded {exp.corpus} {corpus.sha8} from {artifacts_dir}")
        except Exception:
            corpus = None
    if corpus is None:
        corpus = build_corpus(data_dir, exp.corpus)
        save_corpus(corpus, artifacts_dir)
        if verbose:
            print(f"corpus: built {exp.corpus} {corpus.sha8} from {data_dir}")

    manifests: dict[str, dict] = {}
    mismatched: dict[str, tuple[str, str]] = {}
    for spec in exp.splits:
        manifest = write_split(
            corpus,
            artifacts_dir,
            config=spec.config,
            strategy=spec.strategy,
            bible_grouping=spec.bible_grouping,
            purpose=spec.purpose,
        )
        manifests[spec.split_id] = manifest
        sha = manifest["split_sha256"]
        if spec.sha256 and sha != spec.sha256:
            mismatched[spec.split_id] = (spec.sha256, sha)
        if verbose:
            realized = manifest["realized"]
            status = "pinned" if spec.sha256 == sha else ("MISMATCH" if spec.sha256 else "not pinned")
            print(
                f"split : {spec.split_id:34s} {realized['train']['rows']:6d}/"
                f"{realized['dev']['rows']:5d}/{realized['test']['rows']:5d}  "
                f"[{spec.purpose}] {sha[:8]} {status}"
            )

    if mismatched:
        lines = "\n".join(
            f"  {split_id}: pinned {pinned[:8]}, built {actual[:8]}"
            for split_id, (pinned, actual) in mismatched.items()
        )
        raise PinError(
            f"{exp.name}: split(s) differ from their pinned hashes:\n{lines}\n"
            "The data or the split method changed. If that is intended, update the pins; "
            "results on the old splits are no longer comparable with new ones."
        )

    built = Built(exp, corpus, manifests)
    if verbose and built.unpinned:
        print("\nnot pinned yet; pin these before training:")
        for split_id in built.unpinned:
            print("  " + _pin_help(exp.name, split_id, built.split_sha256(split_id)))
    return built


def load_pinned_split(
    built: Built, split_id: str, *, artifacts_dir: str | Path | None = None
) -> Split:
    """Load one of the experiment's splits, asserting the hash its experiment pins."""
    spec = built.experiment.split(split_id)
    if not spec.sha256:
        raise PinError(
            f"{built.experiment.name}: split {split_id} isn't pinned. Agree on it, then add\n  "
            + _pin_help(built.experiment.name, split_id, built.split_sha256(split_id))
        )
    return load_split(
        paths.artifacts_dir(artifacts_dir),
        config=spec.config,
        strategy=spec.strategy,
        bible_grouping=spec.bible_grouping,
        expect_split_sha256=spec.sha256,
        corpus_sha8=built.corpus.sha8,
        expect_corpus_sha256=built.corpus.sha256,
    )


def load_manifests(
    experiment: str | Experiment, *, artifacts_dir: str | Path | None = None
) -> tuple[dict, list[dict]]:
    """The experiment's corpus manifest and split manifests, as last built on this machine."""
    exp = _experiment(experiment)
    artifacts_dir = paths.artifacts_dir(artifacts_dir)
    corpus_path = paths.corpus_dir(exp.corpus, artifacts_dir) / CORPUS_MANIFEST
    if not corpus_path.exists():
        raise FileNotFoundError(f"{exp.corpus} isn't built here; run `qomlaq build {exp.name}`")
    corpus_manifest = json.loads(corpus_path.read_text(encoding="utf-8"))
    sha8 = corpus_manifest["corpus_sha256"][:8]

    manifests = []
    for spec in exp.splits:
        path = paths.split_dir(sha8, spec.split_id, artifacts_dir) / SPLIT_MANIFEST
        if not path.exists():
            raise FileNotFoundError(f"{spec.split_id} isn't built here; run `qomlaq build {exp.name}`")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if spec.sha256 and manifest["split_sha256"] != spec.sha256:
            raise PinError(
                f"{spec.split_id} on disk is {manifest['split_sha256'][:8]}, pinned "
                f"{spec.sha256[:8]}; run `qomlaq build {exp.name}` to see why"
            )
        manifests.append(manifest)
    return corpus_manifest, manifests


# ------------------------------------------------------------------ train, evaluate


def _init_source(exp: Experiment, run: RunSpec, artifacts_dir: Path) -> tuple[str | None, str | None]:
    """What to load weights from, and how to record it in the model card."""
    if exp.init_from is None:
        return None, None
    label = exp.init_from.describe(run.direction)
    if exp.init_from.model:
        return exp.init_from.model, label
    ref = exp.init_from.run
    path = paths.find_model(ref.experiment, ref.run_id.format(direction=run.direction), artifacts_dir)
    return str(path), label


def train(
    experiment: str | Experiment,
    run_id: str,
    *,
    data_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    resume: bool = False,
):
    """Train one run of an experiment. Returns its model card."""
    from .train import train_run

    artifacts_dir = paths.artifacts_dir(artifacts_dir)
    built = build(experiment, data_dir=data_dir, artifacts_dir=artifacts_dir)
    exp = built.experiment
    run = exp.run(run_id)
    split = load_pinned_split(built, run.split_id, artifacts_dir=artifacts_dir)
    init_from, init_label = _init_source(exp, run, artifacts_dir)
    out = paths.model_dir(exp.name, run.run_id, artifacts_dir)
    print(f"\ntrain : {exp.name}/{run.run_id} ({run.src_lang} -> {run.tgt_lang}) -> {out}")
    return train_run(
        run,
        split,
        out,
        profile=exp.profile,
        model_key=exp.model,
        cfg=exp.training,
        init_from=init_from,
        init_label=init_label,
        resume_from_checkpoint=resume,
    )


def _score(owner: Experiment, run: RunSpec, split: Split, partition: str,
           artifacts_dir: Path):
    """Translate one partition with one run's checkpoint, guard, score and save."""
    from .config import GENERATION
    from .evaluate import ModelCard, evaluate
    from .generate import count_truncated, load_model, translate

    model_path = paths.find_model(owner.name, run.run_id, artifacts_dir)
    card = ModelCard.load(model_path)
    # The model's own training pairs, from its card: this is what the guard checks against.
    own = load_split(
        artifacts_dir,
        config=card.config,
        strategy=card.strategy,
        bible_grouping=card.bible_grouping,
        expect_split_sha256=card.split_sha256,
        corpus_sha8=card.corpus_sha8,
    )
    sources = split.partition(partition)[run.src_col].astype(str).tolist()
    loaded = load_model(model_path)
    truncated = count_truncated(loaded.tokenizer, sources, run.src_lang,
                                GENERATION.max_source_length)
    print(f"\neval  : {owner.name}/{run.run_id} on {split.split_id}/{partition} "
          f"({len(sources)} pairs, {truncated} over {GENERATION.max_source_length} tokens)")
    hypotheses = translate(sources, loaded, src_lang=run.src_lang, tgt_lang=run.tgt_lang)
    record, dump = evaluate(
        hypotheses, split=split, partition=partition, model_card=card, train_uids=own.train_uids
    )
    out = paths.results_dir(owner.name, run.run_id, split.split_id, partition, split.sha256,
                            artifacts_dir)
    record.save(out, dump)
    print(f"        {record.primary_metric} {record.primary:.2f} -> {out}")
    return record


def evaluate_run(
    experiment: str | Experiment,
    run_id: str,
    *,
    partition: str = "test",
    data_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
):
    """Score one run on a partition of its own split. Returns the eval record."""
    artifacts_dir = paths.artifacts_dir(artifacts_dir)
    built = build(experiment, data_dir=data_dir, artifacts_dir=artifacts_dir, verbose=False)
    run = built.experiment.run(run_id)
    split = load_pinned_split(built, run.split_id, artifacts_dir=artifacts_dir)
    return _score(built.experiment, run, split, partition, artifacts_dir)


def run_evaluation(
    experiment: str | Experiment,
    name: str,
    *,
    data_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
) -> list:
    """Score every run an evaluation lists on its one shared partition."""
    artifacts_dir = paths.artifacts_dir(artifacts_dir)
    built = build(experiment, data_dir=data_dir, artifacts_dir=artifacts_dir, verbose=False)
    exp = built.experiment
    evaluation = exp.evaluation(name)
    split = load_pinned_split(built, evaluation.split_id, artifacts_dir=artifacts_dir)

    records = []
    for ref in evaluation.runs:
        owner = exp if ref.experiment == exp.name else load_experiment(ref.experiment)
        if owner is not exp:
            build(owner, data_dir=data_dir, artifacts_dir=artifacts_dir, verbose=False)
        records.append(_score(owner, owner.run(ref.run_id), split, evaluation.partition,
                              artifacts_dir))
    return records


# --------------------------------------------------------------------------- status


def split_status(experiment: str | Experiment, *,
                 artifacts_dir: str | Path | None = None) -> pd.DataFrame:
    """Every split of an experiment as last built here: sizes, hash, and whether pinned."""
    exp = _experiment(experiment)
    _, manifests = load_manifests(exp, artifacts_dir=artifacts_dir)
    rows = []
    for spec, manifest in zip(exp.splits, manifests):
        realized = manifest["realized"]
        rows.append({
            "split": spec.split_id,
            "purpose": spec.purpose,
            "train": realized["train"]["rows"],
            "dev": realized["dev"]["rows"],
            "test": realized["test"]["rows"],
            "sha256": manifest["split_sha256"],
            "pinned": "yes" if spec.sha256 else "no",
        })
    return pd.DataFrame(rows)



def run_status(experiment: str | Experiment, *,
               artifacts_dir: str | Path | None = None) -> pd.DataFrame:
    """Every run of an experiment, whether it is trained, and its test score if scored."""
    from .evaluate import RECORD_NAME, EvalRecord

    exp = _experiment(experiment)
    artifacts_dir = paths.artifacts_dir(artifacts_dir)
    try:
        _, manifests = load_manifests(exp, artifacts_dir=artifacts_dir)
        shas = {m["split_id"]: m["split_sha256"] for m in manifests}
    except (FileNotFoundError, PinError):
        shas = {}

    rows = []
    for run in exp.runs:
        trained = (paths.model_dir(exp.name, run.run_id, artifacts_dir) / "model"
                   / "train_manifest.json").exists()
        score = None
        if run.split_id in shas:
            path = paths.results_dir(exp.name, run.run_id, run.split_id, "test",
                                     shas[run.split_id], artifacts_dir)
            if (path / RECORD_NAME).exists():
                record = EvalRecord.load(path)
                score = round(record.primary, 2)
        rows.append({
            "run_id": run.run_id,
            "split": run.split_id,
            "purpose": exp.split(run.split_id).purpose,
            "trained": "yes" if trained else "",
            "test score": score if score is not None else "",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report


def report(experiment: str | Experiment, *,
           artifacts_dir: str | Path | None = None) -> dict[str, Path]:
    """Write every table for one experiment under artifacts/tables/<experiment>/."""
    from .evaluate import RECORD_NAME, EvalRecord
    from .report import write_tables

    exp = _experiment(experiment)
    artifacts_dir = paths.artifacts_dir(artifacts_dir)
    corpus_manifest, manifests = load_manifests(exp, artifacts_dir=artifacts_dir)
    shas = {m["split_id"]: m["split_sha256"] for m in manifests}

    def record_at(owner: str, run_id: str, split_id: str, partition: str):
        path = paths.results_dir(owner, run_id, split_id, partition, shas[split_id], artifacts_dir)
        if not (path / RECORD_NAME).exists():
            return None
        record = EvalRecord.load(path)
        # Records scored before experiments existed don't name one; their folder does.
        return record if record.experiment else dataclasses.replace(record, experiment=owner)

    records = [
        r for r in (
            record_at(exp.name, run.run_id, run.split_id, "test")
            for run in exp.runs
            if exp.split(run.split_id).purpose == "headline"
        )
        if r is not None
    ]
    evaluations = {
        ev.name: [
            r for r in (
                record_at(ref.experiment, ref.run_id, ev.split_id, ev.partition)
                for ref in ev.runs
            )
            if r is not None
        ]
        for ev in exp.evaluations
    }
    return write_tables(
        paths.tables_dir(exp.name, artifacts_dir),
        experiment=exp.name,
        corpus_manifest=corpus_manifest,
        split_manifests=manifests,
        records=records,
        evaluations=evaluations,
    )


def load_tables(experiment: str | Experiment, *,
                artifacts_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """The tables :func:`report` last wrote for an experiment, by name."""
    exp = _experiment(experiment)
    directory = paths.tables_dir(exp.name, artifacts_dir)
    return {path.stem: pd.read_csv(path) for path in sorted(directory.glob("*.csv"))}
