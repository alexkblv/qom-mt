"""Experiments: what to run.

An experiment is one TOML file in ``experiments/``. It names a corpus version, the splits
to build (each pinned to its hash once agreed), a model profile, the directions, any
training overrides and any extra evaluations. Run ids, output paths and report tables
are all derived from that file, so nothing about an experiment is declared anywhere
else. ``experiments/baseline.toml`` is the annotated reference for the format.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from . import paths
from .config import (
    BIBLE_GROUPINGS,
    CORPORA,
    DIRECTIONS,
    MODELS,
    SPLIT_PURPOSES,
    SPLIT_STRATEGIES,
    TRAINING,
    Direction,
    ModelProfile,
    TrainingConfig,
    config_has_verses,
    configs_of,
    corpus_sources,
    split_dir_name,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
EVAL_PARTITIONS = ("dev", "test")


class ExperimentError(ValueError):
    pass


@dataclass(frozen=True)
class SplitSpec:
    config: str
    strategy: str
    bible_grouping: str = "chapter"
    purpose: str = "headline"
    #: The pinned split hash. Training and evaluation refuse any other split.
    sha256: str | None = None

    @property
    def split_id(self) -> str:
        return split_dir_name(self.config, self.strategy, self.bible_grouping)


@dataclass(frozen=True)
class RunSpec:
    """One training run: one split of an experiment, in one direction."""

    experiment: str
    config: str
    strategy: str
    direction: str
    bible_grouping: str = "chapter"

    @property
    def run_id(self) -> str:
        parts = [self.config, self.strategy, self.direction]
        if self.bible_grouping != "chapter" and config_has_verses(self.config):
            parts.append(self.bible_grouping)
        return "__".join(parts)

    @property
    def split_id(self) -> str:
        return split_dir_name(self.config, self.strategy, self.bible_grouping)

    @property
    def lang(self) -> Direction:
        return DIRECTIONS[self.direction]

    @property
    def src_lang(self) -> str:
        return self.lang.src_lang

    @property
    def tgt_lang(self) -> str:
        return self.lang.tgt_lang

    @property
    def src_col(self) -> str:
        return self.lang.src_col

    @property
    def tgt_col(self) -> str:
        return self.lang.tgt_col


@dataclass(frozen=True)
class RunRef:
    """A run named from an experiment file: in the same experiment, or another one."""

    experiment: str
    run_id: str

    def __str__(self) -> str:
        return f"{self.experiment}/{self.run_id}"


@dataclass(frozen=True)
class Evaluation:
    """Several runs scored on one shared partition of one split."""

    name: str
    split_id: str
    partition: str
    runs: tuple[RunRef, ...]


@dataclass(frozen=True)
class InitFrom:
    """Where training starts: a Hugging Face model id or local path, or another run.

    A run reference may contain ``{direction}``, filled in per run, so each direction
    starts from its own checkpoint.
    """

    model: str | None = None
    run: RunRef | None = None

    def describe(self, direction: str) -> str:
        if self.model:
            return self.model
        return f"{self.run.experiment}/{self.run.run_id.format(direction=direction)}"


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    corpus: str
    model: str
    directions: tuple[str, ...]
    splits: tuple[SplitSpec, ...]
    evaluations: tuple[Evaluation, ...] = ()
    training: TrainingConfig = TRAINING
    init_from: InitFrom | None = None

    @property
    def profile(self) -> ModelProfile:
        return MODELS[self.model]

    @property
    def runs(self) -> tuple[RunSpec, ...]:
        """Every split in every direction. The run list is never written by hand."""
        return tuple(
            RunSpec(self.name, s.config, s.strategy, d, s.bible_grouping)
            for d in self.directions
            for s in self.splits
        )

    @property
    def runs_by_id(self) -> dict[str, RunSpec]:
        return {r.run_id: r for r in self.runs}

    def run(self, run_id: str) -> RunSpec:
        try:
            return self.runs_by_id[run_id]
        except KeyError:
            raise ExperimentError(
                f"{self.name} has no run {run_id!r}; `qomlaq runs {self.name}` lists them"
            ) from None

    def split(self, split_id: str) -> SplitSpec:
        for spec in self.splits:
            if spec.split_id == split_id:
                return spec
        raise ExperimentError(f"{self.name} has no split {split_id!r}")

    def evaluation(self, name: str) -> Evaluation:
        for evaluation in self.evaluations:
            if evaluation.name == name:
                return evaluation
        raise ExperimentError(f"{self.name} has no evaluation {name!r}")


# ------------------------------------------------------------------------- loading


def list_experiments(root: str | Path | None = None) -> list[str]:
    directory = paths.experiments_dir(root)
    return sorted(p.stem for p in directory.glob("*.toml"))


def load_experiment(name: str, root: str | Path | None = None) -> Experiment:
    path = paths.experiments_dir(root) / f"{name}.toml"
    if not path.exists():
        available = ", ".join(list_experiments(root)) or "none"
        raise ExperimentError(f"no experiment {name!r} in {path.parent} (available: {available})")
    with path.open("rb") as fh:
        try:
            data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ExperimentError(f"{path.name}: {exc}") from exc
    return parse_experiment(name, data)


def _require_keys(where: str, table: dict, allowed: set[str], required: set[str]) -> None:
    unknown = set(table) - allowed
    if unknown:
        raise ExperimentError(f"{where}: unknown key(s) {sorted(unknown)}")
    missing = required - set(table)
    if missing:
        raise ExperimentError(f"{where}: missing key(s) {sorted(missing)}")


def _choice(where: str, value, allowed) -> str:
    if value not in allowed:
        raise ExperimentError(f"{where}: {value!r} is not one of {sorted(allowed)}")
    return value


def _parse_split(where: str, table: dict, configs: tuple[str, ...]) -> SplitSpec:
    _require_keys(where, table, {"config", "strategy", "bible_grouping", "purpose", "sha256"},
                  {"config", "strategy"})
    config = _choice(f"{where}.config", table["config"], configs)
    grouping = _choice(f"{where}.bible_grouping", table.get("bible_grouping", "chapter"),
                       BIBLE_GROUPINGS)
    if grouping != "chapter" and not config_has_verses(config):
        raise ExperimentError(f"{where}: bible_grouping only applies to configs with the Bible")
    sha = table.get("sha256")
    if sha is not None and not _SHA256.match(str(sha)):
        raise ExperimentError(f"{where}.sha256 must be a 64-character hex digest")
    return SplitSpec(
        config=config,
        strategy=_choice(f"{where}.strategy", table["strategy"], SPLIT_STRATEGIES),
        bible_grouping=grouping,
        purpose=_choice(f"{where}.purpose", table.get("purpose", "headline"), SPLIT_PURPOSES),
        sha256=sha,
    )


def _parse_run_ref(where: str, value: str, experiment: str) -> RunRef:
    if not isinstance(value, str) or not value:
        raise ExperimentError(f"{where}: run references are strings")
    if "/" in value:
        other, run_id = value.split("/", 1)
        return RunRef(other, run_id)
    return RunRef(experiment, value)


def _parse_training(where: str, table: dict) -> TrainingConfig:
    fields = {f.name: f for f in dataclasses.fields(TrainingConfig)}
    _require_keys(where, table, set(fields), set())
    values = {}
    for key, value in table.items():
        default = getattr(TRAINING, key)
        if isinstance(default, bool):
            ok = isinstance(value, bool)
        elif isinstance(default, (int, float)):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            if ok and isinstance(default, float):
                value = float(value)
            elif ok:
                ok = isinstance(value, int)
        else:
            ok = isinstance(value, type(default))
        if not ok:
            raise ExperimentError(f"{where}.{key} must be {type(default).__name__}, got {value!r}")
        values[key] = value
    return dataclasses.replace(TRAINING, **values)


def _parse_init_from(where: str, value, experiment: str) -> InitFrom:
    if isinstance(value, str) and value:
        return InitFrom(model=value)
    if isinstance(value, dict):
        _require_keys(where, value, {"experiment", "run"}, {"experiment", "run"})
        return InitFrom(run=RunRef(value["experiment"], value["run"]))
    raise ExperimentError(
        f"{where}: expected a model id or path, or {{ experiment = ..., run = ... }}"
    )


def parse_experiment(name: str, data: dict) -> Experiment:
    """Validate an experiment's TOML table. Every mistake names the key it is in."""
    if not _NAME.match(name):
        raise ExperimentError(f"experiment name {name!r}: use lowercase letters, digits, - . _")
    where = name
    _require_keys(
        where, data,
        {"description", "corpus", "model", "directions", "splits", "evaluations", "training",
         "init_from"},
        {"corpus", "model", "directions", "splits"},
    )
    corpus = _choice(f"{where}.corpus", data["corpus"], CORPORA)
    configs = configs_of(corpus_sources(corpus))

    directions = tuple(data["directions"])
    if not directions or len(set(directions)) != len(directions):
        raise ExperimentError(f"{where}.directions: list each direction once")
    for direction in directions:
        _choice(f"{where}.directions", direction, DIRECTIONS)

    splits = tuple(
        _parse_split(f"{where}.splits[{i}]", table, configs)
        for i, table in enumerate(data["splits"])
    )
    if not splits:
        raise ExperimentError(f"{where}.splits: an experiment needs at least one split")
    split_ids = [s.split_id for s in splits]
    duplicated = sorted({s for s in split_ids if split_ids.count(s) > 1})
    if duplicated:
        raise ExperimentError(f"{where}.splits: {duplicated} declared more than once")

    run_ids = {
        RunSpec(name, s.config, s.strategy, d, s.bible_grouping).run_id
        for s in splits
        for d in directions
    }
    evaluations = []
    for i, table in enumerate(data.get("evaluations", [])):
        ewhere = f"{where}.evaluations[{i}]"
        _require_keys(ewhere, table, {"name", "split", "partition", "runs"},
                      {"name", "split", "runs"})
        _choice(f"{ewhere}.split", table["split"], split_ids)
        refs = tuple(_parse_run_ref(f"{ewhere}.runs", r, name) for r in table["runs"])
        local_unknown = [r.run_id for r in refs if r.experiment == name and r.run_id not in run_ids]
        if local_unknown:
            raise ExperimentError(f"{ewhere}.runs: no such run(s) {local_unknown}")
        evaluations.append(
            Evaluation(
                name=table["name"],
                split_id=table["split"],
                partition=_choice(f"{ewhere}.partition", table.get("partition", "test"),
                                  EVAL_PARTITIONS),
                runs=refs,
            )
        )
    names = [e.name for e in evaluations]
    if len(set(names)) != len(names):
        raise ExperimentError(f"{where}.evaluations: names must be unique")

    return Experiment(
        name=name,
        description=str(data.get("description", "")).strip(),
        corpus=corpus,
        model=_choice(f"{where}.model", data["model"], MODELS),
        directions=directions,
        splits=splits,
        evaluations=tuple(evaluations),
        training=_parse_training(f"{where}.training", data.get("training", {})),
        init_from=(
            _parse_init_from(f"{where}.init_from", data["init_from"], name)
            if "init_from" in data else None
        ),
    )
