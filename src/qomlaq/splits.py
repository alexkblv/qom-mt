"""Split construction, persistence and verified loading.

A split is an *artifact*, not a value computed on the fly. It is written to disk under a
path that embeds the corpus hash, carries a manifest recording how it was built and what
it contains, and can only be loaded by a caller that states which split it expects.

There is no way to obtain a partition except by loading one, and loading one verifies
it, so a test set can never be quietly re-derived by a different implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from . import __version__, code_fingerprint, paths, runtime_versions, same_except_provenance
from .checks import CheckReport, run_split_checks
from .config import (
    BIBLE_GROUPINGS,
    DEFAULT_SPLITS,
    SEED,
    SOURCES,
    SPLIT_STRATEGIES,
    SplitConfig,
    split_dir_name,
)
from .corpus import CorpusArtifact
from .groups import regroup_by_verse
from .hashing import PARTITION_IDENTITY_COLUMNS, sha256_frame, sha256_text, short

PARTITIONS = ("train", "dev", "test")
MANIFEST_NAME = "split_manifest.json"
SCHEMA_VERSION = "2.0.0"

#: Columns persisted per partition. Text is included so a consumer never has to rejoin
#: to the corpus (and so a later corpus edit cannot retroactively redefine a test set).
PARTITION_COLUMNS: tuple[str, ...] = (
    "pair_uid",
    "group_id",
    "unit_level",
    "source_doc",
    "source_ref",
    "record_kind",
    "qom",
    "es",
    "norm_key",
    "libro",
    "capitulo",
    "versiculo",
)


class SplitError(RuntimeError):
    pass


class SplitIntegrityError(SplitError):
    pass


@dataclass(frozen=True)
class Split:
    """A loaded, verified split. The only object the trainer and scorer accept."""

    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame
    manifest: dict

    @property
    def split_id(self) -> str:
        return self.manifest["split_id"]

    @property
    def sha256(self) -> str:
        return self.manifest["split_sha256"]

    @property
    def corpus_sha256(self) -> str:
        return self.manifest["corpus_sha256"]

    @property
    def train_uids(self) -> set[str]:
        return set(self.train["pair_uid"])

    def partition(self, name: str) -> pd.DataFrame:
        if name not in PARTITIONS:
            raise SplitError(f"unknown partition {name!r}")
        return getattr(self, name)

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "dev": self.dev, "test": self.test}

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in PARTITIONS:
            df = self.partition(name)
            rows.append(
                {
                    "partition": name,
                    "rows": len(df),
                    "groups": df["group_id"].nunique(),
                    "sources": df["source_doc"].nunique(),
                }
            )
        return pd.DataFrame(rows)


def allocate_groups(
    groups: pd.DataFrame,
    fractions: Mapping[str, float],
    *,
    stratify: bool = True,
) -> dict[str, str]:
    """Assign whole groups to partitions, balancing on *rows* rather than group count.

    Deterministic largest-first packing: within each stratum, groups are ordered by
    descending row count (ties broken by id, so the result does not depend on input
    order), and each group goes to whichever partition is furthest below its row quota.
    Strata are packed independently, so adding a source never moves a unit of another
    source when splits are stratified.

    Group sizes vary by two orders of magnitude here (330 rows down to 1), so allocating
    by group count would drift far from the target proportions.
    """
    if groups.empty:
        return {}

    assignment: dict[str, str] = {}
    strata = groups.groupby("stratum", sort=True) if stratify else [("_all", groups)]

    for _, chunk in strata:
        total_rows = int(chunk["n_rows"].sum())
        quota = {name: total_rows * frac for name, frac in fractions.items()}
        filled = {name: 0.0 for name in fractions}

        ordered = chunk.sort_values(
            ["n_rows", "group_id"], ascending=[False, True], kind="mergesort"
        )
        for group_id, n_rows in zip(ordered["group_id"], ordered["n_rows"]):
            # Largest remaining deficit wins; ties broken by the fixed partition order
            # so the outcome is reproducible without consulting the RNG at all.
            target = max(PARTITIONS, key=lambda p: (quota.get(p, 0) - filled[p], -PARTITIONS.index(p)))
            assignment[str(group_id)] = target
            filled[target] += float(n_rows)

    return assignment


def _forced_partitions() -> dict[str, str]:
    return {s.name: s.forced_partition for s in SOURCES if s.forced_partition}


def make_split(
    corpus: CorpusArtifact,
    *,
    config: str,
    strategy: str,
    bible_grouping: str = "chapter",
    fractions: SplitConfig = DEFAULT_SPLITS,
    purpose: str = "headline",
) -> tuple[dict[str, pd.DataFrame], dict, CheckReport]:
    """Build one split. Returns partitions, manifest and the check report.

    Callers should prefer :func:`write_split`, which persists the result; the partitions
    are returned here only so the notebook can display them.

    ``purpose`` is recorded in the manifest. ``"control"`` marks a split built only to
    measure a design choice -- the verse-level Bible grouping exists to measure what
    chapter grouping is worth, and must never back a headline number.
    """
    if strategy not in SPLIT_STRATEGIES:
        raise SplitError(f"unknown strategy {strategy!r}; expected {SPLIT_STRATEGIES}")
    if bible_grouping not in BIBLE_GROUPINGS:
        raise SplitError(f"unknown bible_grouping {bible_grouping!r}; expected {BIBLE_GROUPINGS}")

    frame = corpus.config_frame(config)
    if bible_grouping == "verse":
        frame = regroup_by_verse(frame)

    forced = {k: v for k, v in _forced_partitions().items() if (frame["source_doc"] == k).any()}
    is_forced = frame["source_doc"].isin(forced)
    free = frame[~is_forced]

    groups = (
        free.groupby(["group_id", "source_doc"], observed=True)
        .size()
        .reset_index(name="n_rows")
        .rename(columns={"source_doc": "stratum"})
    )
    assignment = allocate_groups(groups, fractions.as_dict(), stratify=(strategy == "stratified"))

    parts: dict[str, pd.DataFrame] = {}
    partition_of = free["group_id"].map(assignment)
    for name in PARTITIONS:
        selected = free[partition_of == name]
        if name in forced.values():
            extra = frame[is_forced & frame["source_doc"].map(forced).eq(name)]
            selected = pd.concat([selected, extra], ignore_index=True)
        parts[name] = (
            selected.sort_values("pair_uid", kind="mergesort")
            .reset_index(drop=True)
            .reindex(columns=[c for c in PARTITION_COLUMNS if c in frame.columns])
        )

    report = run_split_checks(
        parts,
        target_fractions=fractions.as_dict(),
        forced_partitions=forced,
        expected_uids=set(frame["pair_uid"]),
    )

    partition_hashes = {
        name: sha256_frame(df, PARTITION_IDENTITY_COLUMNS) for name, df in parts.items()
    }
    split_id = split_dir_name(config, strategy, bible_grouping)
    # The seed is part of the split's identity, although the packing itself uses none.
    identity = json.dumps(
        {
            "split_id": split_id,
            "corpus_sha256": corpus.sha256,
            "seed": SEED,
            "fractions": fractions.as_dict(),
            "partition_sha256": partition_hashes,
        },
        sort_keys=True,
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "code_fingerprint": code_fingerprint(),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_id": split_id,
        "purpose": purpose,
        "config": config,
        "strategy": strategy,
        "bible_grouping": bible_grouping,
        "corpus_sha256": corpus.sha256,
        "corpus": corpus.name,
        "corpus_sha8": corpus.sha8,
        "seed": SEED,
        "target_fractions": fractions.as_dict(),
        "forced_partitions": forced,
        "realized": {
            name: {
                "rows": int(len(df)),
                "groups": int(df["group_id"].nunique()),
                "fraction": round(len(df) / max(1, sum(len(d) for d in parts.values())), 4),
            }
            for name, df in parts.items()
        },
        "per_source_rows": {
            name: df["source_doc"].value_counts().to_dict() for name, df in parts.items()
        },
        "partition_sha256": partition_hashes,
        "checks": report.summary(),
        "runtime": runtime_versions(),
        "split_sha256": sha256_text(identity),
    }
    return parts, manifest, report


def write_split(
    corpus: CorpusArtifact,
    artifacts_dir: str | Path,
    *,
    config: str,
    strategy: str,
    bible_grouping: str = "chapter",
    fractions: SplitConfig = DEFAULT_SPLITS,
    purpose: str = "headline",
) -> dict:
    """Build, verify and persist a split. Returns its manifest.

    Deliberately does not return the partitions: the only way to use a split is to load
    it back, which re-verifies it.

    Headline splits must pass every check. ``purpose="control"`` splits may fail the
    checks they were built to violate, but the failures are recorded in the manifest so
    the report stage can refuse to place them in a results table.

    A split already on disk that differs only in provenance (when and by which code
    version it was built) is left as it is, so rebuilding doesn't touch tracked files.
    """
    parts, manifest, report = make_split(
        corpus,
        config=config,
        strategy=strategy,
        bible_grouping=bible_grouping,
        fractions=fractions,
        purpose=purpose,
    )
    if purpose == "headline":
        report.raise_for_failures()

    out_dir = paths.split_dir(corpus.sha8, manifest["split_id"], artifacts_dir)
    existing = _existing_manifest(out_dir)
    if existing is not None and same_except_provenance(existing, manifest):
        return existing
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in parts.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _existing_manifest(out_dir: Path) -> dict | None:
    """The manifest on disk, if its partitions are all there too."""
    path = out_dir / MANIFEST_NAME
    if not path.exists() or not all((out_dir / f"{p}.csv").exists() for p in PARTITIONS):
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(
    artifacts_dir: str | Path,
    *,
    config: str,
    strategy: str,
    expect_split_sha256: str,
    bible_grouping: str = "chapter",
    corpus_sha8: str | None = None,
    expect_corpus_sha256: str | None = None,
) -> Split:
    """Load a split, proving it is the one the caller means.

    ``expect_split_sha256`` has no default on purpose. Stating the hash is what turns
    "I loaded a test set" into "I loaded *this* test set", and it is the reason a stale
    or hand-edited partition cannot quietly become the evaluation set.
    """
    root = paths.splits_root(artifacts_dir)
    if corpus_sha8 is None:
        candidates = sorted(p for p in root.glob("*") if p.is_dir())
        if len(candidates) != 1:
            raise SplitError(
                f"{len(candidates)} corpus hashes present under {root}; pass corpus_sha8 "
                "to say which corpus this split belongs to."
            )
        corpus_sha8 = candidates[0].name

    split_dir = paths.split_dir(corpus_sha8, split_dir_name(config, strategy, bible_grouping),
                                artifacts_dir)
    manifest_path = split_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise SplitError(f"no split at {split_dir}; build it with write_split first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parts: dict[str, pd.DataFrame] = {}
    for name in PARTITIONS:
        df = pd.read_csv(
            split_dir / f"{name}.csv",
            dtype={"pair_uid": "string", "group_id": "string"},
            keep_default_na=False,
            na_values=[""],
        )
        recomputed = sha256_frame(df, PARTITION_IDENTITY_COLUMNS)
        if recomputed != manifest["partition_sha256"][name]:
            raise SplitIntegrityError(
                f"{split_dir.name}/{name}.csv does not match its manifest hash "
                f"(recomputed {short(recomputed)}, manifest "
                f"{short(manifest['partition_sha256'][name])}). The file has been edited "
                "or written by a different pipeline version."
            )
        parts[name] = df

    if manifest["split_sha256"] != expect_split_sha256:
        raise SplitIntegrityError(
            f"split hash mismatch for {manifest['split_id']}: "
            f"expected {short(expect_split_sha256)}, found {short(manifest['split_sha256'])}"
        )
    if expect_corpus_sha256 and manifest["corpus_sha256"] != expect_corpus_sha256:
        raise SplitIntegrityError(
            f"split was built from corpus {short(manifest['corpus_sha256'])}, "
            f"caller expected {short(expect_corpus_sha256)}"
        )

    report = run_split_checks(
        parts,
        target_fractions=manifest["target_fractions"],
        forced_partitions=manifest.get("forced_partitions", {}),
    )
    if manifest.get("purpose", "headline") == "headline":
        report.raise_for_failures()

    return Split(train=parts["train"], dev=parts["dev"], test=parts["test"], manifest=manifest)


def split_index(artifacts_dir: str | Path) -> pd.DataFrame:
    """Every persisted split with its hash."""
    root = paths.splits_root(artifacts_dir)
    rows = []
    for manifest_path in sorted(root.rglob(MANIFEST_NAME)):
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "split_id": m["split_id"],
                "config": m["config"],
                "strategy": m["strategy"],
                "bible_grouping": m["bible_grouping"],
                "train": m["realized"]["train"]["rows"],
                "dev": m["realized"]["dev"]["rows"],
                "test": m["realized"]["test"]["rows"],
                "corpus_sha8": m["corpus_sha8"],
                "split_sha256": m["split_sha256"],
            }
        )
    return pd.DataFrame(rows)
