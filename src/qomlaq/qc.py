"""Quality-control filters, applied identically to every source.

Each stage records what it removed in a *ledger*, so the corpus table's row count can
be reconciled arithmetically from the raw source counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import QCConfig
from .normalize import fold_for_matching, tokens


@dataclass
class QCLedger:
    """Row counts through the pipeline, per stage and per source."""

    stages: list[dict] = field(default_factory=list)

    def record(self, stage: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
        removed_by_source = (
            before.groupby("source_doc", observed=True).size()
            - after.groupby("source_doc", observed=True).size().reindex(
                before["source_doc"].unique(), fill_value=0
            )
        ).fillna(0)
        self.stages.append(
            {
                "stage": stage,
                "rows_before": int(len(before)),
                "rows_after": int(len(after)),
                "removed": int(len(before) - len(after)),
                "removed_by_source": {
                    str(k): int(v) for k, v in removed_by_source.items() if v
                },
            }
        )

    def as_list(self) -> list[dict]:
        return list(self.stages)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{k: v for k, v in s.items() if k != "removed_by_source"} for s in self.stages]
        )


def add_length_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["qom_len_c"] = out["qom"].str.len().astype("int64")
    out["es_len_c"] = out["es"].str.len().astype("int64")
    out["qom_len_t"] = out["qom"].map(lambda s: len(tokens(s))).astype("int64")
    out["es_len_t"] = out["es"].map(lambda s: len(tokens(s))).astype("int64")
    out["len_ratio_t"] = (out["es_len_t"] + 1) / (out["qom_len_t"] + 1)
    out["norm_key"] = [
        fold_for_matching(q) + "\x1f" + fold_for_matching(e)
        for q, e in zip(out["qom"], out["es"])
    ]
    return out


def apply_qc(frame: pd.DataFrame, cfg: QCConfig) -> tuple[pd.DataFrame, QCLedger]:
    """Filter the canonical table, returning the survivors and a ledger.

    Stage order is fixed and reported: dedup, length ratio, minimum tokens, maximum
    characters. Empty sides are already removed during ingestion.
    """
    ledger = QCLedger()
    current = add_length_features(frame)

    subset = ["qom", "es"]
    if cfg.dedup_scope == "within_source":
        subset = subset + ["source_doc"]
    after = current.drop_duplicates(subset=subset, keep="first")
    ledger.record(f"dedup_{cfg.dedup_scope}", current, after)
    current = after

    after = current[
        current["len_ratio_t"].between(cfg.len_ratio_min, cfg.len_ratio_max, inclusive="both")
    ]
    ledger.record("len_ratio", current, after)
    current = after

    after = current[
        (current["qom_len_t"] >= cfg.min_tokens) & (current["es_len_t"] >= cfg.min_tokens)
    ]
    ledger.record("min_tokens", current, after)
    current = after

    after = current[(current["qom_len_c"] < cfg.max_chars) & (current["es_len_c"] < cfg.max_chars)]
    ledger.record("max_chars", current, after)
    current = after

    return current.reset_index(drop=True), ledger


def cross_source_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Pairs that appear in more than one source.

    Deliberately *not* removed -- the paper documents within-source dedup -- but they
    must share a group so a duplicate cannot straddle partitions. Reported so the count
    is visible rather than assumed to be zero.
    """
    counts = frame.groupby("norm_key", observed=True)["source_doc"].nunique()
    keys = counts[counts > 1].index
    return frame[frame["norm_key"].isin(keys)].sort_values("norm_key", kind="mergesort")


def unify_duplicate_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge whole groups that share any identical pair.

    Without this, the same sentence appearing twice can land in train and in test while
    an exact-pair check computed per source still passes.

    Groups are merged *wholesale*, via union-find, rather than by moving the duplicated
    row into the other group. Moving a single row would silently pull one verse out of
    its chapter -- which is how a chapter-grouped Bible split still ended up with a
    handful of adjacent verses straddling partitions. Merging keeps the invariant "every
    verse of a chapter shares a group" intact.
    """
    out = frame.copy()
    duplicated = out.groupby("norm_key", observed=True)["group_id"].transform("nunique") > 1
    if not duplicated.any():
        return out

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Lexicographically smallest root, so the outcome is order-independent.
            lo, hi = sorted((ra, rb))
            parent[hi] = lo

    for _, group_ids in out.loc[duplicated].groupby("norm_key", observed=True)["group_id"]:
        ids = [str(g) for g in group_ids.dropna().unique()]
        for other in ids[1:]:
            union(ids[0], other)

    if parent:
        touched = out["group_id"].astype("string").isin(parent)
        out.loc[touched, "group_id"] = out.loc[touched, "group_id"].map(
            lambda g: find(str(g))
        )
    return out
