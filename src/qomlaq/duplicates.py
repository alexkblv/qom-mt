"""Exact and near-duplicate detection over the whole corpus, before it is split.

Three checks, all on words rather than characters, with the Jaccard coefficient from
``textdistance``:

1. **Exact duplicates.** Rows whose Qom side, Spanish side or whole pair is identical
   after folding (:func:`qomlaq.normalize.fold_for_matching`). QC already drops identical
   pairs within a source; this check also shows one-sided repeats and repeats across
   sources, which a split has to keep on one side.
2. **Copied pairs.** Jaccard between the Qom and the Spanish words of the same row, as in
   Chiruzzo et al. (2022), who searched a Guarani-Spanish corpus, all partitions at once,
   for pairs above 0.6 and reviewed them by hand: reported speech had left the Spanish
   text nearly unchanged on the Guarani side. In Qom transcriptions the same happens with
   code-switching and with a Spanish line pasted twice.
3. **Near-duplicate rows.** Jaccard between the Qom words of two rows, and between their
   Spanish words. Two sentences that differ in one name can otherwise land on both sides
   of a split.

Nothing is removed here. Every match above an inspection floor is written out so the
threshold can be chosen by reading the cases, and the counts at each candidate threshold
say whether a manual review is feasible.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import textdistance

from .config import DUPLICATES, SPEECH_TABLE, DuplicatesConfig, TableSpec
from .corpus import CorpusArtifact
from .hashing import pair_uid, sha256_file
from .ingest import text_mask
from .normalize import fold_for_matching, normalize_text, word_tokens

#: Columns every row carries, whichever input it came from.
ROW_COLUMNS: tuple[str, ...] = (
    "row_id", "source_doc", "source_ref", "group_id", "record_kind", "qom", "es",
)

SIDES: tuple[str, ...] = ("qom", "es")

#: Word-level Jaccard over bags of words. Multiplicities count: a word used twice in one
#: sentence and once in the other contributes one shared occurrence out of two.
#: ``external=False`` keeps textdistance's own implementation, so the numbers do not
#: depend on which accelerator libraries happen to be installed.
JACCARD = textdistance.Jaccard(qval=1, as_set=False, external=False)


def bag_of_words(text: str) -> list[str]:
    """Casefolded words, punctuation dropped, the apostrophe kept inside words."""
    return word_tokens(normalize_text(text).casefold())


def jaccard(a: list[str], b: list[str]) -> float:
    """Jaccard coefficient of two bags of words; 0 when either is empty."""
    if not a or not b:
        return 0.0
    return float(JACCARD(a, b))


# --------------------------------------------------------------------------- exact


def _fold_keys(frame: pd.DataFrame, side: str) -> list[str]:
    if side == "pair":
        return [
            fold_for_matching(q) + "\x1f" + fold_for_matching(e)
            for q, e in zip(frame["qom"], frame["es"])
        ]
    return [fold_for_matching(t) for t in frame[side]]


def _group_keys(frame: pd.DataFrame) -> pd.Series:
    """Group ids with every missing one replaced by a key of its own, so rows whose unit
    is unknown never count as sharing one."""
    groups = frame["group_id"].astype("string")
    return groups.fillna("row:" + frame["row_id"].astype("string"))


def exact_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows whose Qom side, Spanish side or whole pair repeats after folding.

    One block of rows per repeated text, per side; a row can appear under more than one
    side. ``n_sources`` and ``n_groups`` say whether the repeat is confined to one source
    or one discourse unit: a repeat inside one unit cannot straddle a split.
    """
    blocks = []
    for side in (*SIDES, "pair"):
        keys = pd.Series(_fold_keys(frame, side), index=frame.index)
        repeated = keys.map(keys.value_counts()) > 1
        rows = frame.loc[repeated, list(ROW_COLUMNS)].copy()
        rows.insert(0, "side", side)
        rows.insert(1, "key", keys[repeated])
        rows["group_key"] = _group_keys(rows)
        blocks.append(rows)
    out = pd.concat(blocks, ignore_index=True)
    out = out.sort_values(["side", "key", "source_doc", "source_ref"], kind="mergesort")

    per_block = out.groupby(["side", "key"], sort=False)
    out.insert(1, "dup_id", per_block.ngroup().astype("int64"))
    out.insert(2, "n_rows", per_block["row_id"].transform("size").astype("int64"))
    out.insert(3, "n_sources", per_block["source_doc"].transform("nunique").astype("int64"))
    out.insert(4, "n_groups", per_block["group_key"].transform("nunique").astype("int64"))
    return out.drop(columns=["key", "group_key"]).reset_index(drop=True)


def _exact_by_source(exact: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Rows in repeated blocks, per side and source."""
    counts = exact.groupby(["side", "source_doc"], observed=True).size()
    return {
        side: {str(source): int(n) for (s, source), n in counts.items() if s == side}
        for side in (*SIDES, "pair")
    }


def _exact_summary(exact: pd.DataFrame) -> dict[str, dict[str, int]]:
    summary = {}
    for side in (*SIDES, "pair"):
        blocks = exact[exact["side"] == side].drop_duplicates("dup_id")
        summary[side] = {
            "blocks": int(len(blocks)),
            "rows": int(blocks["n_rows"].sum()),
            "cross_source_blocks": int((blocks["n_sources"] > 1).sum()),
            "cross_group_blocks": int((blocks["n_groups"] > 1).sum()),
        }
    return summary


# ---------------------------------------------------------------------------- near


def _occurrences(bag: list[str]) -> list[str]:
    """Each occurrence of a word as an element of its own, so a set comparison of these
    is the bag comparison textdistance makes."""
    seen: Counter[str] = Counter()
    out = []
    for word in bag:
        seen[word] += 1
        out.append(f"{word}\x1f{seen[word]}")
    return out


def near_duplicate_pairs(bags: list[list[str]], floor: float) -> list[tuple[int, int, float]]:
    """Every pair of rows whose word-level Jaccard is at least ``floor``, as
    ``(i, j, score)`` with ``i < j``.

    Scoring every pair of a 30,000-row corpus is out of reach, so candidates come from a
    prefix index (Xiao et al., 2008). Word occurrences are ordered from rarest to most
    frequent and rows are visited from shortest to longest. A row of ``n`` occurrences is
    looked up by its first ``n - ceil(floor * n) + 1`` and indexed under its first
    ``n - ceil(2 * floor / (1 + floor) * n) + 1``: a shorter row that overlaps a longer one
    enough to reach the floor must share one of those. Two bounds rule candidates out on
    the way, a size ratio and a count of what the two rows can still have in common. All
    of this is filtering, not approximation; ``textdistance`` scores every surviving pair
    and the ones below the floor are dropped.
    """
    if not 0 < floor <= 1:
        raise ValueError(f"floor must be in (0, 1], got {floor}")
    eps = 1e-9
    occurrences = [_occurrences(bag) for bag in bags]
    frequency = Counter(e for row in occurrences for e in row)
    rank = {e: i for i, e in enumerate(sorted(frequency, key=lambda e: (frequency[e], e)))}
    ordered = [sorted(row, key=rank.__getitem__) for row in occurrences]
    sizes = [len(row) for row in ordered]

    index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    pairs: list[tuple[int, int, float]] = []
    for j in sorted(range(len(ordered)), key=lambda r: (sizes[r], r)):
        row, n = ordered[j], sizes[j]
        if n == 0:
            continue
        probe = n - math.ceil(floor * n - eps) + 1
        keep = n - math.ceil(2 * floor / (1 + floor) * n - eps) + 1
        # candidate -> occurrences in common so far; -1 once the pair is ruled out
        common: dict[int, int] = {}
        for p in range(probe):
            for i, q in index[row[p]]:
                count = common.get(i, 0)
                if count < 0:
                    continue
                m = sizes[i]
                # Jaccard >= floor needs floor * (n + m) / (1 + floor) occurrences in
                # common, which the shorter row (m <= n) can only supply if m >= floor * n.
                needed = math.ceil(floor / (1 + floor) * (n + m) - eps)
                possible = count + 1 + min(n - p - 1, m - q - 1)
                common[i] = -1 if (m < floor * n - eps or possible < needed) else count + 1
        for p in range(keep):
            index[row[p]].append((j, p))
        for i, count in common.items():
            if count > 0:
                score = jaccard(bags[i], bags[j])
                if score >= floor - eps:
                    pairs.append((min(i, j), max(i, j), score))
    pairs.sort()
    return pairs


_NEAR_COLUMNS: tuple[str, ...] = (
    "row_a", "row_b", "source_a", "source_b", "ref_a", "ref_b", "group_a", "group_b",
    "same_source", "same_group", "jaccard_qom", "jaccard_es", "jaccard_max",
    "qom_a", "qom_b", "es_a", "es_b",
)


def near_duplicates(frame: pd.DataFrame, bags: dict[str, list[list[str]]], floor: float) -> pd.DataFrame:
    """Pairs of rows at or above ``floor`` on either side, with both sides' scores."""
    scores: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for side in SIDES:
        for i, j, score in near_duplicate_pairs(bags[side], floor):
            scores[(i, j)][side] = score
    for (i, j), found in scores.items():
        for side in SIDES:
            if side not in found:
                found[side] = jaccard(bags[side][i], bags[side][j])

    groups = _group_keys(frame)
    records = []
    for (i, j), found in sorted(scores.items()):
        a, b = frame.iloc[i], frame.iloc[j]
        records.append(
            {
                "row_a": a["row_id"], "row_b": b["row_id"],
                "source_a": a["source_doc"], "source_b": b["source_doc"],
                "ref_a": a["source_ref"], "ref_b": b["source_ref"],
                "group_a": a["group_id"], "group_b": b["group_id"],
                "same_source": bool(a["source_doc"] == b["source_doc"]),
                "same_group": bool(groups.iloc[i] == groups.iloc[j]),
                "jaccard_qom": found["qom"], "jaccard_es": found["es"],
                "jaccard_max": max(found.values()),
                "qom_a": a["qom"], "qom_b": b["qom"], "es_a": a["es"], "es_b": b["es"],
            }
        )
    out = pd.DataFrame.from_records(records, columns=list(_NEAR_COLUMNS))
    return out.sort_values(["jaccard_max", "row_a", "row_b"], ascending=[False, True, True],
                           kind="mergesort").reset_index(drop=True)


def _near_by_source(near: pd.DataFrame, thresholds: tuple[float, ...]) -> dict[str, dict[str, int]]:
    """Pairs at or above each threshold on either side, per pair of sources."""
    key = pd.Series(
        [" | ".join(sorted((str(a), str(b)))) for a, b in zip(near["source_a"], near["source_b"])],
        index=near.index, dtype="string",
    )
    out = {}
    for threshold in thresholds:
        hits = key[near["jaccard_max"] >= threshold - 1e-9]
        out[_label(threshold)] = {str(k): int(n) for k, n in hits.value_counts().sort_index().items()}
    return out


def _near_summary(near: pd.DataFrame, thresholds: tuple[float, ...]) -> dict:
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for side, column in (("qom", "jaccard_qom"), ("es", "jaccard_es"), ("either", "jaccard_max")):
        summary[side] = {}
        for threshold in thresholds:
            hits = near[near[column] >= threshold - 1e-9]
            crossing = hits[~hits["same_group"]]
            summary[side][_label(threshold)] = {
                "pairs": int(len(hits)),
                "rows": int(pd.concat([hits["row_a"], hits["row_b"]]).nunique()),
                "cross_group_pairs": int(len(crossing)),
                "cross_group_rows": int(pd.concat([crossing["row_a"], crossing["row_b"]]).nunique()),
            }
    return summary


# -------------------------------------------------------------------------- copied


def copied_scores(bags: dict[str, list[list[str]]]) -> list[float]:
    """Jaccard between the Qom and the Spanish words of each row."""
    return [jaccard(q, e) for q, e in zip(bags["qom"], bags["es"])]


def _label(threshold: float) -> str:
    return f"{threshold:.2f}"


# -------------------------------------------------------------------------- report


@dataclass(frozen=True)
class DuplicatesReport:
    """Every table the check produced, plus the counts that go to ``summary.json``."""

    exact: pd.DataFrame
    copied: pd.DataFrame
    near: pd.DataFrame
    summary: dict

    def describe(self) -> str:
        def pivot(table: dict[str, dict[str, int]], rows: str) -> str:
            frame = pd.DataFrame(table).fillna(0).astype("int64").rename_axis(rows)
            return frame.to_string() if len(frame) else "  none"

        lines = ["inputs"]
        for item in self.summary["inputs"]:
            ident = item.get("corpus_sha256", item.get("sha256", ""))[:8]
            dropped = item.get("dropped_empty", 0)
            note = f"  ({dropped} rows dropped: empty side)" if dropped else ""
            lines.append(f"  {item['name']:40s} {ident:8s} {item['rows']:>7,d} rows{note}")
        lines.append(f"  {'total':40s} {'':8s} {self.summary['rows']:>7,d} rows")

        lines += ["", "exact duplicates after folding (blocks of identical text)"]
        exact = pd.DataFrame(self.summary["exact"]).T.rename_axis("side").reset_index()
        lines.append(exact.to_string(index=False))
        lines += ["rows in repeated blocks, per source:", pivot(self.summary["exact_by_source"], "source")]

        lines += ["", "copied pairs: Jaccard(qom words, es words) of one row >= threshold"]
        copied = pd.DataFrame(
            [{"threshold": t, "rows": n} for t, n in self.summary["copied"]["rows"].items()]
        )
        lines.append(copied.to_string(index=False))
        lines += ["rows per source:", pivot(self.summary["copied"]["rows_by_source"], "source")]

        lines += ["", "near-duplicate rows: Jaccard between two rows >= threshold "
                      "(cross_group: the two rows are not known to share a discourse unit)"]
        rows = []
        for side, per_threshold in self.summary["near"].items():
            for threshold, counts in per_threshold.items():
                rows.append({"side": side, "threshold": threshold, **counts})
        lines.append(pd.DataFrame(rows).to_string(index=False))
        lines += ["pairs per pair of sources, either side:", pivot(self.summary["near_by_source"], "sources")]
        return "\n".join(lines)


def _check_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ROW_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"duplicate check needs the columns {missing}")
    if frame["row_id"].duplicated().any():
        raise ValueError("row_id must be unique across every input")
    return frame.reset_index(drop=True)


def duplicates_report(
    frame: pd.DataFrame,
    *,
    config: DuplicatesConfig = DUPLICATES,
    inputs: tuple[dict, ...] = (),
) -> DuplicatesReport:
    """Run the three checks on rows shaped like :data:`ROW_COLUMNS`."""
    frame = _check_frame(frame)
    thresholds = tuple(sorted({t for t in config.thresholds if t >= config.floor - 1e-9} | {config.floor}))
    bags = {side: [bag_of_words(t) for t in frame[side]] for side in SIDES}

    exact = exact_duplicates(frame)

    scores = pd.Series(copied_scores(bags), index=frame.index, name="jaccard_qom_es")
    copied = frame.loc[scores >= config.floor - 1e-9, list(ROW_COLUMNS)].copy()
    copied.insert(5, "jaccard_qom_es", scores[copied.index])
    copied = copied.sort_values(["jaccard_qom_es", "row_id"], ascending=[False, True],
                                kind="mergesort").reset_index(drop=True)

    near = near_duplicates(frame, bags, config.floor)

    summary = {
        "inputs": list(inputs),
        "rows": int(len(frame)),
        "method": {
            "words": "word_tokens(normalize_text(text).casefold())",
            "similarity": "textdistance.Jaccard(qval=1, as_set=False): bag of words, multiplicities kept",
            "exact": "fold_for_matching(text) on each side and on the pair",
            "textdistance": importlib.metadata.version("textdistance"),
            "floor": config.floor,
            "thresholds": [_label(t) for t in thresholds],
        },
        "exact": _exact_summary(exact),
        "exact_by_source": _exact_by_source(exact),
        "copied": {
            "rows": {_label(t): int((scores >= t - 1e-9).sum()) for t in thresholds},
            "rows_by_source": {
                _label(t): {
                    str(k): int(n)
                    for k, n in frame.loc[scores >= t - 1e-9, "source_doc"].value_counts().sort_index().items()
                }
                for t in thresholds
            },
        },
        "near": _near_summary(near, thresholds),
        "near_by_source": _near_by_source(near, thresholds),
    }
    return DuplicatesReport(exact=exact, copied=copied, near=near, summary=summary)


EXACT_NAME = "exact_duplicates.csv"
COPIED_NAME = "copied_pairs.csv"
NEAR_NAME = "near_duplicates.csv"
SUMMARY_NAME = "summary.json"


def write_report(report: DuplicatesReport, out_dir: str | Path) -> Path:
    """Write the three tables (ignored by git: they carry corpus text) and the summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.exact.to_csv(out_dir / EXACT_NAME, index=False)
    report.copied.to_csv(out_dir / COPIED_NAME, index=False)
    report.near.to_csv(out_dir / NEAR_NAME, index=False)
    (out_dir / SUMMARY_NAME).write_text(
        json.dumps(report.summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out_dir


# -------------------------------------------------------------------------- inputs


def corpus_rows(artifact: CorpusArtifact) -> tuple[pd.DataFrame, dict]:
    """A built corpus as check rows, with its identity."""
    frame = artifact.frame
    rows = pd.DataFrame(
        {
            "row_id": frame["pair_uid"].astype("string"),
            "source_doc": frame["source_doc"].astype("string"),
            "source_ref": frame["source_ref"].astype("string"),
            "group_id": frame["group_id"].astype("string"),
            "record_kind": frame["record_kind"].astype("string"),
            "qom": frame["qom"].astype("string"),
            "es": frame["es"].astype("string"),
        }
    )
    info = {"name": artifact.name, "corpus_sha256": artifact.sha256, "rows": int(len(rows))}
    return rows, info


def read_table(path: str | Path, spec: TableSpec = SPEECH_TABLE) -> tuple[pd.DataFrame, dict]:
    """A parallel table outside the corpus registry, normalized the way ingestion
    normalizes, with rows missing a side dropped and counted.

    The rows carry no ``group_id``: how the table divides into discourse units is not
    decided by the table itself.
    """
    path = Path(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in (spec.col_qom, spec.col_es) if c not in raw.columns]
    if missing:
        raise KeyError(f"{path.name} lacks the columns {missing}; it has {list(raw.columns)}")
    kept = raw.loc[text_mask(raw, spec.col_qom, spec.col_es)]

    present = [c for c in spec.ref_columns if c in kept.columns]
    if present:
        refs = [" ".join(str(v).strip() for v in values) for values in kept[present].itertuples(index=False)]
    else:
        refs = [str(i) for i in kept.index]
    qom_raw = kept[spec.col_qom].str.strip().tolist()
    es_raw = kept[spec.col_es].str.strip().tolist()
    rows = pd.DataFrame(
        {
            "row_id": [pair_uid(path.stem, r, q, e) for r, q, e in zip(refs, qom_raw, es_raw)],
            "source_doc": path.stem,
            "source_ref": refs,
            "group_id": pd.Series(pd.NA, index=range(len(refs)), dtype="string"),
            "record_kind": "line",
            "qom": [normalize_text(q) for q in qom_raw],
            "es": [normalize_text(e) for e in es_raw],
        }
    )
    for column in ("row_id", "source_doc", "source_ref", "record_kind", "qom", "es"):
        rows[column] = rows[column].astype("string")
    info = {
        "name": path.name,
        "sha256": sha256_file(path),
        "rows": int(len(rows)),
        "dropped_empty": int(len(raw) - len(rows)),
    }
    return rows, info
