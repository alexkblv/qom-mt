"""Every assertion the pipeline makes, in one place.

Notebooks call :func:`run_split_checks` and print the report; the test suite calls the
same functions on the same artifacts. There is deliberately no second implementation of
a check written inline in a notebook.

Checks return results rather than raising, so a report can show every outcome at once;
:meth:`CheckReport.raise_for_failures` turns the report into a hard failure.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

PARTITIONS = ("train", "dev", "test")


class CheckFailure(AssertionError):
    pass


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str
    value: object = None
    severity: str = "error"  # "error" fails the report; "info" is reported only

    def __str__(self) -> str:
        mark = "PASS" if self.passed else ("WARN" if self.severity == "info" else "FAIL")
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class CheckReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, result: CheckResult) -> CheckResult:
        self.results.append(result)
        return result

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def passed(self) -> bool:
        return not self.failures

    def raise_for_failures(self) -> None:
        if self.failures:
            lines = "\n".join(f"  - {r}" for r in self.failures)
            raise CheckFailure(f"{len(self.failures)} check(s) failed:\n{lines}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"check": r.name, "result": "pass" if r.passed else "FAIL", "detail": r.detail}
                for r in self.results
            ]
        )

    def summary(self) -> dict:
        return {r.name: {"passed": r.passed, "value": r.value} for r in self.results}

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self.results)


def _pairs(parts: Mapping[str, pd.DataFrame]):
    return itertools.combinations([p for p in PARTITIONS if p in parts], 2)


def _values(df: pd.DataFrame, column: str) -> set:
    """Column contents as a set, tolerating an empty partition (which has no columns)."""
    if df.empty or column not in df.columns:
        return set()
    return set(df[column].dropna())


def check_group_disjoint(parts: Mapping[str, pd.DataFrame]) -> CheckResult:
    """No discourse unit may appear in two partitions."""
    offenders: dict[str, int] = {}
    for a, b in _pairs(parts):
        shared = _values(parts[a], "group_id") & _values(parts[b], "group_id")
        if shared:
            offenders[f"{a}/{b}"] = len(shared)
    return CheckResult(
        "group_disjoint",
        not offenders,
        "no group spans partitions" if not offenders else f"shared groups: {offenders}",
        value=offenders,
    )


def check_pair_disjoint(parts: Mapping[str, pd.DataFrame]) -> CheckResult:
    """No identical pair may appear in two partitions."""
    offenders: dict[str, int] = {}
    for a, b in _pairs(parts):
        shared = _values(parts[a], "pair_uid") & _values(parts[b], "pair_uid")
        if shared:
            offenders[f"{a}/{b}"] = len(shared)
    return CheckResult(
        "pair_disjoint_exact",
        not offenders,
        "no exact pair spans partitions" if not offenders else f"shared pairs: {offenders}",
        value=offenders,
    )


def check_pair_disjoint_normalized(parts: Mapping[str, pd.DataFrame]) -> CheckResult:
    """Same, after casefolding, apostrophe unification and punctuation stripping.

    Two copies of a pair that differ only in quoting slip past an exact-match check.
    """
    if not all("norm_key" in df.columns for df in parts.values() if not df.empty):
        return CheckResult("pair_disjoint_normalized", True, "norm_key absent, skipped")
    offenders: dict[str, int] = {}
    for a, b in _pairs(parts):
        shared = _values(parts[a], "norm_key") & _values(parts[b], "norm_key")
        if shared:
            offenders[f"{a}/{b}"] = len(shared)
    return CheckResult(
        "pair_disjoint_normalized",
        not offenders,
        "no normalized duplicate spans partitions"
        if not offenders
        else f"normalized duplicates: {offenders}",
        value=offenders,
    )


def check_single_side_overlap(parts: Mapping[str, pd.DataFrame]) -> CheckResult:
    """Count rows whose Qom side *or* Spanish side alone also occurs in train.

    Reported rather than enforced: a shared short sentence is not necessarily leakage,
    but the count must be visible.
    """
    if "train" not in parts:
        return CheckResult("single_side_overlap", True, "no train partition")
    train_qom = _values(parts["train"], "qom")
    train_es = _values(parts["train"], "es")
    counts: dict[str, dict[str, int]] = {}
    for name in ("dev", "test"):
        if name not in parts:
            continue
        df = parts[name]
        if df.empty or "qom" not in df.columns:
            counts[name] = {"qom": 0, "es": 0}
            continue
        counts[name] = {
            "qom": int(df["qom"].isin(train_qom).sum()),
            "es": int(df["es"].isin(train_es).sum()),
        }
    total = sum(v["qom"] + v["es"] for v in counts.values())
    return CheckResult(
        "single_side_overlap",
        True,  # informational
        f"rows sharing one side with train: {counts}",
        value=counts,
        severity="info" if total else "error",
    )


def check_bible_adjacency(parts: Mapping[str, pd.DataFrame]) -> CheckResult:
    """Share of held-out Bible verses whose neighboring verse is in train.

    Chapter grouping should drive this to zero. Verse-level grouping leaves most
    held-out verses one verse away from a training example.
    """
    needed = {"libro", "capitulo", "versiculo"}
    if "train" not in parts or parts["train"].empty or not needed.issubset(parts["train"].columns):
        return CheckResult("bible_verse_adjacency", True, "no Bible columns, skipped")

    def verses(df: pd.DataFrame) -> set[tuple[str, str, int]]:
        if df.empty:
            return set()
        # Only verse-structured rows carry all three coordinates.
        bible = df.dropna(subset=sorted(needed))
        out: set[tuple[str, str, int]] = set()
        for libro, cap, ver in zip(bible["libro"], bible["capitulo"], bible["versiculo"]):
            try:
                out.add((str(libro), str(cap), int(float(ver))))
            except (TypeError, ValueError):
                continue
        return out

    train = verses(parts["train"])
    if not train:
        return CheckResult("bible_verse_adjacency", True, "no Bible rows in train")

    detail: dict[str, float] = {}
    worst = 0.0
    for name in ("dev", "test"):
        if name not in parts:
            continue
        held = verses(parts[name])
        if not held:
            continue
        adjacent = sum(
            1
            for (libro, cap, ver) in held
            if (libro, cap, ver - 1) in train or (libro, cap, ver + 1) in train
        )
        rate = adjacent / len(held)
        detail[name] = round(rate, 4)
        worst = max(worst, rate)
    return CheckResult(
        "bible_verse_adjacency",
        worst == 0.0,
        f"adjacent-verse-in-train rate: {detail}",
        value=detail,
    )


def check_forced_partitions(
    parts: Mapping[str, pd.DataFrame], forced: Mapping[str, str]
) -> CheckResult:
    """Sources declared as held-out probes must appear only in their partition."""
    problems: dict[str, dict[str, int]] = {}
    for source, partition in forced.items():
        for name, df in parts.items():
            if name == partition or df.empty or "source_doc" not in df.columns:
                continue
            n = int((df["source_doc"] == source).sum())
            if n:
                problems.setdefault(source, {})[name] = n
    return CheckResult(
        "forced_partitions",
        not problems,
        f"forced sources confined to their partition: {dict(forced)}"
        if not problems
        else f"leaked out of their partition: {problems}",
        value=problems,
    )


def check_realized_fractions(
    parts: Mapping[str, pd.DataFrame], target: Mapping[str, float], tolerance: float = 0.02
) -> CheckResult:
    """Realized row proportions must be close to the configured ones."""
    total = sum(len(df) for df in parts.values())
    if not total:
        return CheckResult("realized_fractions", False, "empty split")
    realized = {name: len(df) / total for name, df in parts.items()}
    drift = {name: round(realized[name] - target.get(name, 0), 4) for name in realized}
    worst = max(abs(v) for v in drift.values())
    return CheckResult(
        "realized_fractions",
        worst <= tolerance,
        f"realized {{{', '.join(f'{k}: {v:.3f}' for k, v in realized.items())}}} "
        f"vs target {dict(target)} (max drift {worst:.3f})",
        value={"realized": realized, "drift": drift},
    )


def check_partition_coverage(
    parts: Mapping[str, pd.DataFrame], expected_uids: set[str]
) -> CheckResult:
    """Partitions must together reproduce the configuration's rows exactly."""
    seen: set[str] = set()
    duplicated = 0
    for df in parts.values():
        uids = _values(df, "pair_uid")
        duplicated += len(seen & uids)
        seen |= uids
    missing = len(expected_uids - seen)
    extra = len(seen - expected_uids)
    ok = missing == 0 and extra == 0 and duplicated == 0
    return CheckResult(
        "partition_coverage",
        ok,
        "partitions exactly cover the configuration"
        if ok
        else f"missing={missing} extra={extra} duplicated={duplicated}",
        value={"missing": missing, "extra": extra, "duplicated": duplicated},
    )


def check_no_line_groups(frame: pd.DataFrame) -> CheckResult:
    """No row may be its own discourse unit."""
    n = int((frame.get("unit_level") == "line").sum()) if "unit_level" in frame else 0
    return CheckResult(
        "no_line_level_groups",
        n == 0,
        "no one-row groups" if n == 0 else f"{n} rows are their own group",
        value=n,
    )


def check_titles_have_content(frame: pd.DataFrame) -> CheckResult:
    """Every title shares a group with the lines it names."""
    if "record_kind" not in frame.columns:
        return CheckResult("titles_share_parent_group", True, "no title rows")
    titles = frame[frame["record_kind"] == "title"]
    if titles.empty:
        return CheckResult("titles_share_parent_group", True, "no title rows")
    line_groups = set(frame.loc[frame["record_kind"] == "line", "group_id"])
    orphans = int((~titles["group_id"].isin(line_groups)).sum())
    return CheckResult(
        "titles_share_parent_group",
        orphans == 0,
        f"all {len(titles)} titles grouped with their content"
        if orphans == 0
        else f"{orphans} titles form groups of their own",
        value=orphans,
    )


def check_overlap_with_train(
    evaluated: pd.DataFrame, train_uids: set[str], *, label: str = "eval"
) -> CheckResult:
    """Is this evaluation set actually held out from the model's training data?"""
    uids = _values(evaluated, "pair_uid")
    overlap = uids & train_uids
    frac = len(overlap) / len(uids) if uids else 0.0
    return CheckResult(
        "eval_disjoint_from_train",
        not overlap,
        f"{label}: {len(overlap)}/{len(uids)} evaluated pairs are in the model's "
        f"training data ({frac:.1%})",
        value={"overlap": len(overlap), "n": len(uids), "frac": round(frac, 4)},
    )


def run_split_checks(
    parts: Mapping[str, pd.DataFrame],
    *,
    target_fractions: Mapping[str, float],
    forced_partitions: Mapping[str, str],
    expected_uids: set[str] | None = None,
) -> CheckReport:
    """The full battery run on every split at build time and in the test suite."""
    report = CheckReport()
    report.add(check_group_disjoint(parts))
    report.add(check_pair_disjoint(parts))
    report.add(check_pair_disjoint_normalized(parts))
    report.add(check_single_side_overlap(parts))
    report.add(check_bible_adjacency(parts))
    report.add(check_forced_partitions(parts, forced_partitions))
    report.add(check_realized_fractions(parts, target_fractions))
    if expected_uids is not None:
        report.add(check_partition_coverage(parts, expected_uids))
    combined = pd.concat([d for d in parts.values() if not d.empty], ignore_index=True)
    report.add(check_no_line_groups(combined))
    report.add(check_titles_have_content(combined))
    return report
