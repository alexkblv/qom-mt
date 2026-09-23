"""Every check is fed the failure it exists to catch.

A check that cannot fail is worth nothing, so each one here is proven to fail on a
deliberately broken input as well as pass on a good one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qomlaq.checks import (
    CheckFailure,
    CheckReport,
    check_bible_adjacency,
    check_forced_partitions,
    check_group_disjoint,
    check_no_line_groups,
    check_overlap_with_train,
    check_pair_disjoint,
    check_pair_disjoint_normalized,
    check_partition_coverage,
    check_realized_fractions,
    check_titles_have_content,
    run_split_checks,
)


def _row(uid, group, qom="q", es="e", source="S", kind="line", **extra):
    row = {
        "pair_uid": uid,
        "group_id": group,
        "qom": qom,
        "es": es,
        "source_doc": source,
        "record_kind": kind,
        "unit_level": "fragmento",
        "norm_key": f"{qom}\x1f{es}",
    }
    row.update(extra)
    return row


def _parts(train, dev, test):
    return {
        "train": pd.DataFrame(train),
        "dev": pd.DataFrame(dev),
        "test": pd.DataFrame(test),
    }


def test_group_disjoint_catches_a_shared_group():
    good = _parts([_row("a", "g1")], [_row("b", "g2")], [_row("c", "g3")])
    assert check_group_disjoint(good).passed

    bad = _parts([_row("a", "g1")], [_row("b", "g2")], [_row("c", "g1")])
    result = check_group_disjoint(bad)
    assert not result.passed and result.value == {"train/test": 1}


def test_pair_disjoint_catches_a_repeated_pair():
    good = _parts([_row("a", "g1")], [_row("b", "g2")], [_row("c", "g3")])
    assert check_pair_disjoint(good).passed

    bad = _parts([_row("a", "g1")], [_row("b", "g2")], [_row("a", "g3")])
    assert not check_pair_disjoint(bad).passed


def test_normalized_check_catches_what_exact_matching_misses():
    """Two renderings of one sentence differing only in quoting and case."""
    train = [_row("a", "g1", qom="Na’aq", es="El día")]
    test = [_row("b", "g2", qom="na'aq", es="el dia")]
    parts = _parts(train, [_row("c", "g3", qom="x", es="y")], test)

    assert check_pair_disjoint(parts).passed  # exact match sees nothing

    parts["train"]["norm_key"] = ["na'aq\x1fel dia"]
    parts["test"]["norm_key"] = ["na'aq\x1fel dia"]
    assert not check_pair_disjoint_normalized(parts).passed


def test_bible_adjacency_catches_neighbouring_verses():
    def bible(uid, group, verse):
        return _row(uid, group, source="La Biblia", libro="GEN", capitulo="1", versiculo=verse)

    apart = _parts([bible("a", "g1", 1)], [bible("b", "g2", 50)], [bible("c", "g3", 99)])
    assert check_bible_adjacency(apart).passed

    adjacent = _parts([bible("a", "g1", 1)], [bible("b", "g2", 50)], [bible("c", "g3", 2)])
    result = check_bible_adjacency(adjacent)
    assert not result.passed and result.value["test"] == 1.0


def test_forced_partition_check_catches_a_leaked_probe():
    forced = {"Las Aventuras de Copaic": "test"}
    good = _parts([_row("a", "g1")], [_row("b", "g2")], [_row("c", "g3", source="Las Aventuras de Copaic")])
    assert check_forced_partitions(good, forced).passed

    bad = _parts([_row("a", "g1", source="Las Aventuras de Copaic")], [_row("b", "g2")], [_row("c", "g3")])
    assert not check_forced_partitions(bad, forced).passed


def test_realized_fraction_check_catches_drift():
    target = {"train": 0.8, "dev": 0.1, "test": 0.1}
    balanced = _parts([_row(str(i), "g1") for i in range(80)],
                      [_row(f"d{i}", "g2") for i in range(10)],
                      [_row(f"t{i}", "g3") for i in range(10)])
    assert check_realized_fractions(balanced, target).passed

    skewed = _parts([_row(str(i), "g1") for i in range(76)],
                    [_row(f"d{i}", "g2") for i in range(11)],
                    [_row(f"t{i}", "g3") for i in range(13)])
    assert not check_realized_fractions(skewed, target).passed


def test_coverage_check_catches_missing_and_duplicated_rows():
    expected = {"a", "b", "c"}
    complete = _parts([_row("a", "g1")], [_row("b", "g2")], [_row("c", "g3")])
    assert check_partition_coverage(complete, expected).passed

    missing = _parts([_row("a", "g1")], [_row("b", "g2")], [])
    assert not check_partition_coverage(missing, expected).passed

    duplicated = _parts([_row("a", "g1")], [_row("b", "g2")], [_row("a", "g3"), _row("c", "g4")])
    assert not check_partition_coverage(duplicated, expected).passed


def test_line_group_check_catches_one_row_groups():
    good = pd.DataFrame([_row("a", "g1")])
    assert check_no_line_groups(good).passed

    bad = pd.DataFrame([_row("a", "g1", unit_level="line")])
    assert not check_no_line_groups(bad).passed


def test_title_check_catches_a_title_alone_in_its_group():
    good = pd.DataFrame([_row("a", "g1"), _row("t", "g1", kind="title")])
    assert check_titles_have_content(good).passed

    bad = pd.DataFrame([_row("a", "g1"), _row("t", "g_orphan", kind="title")])
    assert not check_titles_have_content(bad).passed


def test_train_overlap_check_catches_a_contaminated_eval_set():
    evaluated = pd.DataFrame([_row("a", "g1"), _row("b", "g2")])
    assert check_overlap_with_train(evaluated, {"z"}).passed

    result = check_overlap_with_train(evaluated, {"a", "z"})
    assert not result.passed
    assert result.value == {"overlap": 1, "n": 2, "frac": 0.5}


def test_report_raises_only_on_error_severity():
    report = CheckReport()
    report.add(check_group_disjoint(_parts([_row("a", "g1")], [_row("b", "g2")], [_row("c", "g1")])))
    assert not report.passed
    with pytest.raises(CheckFailure):
        report.raise_for_failures()


def test_full_battery_passes_on_a_clean_split():
    parts = _parts(
        [_row(f"tr{i}", f"g{i}") for i in range(80)],
        [_row(f"dv{i}", f"h{i}") for i in range(10)],
        [_row(f"te{i}", f"k{i}") for i in range(10)],
    )
    for name, df in parts.items():
        df["norm_key"] = [f"{name}{i}" for i in range(len(df))]
    report = run_split_checks(
        parts,
        target_fractions={"train": 0.8, "dev": 0.1, "test": 0.1},
        forced_partitions={},
        expected_uids=set(pd.concat(parts.values())["pair_uid"]),
    )
    assert report.passed, str(report)
