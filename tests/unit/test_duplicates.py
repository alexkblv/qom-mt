"""Duplicate detection: exact repeats after folding, word-level Jaccard through textdistance."""

from __future__ import annotations

import json
import random

import pandas as pd
import pytest
import textdistance

from qomlaq.config import DuplicatesConfig
from qomlaq.duplicates import (
    ROW_COLUMNS,
    bag_of_words,
    duplicates_report,
    exact_duplicates,
    jaccard,
    near_duplicate_pairs,
    read_table,
    write_report,
)


def rows(*items: tuple[str, str, str, str | None]) -> pd.DataFrame:
    """(qom, es, source, group) tuples as check rows."""
    frame = pd.DataFrame(
        [
            (f"r{i}", source, f"{source}_{i}", group, "line", qom, es)
            for i, (qom, es, source, group) in enumerate(items)
        ],
        columns=list(ROW_COLUMNS),
    )
    frame["group_id"] = frame["group_id"].astype("string")
    return frame


def test_bag_of_words_casefolds_drops_punctuation_and_keeps_the_apostrophe_inside_words():
    assert bag_of_words("¡Hola, Alex! ¿Cómo estás?") == ["hola", "alex", "cómo", "estás"]
    assert bag_of_words("Qom l’aqtaqa na’a’q") == ["qom", "l'aqtaqa", "na'a'q"]


def test_jaccard_is_textdistance_on_bags_of_words():
    a, b = bag_of_words("Hola Alex, ¿cómo estás?"), bag_of_words("Hola Pau, ¿cómo estás?")
    assert jaccard(a, b) == pytest.approx(3 / 5)
    assert jaccard(a, b) == textdistance.jaccard(a, b)
    assert jaccard(["na", "na", "da"], ["na", "da"]) == pytest.approx(2 / 3), "multiplicities count"
    assert jaccard([], ["na"]) == 0.0
    assert jaccard([], []) == 0.0


def test_exact_duplicates_fold_case_and_punctuation_and_count_sources_and_groups():
    frame = rows(
        ("Na’aq", "Buen día", "A", "A_1"),
        ("¡NA'AQ!", "otra cosa", "B", "B_7"),
        ("qom", "es", "A", "A_1"),
    )
    out = exact_duplicates(frame)
    assert set(out["side"]) == {"qom"}
    assert sorted(out["row_id"]) == ["r0", "r1"]
    assert out["n_rows"].tolist() == [2, 2]
    assert out["n_sources"].tolist() == [2, 2]
    assert out["n_groups"].tolist() == [2, 2]


def test_an_identical_pair_shows_under_every_side_and_a_missing_group_never_matches_another():
    frame = rows(
        ("na qom", "la persona", "A", "A_1"),
        ("na qom", "la persona", "A", "A_1"),
        ("na qom", "la persona", "T", None),
        ("na qom", "la persona", "T", None),
    )
    out = exact_duplicates(frame)
    assert set(out["side"]) == {"qom", "es", "pair"}
    block = out[out["side"] == "pair"]
    assert len(block) == 4
    # rows without a group id count as four units: two of A_1 share one, the others are their own
    assert block["n_groups"].iloc[0] == 3


def test_exact_duplicates_of_a_clean_frame_is_empty_with_the_same_columns():
    out = exact_duplicates(rows(("a b", "c d", "A", "A_1"), ("e f", "g h", "A", "A_1")))
    assert out.empty
    assert {"side", "dup_id", "n_rows", "n_sources", "n_groups", *ROW_COLUMNS} <= set(out.columns)


@pytest.mark.parametrize("floor", [0.5, 0.6, 0.8, 0.85, 1.0])
def test_the_prefix_index_finds_exactly_the_pairs_a_full_scan_finds(floor):
    """The index only prunes; textdistance over every pair is the reference."""
    rng = random.Random(int(floor * 100))
    vocabulary = [f"w{i}" for i in range(14)]
    bags = [[rng.choice(vocabulary) for _ in range(rng.randint(0, 12))] for _ in range(220)]
    expected = {}
    for i in range(len(bags)):
        for j in range(i + 1, len(bags)):
            score = jaccard(bags[i], bags[j])
            if score >= floor - 1e-9:
                expected[(i, j)] = score
    assert expected, "the random corpus must contain pairs at this floor"
    found = {(i, j): score for i, j, score in near_duplicate_pairs(bags, floor)}
    assert found == expected


def test_near_duplicate_pairs_rejects_an_impossible_floor():
    with pytest.raises(ValueError):
        near_duplicate_pairs([["a"]], 0.0)


def test_report_counts_pairs_and_rows_per_threshold_and_flags_pairs_across_groups():
    frame = rows(
        ("so shiỹaxaua", "Hola Alex, ¿cómo estás?", "A", "A_1"),
        ("so shiỹaxauapi", "Hola Pau, ¿cómo estás?", "A", "A_2"),
        ("na qom na'aq", "los qom hoy", "A", "A_3"),
        ("na qom na'aq", "la gente qom hoy", "A", "A_3"),
        ("estamos en la universidad", "estamos en la universidad hoy", "T", None),
        ("huo'o da ña'achec", "agradecemos", "A", "A_1"),
    )
    report = duplicates_report(frame, config=DuplicatesConfig(floor=0.6, thresholds=(0.5, 0.6, 0.8, 1.0)))

    assert report.summary["method"]["thresholds"] == ["0.60", "0.80", "1.00"], "below the floor is unknown"

    near = report.summary["near"]
    assert near["es"]["0.60"] == {"pairs": 1, "rows": 2, "cross_group_pairs": 1, "cross_group_rows": 2}
    assert near["es"]["0.80"]["pairs"] == 0
    assert near["qom"]["1.00"] == {"pairs": 1, "rows": 2, "cross_group_pairs": 0, "cross_group_rows": 0}
    assert near["either"]["0.60"]["pairs"] == 2

    pair = report.near.set_index(["row_a", "row_b"])
    assert pair.loc[("r0", "r1"), "jaccard_es"] == pytest.approx(0.6)
    assert pair.loc[("r0", "r1"), "jaccard_qom"] == pytest.approx(1 / 3)
    assert not pair.loc[("r0", "r1"), "same_group"]
    assert pair.loc[("r2", "r3"), "same_group"]

    assert report.summary["copied"]["rows"] == {"0.60": 1, "0.80": 1, "1.00": 0}
    assert report.copied["row_id"].tolist() == ["r4"]
    assert report.copied["jaccard_qom_es"].iloc[0] == pytest.approx(4 / 5)

    assert report.summary["exact"]["qom"] == {
        "blocks": 1, "rows": 2, "cross_source_blocks": 0, "cross_group_blocks": 0,
    }
    assert report.summary["rows"] == 6
    assert report.summary["exact_by_source"] == {"qom": {"A": 2}, "es": {}, "pair": {}}
    assert report.summary["copied"]["rows_by_source"]["0.60"] == {"T": 1}
    assert report.summary["near_by_source"]["0.60"] == {"A | A": 2}
    assert report.summary["near_by_source"]["1.00"] == {"A | A": 1}


def test_report_refuses_repeated_row_ids():
    frame = rows(("a b", "c d", "A", "A_1"), ("e f", "g h", "A", "A_1"))
    frame["row_id"] = "same"
    with pytest.raises(ValueError, match="row_id"):
        duplicates_report(frame)


def test_write_report_writes_the_tables_and_a_summary(tmp_path):
    report = duplicates_report(rows(("na qom", "los qom", "A", "A_1"), ("na qom", "la gente", "A", "A_2")))
    out = write_report(report, tmp_path / "dup")
    names = {p.name for p in out.iterdir()}
    assert names == {"exact_duplicates.csv", "copied_pairs.csv", "near_duplicates.csv", "summary.json"}
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["exact"]["qom"]["blocks"] == 1
    assert len(pd.read_csv(out / "exact_duplicates.csv")) == len(report.exact) == 2


def test_read_table_uses_the_speech_schema_and_drops_rows_missing_a_side(tmp_path):
    path = tmp_path / "corpus_x.csv"
    pd.DataFrame(
        {
            "index": [1, 2, 3],
            "location": ["ciudad universitaria"] * 3,
            "speaker": ["Felipe Ceron"] * 3,
            "audio": ["audio 1"] * 3,
            "qom": ["La’ enauac  na", "", "mashe qomi’"],
            "español": ["Hola a todos", "sin qom", "que nos conoce"],
            "revisión": ["", "", ""],
        }
    ).to_csv(path, index=False)
    table, info = read_table(path)
    assert table["source_ref"].tolist() == ["Felipe Ceron audio 1 1", "Felipe Ceron audio 1 3"]
    assert table["qom"].tolist() == ["La' enauac na", "mashe qomi'"], "ingestion's normalization"
    assert table["source_doc"].eq("corpus_x").all()
    assert table["group_id"].isna().all()
    assert table["row_id"].is_unique
    assert info["rows"] == 2 and info["dropped_empty"] == 1 and info["name"] == "corpus_x.csv"


def test_read_table_names_the_missing_columns(tmp_path):
    path = tmp_path / "other.csv"
    pd.DataFrame({"a": ["x"], "b": ["y"]}).to_csv(path, index=False)
    with pytest.raises(KeyError, match="español"):
        read_table(path)
