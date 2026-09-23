"""Regression tests against the real corpus.

Skipped automatically when ``corpus/`` is absent (it is gitignored for copyright), so a
fresh clone still runs the unit suite.
"""

from __future__ import annotations

import pandas as pd
import pytest

from qomlaq.checks import CheckFailure
from qomlaq.config import SOURCES
from qomlaq.corpus import CorpusArtifact, CorpusError, build_corpus, load_corpus, save_corpus
from qomlaq.hashing import CORPUS_IDENTITY_COLUMNS, sha256_frame
from qomlaq.ingest import read_lines
from qomlaq.splits import SplitIntegrityError, load_split, write_split

pytestmark = pytest.mark.corpus


# --------------------------------------------------------------------------- ingestion


def test_metadata_is_index_aligned(corpus_dir):
    """Each kept row carries its own source row's ids.

    Arte verbal drops 4 rows mid-sheet, so attaching ids by position would shift the
    fragment id of every later row.
    """
    spec = next(s for s in SOURCES if s.name == "Arte verbal qom")
    raw = pd.read_excel(corpus_dir / spec.filename, sheet_name=spec.sheet)
    ingested = read_lines(spec, corpus_dir)

    assert len(ingested) < len(raw), "expected some rows to be dropped, otherwise this is vacuous"

    from qomlaq.normalize import clean_id

    def as_optional(value):
        return None if pd.isna(value) else value

    for _, row in ingested.iterrows():
        source_row = int(row["source_row"])
        for column in ("id_fragmento", "id_seccion", "id_capitulo"):
            expected = clean_id(raw.iloc[source_row][column])
            assert as_optional(row[column]) == expected, (
                f"row {source_row}: {column} is {row[column]!r}, source says {expected!r}"
            )


def test_every_declared_source_is_present_and_used(built_corpus):
    ingested = set(built_corpus.frame["source_doc"])
    assert ingested == {s.name for s in SOURCES}


def test_missing_source_is_an_error(tmp_path):
    """A missing file must fail the build, never produce a smaller corpus."""
    with pytest.raises(CorpusError):
        build_corpus(tmp_path, "text-v1")


def test_corpus_hash_survives_ascii_folded_filenames(corpus_dir, built_corpus, tmp_path):
    """Some copy tools rewrite filenames to ASCII.

    'Educación Sanitaria Intercultural.xlsx' can arrive as 'Educacion ...', so a registry
    matched on exact names would fail to find two of the seven sources. The same corpus
    must build, byte for byte, from either spelling.
    """
    import shutil
    import unicodedata

    def ascii_fold(name: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", name) if not unicodedata.combining(c)
        )

    for path in corpus_dir.iterdir():
        if path.suffix.lower() in (".xlsx", ".csv"):
            shutil.copy2(path, tmp_path / ascii_fold(path.name))

    folded_names = {p.name for p in tmp_path.iterdir()}
    assert "Educacion Sanitaria Intercultural.xlsx" in folded_names, "fixture is vacuous"

    assert build_corpus(tmp_path, built_corpus.name).sha256 == built_corpus.sha256


# ------------------------------------------------------------------------ normalization


def test_no_curly_apostrophes_survive(built_corpus):
    text = built_corpus.frame["qom"].str.cat(built_corpus.frame["es"], sep=" ")
    for variant in ("’", "‘", "ʼ"):
        assert not text.str.contains(variant, regex=False).any()


def test_normalization_actually_had_work_to_do(built_corpus):
    """Guards against the test above passing vacuously."""
    raw = built_corpus.frame["qom_raw"].str.cat(built_corpus.frame["es_raw"], sep=" ")
    assert raw.str.contains("’", regex=False).any()


# ------------------------------------------------------------------------------- QC


def test_qc_ledger_reconciles(built_corpus):
    ledger = built_corpus.manifest["qc_ledger"]
    for stage in ledger:
        assert stage["rows_before"] - stage["removed"] == stage["rows_after"]
    for earlier, later in zip(ledger, ledger[1:]):
        assert earlier["rows_after"] == later["rows_before"]
    assert ledger[-1]["rows_after"] == built_corpus.manifest["counts"]["rows"]


def test_min_token_filter_is_applied(built_corpus):
    frame = built_corpus.frame
    assert frame["qom_len_t"].min() >= 2
    assert frame["es_len_t"].min() >= 2


def test_length_ratio_within_bounds(built_corpus):
    ratio = built_corpus.frame["len_ratio_t"]
    assert ratio.min() >= 0.15 and ratio.max() <= 6.0


# ---------------------------------------------------------------------------- groups


def test_no_line_level_groups(built_corpus):
    assert (built_corpus.frame["unit_level"] == "line").sum() == 0


def test_titles_share_a_group_with_their_content(built_corpus):
    frame = built_corpus.frame
    titles = frame[frame["record_kind"] == "title"]
    assert len(titles) > 0
    line_groups = set(frame.loc[frame["record_kind"] == "line", "group_id"])
    assert titles["group_id"].isin(line_groups).all()


def test_bible_groups_are_book_chapters(built_corpus):
    bible = built_corpus.frame[built_corpus.frame["source_doc"] == "La Biblia"]
    expected = bible.groupby(["libro", "capitulo"], observed=True).ngroups
    # Union-find may merge chapters that share an identical verse, so groups <= chapters.
    assert bible["group_id"].nunique() <= expected
    assert bible["group_id"].str.startswith("la_biblia__chap__").all()


def test_principito_groups_are_chapters(built_corpus):
    frame = built_corpus.frame
    principito = frame[frame["source_doc"] == "El Principito"]
    assert principito["group_id"].nunique() == principito["capitulo"].nunique()


def test_udhr_uses_its_parte_column(built_corpus):
    """UDHR articles ('parte') are its discourse units."""
    udhr = built_corpus.frame[
        built_corpus.frame["source_doc"] == "La Declaración Universal de los Derechos Humanos"
    ]
    assert udhr["group_id"].nunique() > 1
    assert (udhr["unit_level"] == "parte").any()


def test_no_group_spans_two_sources(built_corpus):
    counts = built_corpus.frame.groupby("group_id", observed=True)["source_doc"].nunique()
    assert counts.max() == 1


# --------------------------------------------------------------------------- artifacts


def test_corpus_round_trips_through_disk(built_corpus, tmp_path):
    save_corpus(built_corpus, tmp_path)
    reloaded = load_corpus(tmp_path, built_corpus.name, expect_corpus_sha256=built_corpus.sha256)
    assert reloaded.sha256 == built_corpus.sha256
    assert len(reloaded.frame) == len(built_corpus.frame)


def test_tampered_corpus_is_rejected(built_corpus, tmp_path):
    save_corpus(built_corpus, tmp_path)
    table = tmp_path / "corpus" / built_corpus.name / "corpus.csv"
    frame = pd.read_csv(table)
    frame.loc[0, "es"] = "tampered"
    frame.to_csv(table, index=False)
    with pytest.raises(CorpusError):
        load_corpus(tmp_path, built_corpus.name)


def test_corpus_artifact_cannot_be_constructed_directly(built_corpus):
    """Bypassing the constructor would bypass hash verification."""
    with pytest.raises(CorpusError):
        CorpusArtifact(frame=built_corpus.frame, manifest=built_corpus.manifest)


def test_corpus_hash_is_deterministic(built_corpus):
    recomputed = sha256_frame(built_corpus.frame, CORPUS_IDENTITY_COLUMNS)
    assert recomputed == built_corpus.sha256


# ------------------------------------------------------------------------------ splits


def test_headline_splits_pass_every_check(headline_splits):
    for split_id, (_parts, _manifest, report) in headline_splits.items():
        assert report.passed, f"{split_id}:\n{report}"


def test_chapter_grouping_eliminates_verse_adjacency(headline_splits):
    for split_id, (_parts, manifest, _report) in headline_splits.items():
        rates = manifest["checks"]["bible_verse_adjacency"]["value"] or {}
        for partition, rate in rates.items():
            assert rate == 0.0, f"{split_id} {partition}: adjacency {rate}"


def test_verse_control_shows_verse_adjacency(splits):
    """The control exists to quantify what chapter grouping is worth."""
    _parts, manifest, _report = splits["base_bible/stratified__verse"]
    rates = manifest["checks"]["bible_verse_adjacency"]["value"]
    assert rates["test"] > 0.9, "verse-level grouping should leave most test verses adjacent"
    assert manifest["purpose"] == "control"


def test_forced_holdout_never_appears_in_training(headline_splits):
    for split_id, (parts, _manifest, _report) in headline_splits.items():
        if "base" not in split_id:
            continue
        assert (parts["train"]["source_doc"] == "Las Aventuras de Copaic").sum() == 0
        assert (parts["dev"]["source_doc"] == "Las Aventuras de Copaic").sum() == 0


def test_realized_fractions_are_close_to_target(headline_splits):
    for split_id, (_parts, manifest, _report) in headline_splits.items():
        drift = manifest["checks"]["realized_fractions"]["value"]["drift"]
        assert max(abs(v) for v in drift.values()) <= 0.02, f"{split_id}: {drift}"


def test_splits_are_reproducible(built_corpus):
    from qomlaq.splits import make_split

    first = make_split(built_corpus, config="base", strategy="stratified")[1]
    second = make_split(built_corpus, config="base", strategy="stratified")[1]
    assert first["split_sha256"] == second["split_sha256"]
    assert first["partition_sha256"] == second["partition_sha256"]


def test_split_round_trips_and_rejects_a_wrong_hash(built_corpus, tmp_path):
    manifest = write_split(built_corpus, tmp_path, config="base", strategy="stratified")
    loaded = load_split(
        tmp_path,
        config="base",
        strategy="stratified",
        expect_split_sha256=manifest["split_sha256"],
    )
    assert loaded.sha256 == manifest["split_sha256"]

    with pytest.raises(SplitIntegrityError):
        load_split(
            tmp_path, config="base", strategy="stratified", expect_split_sha256="0" * 64
        )


def test_tampered_partition_is_rejected(built_corpus, tmp_path):
    manifest = write_split(built_corpus, tmp_path, config="base", strategy="stratified")
    test_csv = tmp_path / "splits" / built_corpus.sha8 / "base" / "stratified" / "test.csv"
    frame = pd.read_csv(test_csv)
    frame = frame.iloc[:-1]  # drop a row
    frame.to_csv(test_csv, index=False)
    with pytest.raises(SplitIntegrityError):
        load_split(
            tmp_path,
            config="base",
            strategy="stratified",
            expect_split_sha256=manifest["split_sha256"],
        )


def test_split_path_encodes_the_corpus_hash(built_corpus, tmp_path):
    write_split(built_corpus, tmp_path, config="base", strategy="stratified")
    assert (tmp_path / "splits" / built_corpus.sha8).is_dir()


def test_evaluation_sets_are_disjoint_from_their_train_partition(headline_splits):
    from qomlaq.checks import check_overlap_with_train

    for split_id, (parts, _manifest, _report) in headline_splits.items():
        train_uids = set(parts["train"]["pair_uid"])
        for partition in ("dev", "test"):
            result = check_overlap_with_train(parts[partition], train_uids, label=split_id)
            assert result.passed, str(result)
