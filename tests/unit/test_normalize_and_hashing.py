"""Normalization must unify the glottal stop without touching Qom orthography."""

from __future__ import annotations

import unicodedata

import pandas as pd
import pytest

from qomlaq.hashing import canonical_bytes, pair_uid, sha256_frame
from qomlaq.normalize import (
    PROTECTED_GRAPHEMES,
    clean_id,
    fold_for_matching,
    normalize_text,
    tokens,
)


@pytest.mark.parametrize("variant", ["’", "‘", "ʼ", "´", "`", "′"])
def test_apostrophe_variants_unify_to_u0027(variant):
    assert normalize_text(f"na{variant}aq") == "na'aq"


def test_protected_graphemes_survive_normalization():
    """The corpus keeps orthographic variation on purpose; normalization must not merge it."""
    for grapheme in PROTECTED_GRAPHEMES:
        text = f"a{grapheme}b"
        assert normalize_text(text) == unicodedata.normalize("NFC", text)


def test_orthographic_variants_stay_distinct():
    """NFC composes, but must not turn one Qom variant into another."""
    assert normalize_text("ỹ") != normalize_text("ȳ")
    assert normalize_text("ñ") != normalize_text("n")


def test_decomposed_and_precomposed_forms_converge():
    decomposed = "ỹ"  # y + combining tilde
    assert normalize_text(decomposed) == normalize_text("ỹ")


def test_whitespace_is_collapsed_and_stripped():
    assert normalize_text("  a\t b\n c  ") == "a b c"


def test_normalize_is_idempotent():
    text = "  N’axã ỹ  qo'  "
    assert normalize_text(normalize_text(text)) == normalize_text(text)


def test_fold_for_matching_ignores_case_and_punctuation():
    assert fold_for_matching("¡Dibújame un cordero!") == fold_for_matching("dibújame un cordero")


def test_fold_for_matching_keeps_diacritics():
    """Folding must not strip accents: 'ỹ' vs 'y' is a real Qom orthographic contrast."""
    assert fold_for_matching("ỹale") != fold_for_matching("yale")


def test_fold_for_matching_unifies_apostrophes():
    assert fold_for_matching("na’aq") == fold_for_matching("na'aq")


@pytest.mark.parametrize("placeholder", ["N/A", "n/a", " none ", "NULL", "-", "", "nan"])
def test_clean_id_rejects_placeholders(placeholder):
    assert clean_id(placeholder) is None


def test_clean_id_strips_trailing_separator():
    """Arte verbal writes 'C.1.' in Lineas and 'C.1' in Fragmentos for the same unit."""
    assert clean_id("C.1.") == clean_id("C.1") == "C.1"


def test_clean_id_normalizes_excel_floats():
    assert clean_id("3.0") == "3"


def test_pair_uid_is_stable_and_distinguishes_sources():
    a = pair_uid("A", "ref", "qom", "es")
    assert a == pair_uid("A", "ref", "qom", "es")
    assert a != pair_uid("B", "ref", "qom", "es")


def test_canonical_bytes_is_row_and_column_order_independent(toy_frame):
    cols = ["pair_uid", "qom", "es"]
    shuffled = toy_frame.iloc[::-1].loc[:, ["es", "qom", "pair_uid", "group_id"]]
    assert canonical_bytes(toy_frame, cols) == canonical_bytes(shuffled, cols)


def test_frame_hash_changes_when_content_changes(toy_frame):
    cols = ["pair_uid", "qom", "es"]
    before = sha256_frame(toy_frame, cols)
    edited = toy_frame.copy()
    edited.loc[0, "qom"] = "different"
    assert sha256_frame(edited, cols) != before


def test_canonical_bytes_rejects_missing_columns(toy_frame):
    with pytest.raises(KeyError):
        canonical_bytes(toy_frame, ["pair_uid", "absent"])


def test_tokens_split_apostrophe_as_its_own_token():
    """Documents the reporting convention behind the corpus token counts."""
    assert tokens("l'aqtaqa") == ["l", "'", "aqtaqa"]
