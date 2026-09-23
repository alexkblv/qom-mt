"""Text normalization.

Scope is deliberately minimal. The corpus intentionally preserves Qom orthographic
variation (``ỹ``/``ȳ``, ``ñ``/n-with-bar, d/r, h/j, e/i), and normalization must not
"correct" any of it. Exactly three things happen here:

1. Unicode NFC — a canonical-equivalence change only. ``ỹ`` written as ``y`` + combining
   tilde becomes U+1EF9; it does *not* become ``ȳ`` (U+0233), so the variants the corpus
   cares about stay distinct.
2. Apostrophe unification to U+0027. The apostrophe writes the glottal stop, a phoneme,
   and the sources disagree on which codepoint to use (El Principito uses U+2019
   throughout, Arte verbal and La Biblia use U+0027). Leaving them mixed means the model
   and chrF treat the same phoneme as two different characters.
3. Whitespace collapse and strip.

:func:`fold_for_matching` is a separate, more aggressive transform used only for
*detecting* near-duplicates across splits. It never touches stored text.
"""

from __future__ import annotations

import re
import unicodedata

#: Codepoints that occur in the sources as the glottal-stop apostrophe, plus visually
#: identical neighbours that would otherwise sneak through. All map to U+0027.
APOSTROPHE_VARIANTS: dict[str, str] = {
    "‘": "'",  # LEFT SINGLE QUOTATION MARK
    "’": "'",  # RIGHT SINGLE QUOTATION MARK
    "ʼ": "'",  # MODIFIER LETTER APOSTROPHE
    "ʹ": "'",  # MODIFIER LETTER PRIME
    "`": "'",  # GRAVE ACCENT
    "´": "'",  # ACUTE ACCENT
    "′": "'",  # PRIME
}

_APOSTROPHE_RE = re.compile("[" + "".join(APOSTROPHE_VARIANTS) + "]")
_WHITESPACE_RE = re.compile(r"\s+")

#: Characters that carry Qom orthographic distinctions. Guarded by a unit test that
#: asserts normalization leaves every one of them byte-identical.
PROTECTED_GRAPHEMES: tuple[str, ...] = (
    "ỹ", "Ỹ", "ȳ", "Ȳ", "ŷ", "ÿ", "ý",
    "ñ", "Ñ", "n̄",  # n with combining macron
    "'",
)


def normalize_text(value: str) -> str:
    """Normalize one side of a parallel pair. Idempotent."""
    text = unicodedata.normalize("NFC", str(value))
    text = _APOSTROPHE_RE.sub("'", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokens(value: str) -> list[str]:
    """Orthographic tokenization used for QC length ratios and corpus statistics.

    Punctuation is split off as its own token, which means the glottal-stop apostrophe
    counts separately: ``l'aqtaqa`` -> ``["l", "'", "aqtaqa"]``. That is a reporting
    convention, not a linguistic claim; :func:`word_tokens` gives the alternative.
    """
    return [t for t in _TOKEN_RE.findall(value) if t.strip()]


_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def word_tokens(value: str) -> list[str]:
    """Whitespace/punctuation tokenization that keeps the apostrophe word-internal."""
    return [t for t in _WORD_RE.findall(value) if t.strip("'")]


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def fold_for_matching(value: str) -> str:
    """Aggressive fold used only to detect near-duplicates across splits.

    Casefolds, strips punctuation (after apostrophe unification) and collapses
    whitespace, so that two renderings of the same sentence that differ only in
    quoting or capitalisation compare equal.
    """
    text = normalize_text(value).casefold()
    text = _PUNCT_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


#: Values that appear in id columns to mean "no value". Compared case-insensitively
#: after stripping. Kept explicit so a new placeholder spelling fails loudly in tests
#: rather than silently becoming a group named after it.
ID_PLACEHOLDERS: frozenset[str] = frozenset(
    {"", "n/a", "n.a.", "na", "none", "nan", "null", "-", "--", "?", "s/d", "sin dato"}
)


def clean_id(value) -> str | None:
    """Normalize a structural id, mapping placeholders and NaN to ``None``.

    Also reconciles a real inconsistency in the sources: Arte verbal's ``Lineas`` sheet
    writes fragment ids as ``C.1.`` while its ``Fragmentos`` sheet writes ``C.1``. Both
    sheets list the same 70 fragments, so the trailing separator is a typing artifact,
    not a distinction. Stripping trailing dots lets a fragment's title join its lines
    instead of forming a group of its own.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in ID_PLACEHOLDERS:
        return None
    # Excel turns integer-like ids into floats ("3.0"); make them stable strings.
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    text = text.rstrip(". ")
    if not text or text.casefold() in ID_PLACEHOLDERS:
        return None
    return text
