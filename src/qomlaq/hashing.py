"""Content hashing for corpus and split artifacts.

Every hash in this project is taken over a *canonical serialization* rather than
over file bytes, so that a hash is stable across pandas versions, parquet codecs,
column ordering and row ordering. See :func:`canonical_bytes`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

# Field and record separators chosen to be absent from the corpus text.
_FIELD_SEP = "\x1f"  # UNIT SEPARATOR
_RECORD_SEP = "\n"

#: Columns that define the identity of the canonical corpus. Deliberately narrow:
#: adding a derived column (a length, a QC flag) must not change the corpus hash.
CORPUS_IDENTITY_COLUMNS: tuple[str, ...] = (
    "pair_uid",
    "qom",
    "es",
    "source_doc",
    "source_ref",
    "group_id",
    "unit_level",
)

#: Columns that define the identity of a split partition.
PARTITION_IDENTITY_COLUMNS: tuple[str, ...] = ("pair_uid", "group_id")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pair_uid(source_doc: str, source_ref: str, qom_raw: str, es_raw: str) -> str:
    """Stable identity for a parallel pair.

    Derived from provenance plus the *raw* (pre-normalization) text, so that a pair
    keeps its identity if normalization rules later change, and two identical
    sentences from different sources remain distinguishable.
    """
    joined = _FIELD_SEP.join((source_doc, source_ref, qom_raw, es_raw))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def canonical_bytes(frame: pd.DataFrame, columns: Sequence[str]) -> bytes:
    """Serialize a frame to bytes that depend only on its content.

    Rows are sorted by the first column, columns are emitted in the given order,
    values are stringified with NA rendered as the empty string, and lines are
    joined with ``\\n``. No index, no dtypes, no library metadata.
    """
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"cannot hash frame, missing columns: {missing}")

    ordered = frame.loc[:, list(columns)].sort_values(
        by=list(columns), kind="mergesort", na_position="first"
    )
    lines: list[str] = []
    for row in ordered.itertuples(index=False, name=None):
        lines.append(_FIELD_SEP.join("" if pd.isna(v) else str(v) for v in row))
    return _RECORD_SEP.join(lines).encode("utf-8")


def sha256_frame(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    return hashlib.sha256(canonical_bytes(frame, columns)).hexdigest()


def combine_hashes(parts: Iterable[str]) -> str:
    """Order-independent combination of component hashes."""
    joined = _FIELD_SEP.join(sorted(parts))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def short(sha: str, length: int = 8) -> str:
    return sha[:length]
