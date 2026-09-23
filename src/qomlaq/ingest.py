"""Source readers.

One reader per source *kind*, driven by the registry in :mod:`qomlaq.config`.

The central rule: text and structural ids are selected together in a single indexed
pass, and filtering happens once, on a boolean mask that preserves the index, so every
kept row keeps its own source row's ids.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SourceSpec
from .hashing import pair_uid
from .normalize import clean_id, normalize_text

#: Emitted by every reader, in this order.
COLUMNS: tuple[str, ...] = (
    "pair_uid",
    "qom",
    "es",
    "qom_raw",
    "es_raw",
    "source_doc",
    "source_file",
    "source_sheet",
    "source_row",
    "source_ref",
    "record_kind",
)


class IngestError(RuntimeError):
    pass


def _fold_filename(name: str) -> str:
    """Filename key that ignores diacritics and case."""
    decomposed = unicodedata.normalize("NFD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def resolve_source_path(data_dir: Path, filename: str) -> Path:
    """Locate a declared source file, tolerating diacritic-stripped names.

    Some upload and archive tools rewrite filenames to ASCII, so ``Educación Sanitaria
    Intercultural.xlsx`` can arrive as ``Educacion Sanitaria Intercultural.xlsx``.
    Matching on a fold of the name keeps the registry working after such a copy, while a
    genuinely absent file still fails loudly instead of shrinking the corpus in silence.
    """
    exact = data_dir / filename
    if exact.exists():
        return exact
    if not data_dir.is_dir():
        raise IngestError(f"data directory not found: {data_dir}")

    target = _fold_filename(filename)
    matches = sorted(p for p in data_dir.iterdir() if p.is_file() and _fold_filename(p.name) == target)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise IngestError(
            f"{filename!r} matches several files in {data_dir}: {[p.name for p in matches]}"
        )
    present = sorted(p.name for p in data_dir.iterdir() if p.suffix.lower() in (".xlsx", ".csv"))
    raise IngestError(f"source file {filename!r} not found in {data_dir}. Present: {present}")


def _text_mask(frame: pd.DataFrame, col_qom: str, col_es: str) -> pd.Series:
    """Rows where both sides carry real text. Index-preserving."""
    mask = pd.Series(True, index=frame.index)
    for col in (col_qom, col_es):
        values = frame[col]
        text = values.astype("string").str.strip()
        mask &= values.notna() & text.ne("") & ~text.str.casefold().isin(["nan", "none"])
    return mask


def _finalize(
    rows: pd.DataFrame,
    spec: SourceSpec,
    *,
    sheet: str | None,
    record_kind: str,
    col_qom: str,
    col_es: str,
    id_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Shared tail of every reader: filter once, normalize, build ids."""
    rows = rows.copy()
    rows["source_row"] = np.arange(len(rows), dtype="int64")

    mask = _text_mask(rows, col_qom, col_es)
    kept = rows.loc[mask]  # single filter; index (and therefore metadata) preserved

    out = pd.DataFrame(index=kept.index)
    out["qom_raw"] = kept[col_qom].astype("string").str.strip()
    out["es_raw"] = kept[col_es].astype("string").str.strip()
    out["qom"] = out["qom_raw"].map(normalize_text)
    out["es"] = out["es_raw"].map(normalize_text)
    out["source_doc"] = spec.name
    out["source_file"] = spec.filename
    out["source_sheet"] = sheet or ""
    out["source_row"] = kept["source_row"]
    out["record_kind"] = record_kind

    for col in id_columns:
        # Nullable string dtype throughout, so an absent id is pd.NA everywhere rather
        # than None in one place and NaN in another depending on the column's contents.
        if col in kept.columns:
            out[col] = kept[col].map(clean_id).astype("string")
        else:
            out[col] = pd.Series(pd.NA, index=kept.index, dtype="string")

    out["source_ref"] = _build_refs(out, spec, record_kind)
    out["pair_uid"] = [
        pair_uid(spec.name, ref, q, e)
        for ref, q, e in zip(out["source_ref"], out["qom_raw"], out["es_raw"])
    ]
    return out.reset_index(drop=True)


def _build_refs(frame: pd.DataFrame, spec: SourceSpec, record_kind: str) -> list[str]:
    refs: list[str] = []
    for _, row in frame.iterrows():
        fields = {k: ("" if v is None or pd.isna(v) else v) for k, v in row.items()}
        fields["source_slug"] = spec.slug
        fields["row"] = row["source_row"]
        try:
            ref = spec.ref_template.format(**fields)
        except KeyError:
            ref = f"{spec.slug}_{row['source_row']}"
        if record_kind == "title":
            ref = f"{ref}#title"
        refs.append(ref)
    return refs


def read_lines(spec: SourceSpec, data_dir: Path) -> pd.DataFrame:
    """Read the parallel text rows of a source."""
    path = resolve_source_path(data_dir, spec.filename)

    if spec.kind == "xlsx":
        raw = pd.read_excel(path, sheet_name=spec.sheet)
    else:
        raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]

    for col in (spec.col_qom, spec.col_es):
        if col not in raw.columns:
            raise IngestError(
                f"{spec.name}: expected column {col!r}, found {list(raw.columns)}"
            )

    id_columns = spec.id_columns
    for name, columns in spec.derived_ids:
        # Built here so the group rule stays declarative.
        parts = [raw[column].astype("string").str.strip() for column in columns]
        joined = parts[0]
        for part in parts[1:]:
            joined = joined + "_" + part
        raw[name] = joined
        id_columns = id_columns + (name,)

    return _finalize(
        raw,
        spec,
        sheet=spec.sheet,
        record_kind="line",
        col_qom=spec.col_qom,
        col_es=spec.col_es,
        id_columns=id_columns,
    )


def read_titles(spec: SourceSpec, data_dir: Path) -> pd.DataFrame:
    """Read chapter/section/fragment title pairs from a source's metadata sheets.

    A sheet contributes a pair for every ``nombre_<unit>_qom`` column that has a
    matching ``nombre_<unit>_es``; the row's ``id_<unit>`` is carried along so the title
    can be attached to the group of the unit it names.
    """
    if not spec.title_sheets:
        return pd.DataFrame(columns=list(COLUMNS))

    path = resolve_source_path(data_dir, spec.filename)
    frames: list[pd.DataFrame] = []
    for sheet in spec.title_sheets:
        try:
            raw = pd.read_excel(path, sheet_name=sheet)
        except ValueError:  # sheet absent in this workbook
            continue
        raw.columns = [str(c).strip() for c in raw.columns]

        for qom_col in [c for c in raw.columns if c.startswith("nombre_") and c.endswith("_qom")]:
            unit = qom_col[len("nombre_") : -len("_qom")]
            es_col = f"nombre_{unit}_es"
            if es_col not in raw.columns:
                continue  # e.g. Arte's Secciones sheet has a single untranslated name
            id_columns = tuple(c for c in raw.columns if c.startswith("id_"))
            frame = _finalize(
                raw,
                spec,
                sheet=sheet,
                record_kind="title",
                col_qom=qom_col,
                col_es=es_col,
                id_columns=id_columns,
            )
            if len(frame):
                frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=list(COLUMNS))
    return pd.concat(frames, ignore_index=True)


def read_source(spec: SourceSpec, data_dir: Path) -> pd.DataFrame:
    """Lines plus title pairs for one source."""
    parts = [read_lines(spec, data_dir)]
    titles = read_titles(spec, data_dir)
    if len(titles):
        parts.append(titles)
    return pd.concat(parts, ignore_index=True)
