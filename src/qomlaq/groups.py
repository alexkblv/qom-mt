"""Discourse-unit group ids.

A *group* is the unit that must never be split across partitions: a fragment, a
section, a chapter, a UDHR article, a Bible chapter. Splitting happens over groups, so
this module decides how much leakage protection the split actually provides.

Two rules carry the design:

* Every source has an explicit :class:`~qomlaq.config.GroupRule`; there is no shared
  fallback chain ending in "one group per line". A source whose structure is missing
  raises rather than silently degrading, unless it declares ``terminal="document"``.
* Title rows never form their own group. A title is attached to the unit it names, and
  any title left in a group with no line rows is reattached to a sibling unit.
"""

from __future__ import annotations

import pandas as pd

from .config import SourceSpec, SOURCES_BY_NAME

#: Values of the ``unit_level`` column, coarse to fine.
UNIT_LEVELS: tuple[str, ...] = (
    "document",
    "capitulo",
    "parte",
    "libro_capitulo",
    "seccion",
    "fragmento",
    "verse",
)

_LEVEL_TAGS = {
    "id_fragmento": "frag",
    "id_seccion": "sec",
    "id_capitulo": "cap",
    "capitulo": "cap",
    "parte": "parte",
    "libro_capitulo": "chap",
    "versiculo": "verse",
}


class GroupError(RuntimeError):
    pass


def _level_name(column: str) -> str:
    return column.removeprefix("id_")


def assign_groups(frame: pd.DataFrame, *, allow_line_groups: bool = False) -> pd.DataFrame:
    """Add ``group_id`` and ``unit_level`` columns.

    Raises if any source cannot supply a group for a row and does not declare a
    document-level terminal, so a structural regression fails loudly at build time.
    """
    out = frame.copy()
    group_ids = pd.Series(pd.NA, index=out.index, dtype="string")
    unit_levels = pd.Series(pd.NA, index=out.index, dtype="string")

    for source_doc, chunk in out.groupby("source_doc", sort=False):
        spec = SOURCES_BY_NAME.get(str(source_doc))
        if spec is None:
            raise GroupError(f"no registry entry for source {source_doc!r}")

        resolved = pd.Series(pd.NA, index=chunk.index, dtype="string")
        levels = pd.Series(pd.NA, index=chunk.index, dtype="string")

        for column in spec.group.levels:
            if column not in chunk.columns:
                continue
            values = chunk[column]
            usable = resolved.isna() & values.notna()
            if not usable.any():
                continue
            tag = _LEVEL_TAGS.get(column, _level_name(column))
            resolved.loc[usable] = (
                f"{spec.slug}__{tag}__" + values.loc[usable].astype("string")
            )
            levels.loc[usable] = _level_name(column)

        unresolved = resolved.isna()
        if unresolved.any():
            if spec.group.terminal == "document":
                resolved.loc[unresolved] = f"{spec.slug}__doc"
                levels.loc[unresolved] = "document"
            elif allow_line_groups:
                resolved.loc[unresolved] = (
                    f"{spec.slug}__line__" + chunk.loc[unresolved, "pair_uid"]
                )
                levels.loc[unresolved] = "line"
            else:
                sample = chunk.loc[unresolved, "source_ref"].head(3).tolist()
                raise GroupError(
                    f"{spec.name}: {int(unresolved.sum())} rows have no value for any of "
                    f"{spec.group.levels} and the source does not declare a document "
                    f"terminal. Examples: {sample}. Pass allow_line_groups=True only if "
                    "one-row groups are genuinely intended."
                )

        group_ids.loc[chunk.index] = resolved
        unit_levels.loc[chunk.index] = levels

    out["group_id"] = group_ids
    out["unit_level"] = unit_levels
    return _reattach_orphan_titles(out)


def _reattach_orphan_titles(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure no title row sits in a group made up only of titles.

    A fragment title lands in its fragment's group directly. A chapter or section title
    names a unit whose lines may be grouped more finely, which would leave the title
    alone in its own group and let it reach the test set while its content trains.

    Such a title is reattached to a line group *from the same chapter* (deterministically,
    the first by sorted id), falling back to the same section, then to the source's first
    line group. Attaching within the chapter keeps the title next to the content it names
    rather than lumping every orphan into one arbitrary group.
    """
    if "record_kind" not in frame.columns or not (frame["record_kind"] == "title").any():
        return frame

    lines = frame["record_kind"] == "line"
    line_groups = set(frame.loc[lines, "group_id"].dropna())
    orphaned = (frame["record_kind"] == "title") & ~frame["group_id"].isin(line_groups)
    if not orphaned.any():
        return frame

    out = frame.copy()
    level_of = (
        out.loc[lines].groupby("group_id", observed=True)["unit_level"].first().to_dict()
    )

    def _line_groups_where(source_doc: str, column: str, value) -> list[str]:
        if value is None or pd.isna(value) or column not in out.columns:
            return []
        sel = lines & (out["source_doc"] == source_doc) & (out[column] == value)
        return sorted(set(out.loc[sel, "group_id"].dropna()))

    for idx in out.index[orphaned]:
        row = out.loc[idx]
        source_doc = row["source_doc"]
        candidates = (
            _line_groups_where(source_doc, "id_capitulo", row.get("id_capitulo"))
            or _line_groups_where(source_doc, "id_seccion", row.get("id_seccion"))
            or sorted(set(out.loc[lines & (out["source_doc"] == source_doc), "group_id"].dropna()))
        )
        if not candidates:
            continue
        target = candidates[0]
        out.at[idx, "group_id"] = target
        out.at[idx, "unit_level"] = level_of.get(target, row["unit_level"])
    return out


def regroup_by_verse(frame: pd.DataFrame) -> pd.DataFrame:
    """Group verse-structured sources verse by verse, for the control split.

    Used only by ``bible_grouping="verse"`` splits, which exist to measure what chapter
    grouping is worth.
    """
    out = frame.copy()
    for spec in (s for s in SOURCES_BY_NAME.values() if s.verses):
        rows = out["source_doc"] == spec.name
        if not rows.any():
            continue
        out.loc[rows, "group_id"] = f"{spec.slug}__verse__" + out.loc[rows, "pair_uid"]
        out.loc[rows, "unit_level"] = "verse"
    return out


def group_sizes(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows per group, with the source, ordered largest first."""
    sizes = (
        frame.groupby(["source_doc", "group_id"], observed=True)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["n_rows", "group_id"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )
    return sizes
