"""Building and loading the canonical corpus artifact.

The corpus is built once and identified by a content hash. Splits, models and results
all record that hash, so a question like "was this evaluated on the same corpus?" is
answered by comparing two strings rather than by re-deriving anything.

:class:`CorpusArtifact` cannot be constructed directly -- only :func:`build_corpus` and
:func:`load_corpus` can produce one, and both verify what they return. Downstream code
accepts ``CorpusArtifact`` rather than ``DataFrame``, which is what stops a caller from
quietly assembling their own corpus and splitting it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import __version__, code_fingerprint, paths, runtime_versions, same_except_provenance
from .config import QCConfig, SourceSpec, configs_of, corpus_sources
from .groups import assign_groups, group_sizes
from .hashing import CORPUS_IDENTITY_COLUMNS, sha256_file, sha256_frame, short
from .ingest import IngestError, read_source, resolve_source_path
from .qc import apply_qc, cross_source_duplicates, unify_duplicate_groups

_CONSTRUCTOR_TOKEN = object()

MANIFEST_NAME = "corpus_manifest.json"
TABLE_NAME = "corpus.csv"
SCHEMA_VERSION = "2.0.0"


class CorpusError(RuntimeError):
    pass


@dataclass(frozen=True)
class CorpusArtifact:
    """A built, hash-identified corpus."""

    frame: pd.DataFrame
    manifest: dict
    _token: object = None

    def __post_init__(self) -> None:
        if self._token is not _CONSTRUCTOR_TOKEN:
            raise CorpusError(
                "CorpusArtifact must come from build_corpus() or load_corpus(); "
                "constructing one directly would bypass hash verification."
            )

    @property
    def sha256(self) -> str:
        return self.manifest["corpus_sha256"]

    @property
    def sha8(self) -> str:
        return short(self.sha256)

    @property
    def name(self) -> str:
        """The corpus version this artifact was built from."""
        return self.manifest["corpus"]

    @property
    def configs(self) -> tuple[str, ...]:
        return tuple(self.manifest["configs"])

    def config_frame(self, config: str) -> pd.DataFrame:
        """Rows belonging to one corpus configuration."""
        if config not in self.configs:
            raise CorpusError(f"{self.name} has no config {config!r}; expected {self.configs}")
        return self.frame[self.frame[f"in_{config}"]].reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        rows = []
        for source in self.manifest["sources"]:
            rows.append(
                {
                    "source": source["name"],
                    "segments": source["after_qc"],
                    "lines": source["lines"],
                    "titles": source["titles"],
                    "groups": source["groups"],
                    "unit_levels": ", ".join(source["unit_levels"]),
                    "qom_tokens": source["qom_tokens"],
                    "es_tokens": source["es_tokens"],
                }
            )
        return pd.DataFrame(rows)


def _source_stats(spec: SourceSpec, raw_rows: int, kept: pd.DataFrame) -> dict:
    chunk = kept[kept["source_doc"] == spec.name]
    return {
        "name": spec.name,
        "file": spec.filename,
        "raw_rows": int(raw_rows),
        "after_qc": int(len(chunk)),
        "lines": int((chunk["record_kind"] == "line").sum()),
        "titles": int((chunk["record_kind"] == "title").sum()),
        "groups": int(chunk["group_id"].nunique()),
        "unit_levels": sorted(chunk["unit_level"].dropna().unique().tolist()),
        "qom_tokens": int(chunk["qom_len_t"].sum()),
        "es_tokens": int(chunk["es_len_t"].sum()),
        "qom_len_mean": round(float(chunk["qom_len_t"].mean()), 2) if len(chunk) else 0.0,
        "qom_len_std": round(float(chunk["qom_len_t"].std(ddof=1)), 2) if len(chunk) > 1 else 0.0,
        "es_len_mean": round(float(chunk["es_len_t"].mean()), 2) if len(chunk) else 0.0,
        "es_len_std": round(float(chunk["es_len_t"].std(ddof=1)), 2) if len(chunk) > 1 else 0.0,
        "configs": list(spec.configs),
        "forced_partition": spec.forced_partition,
    }


def build_corpus(
    data_dir: str | Path,
    corpus: str,
    *,
    qc: QCConfig | None = None,
    allow_line_groups: bool = False,
) -> CorpusArtifact:
    """Ingest a corpus version's sources, normalize, group, filter, and hash the result."""
    data_dir = Path(data_dir)
    qc = qc or QCConfig()
    sources = corpus_sources(corpus)
    configs = configs_of(sources)

    frames: list[pd.DataFrame] = []
    raw_counts: dict[str, int] = {}
    file_hashes: dict[str, str] = {}

    for spec in sources:
        try:
            path = resolve_source_path(data_dir, spec.filename)
        except IngestError as exc:
            raise CorpusError(
                f"missing source {spec.name!r}: {exc}. The corpus is declared in "
                "qomlaq.config; an absent file is an error, never a silently smaller "
                "corpus."
            ) from exc
        file_hashes[path.name] = sha256_file(path)
        frame = read_source(spec, data_dir)
        raw_counts[spec.name] = int(len(frame))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined = assign_groups(combined, allow_line_groups=allow_line_groups)
    kept, ledger = apply_qc(combined, qc)
    kept = unify_duplicate_groups(kept)

    for config in configs:
        members = {s.name for s in sources if config in s.configs}
        kept[f"in_{config}"] = kept["source_doc"].isin(members)

    kept = kept.sort_values("pair_uid", kind="mergesort").reset_index(drop=True)
    corpus_sha256 = sha256_frame(kept, CORPUS_IDENTITY_COLUMNS)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "code_fingerprint": code_fingerprint(),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": corpus,
        "configs": list(configs),
        "data_dir": paths.for_manifest(data_dir),
        "qc": qc.__dict__,
        "source_files": file_hashes,
        "sources": [
            _source_stats(spec, raw_counts[spec.name], kept)
            for spec in sources
            if (kept["source_doc"] == spec.name).any()
        ],
        "qc_ledger": ledger.as_list(),
        "counts": {
            "rows": int(len(kept)),
            "lines": int((kept["record_kind"] == "line").sum()),
            "titles": int((kept["record_kind"] == "title").sum()),
            "groups": int(kept["group_id"].nunique()),
            **{f"{c}_rows": int(kept[f"in_{c}"].sum()) for c in configs},
        },
        "cross_source_duplicate_rows": int(len(cross_source_duplicates(kept))),
        "runtime": runtime_versions(),
        "corpus_sha256": corpus_sha256,
    }
    return CorpusArtifact(frame=kept, manifest=manifest, _token=_CONSTRUCTOR_TOKEN)


def save_corpus(artifact: CorpusArtifact, artifacts_dir: str | Path) -> Path:
    """Write the corpus table and its manifest.

    The manifest is only rewritten when the corpus itself changed. A rebuild on another
    machine, from another data directory or with newer code changes nothing but
    provenance, so the tracked manifest stays as it is.
    """
    out_dir = paths.corpus_dir(artifact.name, artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact.frame.to_csv(out_dir / TABLE_NAME, index=False)
    manifest_path = out_dir / MANIFEST_NAME
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if same_except_provenance(existing, artifact.manifest):
            return out_dir
    manifest_path.write_text(
        json.dumps(artifact.manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_dir


def load_corpus(
    artifacts_dir: str | Path, corpus: str, *, expect_corpus_sha256: str | None = None
) -> CorpusArtifact:
    """Load a built corpus version, re-verifying its hash.

    ``expect_corpus_sha256`` is optional here (the manifest is self-describing) but
    passing it turns "I loaded the corpus" into "I loaded *this* corpus".
    """
    corpus_dir = paths.corpus_dir(corpus, artifacts_dir)
    manifest_path = corpus_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise CorpusError(f"no corpus manifest at {manifest_path}; run build_corpus first")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(corpus_dir / TABLE_NAME, dtype={"pair_uid": "string"}, keep_default_na=False, na_values=[""])
    for flag in [c for c in frame.columns if c.startswith("in_")]:
        frame[flag] = frame[flag].astype(str).str.lower().isin(["true", "1"])

    recomputed = sha256_frame(frame, CORPUS_IDENTITY_COLUMNS)
    if recomputed != manifest["corpus_sha256"]:
        raise CorpusError(
            "corpus table does not match its manifest hash "
            f"(recomputed {short(recomputed)}, manifest {short(manifest['corpus_sha256'])}). "
            "The table on disk has been edited or written by a different pipeline version."
        )
    if expect_corpus_sha256 and recomputed != expect_corpus_sha256:
        raise CorpusError(
            f"corpus hash mismatch: expected {short(expect_corpus_sha256)}, "
            f"found {short(recomputed)}"
        )
    return CorpusArtifact(frame=frame, manifest=manifest, _token=_CONSTRUCTOR_TOKEN)


def largest_groups(artifact: CorpusArtifact, n: int = 10) -> pd.DataFrame:
    return group_sizes(artifact.frame).head(n)
