"""QomL'aqtaqa Qom-Spanish MT pipeline.

Import surface is deliberately small: build or load a corpus, make or load splits,
score with the one metric module, and report. ``qomlaq --help`` lists the commands.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

__version__ = "2.1.0"

_SOURCE_DIR = Path(__file__).resolve().parent


def code_fingerprint() -> str:
    """sha256 over the package's own sources.

    Recorded in every artifact so a result names the exact pipeline that produced it,
    including uncommitted local edits that a git tag would not capture.
    """
    digest = hashlib.sha256()
    for path in sorted(_SOURCE_DIR.rglob("*.py")):
        digest.update(path.relative_to(_SOURCE_DIR).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def runtime_versions() -> dict[str, str]:
    """Versions of everything that can change a number."""
    versions: dict[str, str] = {
        "qomlaq": __version__,
        "code_fingerprint": code_fingerprint(),
        "python": platform.python_version(),
    }
    for package in ("pandas", "numpy", "sacrebleu", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


#: Manifest fields that record how, when and where an artifact was built, not what it
#: contains. A rebuild that changes only these leaves the artifact on disk as it is.
PROVENANCE_FIELDS: tuple[str, ...] = (
    "built_utc",
    "code_fingerprint",
    "package_version",
    "runtime",
    "data_dir",
)


def same_except_provenance(a: dict, b: dict) -> bool:
    """Whether two manifests describe the same artifact."""
    def strip(manifest: dict) -> str:
        content = {k: v for k, v in manifest.items() if k not in PROVENANCE_FIELDS}
        return json.dumps(content, sort_keys=True, ensure_ascii=False)

    return strip(a) == strip(b)


__all__ = [
    "__version__",
    "PROVENANCE_FIELDS",
    "code_fingerprint",
    "runtime_versions",
    "same_except_provenance",
]
