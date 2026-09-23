"""Which manifest fields describe an artifact, and which only record how it was built."""

from __future__ import annotations

from qomlaq import PROVENANCE_FIELDS, same_except_provenance

MANIFEST = {
    "corpus_sha256": "a" * 64,
    "counts": {"rows": 10},
    "built_utc": "2026-09-23T12:00:00+00:00",
    "code_fingerprint": "sha256:" + "1" * 64,
    "package_version": "2.1.0",
    "runtime": {"python": "3.14.6"},
    "data_dir": "corpus",
}


def test_a_rebuild_elsewhere_is_the_same_artifact():
    elsewhere = dict(
        MANIFEST,
        built_utc="2026-10-01T09:00:00+00:00",
        code_fingerprint="sha256:" + "2" * 64,
        package_version="2.2.0",
        runtime={"python": "3.12.1"},
        data_dir="/data/qom/corpus",
    )
    assert same_except_provenance(MANIFEST, elsewhere)


def test_a_content_change_is_a_different_artifact():
    assert not same_except_provenance(MANIFEST, dict(MANIFEST, counts={"rows": 11}))
    assert not same_except_provenance(MANIFEST, dict(MANIFEST, corpus_sha256="b" * 64))


def test_provenance_fields_are_only_about_the_build():
    assert "corpus_sha256" not in PROVENANCE_FIELDS
    assert "split_sha256" not in PROVENANCE_FIELDS
