"""The published SEP-2828 conformance corpus matches its source and itself.

The corpus under ``conformance/sep2828/`` is a standalone, integrity-pinned
mirror of two suites from ``tests/vectors/``. These tests guarantee it never
drifts: every fixture is byte-identical to the source vector it mirrors, the
manifest digests match the bytes on disk, and the standalone runner reaches a
clean verdict on its own. They re-derive everything here rather than importing
the build script or the runner, so the check is independent of the code that
produced the corpus.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "tests" / "vectors"
CORPUS = REPO / "conformance" / "sep2828"
SUITES = ["record_conformance_v0", "record_set_v0"]


def _fixture_files(suite_root: Path) -> dict[str, bytes]:
    """Map of POSIX relpath -> bytes for every fixture file under a suite dir."""
    out: dict[str, bytes] = {}
    for path in sorted(suite_root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        out[path.relative_to(suite_root).as_posix()] = path.read_bytes()
    return out


def test_corpus_is_byte_identical_to_source_vectors() -> None:
    for suite in SUITES:
        src = _fixture_files(SRC / suite)
        mirror = _fixture_files(CORPUS / suite)
        assert src.keys() == mirror.keys(), f"{suite}: file set differs from source vectors"
        for rel, data in src.items():
            assert mirror[rel] == data, f"{suite}/{rel} differs from the source vector"


def test_manifest_matches_files_on_disk() -> None:
    manifest = json.loads((CORPUS / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["suites"] == SUITES

    computed: dict[str, str] = {}
    for suite in SUITES:
        for rel, data in _fixture_files(CORPUS / suite).items():
            key = f"{suite}/{rel}"
            computed[key] = "sha256:" + hashlib.sha256(data).hexdigest()

    assert manifest["files"] == computed, "MANIFEST.json file digests are stale"

    lines = [f"{computed[k].split(':', 1)[1]}  {k}" for k in sorted(computed)]
    blob = ("\n".join(lines) + "\n").encode("utf-8")
    expected_corpus = "sha256:" + hashlib.sha256(blob).hexdigest()
    assert manifest["corpusDigest"] == expected_corpus


def test_manifest_version_matches_version_file() -> None:
    manifest = json.loads((CORPUS / "MANIFEST.json").read_text(encoding="utf-8"))
    version = (CORPUS / "VERSION").read_text(encoding="utf-8").strip()
    assert manifest["version"] == version
    assert version.count(".") == 2, "VERSION should be semver MAJOR.MINOR.PATCH"


def test_runner_passes_both_suites() -> None:
    result = subprocess.run(
        [sys.executable, "run.py"],
        cwd=CORPUS, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: all 2 suite(s) matched expected." in result.stdout


def test_runner_verify_manifest_clean() -> None:
    result = subprocess.run(
        [sys.executable, "run.py", "--verify-manifest"],
        cwd=CORPUS, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MANIFEST OK" in result.stdout
