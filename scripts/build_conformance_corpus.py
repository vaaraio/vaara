#!/usr/bin/env python3
"""Assemble the standalone SEP-2828 conformance corpus from the source vectors.

The canonical fixtures live under ``tests/vectors/`` where the test suite and
the cross-suite checkers reference them. The published corpus under
``conformance/sep2828/`` is a faithful, integrity-pinned mirror of two of
those suites, packaged so an outside implementation can download the corpus
directory alone and run it with no Vaara dependency.

This script regenerates that mirror:

1. byte-copies each suite directory from ``tests/vectors`` into the corpus,
2. writes ``MANIFEST.json`` (per-file sha256 plus a single ``corpusDigest``),
   stamping it with the version read from the corpus ``VERSION`` file.

It does not touch the authored packaging files (``VERSION``, ``run.py``,
``README.md``). Run it after changing any source fixture, then commit the
result; ``tests/test_conformance_corpus.py`` fails if the corpus drifts from
the source vectors or from the manifest.

Run: ``python scripts/build_conformance_corpus.py``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "tests" / "vectors"
DST = REPO / "conformance" / "sep2828"
SUITES = ["record_conformance_v0", "record_set_v0"]
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def file_digests(root: Path, suites: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for suite in suites:
        for path in sorted((root / suite).rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            rel = path.relative_to(root).as_posix()
            out[rel] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def corpus_digest(files: dict[str, str]) -> str:
    lines = [f"{files[k].split(':', 1)[1]}  {k}" for k in sorted(files)]
    blob = ("\n".join(lines) + "\n").encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def build() -> None:
    version = (DST / "VERSION").read_text(encoding="utf-8").strip()
    for suite in SUITES:
        target = DST / suite
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(SRC / suite, target, ignore=_IGNORE)

    files = file_digests(DST, SUITES)
    manifest = {
        "corpus": "sep2828-execution-record-conformance",
        "spec": "SEP-2828",
        "version": version,
        "suites": SUITES,
        "files": files,
        "corpusDigest": corpus_digest(files),
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (DST / "MANIFEST.json").write_text(text, encoding="utf-8")
    print(f"corpus {version}: {len(files)} files, corpusDigest {manifest['corpusDigest']}")


if __name__ == "__main__":
    build()
    sys.exit(0)
