#!/usr/bin/env python3
"""Run the SEP-2828 execution-record conformance corpus.

This directory is the standalone conformance corpus for the SEP-2828
server-side signed execution record. It is self-contained: it imports no
Vaara code and depends on nothing but the Python standard library, so an
independent implementation can download this directory alone and check
itself against the same fixtures the reference implementation ships.

Two suites make up the corpus:

- ``record_conformance_v0`` — is a single record well-formed, and does the
  one binding it proves about itself (``projectionDigest`` over the
  projection bytes) recompute?
- ``record_set_v0`` — across a directory of records, how many conform and
  where are the gaps in the chain (duplicate calls, decisions with no
  outcome, executed actions with no committed result)?

Each suite carries its own independent checker (``_check_independent.py``),
its fixtures, and an ``expected.json`` of the verdicts. This runner invokes
both and aggregates the result.

Usage::

    python run.py                 # run both suites, exit 0 iff all pass
    python run.py --verify-manifest   # confirm the bytes match MANIFEST.json
    python run.py --version       # print the corpus version

Exit 0 means every case matched its expected verdict (or, with
``--verify-manifest``, that every file matches the published digest).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _load_manifest() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))
    return data


def file_digests(root: Path, suites: list[str]) -> dict[str, str]:
    """sha256 of every fixture file under each suite, keyed by POSIX relpath.

    ``__pycache__`` and compiled ``.pyc`` files are skipped so the digest set
    is the source fixtures only, independent of any local test run.
    """
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
    """A single digest over the whole corpus: sha256 of the sorted file list.

    Each line is ``<hexdigest>  <relpath>`` (the ``sha256sum`` shape), sorted
    by path, newline-joined with a trailing newline. Recomputable by anyone
    who holds the files, so the corpus version pins to an exact byte set.
    """
    lines = [f"{files[k].split(':', 1)[1]}  {k}" for k in sorted(files)]
    blob = ("\n".join(lines) + "\n").encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def run_suites(manifest: dict[str, Any]) -> int:
    suites: list[str] = manifest["suites"]
    failed: list[str] = []
    for suite in suites:
        checker = HERE / suite / "_check_independent.py"
        print(f"== {suite} ==")
        result = subprocess.run([sys.executable, str(checker)])
        if result.returncode != 0:
            failed.append(suite)
        print()
    if failed:
        print(f"FAIL: {len(failed)} suite(s) did not match expected: {', '.join(failed)}")
        return 1
    print(f"PASS: all {len(suites)} suite(s) matched expected.")
    return 0


def verify_manifest(manifest: dict[str, Any]) -> int:
    want: dict[str, str] = manifest["files"]
    got = file_digests(HERE, manifest["suites"])
    problems: list[str] = []
    for rel in sorted(set(want) | set(got)):
        if rel not in got:
            problems.append(f"missing file: {rel}")
        elif rel not in want:
            problems.append(f"unexpected file: {rel}")
        elif want[rel] != got[rel]:
            problems.append(f"digest mismatch: {rel}")
    want_corpus = manifest["corpusDigest"]
    got_corpus = corpus_digest(got)
    if want_corpus != got_corpus:
        problems.append(f"corpusDigest mismatch: manifest {want_corpus} != computed {got_corpus}")
    if problems:
        print("MANIFEST verification FAILED:")
        for line in problems:
            print(f"  {line}")
        return 1
    print(f"MANIFEST OK: {len(got)} files match, corpusDigest {got_corpus}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SEP-2828 conformance corpus runner")
    parser.add_argument("--verify-manifest", action="store_true",
                        help="confirm every file matches the digest in MANIFEST.json")
    parser.add_argument("--version", action="store_true",
                        help="print the corpus version and exit")
    args = parser.parse_args(argv)

    manifest = _load_manifest()
    if args.version:
        print(manifest["version"])
        return 0
    if args.verify_manifest:
        return verify_manifest(manifest)
    return run_suites(manifest)


if __name__ == "__main__":
    sys.exit(main())
