# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The envelope schema in the Internet-Draft, held against the vectors.

ietf/vaara-receipt.cddl is the CDDL appendix of the draft, byte for byte.
Every receipt in tests/vectors/ is validated against it with the `cddl`
tool (the Rust crate of that name, `cargo install cddl`) when the tool is
on PATH: the positives must pass, and the receipts that fail are exactly
the named negative cases whose fault is structural. Without the tool the
validation is skipped and the copy check still runs.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CDDL = ROOT / "ietf" / "vaara-receipt.cddl"
VECTORS = ROOT / "tests" / "vectors"

#: Receipts in the corpus that are meant to fail the schema, and why.
STRUCTURAL_NEGATIVES = {
    "conformance_statement_v0/emitter_records/flawed/bad.json": "status outside the set",
    "pq_hybrid_v0/cases.json#7": "sigSuite outside the allowlist",
    "record_conformance_v0/records/neg_bad_status.json": "status outside the set",
    "record_conformance_v0/records/neg_malformed_backlink_digest.json": "not a sha256 digest",
    "record_conformance_v0/records/neg_unsupported_alg.json": "alg ES512",
    "record_set_v0/sets/mixed_nonconforming/bad.json": "status outside the set",
}


def _drafts() -> list[Path]:
    return sorted((ROOT / "ietf").glob("draft-sirkkavaara-vaara-receipt-*.xml"),
                  key=lambda p: int(p.stem.rsplit("-", 1)[1]))


def test_the_latest_draft_carries_the_schema_file_verbatim():
    xml = _drafts()[-1].read_text(encoding="utf-8")
    block = re.search(r'<sourcecode type="cddl"[^>]*><!\[CDATA\[\n(.*?)\]\]></sourcecode>', xml, re.S)
    assert block, f"{_drafts()[-1].name} has no CDDL appendix"
    assert block.group(1).rstrip() == CDDL.read_text(encoding="utf-8").rstrip()


def _receipts():
    for path in sorted(VECTORS.rglob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        rel = path.relative_to(VECTORS).as_posix()
        stack = [(doc, rel)]
        while stack:
            node, where = stack.pop()
            if isinstance(node, dict):
                if {"signature", "version"} <= node.keys() and (
                        "decisionDerived" in node or "outcomeDerived" in node):
                    yield where, node
                stack.extend((v, where) for v in node.values())
            elif isinstance(node, list):
                if where.endswith("cases.json"):
                    stack.extend((v, f"{where}#{i}") for i, v in enumerate(node))
                else:
                    stack.extend((v, where) for v in node)


@pytest.mark.skipif(shutil.which("cddl") is None, reason="the cddl tool is not installed")
def test_every_positive_receipt_fits_the_schema_and_only_the_negatives_do_not(tmp_path):
    failed = set()
    for i, (where, receipt) in enumerate(_receipts()):
        doc = tmp_path / f"{i}.json"
        doc.write_text(json.dumps(receipt), encoding="utf-8")
        run = subprocess.run(["cddl", "validate", "--cddl", str(CDDL), "--json", str(doc)],
                             capture_output=True, text=True)
        if "is successful" not in run.stdout + run.stderr:
            failed.add(where)
    assert failed == set(STRUCTURAL_NEGATIVES)
