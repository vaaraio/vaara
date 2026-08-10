# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SPEC.md is normative, so its claims are checked against what ships.

Two of them were not true.

Section 2 offered `ML-DSA-65` as an `alg` value "as a post-quantum
scheme". No producer emits it, no vector carries it, and the reference
checkers reject any receipt whose `alg` is not `ES256`. An implementer
who took the table at its word would have emitted receipts Vaara refuses.
The post-quantum path that does ship is a `pqSignature` sibling block
under a hybrid suite, which is a different construction in a different
place.

Section 6 gave the ingest stream a `completeness` block with the same two
field names as Section 5.3 and the opposite base: ingest counts from 1,
authorization counts from 0. Both are true of the shipped vectors, and
the document said neither.

The tests below pin the envelope shape, the algorithm set, both
completeness conventions, and the standing claim that every named vector
directory carries a checker that runs on the standard library plus
`cryptography` and `rfc8785`.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.attestation._decision_conformance import VALID_VERDICTS
from vaara.attestation._receipt_conformance import VALID_ALGS

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "SPEC.md"
VECTORS = ROOT / "tests" / "vectors"

#: Third-party imports SPEC.md Section "This document packages..." allows a
#: checker to make. Everything else has to be the standard library.
ALLOWED_THIRD_PARTY = {"cryptography", "rfc8785", "dilithium_py", "asn1crypto"}

#: Exit code the checkers use for "optional dependency missing".
SKIP = 77


def _spec() -> str:
    return SPEC.read_text(encoding="utf-8")


def _named_vector_dirs() -> list[Path]:
    """Every `tests/vectors/<name>/` path SPEC.md points a reader at."""
    names = sorted(set(re.findall(r"tests/vectors/(\w+)/", _spec())))
    assert names, "SPEC.md no longer names any vector directory"
    return [VECTORS / name for name in names]


def _receipts(obj, where: str):
    """Every receipt envelope inside a vector file, at any nesting."""
    if isinstance(obj, dict):
        if isinstance(obj.get("decisionDerived"), dict) and "issuerAsserted" in obj:
            yield where, obj
        for key, value in obj.items():
            yield from _receipts(value, f"{where}.{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from _receipts(value, f"{where}[{index}]")


def _profile_receipts() -> list[tuple[str, dict]]:
    """Receipts under the profiles Section 5.1 registers."""
    registered = ("x402_settlement_v0", "authorization_v0", "contiguity_v0",
                  "ap2_v0", "tap_v0", "external_evidence_v0", "class_gate_v0")
    found = []
    for name in registered:
        for path in sorted((VECTORS / name).rglob("*.json")):
            try:
                doc = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            found.extend(
                (f"{path.relative_to(VECTORS)}:{where}", receipt)
                for where, receipt in _receipts(doc, "")
            )
    assert found, "no receipts found under the registered profiles"
    return found


@pytest.mark.parametrize("directory", _named_vector_dirs(), ids=lambda p: p.name)
def test_every_vector_directory_spec_names_exists(directory):
    assert directory.is_dir(), f"SPEC.md points at {directory}, which is absent"
    assert (directory / "_check_independent.py").exists(), (
        f"{directory.name} has no independent checker; SPEC.md Section 7 calls "
        f"the committed vectors plus _check_independent.py the reference suite"
    )


@pytest.mark.parametrize("directory", _named_vector_dirs(), ids=lambda p: p.name)
def test_checkers_import_only_what_the_spec_promises(directory):
    """"imports only the standard library, cryptography, and rfc8785"."""
    source = (directory / "_check_independent.py").read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    third_party = {
        name for name in imported
        if name not in sys.stdlib_module_names and not name.startswith("_")
    }
    assert third_party <= ALLOWED_THIRD_PARTY, (
        f"{directory.name}/_check_independent.py imports {sorted(third_party - ALLOWED_THIRD_PARTY)}, "
        f"which breaks the claim that a third party can replay the vectors "
        f"with no Vaara installed"
    )
    assert "vaara" not in imported


@pytest.mark.parametrize("directory", _named_vector_dirs(), ids=lambda p: p.name)
def test_checkers_pass_without_vaara_importable(directory, tmp_path):
    """Run each checker with Vaara removed from the import path."""
    checker = directory / "_check_independent.py"
    if "sys.argv" in checker.read_text() and directory.name == "article12_fold_v0":
        pytest.skip("takes a package path as an argument, not a self-contained run")
    done = subprocess.run(
        [sys.executable, "-I", str(checker)],
        cwd=directory, capture_output=True, text=True, timeout=300,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    if done.returncode == SKIP:
        pytest.skip(done.stdout.strip() or "optional dependency missing")
    # `cryptography` and `rfc8785` are the two the checkers are allowed to
    # need, and both are extras. On a base install they are absent, which is
    # the environment's business and not a claim failing.
    missing = re.search(r"No module named '([\w.]+)'", done.stderr)
    if missing and missing.group(1).split(".")[0] in ALLOWED_THIRD_PARTY:
        pytest.skip(f"{missing.group(1)} not installed in this environment")
    assert done.returncode == 0, (
        f"{directory.name}/_check_independent.py exited {done.returncode}\n"
        f"{done.stdout[-2000:]}\n{done.stderr[-2000:]}"
    )


def test_registered_profile_receipts_have_the_documented_envelope():
    documented = {"version", "alg", "backLink", "decisionDerived",
                  "issuerAsserted", "signature"}
    for where, receipt in _profile_receipts():
        assert documented <= set(receipt), f"{where} is missing {sorted(documented - set(receipt))}"
        assert set(receipt) - documented <= {"timestampAnchors"}, (
            f"{where} carries members Section 2 does not document: "
            f"{sorted(set(receipt) - documented - {'timestampAnchors'})}"
        )
        assert receipt["version"] == 1
        assert receipt["alg"] == "ES256", f"{where} uses {receipt['alg']}"
        assert re.fullmatch(r"[0-9a-f]{128}", receipt["signature"]), (
            f"{where} signature is not the documented 64-byte r||s pair"
        )
        assert receipt["decisionDerived"]["decision"] in VALID_VERDICTS


def test_spec_offers_no_algorithm_the_checkers_reject():
    """Regression: Section 2 offered ML-DSA-65 as an `alg` value."""
    row = re.search(r"^\|\s*`alg`\s*\|.*$", _spec(), re.MULTILINE)
    assert row, "SPEC.md no longer documents the alg field"
    offered = set(re.findall(r"`([A-Za-z0-9+-]+)`", row.group(0))) - {"alg", "string"}
    assert offered, "the alg row names no algorithm"
    assert offered <= VALID_ALGS, (
        f"SPEC.md offers {sorted(offered - VALID_ALGS)} as a receipt `alg`, "
        f"which the reference checkers reject"
    )


def test_both_completeness_conventions_are_the_shipped_ones():
    """Section 5.3 counts from 0; Section 6 counts from 1. Both are real."""
    authorization = [
        block
        for path in sorted((VECTORS / "contiguity_v0" / "complete").glob("*.json"))
        for block in [json.loads(path.read_text())["evidence"]["completeness"]]
    ]
    assert authorization, "no contiguity vectors found"
    assert min(b["seq"] for b in authorization) == 0
    assert all(b["runningCount"] == b["seq"] + 1 for b in authorization)

    ingest = [
        json.loads(path.read_text())["record"]["completeness"]
        for path in sorted((VECTORS / "ingest_v0" / "cases").glob("*.json"))
    ]
    assert ingest, "no ingest vectors carry a completeness block"
    assert all(b["seq"] == 1 and b["runningCount"] == 1 for b in ingest), (
        "SPEC.md Section 6 says a lone ingest is seq 1 of a one-record stream"
    )

    spec = " ".join(_spec().split())
    assert "the ingest stream counts from 1" in spec, (
        "SPEC.md no longer warns that the two conventions differ"
    )
