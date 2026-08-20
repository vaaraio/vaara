# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gate the release_condition_v0 vectors through both graders.

Two things have to hold, and they are different things. The independent checker
imports no Vaara and recomputes every verdict from the case bytes, so a pass
there means a stranger with ``rfc8785`` and ``cryptography`` reaches the same
answer. Then the shipped module is run over the same committed bytes, so the
reference implementation is held to the corpus rather than the corpus being
whatever the implementation happens to say today.

The same cases also run through the ``vaara release-check`` CLI, which is what a
settlement agent would actually invoke.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

VECTORS = Path(__file__).resolve().parent / "vectors" / "release_condition_v0"
CASES = sorted(p.stem for p in (VECTORS / "cases").glob("*.json"))

# Exit codes the CLI contracts on: 0 released, 1 not released, 2 usage error.
_NOT_RELEASED = 1


def _case(name: str) -> dict:
    return json.loads((VECTORS / "cases" / f"{name}.json").read_text(encoding="utf-8"))


def test_the_corpus_is_not_only_positive():
    # A suite of one green case would prove nothing about the distinctions this
    # profile exists to make. All four states must appear in the shipped cases.
    states = {_case(name)["expected_state"] for name in CASES}
    assert states == {"released", "held", "expired", "refused"}


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("name", CASES)
def test_reference_implementation_agrees_with_the_corpus(name):
    from cryptography.hazmat.primitives import serialization

    from vaara.audit.signer import Ed25519Verifier
    from vaara.settlement.release import ReleaseBundle, evaluate

    case = _case(name)
    condition_key = serialization.load_pem_public_key(
        (VECTORS / case["condition_key"]).read_bytes()
    )
    receipt_key_pem = (
        (VECTORS / case["receipt_key"]).read_bytes() if case.get("receipt_key") else None
    )
    decision = evaluate(
        case["condition"],
        ReleaseBundle(
            now=case["now"],
            receipt=case.get("receipt"),
            evidence=case.get("evidence"),
            condition_verifier=Ed25519Verifier(condition_key.public_bytes_raw()),
            receipt_public_key_pem=receipt_key_pem,
        ),
    )
    assert decision.state.value == case["expected_state"]
    assert decision.reason.value == case["expected_reason"]


@pytest.mark.parametrize("name", CASES)
def test_cli_release_check(name):
    case = _case(name)
    cmd = [
        sys.executable, "-m", "vaara.cli", "release-check",
        str(VECTORS / "cases" / f"{name}.json"),
        "--condition-key", str(VECTORS / case["condition_key"]),
        "--now", case["now"],
        "--json",
    ]
    if case.get("receipt_key"):
        cmd += ["--receipt-key", str(VECTORS / case["receipt_key"])]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    expected_code = 0 if case["expected_state"] == "released" else _NOT_RELEASED
    assert proc.returncode == expected_code, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["state"] == case["expected_state"]
    assert report["reason"] == case["expected_reason"]


def test_cli_refuses_a_condition_it_cannot_verify():
    # No --condition-key: the money must not move, and the answer must not read
    # as "still waiting" either.
    proc = subprocess.run(
        [
            sys.executable, "-m", "vaara.cli", "release-check",
            str(VECTORS / "cases" / "pos_matching_receipt.json"),
            "--receipt-key", str(VECTORS / "keys" / "es256_public.pem"),
            "--now", "2026-08-21T12:00:00Z",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == _NOT_RELEASED, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["state"] == "refused"
    assert report["reason"] == "condition_key_absent"
