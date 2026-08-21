# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gate the attribute_attestation_zk_v0 vectors through both graders.

The independent checker rebuilds P-256, the Pedersen commitments and the range
argument from the published parameters, imports no Vaara, and recomputes every
verdict from the case bytes. A pass there means a stranger with rfc8785 and
cryptography reaches the same answer. The shipped module is then run over the
same committed bytes, which holds the implementation to the corpus rather than
the other way round.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

VECTORS = Path(__file__).resolve().parent / "vectors" / "attribute_attestation_zk_v0"
CASES = sorted(p.stem for p in (VECTORS / "cases").glob("*.json"))


def _case(name: str) -> dict:
    return json.loads((VECTORS / "cases" / f"{name}.json").read_text(encoding="utf-8"))


def test_all_four_states_appear_in_the_corpus():
    states = {_case(n)["expected_state"] for n in CASES}
    assert states == {"accepted", "withheld", "expired", "refused"}


def test_no_case_carries_a_plaintext_value():
    """The corpus would be worthless if a fixture leaked what it commits to."""
    for name in CASES:
        for attribute in _case(name)["attestation"]["attributes"]:
            assert "value" not in attribute
            assert set(attribute) <= {"name", "commitment", "source", "sourceDetail"}


def test_both_predicate_directions_are_exercised():
    """A corpus that only ever proves >= never tests the negated blind."""
    kinds = {_case(n)["query"]["predicate"]["kind"] for n in CASES}
    assert "in_range" in kinds and "at_least" in kinds


def test_nothing_proved_and_something_forged_are_separate_cases():
    reasons = {_case(n)["expected_reason"] for n in CASES}
    assert {"proof_absent", "proof_invalid", "proof_not_bound"} <= reasons


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("name", CASES)
def test_reference_implementation_agrees_with_the_corpus(name):
    from cryptography.hazmat.primitives import serialization

    from vaara.attestation.attribute_zk import (
        Predicate,
        PredicateQuery,
        SourceStanding,
        evaluate,
    )
    from vaara.audit.signer import Ed25519Verifier

    case = _case(name)
    pub = serialization.load_pem_public_key((VECTORS / case["issuer_key"]).read_bytes())
    q = case["query"]
    accepted = q.get("accepted_issuers")
    decision = evaluate(
        case["attestation"],
        PredicateQuery(
            name=q["name"],
            predicate=Predicate.from_dict(q["predicate"]),
            minimum_source=SourceStanding(q["minimum_source"]),
            subject_id=q.get("subject_id"),
            accepted_issuers=frozenset(accepted) if accepted else None,
        ),
        proof=case.get("proof"),
        now=case["now"],
        verifier=Ed25519Verifier(pub.public_bytes_raw()),
    )
    assert decision.state.value == case["expected_state"]
    assert decision.reason.value == case["expected_reason"]


def test_regeneration_is_byte_stable():
    """The blinds and the prover's scalars are pinned, so the bytes reproduce."""
    before = {p.name: p.read_bytes() for p in sorted((VECTORS / "cases").glob("*.json"))}
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_generate.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    after = {p.name: p.read_bytes() for p in sorted((VECTORS / "cases").glob("*.json"))}
    assert before == after
