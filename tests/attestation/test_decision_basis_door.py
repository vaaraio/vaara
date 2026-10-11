# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The decision basis is reachable where the CHANGELOG says it lives.

CHANGELOG 1.34.0 documents ``vaara.attestation.build_decision_basis(...,
with_proof=True)``. The function lived in a private module and neither
``vaara.attestation`` nor ``vaara.attestation.decision`` exported it, so the
documented call raised AttributeError and the zero-knowledge decision proof
had no door a reader could open. Both packages export it now, with the
proof verifier and the fixed-point scaling a caller needs beside it.
"""

from __future__ import annotations

import vaara.attestation as attestation
import vaara.attestation.decision as decision

POLICY = {"version": 1, "deny": 0.8, "escalate": 0.5}
INTENT = "refund <= 50 EUR to verified customer"


def test_the_changelog_door_exists():
    assert attestation.build_decision_basis is decision.build_decision_basis
    assert "build_decision_basis" in attestation.__all__
    assert {"build_decision_basis", "verify_decision_proof", "to_fixed"} <= set(decision.__all__)


def test_the_documented_call_builds_the_three_blocks_and_the_proof_verifies():
    score, deny, escalate = decision.to_fixed(0.2), decision.to_fixed(0.8), decision.to_fixed(0.5)
    basis = attestation.build_decision_basis(
        rule="R1",
        reason="under threshold",
        declared_intent=INTENT,
        canonical_policy=POLICY,
        canonical_inputs={"riskScore": 0.2, "deny": 0.8, "escalate": 0.5},
        verdict="allow",
        score_fp=score,
        deny_fp=deny,
        escalate_fp=escalate,
        with_proof=True,
    )
    assert set(basis) == {"rationale", "binding", "decisionProof"}
    assert basis["rationale"]["rule"] == "R1"
    ok, why = decision.verify_decision_proof({
        "decision": "allow",
        "binding": basis["binding"],
        "decisionProof": basis["decisionProof"],
    })
    assert ok, why


def test_without_the_proof_the_basis_is_two_blocks():
    basis = decision.build_decision_basis(
        rule="R1", reason="under threshold", declared_intent=INTENT,
        canonical_policy=POLICY, canonical_inputs={"riskScore": 0.2},
        verdict="allow",
    )
    assert set(basis) == {"rationale", "binding"}
