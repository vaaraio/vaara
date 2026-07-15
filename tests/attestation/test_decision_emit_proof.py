from vaara.attestation._decision_conformance import check_decision_conformance
from vaara.attestation._decision_emit import build_decision_basis
from vaara.attestation._decision_proof_verify import verify_decision_proof
from vaara.attestation._decision_types import DecisionDerived, decision_to_dict
from vaara.attestation.zk._circuit import to_fixed

POLICY = {"version": 1, "deny": 0.8, "escalate": 0.5}
INPUTS = {"riskScore": 0.2, "deny": 0.8, "escalate": 0.5}
INTENT = "refund <= 50 EUR to verified customer"


def _dd(basis):
    return decision_to_dict(
        DecisionDerived(
            decision="allow",
            decided_at="2026-07-15T00:00:00Z",
            rationale=basis.get("rationale"),
            binding=basis.get("binding"),
            decision_proof=basis.get("decisionProof"),
        )
    )


def test_basis_without_proof_is_keyless_and_conformant():
    basis = build_decision_basis(
        rule="R1",
        reason="under threshold",
        declared_intent=INTENT,
        canonical_policy=POLICY,
        canonical_inputs=INPUTS,
        verdict="allow",
    )
    assert "decisionProof" not in basis
    dd = _dd(basis)
    report = check_decision_conformance(
        {
            "version": 1,
            "alg": "HS256",
            "backLink": {"attestationDigest": "sha256:" + "aa" * 32, "attestationNonce": "n"},
            "decisionDerived": dd,
            "issuerAsserted": {
                "iss": "x", "sub": "y", "iat": "2026-07-15T00:00:00Z",
                "nonce": "n", "secretVersion": "1", "alg": "HS256",
            },
            "signature": "ab",
        }
    )
    assert report.conforms, [c.detail for c in report.checks if not c.ok]


def test_basis_with_proof_verifies_and_binds():
    basis = build_decision_basis(
        rule="R1",
        reason="under threshold",
        declared_intent=INTENT,
        canonical_policy=POLICY,
        canonical_inputs=INPUTS,
        verdict="allow",
        score_fp=to_fixed(0.2),
        deny_fp=to_fixed(0.8),
        escalate_fp=to_fixed(0.5),
        with_proof=True,
    )
    env = basis["decisionProof"]
    assert env["publicInputs"]["bindingDigest"] == basis["binding"]["bindingDigest"]
    dd = _dd(basis)
    ok, reason = verify_decision_proof(dd)
    assert ok is True, reason
