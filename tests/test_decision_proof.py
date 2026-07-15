"""The decisionProof envelope on the SEP-2828 decision record (v2a: shape).

``decisionDerived.decisionProof`` is Vaara's wire format for a succinct
proof that the verdict is the correct output of the committed policy on the
committed intent and inputs, without revealing them. This layer checks the
envelope shape only: it is keyless and runs in the base install, the same
split the receipt uses (conformance checks shape; the signature/anchor
verify needs external material). Verifying the proof itself lives behind the
attestation extra (v2b). Where a ``binding`` is present, the proof's public
bindingDigest must equal it, so the record self-attests that the proof is
about this exact commitment, no proving system needed to see that.
"""

from __future__ import annotations

import hashlib

from vaara.attestation._decision_conformance import check_decision_conformance


def _sha(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


BINDING_DIGEST = _sha("commit")


def _decision() -> dict:
    return {
        "alg": "HS256",
        "backLink": {"attestationDigest": "sha256:" + "9" * 64, "attestationNonce": "call-A"},
        "decisionDerived": {"decidedAt": "2026-06-01T10:00:00Z", "decision": "allow"},
        "issuerAsserted": {
            "alg": "HS256", "iat": "2026-06-01T10:00:00Z", "iss": "issuer://test",
            "nonce": "d1", "secretVersion": "v1", "sub": "agent:reader",
        },
        "signature": "a" * 64,
        "version": 1,
    }


def _proof() -> dict:
    return {
        "proofSystem": "vaara-p256-cap-v0",
        "publicInputs": {"bindingDigest": BINDING_DIGEST, "verdict": "allow"},
        "proof": "ab" * 128,
        "verifierParamsDigest": _sha("policy-circuit@1"),
    }


def test_absent_proof_still_conforms():
    report = check_decision_conformance(_decision())
    assert report.conforms
    assert not any(c.id.startswith("decision_proof") for c in report.checks)


def test_valid_proof_conforms_and_is_checked():
    doc = _decision()
    doc["decisionDerived"]["decisionProof"] = _proof()
    report = check_decision_conformance(doc)
    assert report.conforms
    ids = {c.id for c in report.checks}
    assert {"decision_proof_object", "decision_proof_system",
            "decision_proof_public_inputs_object", "decision_proof_binding_digest_format",
            "decision_proof_bytes", "decision_proof_params_digest"} <= ids


def test_proof_not_object_fails():
    doc = _decision()
    doc["decisionDerived"]["decisionProof"] = "deadbeef"
    report = check_decision_conformance(doc)
    assert not report.conforms
    assert "decision_proof_object" in report.required_failed


def test_missing_proof_system_fails():
    doc = _decision()
    p = _proof()
    del p["proofSystem"]
    doc["decisionDerived"]["decisionProof"] = p
    report = check_decision_conformance(doc)
    assert not report.conforms
    assert "decision_proof_system" in report.required_failed


def test_bad_proof_hex_fails():
    doc = _decision()
    p = _proof()
    p["proof"] = "xyz"  # not hex
    doc["decisionDerived"]["decisionProof"] = p
    report = check_decision_conformance(doc)
    assert not report.conforms
    assert "decision_proof_bytes" in report.required_failed


def test_bad_params_digest_fails():
    doc = _decision()
    p = _proof()
    p["verifierParamsDigest"] = "nope"
    doc["decisionDerived"]["decisionProof"] = p
    report = check_decision_conformance(doc)
    assert not report.conforms
    assert "decision_proof_params_digest" in report.required_failed


def test_proof_must_bind_to_the_records_commitment():
    """When binding is present, the proof's public bindingDigest must match it."""
    doc = _decision()
    doc["decisionDerived"]["binding"] = {
        "policyDigest": _sha("p"), "intentDigest": _sha("i"),
        "inputsDigest": _sha("in"), "bindingDigest": BINDING_DIGEST,
    }
    p = _proof()
    p["publicInputs"]["bindingDigest"] = _sha("a different commitment")
    doc["decisionDerived"]["decisionProof"] = p
    report = check_decision_conformance(doc)
    assert not report.conforms
    assert "decision_proof_binds_commitment" in report.required_failed
