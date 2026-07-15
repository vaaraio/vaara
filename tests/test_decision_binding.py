"""Commitment binding on the SEP-2828 decision record (the pre-proof layer).

``decisionDerived.binding`` pins a verdict to an exact policy, intent, and
input byte set by digest, so the sensitive content need not ship. Where the
declared intent is also present in the record, the intent digest is
recomputed from it (keyless), the same self-proving pattern the receipt uses
for its projection digest. This is the substrate a succinct proof of
decision correctness (v2) opens. Additive: a record without ``binding``
still conforms; when present, every commitment is checked.
"""

from __future__ import annotations

import hashlib

from vaara.attestation._decision_conformance import check_decision_conformance


def _sha(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


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


INTENT = "read the project README to summarize it"


def _binding() -> dict:
    return {
        "policyDigest": _sha("policy://tool-allowlist@3"),
        "intentDigest": _sha(INTENT),
        "inputsDigest": _sha('{"tool":"fs.read","path":"/README.md"}'),
        "bindingDigest": _sha("commit"),
    }


def test_absent_binding_still_conforms():
    report = check_decision_conformance(_decision())
    assert report.conforms
    assert not any(c.id.startswith("binding") for c in report.checks)


def test_valid_binding_conforms_and_is_checked():
    doc = _decision()
    doc["decisionDerived"]["binding"] = _binding()
    report = check_decision_conformance(doc)
    assert report.conforms
    ids = {c.id for c in report.checks}
    assert {"binding_object", "binding_policy_digest", "binding_intent_digest",
            "binding_inputs_digest", "binding_commitment_digest"} <= ids


def test_binding_not_object_fails():
    doc = _decision()
    doc["decisionDerived"]["binding"] = "sha256:whatever"
    report = check_decision_conformance(doc)
    assert not report.conforms
    assert "binding_object" in report.required_failed


def test_bad_policy_digest_fails():
    doc = _decision()
    b = _binding()
    b["policyDigest"] = "not-a-digest"
    doc["decisionDerived"]["binding"] = b
    report = check_decision_conformance(doc)
    assert not report.conforms
    assert "binding_policy_digest" in report.required_failed


def test_missing_intent_digest_fails():
    doc = _decision()
    b = _binding()
    del b["intentDigest"]
    doc["decisionDerived"]["binding"] = b
    report = check_decision_conformance(doc)
    assert not report.conforms
    assert "binding_intent_digest" in report.required_failed


def test_intent_digest_recomputes_from_declared_intent():
    """Keyless self-check: intentDigest must equal sha256 over declaredIntent."""
    doc = _decision()
    doc["decisionDerived"]["rationale"] = {
        "rule": "policy://tool-allowlist#fs.read",
        "reason": "allowlisted read",
        "declaredIntent": INTENT,
    }
    doc["decisionDerived"]["binding"] = _binding()
    report = check_decision_conformance(doc)
    assert report.conforms
    assert "binding_intent_self_consistent" in {c.id for c in report.checks}


def test_intent_digest_mismatch_gates():
    doc = _decision()
    doc["decisionDerived"]["rationale"] = {
        "rule": "r", "reason": "x", "declaredIntent": INTENT,
    }
    b = _binding()
    b["intentDigest"] = _sha("a different intent entirely")
    doc["decisionDerived"]["binding"] = b
    report = check_decision_conformance(doc)
    assert not report.conforms
    assert "binding_intent_self_consistent" in report.required_failed
