"""Native intent + legible rationale on the SEP-2828 decision record.

The decision record used to carry only the verdict, risk scores, and an
optional content-addressed evidenceRef - the *why* had to be dereferenced
(and the intent lived in a separate attestation). This adds a native,
human-legible ``decisionDerived.rationale``: the rule that fired, a reason
line, and the declared intent the call was judged against, all inside the
2828 record itself. Additive and keyless: a record without ``rationale``
still conforms; when present, every field is checked.
"""

from __future__ import annotations

from vaara.attestation._decision_conformance import check_decision_conformance


def _decision() -> dict:
    """A minimal conforming SEP-2828 decision record (no rationale yet)."""
    return {
        "alg": "HS256",
        "backLink": {
            "attestationDigest": "sha256:" + "9" * 64,
            "attestationNonce": "call-A",
        },
        "decisionDerived": {
            "decidedAt": "2026-06-01T10:00:00Z",
            "decision": "allow",
        },
        "issuerAsserted": {
            "alg": "HS256",
            "iat": "2026-06-01T10:00:00Z",
            "iss": "issuer://test",
            "nonce": "d1",
            "secretVersion": "v1",
            "sub": "agent:reader",
        },
        "signature": "a" * 64,
        "version": 1,
    }


def _rationale() -> dict:
    return {
        "rule": "policy://tool-allowlist#fs.read",
        "reason": "fs.read on an allowlisted path within the approved scope",
        "declaredIntent": "read the project README to summarize it",
    }


def test_absent_rationale_still_conforms():
    """Non-breaking: an old record with no rationale is still conforming."""
    report = check_decision_conformance(_decision())
    assert report.conforms
    assert not any(c.id.startswith("rationale") for c in report.checks)


def test_valid_rationale_conforms_and_is_checked():
    doc = _decision()
    doc["decisionDerived"]["rationale"] = _rationale()
    report = check_decision_conformance(doc)
    assert report.conforms
    ids = {c.id for c in report.checks}
    assert {"rationale_object", "rationale_rule", "rationale_reason",
            "rationale_declared_intent"} <= ids


def test_rationale_not_object_fails():
    doc = _decision()
    doc["decisionDerived"]["rationale"] = "because"
    report = check_decision_conformance(doc)
    assert not report.conforms
    assert "rationale_object" in report.required_failed


def test_rationale_missing_rule_fails():
    doc = _decision()
    r = _rationale()
    del r["rule"]
    doc["decisionDerived"]["rationale"] = r
    report = check_decision_conformance(doc)
    assert not report.conforms
    assert "rationale_rule" in report.required_failed


def test_rationale_empty_reason_fails():
    doc = _decision()
    r = _rationale()
    r["reason"] = ""
    doc["decisionDerived"]["rationale"] = r
    report = check_decision_conformance(doc)
    assert not report.conforms
    assert "rationale_reason" in report.required_failed


def test_rationale_missing_declared_intent_fails():
    doc = _decision()
    r = _rationale()
    del r["declaredIntent"]
    doc["decisionDerived"]["rationale"] = r
    report = check_decision_conformance(doc)
    assert not report.conforms
    assert "rationale_declared_intent" in report.required_failed


def test_intent_satisfied_wrong_type_is_advisory():
    doc = _decision()
    r = _rationale()
    r["intentSatisfied"] = "yes"  # should be a bool
    doc["decisionDerived"]["rationale"] = r
    report = check_decision_conformance(doc)
    assert report.conforms  # advisory does not gate
    assert "rationale_intent_satisfied_type" in report.advisories
