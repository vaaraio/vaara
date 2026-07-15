import hashlib

from vaara.attestation._decision_binding import (
    intent_digest,
    binding_digest,
    build_binding,
    build_rationale,
)


def test_intent_digest_matches_checker_rule():
    intent = "refund <= 50 EUR to verified customer"
    expected = "sha256:" + hashlib.sha256(intent.encode()).hexdigest()
    assert intent_digest(intent) == expected


def test_binding_digest_stable():
    d = binding_digest(
        "sha256:" + "a" * 64, "sha256:" + "b" * 64, "sha256:" + "c" * 64, "allow"
    )
    assert d.startswith("sha256:") and len(d) == 71


def test_build_binding_self_consistent():
    b = build_binding({"v": 1}, "do x", {"risk": 0.2, "deny": 0.8}, "allow")
    assert b["intentDigest"] == intent_digest("do x")
    assert set(b) == {"policyDigest", "intentDigest", "inputsDigest", "bindingDigest"}


def test_build_rationale_shape():
    r = build_rationale("R1", "under threshold", "do x", True)
    assert r == {
        "rule": "R1",
        "reason": "under threshold",
        "declaredIntent": "do x",
        "intentSatisfied": True,
    }
