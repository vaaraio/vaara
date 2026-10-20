"""Reviewer-signed review resolutions: the person who resolved an escalation
signs it, bound to the decision record it overrides, and the verifier checks
the signature against a reviewer key the principal pinned in the permission."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation._attest_canonical import iso8601_to_epoch  # noqa: E402
from vaara.attestation._attest_types import AttestationError  # noqa: E402
from vaara.audit.review_queue import ReviewQueue  # noqa: E402
from vaara.audit.trail import AuditTrail, EventType  # noqa: E402
from vaara.credential import (  # noqa: E402
    REVIEW_RESOLUTION_TYPE,
    Reviewer,
    check_review_resolutions,
    permission_digest,
    permission_from_dict,
    sign_permission,
    verify_permission_signature,
)

PRINCIPAL = ec.generate_private_key(ec.SECP256R1())
BOB = ec.generate_private_key(ec.SECP256R1())
MALLORY = ec.generate_private_key(ec.SECP256R1())
START = "2026-10-01T00:00:00Z"
END = "2026-11-01T00:00:00Z"
T0 = iso8601_to_epoch(START)


def _pem(key) -> str:
    return key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()


BOB_REVIEWER = Reviewer(reviewer="did:example:bob", alg="ES256", public_key=_pem(BOB))


def _perm(reviewers=(BOB_REVIEWER,)):
    return sign_permission(
        principal="alice@example.org", key_id="alice-2026", agent="agent:buyer",
        tool_name="pay.send", not_before=START, not_after=END, alg="ES256",
        signing_material=PRINCIPAL, reviewers=reviewers,
    )


def _escalated(trail, *, decision="escalate"):
    from vaara.taxonomy.actions import ActionRequest, create_default_registry

    args = {"amount": 900}
    req = ActionRequest(agent_id="agent:buyer", tool_name="pay.send",
                        action_type=create_default_registry().classify("pay.send", args),
                        parameters=args)
    aid = trail.record_action_requested(req)
    trail.record_decision(action_id=aid, agent_id="agent:buyer", tool_name="pay.send",
                          decision=decision, reason="t", risk_score=0.6)
    return aid


def _resolve(trail, aid, *, reviewer="did:example:bob", key=BOB, resolution="allow"):
    q = ReviewQueue(":memory:")
    qid = q.enqueue(
        action_id=aid, agent_id="agent:buyer", tool_name="pay.send",
        risk_score=0.6, conformal_lower=0.5, conformal_upper=0.7,
        action_type="pay", bucket_category="finance", reason="borderline",
        parameters={"amount": 900}, context={}, signals={},
    )
    return q.resolve(qid, reviewer=reviewer, resolution=resolution,
                     justification="vendor confirmed by phone", trail=trail,
                     reviewer_key=key)


def _resolution_row(trail):
    return [r for r in trail._records if r.event_type == EventType.ESCALATION_RESOLVED][-1]


def _check(trail, p=None):
    return check_review_resolutions(
        trail, p or _perm(), verifying_material=PRINCIPAL.public_key(), verify_chain=False)


def test_permission_without_reviewers_signs_and_digests_as_before():
    # No reviewers = the dict carries no "reviewers" key, so every permission
    # signed before this field existed keeps its bytes, signature and digest.
    p = _perm(reviewers=())
    assert "reviewers" not in p.to_dict()
    assert permission_from_dict(json.loads(json.dumps(p.to_dict()))) == p


def test_reviewers_ride_inside_the_principal_signature():
    p = _perm()
    d = json.loads(json.dumps(p.to_dict()))
    assert d["reviewers"] == [{"reviewer": "did:example:bob", "alg": "ES256",
                               "publicKey": _pem(BOB)}]
    again = permission_from_dict(d)
    assert again == p and permission_digest(again) == permission_digest(p)
    d["reviewers"][0]["publicKey"] = _pem(MALLORY)
    swapped = permission_from_dict(d)
    assert not verify_permission_signature(swapped, verifying_material=PRINCIPAL.public_key())


@pytest.mark.parametrize("entry,needle", [
    ({"reviewer": "did:example:bob", "alg": "HS256", "publicKey": "x"}, "alg"),
    ({"reviewer": "", "alg": "ES256", "publicKey": _pem(BOB)}, "reviewer"),
    ({"reviewer": "did:example:bob", "alg": "ES256", "publicKey": "not a key"}, "publicKey"),
    ({"reviewer": "did:example:bob", "alg": "ES256", "publicKey": _pem(BOB), "x": 1},
     "unrecognized"),
])
def test_a_malformed_reviewer_entry_is_refused(entry, needle):
    d = _perm().to_dict()
    d["reviewers"] = [entry]
    with pytest.raises(AttestationError, match=needle):
        permission_from_dict(d)


def test_resolve_signs_over_the_decision_record_hash():
    trail = AuditTrail()
    aid = _escalated(trail)
    decision = [r for r in trail.get_action_trail(aid)
                if r.event_type == EventType.DECISION_MADE][-1]
    _resolve(trail, aid)
    block = _resolution_row(trail).data["reviewer_signature"]
    assert block["type"] == REVIEW_RESOLUTION_TYPE
    assert block["alg"] == "ES256"
    assert block["decisionRecordHash"] == decision.record_hash
    assert "publicKey" not in block


def test_an_unsigned_resolution_row_hashes_exactly_as_before():
    trail = AuditTrail()
    aid = _escalated(trail)
    q = ReviewQueue(":memory:")
    qid = q.enqueue(
        action_id=aid, agent_id="agent:buyer", tool_name="pay.send",
        risk_score=0.6, conformal_lower=0.5, conformal_upper=0.7,
        action_type="pay", bucket_category="finance", reason="borderline",
        parameters={}, context={}, signals={},
    )
    q.resolve(qid, reviewer="bob", resolution="allow", trail=trail)
    assert "reviewer_signature" not in _resolution_row(trail).data


def test_signing_without_a_trail_or_a_decision_is_refused_before_anything_changes():
    q = ReviewQueue(":memory:")
    qid = q.enqueue(
        action_id="no-such-action", agent_id="agent:buyer", tool_name="pay.send",
        risk_score=0.6, conformal_lower=0.5, conformal_upper=0.7,
        action_type="pay", bucket_category="finance", reason="borderline",
        parameters={}, context={}, signals={},
    )
    with pytest.raises(ValueError, match="trail"):
        q.resolve(qid, reviewer="bob", resolution="allow", reviewer_key=BOB)
    with pytest.raises(ValueError, match="decision"):
        q.resolve(qid, reviewer="bob", resolution="allow", reviewer_key=BOB,
                  trail=AuditTrail())
    assert q.get(qid).status == "pending"


def test_a_pinned_reviewer_signature_verifies_and_the_override_is_named():
    trail = AuditTrail()
    aid = _escalated(trail)
    _resolve(trail, aid)
    report = _check(trail)
    assert report.ok
    [rev] = report.reviews
    assert (rev.action_id, rev.reviewer, rev.resolution, rev.state) == (
        aid, "did:example:bob", "allow", "signed")
    # The human overrode an escalation: reported as an override for this one
    # action, never folded into the policy's allows.
    assert rev.override is True and rev.decision == "escalate"


def test_a_deny_resolution_is_not_an_override():
    trail = AuditTrail()
    aid = _escalated(trail)
    _resolve(trail, aid, resolution="deny")
    [rev] = _check(trail).reviews
    assert rev.state == "signed" and rev.override is False


def test_a_reviewer_the_permission_does_not_name_is_a_finding():
    trail = AuditTrail()
    aid = _escalated(trail)
    _resolve(trail, aid, reviewer="did:example:mallory", key=MALLORY)
    report = _check(trail)
    assert not report.ok and report.reviews[0].state == "unknown_reviewer"


def test_a_named_reviewer_signing_with_another_key_is_a_finding():
    trail = AuditTrail()
    aid = _escalated(trail)
    _resolve(trail, aid, key=MALLORY)
    report = _check(trail)
    assert not report.ok and report.reviews[0].state == "bad_signature"


def test_a_resolution_edited_after_signing_fails_its_signature():
    trail = AuditTrail()
    aid = _escalated(trail)
    _resolve(trail, aid, resolution="deny")
    _resolution_row(trail).data["resolution"] = "allow"
    assert _check(trail).reviews[0].state == "bad_signature"


def test_a_signature_bound_to_another_decision_is_a_finding():
    trail = AuditTrail()
    aid = _escalated(trail)
    _resolve(trail, aid)
    _resolution_row(trail).data["reviewer_signature"]["decisionRecordHash"] = "0" * 64
    report = _check(trail)
    assert not report.ok and report.reviews[0].state == "decision_mismatch"


def test_a_key_carried_in_the_row_is_never_used():
    trail = AuditTrail()
    aid = _escalated(trail)
    _resolve(trail, aid, key=MALLORY)
    _resolution_row(trail).data["reviewer_signature"]["publicKey"] = _pem(MALLORY)
    report = _check(trail)
    assert not report.ok and report.reviews[0].state == "malformed"


def test_unsigned_is_a_finding_only_when_the_principal_named_reviewers():
    trail = AuditTrail()
    aid = _escalated(trail)
    _resolve(trail, aid, key=None)
    named = _check(trail)
    assert not named.ok and named.reviews[0].state == "unsigned"
    unnamed = _check(trail, _perm(reviewers=()))
    assert unnamed.ok and unnamed.reviews[0].state == "unsigned"


def test_a_signature_with_no_pinned_key_is_reported_unpinned_not_signed():
    trail = AuditTrail()
    aid = _escalated(trail)
    _resolve(trail, aid)
    report = _check(trail, _perm(reviewers=()))
    assert report.ok and report.reviews[0].state == "unpinned"


def test_a_bad_principal_signature_or_chain_judges_nothing():
    trail = AuditTrail()
    aid = _escalated(trail)
    _resolve(trail, aid)
    report = check_review_resolutions(
        trail, _perm(), verifying_material=MALLORY.public_key(), verify_chain=False)
    assert report.grant_problem == "bad_signature" and report.reviews == ()
    assert not report.ok


def test_resolutions_for_other_agents_are_left_out():
    from vaara.taxonomy.actions import ActionRequest, create_default_registry

    trail = AuditTrail()
    req = ActionRequest(agent_id="agent:other", tool_name="pay.send",
                        action_type=create_default_registry().classify("pay.send", {}),
                        parameters={})
    aid = trail.record_action_requested(req)
    trail.record_decision(action_id=aid, agent_id="agent:other", tool_name="pay.send",
                          decision="escalate", reason="t", risk_score=0.6)
    trail.record_escalation_resolved(action_id=aid, agent_id="agent:other",
                                     tool_name="pay.send", resolution="allow",
                                     reviewer="x")
    assert _check(trail).reviews == ()


def test_cli_resolves_with_a_signing_key_and_check_permission_reports_it(tmp_path, capsys):
    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.cli import main

    def priv(key, name):
        path = tmp_path / name
        path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                           serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
        return str(path)

    alice_pub = tmp_path / "alice.pub.pem"
    alice_pub.write_text(_pem(PRINCIPAL))
    bob_pub = tmp_path / "bob.pub.pem"
    bob_pub.write_text(_pem(BOB))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "principal": "alice@example.org", "agent": "agent:buyer", "toolName": "pay.send",
        "notBefore": "2026-01-01T00:00:00Z", "notAfter": "2099-01-01T00:00:00Z",
        "reviewers": [{"reviewer": "did:example:bob", "publicKeyFile": str(bob_pub)}],
    }))
    perm = tmp_path / "permission.json"
    assert main(["permission", "sign", "--spec", str(spec),
                 "--key", priv(PRINCIPAL, "alice.pem"), "--out", str(perm)]) in (None, 0)
    assert json.loads(perm.read_text())["reviewers"][0]["alg"] == "ES256"

    db = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(db)
    aid = _escalated(backend.load_trail())
    backend.close()
    queue = tmp_path / "queue.db"
    with ReviewQueue(queue) as q:
        qid = q.enqueue(
            action_id=aid, agent_id="agent:buyer", tool_name="pay.send",
            risk_score=0.6, conformal_lower=0.5, conformal_upper=0.7,
            action_type="pay", bucket_category="finance", reason="borderline",
            parameters={}, context={}, signals={},
        )
    assert main(["review", "resolve", "--queue-id", qid, "--db", str(queue), "--audit-db", str(db),
                 "--reviewer", "did:example:bob", "--resolution", "allow",
                 "--sign-key", priv(BOB, "bob.pem")]) in (None, 0)
    capsys.readouterr()

    code = main(["trail", "check-permission", "--db", str(db), "--permission", str(perm),
                 "--pubkey-file", str(alice_pub), "--format", "json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["reviews"]["reviews"][0]["state"] == "signed"
    assert report["reviews"]["reviews"][0]["override"] is True
