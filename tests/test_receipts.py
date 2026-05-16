"""Article 12 commit-prove receipt-pair tests."""

from __future__ import annotations

import json

from vaara.audit.receipts import (
    CommitPayload,
    OutcomePayload,
    Receipt,
    extract_receipt,
    verify_receipt,
    verify_receipt_dict,
)
from vaara.audit.trail import AuditTrail, EventType
from vaara.taxonomy.actions import (
    ActionCategory,
    ActionRequest,
    ActionType,
    BlastRadius,
    RegulatoryDomain,
    Reversibility,
    UrgencyClass,
)


def _trail_with_action(decision: str = "allow", with_outcome: bool = True):
    trail = AuditTrail()
    action_type = ActionType(
        name="data.read",
        category=ActionCategory.DATA,
        reversibility=Reversibility.FULLY,
        blast_radius=BlastRadius.LOCAL,
        urgency=UrgencyClass.DEFERRABLE,
        regulatory_domains=frozenset({RegulatoryDomain.EU_AI_ACT}),
    )
    req = ActionRequest(
        agent_id="a-1",
        tool_name="data.read",
        action_type=action_type,
        confidence=0.9,
    )
    action_id = trail.record_action_requested(req)
    trail.record_risk_scored(
        action_id=action_id,
        agent_id="a-1",
        tool_name="data.read",
        assessment={
            "point_estimate": 0.3,
            "threshold_allow": 0.4,
            "threshold_deny": 0.7,
        },
        regulatory_domains=frozenset({RegulatoryDomain.EU_AI_ACT}),
    )
    trail.record_decision(
        action_id=action_id,
        agent_id="a-1",
        tool_name="data.read",
        decision=decision,
        reason="below threshold",
        risk_score=0.3,
        regulatory_domains=frozenset({RegulatoryDomain.EU_AI_ACT}),
    )
    if with_outcome:
        trail.record_outcome(
            action_id=action_id,
            agent_id="a-1",
            tool_name="data.read",
            outcome_severity=0.0,
            description="benign read completed",
        )
    return trail, action_id


def test_commit_hash_is_deterministic_64_hex():
    payload = CommitPayload(
        action_id="a", decision="allow",
        risk_score=0.3, threshold_allow=0.4, threshold_deny=0.7,
        decided_at=1700000000.0,
    )
    h1 = payload.hash()
    h2 = payload.hash()
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_commit_hash_changes_when_any_field_changes():
    base = CommitPayload(
        action_id="a", decision="allow", risk_score=0.3,
        threshold_allow=0.4, threshold_deny=0.7, decided_at=1.0,
    )
    flipped_decision = CommitPayload(
        action_id="a", decision="deny", risk_score=0.3,
        threshold_allow=0.4, threshold_deny=0.7, decided_at=1.0,
    )
    flipped_risk = CommitPayload(
        action_id="a", decision="allow", risk_score=0.31,
        threshold_allow=0.4, threshold_deny=0.7, decided_at=1.0,
    )
    flipped_time = CommitPayload(
        action_id="a", decision="allow", risk_score=0.3,
        threshold_allow=0.4, threshold_deny=0.7, decided_at=2.0,
    )
    h = base.hash()
    assert h != flipped_decision.hash()
    assert h != flipped_risk.hash()
    assert h != flipped_time.hash()


def test_outcome_payload_embeds_commit_hash():
    commit = CommitPayload(
        action_id="a", decision="allow", risk_score=0.3,
        threshold_allow=0.4, threshold_deny=0.7, decided_at=1.0,
    )
    outcome = OutcomePayload(
        action_id="a", commit_hash=commit.hash(),
        outcome_severity=0.0, recorded_at=2.0,
    )
    receipt = Receipt(commit=commit, outcome=outcome)
    assert verify_receipt(receipt) is True


def test_verify_receipt_detects_tampered_outcome_commit_hash():
    commit = CommitPayload(
        action_id="a", decision="allow", risk_score=0.3,
        threshold_allow=0.4, threshold_deny=0.7, decided_at=1.0,
    )
    outcome = OutcomePayload(
        action_id="a", commit_hash="0" * 64,
        outcome_severity=0.0, recorded_at=2.0,
    )
    receipt = Receipt(commit=commit, outcome=outcome)
    assert verify_receipt(receipt) is False


def test_extract_receipt_from_trail():
    trail, action_id = _trail_with_action(decision="allow")
    receipt = extract_receipt(trail, action_id)
    assert receipt is not None
    assert receipt.commit.action_id == action_id
    assert receipt.commit.decision == "allow"
    assert receipt.commit.risk_score == 0.3
    assert receipt.commit.threshold_allow == 0.4
    assert receipt.commit.threshold_deny == 0.7
    assert receipt.outcome is not None
    assert receipt.outcome.commit_hash == receipt.commit_hash
    assert verify_receipt(receipt) is True


def test_extract_receipt_no_outcome_yet():
    trail, action_id = _trail_with_action(with_outcome=False)
    receipt = extract_receipt(trail, action_id)
    assert receipt is not None
    assert receipt.outcome is None
    assert verify_receipt(receipt) is True


def test_extract_receipt_denied_decision():
    trail, action_id = _trail_with_action(decision="deny", with_outcome=False)
    receipt = extract_receipt(trail, action_id)
    assert receipt is not None
    assert receipt.commit.decision == "deny"


def test_extract_receipt_returns_none_for_unknown_action():
    trail, _ = _trail_with_action()
    assert extract_receipt(trail, "no-such-action") is None


def test_receipt_to_dict_round_trips_through_verify_receipt_dict():
    trail, action_id = _trail_with_action()
    receipt = extract_receipt(trail, action_id)
    assert receipt is not None
    d = receipt.to_dict()
    # Round-trip through JSON to catch any non-serializable surprises.
    d2 = json.loads(json.dumps(d))
    assert verify_receipt_dict(d2) is True


def test_verify_receipt_dict_rejects_tampered_serialized_form():
    trail, action_id = _trail_with_action()
    receipt = extract_receipt(trail, action_id)
    d = receipt.to_dict()
    d["commit"]["payload"]["decision"] = "deny"  # tamper but keep hash
    assert verify_receipt_dict(d) is False


def test_verify_receipt_dict_handles_garbage():
    assert verify_receipt_dict({}) is False
    assert verify_receipt_dict({"commit": {}}) is False
    assert verify_receipt_dict({"commit": {"payload": {}, "hash": "x"}}) is False
