"""Regression tests for prior-approval auto-allow (2026-08 audit C2).

Three invariants pin the audited defects:
1. Approving ``tx.transfer`` amount=10 must NOT auto-allow amount=999999
   (the args-digest guard was vacuous because the digest was never stored).
2. An auto-allowed action's trail must show decision=allow with NO
   ESCALATION_SENT — the record and the behaviour must agree.
3. Tenant A's approval must never auto-allow tenant B's same-named
   agent + tool (the lookup was not tenant-scoped).
"""

import hashlib
import json

import pytest

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import EventType
from vaara.pipeline import InterceptionPipeline


@pytest.fixture
def pipeline():
    backend = SQLiteAuditBackend(":memory:")
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    return InterceptionPipeline(trail=trail)


def _escalate_and_approve(pipeline, amount, *, agent="agent-1", tenant=""):
    result = pipeline.intercept(
        agent_id=agent,
        tool_name="tx.transfer",
        parameters={"amount": amount},
        tenant_id=tenant,
    )
    assert result.decision == "escalate", (
        f"precondition: tx.transfer amount={amount} must escalate, "
        f"got {result.decision}"
    )
    pipeline.resolve_escalation(
        action_id=result.action_id,
        resolution="allow",
        reviewer="admin@example.com",
        justification="approved in review",
    )
    return result


def _expected_digest(params: dict) -> str:
    return hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()


class TestPriorApprovalAutoAllow:
    def test_same_shape_auto_allows(self, pipeline):
        _escalate_and_approve(pipeline, amount=10)
        second = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 10},
        )
        assert second.allowed is True
        assert second.decision == "allow"
        assert "prior approval" in second.reason

    def test_different_amount_does_not_auto_allow(self, pipeline):
        _escalate_and_approve(pipeline, amount=10)
        second = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 999999},
        )
        assert second.allowed is False
        assert second.decision == "escalate"

    def test_auto_allow_trail_agrees_with_behaviour(self, pipeline):
        _escalate_and_approve(pipeline, amount=10)
        second = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 10},
        )
        trail = pipeline.trail.get_action_trail(second.action_id)
        event_types = [r.event_type for r in trail]
        decisions = [
            r for r in trail if r.event_type == EventType.DECISION_MADE
        ]
        # Caller was allowed; the chain must record allow, never a
        # dangling escalate with no ESCALATION_SENT behind it.
        assert EventType.ESCALATION_SENT not in event_types
        assert len(decisions) == 1
        assert decisions[0].data["decision"] == "allow"
        assert "prior approval" in decisions[0].data["reason"]

    def test_real_escalation_still_records_sent(self, pipeline):
        _escalate_and_approve(pipeline, amount=10)
        second = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 999999},
        )
        trail = pipeline.trail.get_action_trail(second.action_id)
        event_types = [r.event_type for r in trail]
        assert EventType.ESCALATION_SENT in event_types

    def test_tenant_isolation(self, pipeline):
        _escalate_and_approve(pipeline, amount=10, tenant="tenant-a")
        second = pipeline.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 10},
            tenant_id="tenant-b",
        )
        assert second.decision == "escalate"
        assert second.allowed is False

    def test_resolution_record_carries_args_digest(self, pipeline):
        approved = _escalate_and_approve(pipeline, amount=10)
        trail = pipeline.trail.get_action_trail(approved.action_id)
        resolved = [
            r for r in trail
            if r.event_type == EventType.ESCALATION_RESOLVED
        ]
        assert len(resolved) == 1
        assert resolved[0].data["args_digest"] == _expected_digest(
            {"amount": 10}
        )

    def test_digest_persists_across_restart(self, tmp_path):
        db = str(tmp_path / "audit.db")
        backend = SQLiteAuditBackend(db)
        trail = backend.load_trail()
        trail._on_record = backend.write_record
        p1 = InterceptionPipeline(trail=trail)
        _escalate_and_approve(p1, amount=10)

        backend2 = SQLiteAuditBackend(db)
        trail2 = backend2.load_trail()
        trail2._on_record = backend2.write_record
        p2 = InterceptionPipeline(trail=trail2)
        second = p2.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 10},
        )
        assert second.decision == "allow"
        # ...and the guard survives the restart too:
        third = p2.intercept(
            agent_id="agent-1",
            tool_name="tx.transfer",
            parameters={"amount": 999999},
        )
        assert third.decision == "escalate"
