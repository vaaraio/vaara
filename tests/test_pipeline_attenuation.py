"""Intercept-time enforcement of delegated-privilege attenuation.

A child action whose capability grant broadens the grant of the parent that
delegated to it is denied fail-closed, regardless of risk score, and the deny
is recorded in the audit trail. Actions with no capabilities behave exactly as
before (backward compatibility).
"""

from __future__ import annotations

from vaara import Pipeline
from vaara.audit.trail import EventType
from vaara.credential import Capability

PARENT_GRANT = (
    Capability("amount", "le", "500"),
    Capability("vendor", "in", ("acme", "globex")),
)
NARROWER = (  # subset of PARENT_GRANT -> attenuation, allowed
    Capability("amount", "le", "100"),
    Capability("vendor", "eq", "acme"),
)
BROADER = (  # raises the amount ceiling above the parent -> denied
    Capability("amount", "le", "5000"),
    Capability("vendor", "eq", "acme"),
)


def _parent(pipe: Pipeline):
    return pipe.intercept(
        agent_id="planner", tool_name="plan_task",
        parameters={}, capabilities=PARENT_GRANT,
    )


def test_attenuating_child_is_not_forced_denied():
    pipe = Pipeline()
    p = _parent(pipe)
    child = pipe.intercept(
        agent_id="worker", tool_name="read_document",
        parameters={}, parent_action_id=p.action_id, capabilities=NARROWER,
    )
    assert "privilege attenuation" not in child.reason


def test_broadening_child_is_denied_fail_closed():
    pipe = Pipeline()
    p = _parent(pipe)
    child = pipe.intercept(
        agent_id="worker", tool_name="read_document",
        parameters={}, parent_action_id=p.action_id, capabilities=BROADER,
    )
    assert child.allowed is False
    assert child.decision == "deny"
    assert "privilege attenuation violation" in child.reason
    assert "broadens:amount" in child.reason


def test_denial_is_recorded_in_audit_trail():
    pipe = Pipeline()
    p = _parent(pipe)
    child = pipe.intercept(
        agent_id="worker", tool_name="read_document",
        parameters={}, parent_action_id=p.action_id, capabilities=BROADER,
    )
    decisions = [
        r for r in pipe.trail.snapshot()
        if r.action_id == child.action_id
        and r.event_type == EventType.ACTION_BLOCKED
    ]
    assert decisions, "a blocked decision record should exist for the child"
    assert "privilege attenuation violation" in decisions[0].data.get("reason", "")
    # The tamper-evident chain still verifies.
    assert pipe.trail.verify_chain() is None


def test_no_capabilities_is_unchanged_behavior():
    pipe = Pipeline()
    p = pipe.intercept(agent_id="planner", tool_name="plan_task", parameters={})
    child = pipe.intercept(
        agent_id="worker", tool_name="read_document",
        parameters={}, parent_action_id=p.action_id,
    )
    assert "privilege attenuation" not in child.reason


def test_unknown_parent_grant_fails_open():
    # Parent action carried no capabilities, so there is no parent grant to
    # check against. The child is not denied on attenuation grounds.
    pipe = Pipeline()
    p = pipe.intercept(agent_id="planner", tool_name="plan_task", parameters={})
    child = pipe.intercept(
        agent_id="worker", tool_name="read_document",
        parameters={}, parent_action_id=p.action_id, capabilities=BROADER,
    )
    assert "privilege attenuation" not in child.reason
