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
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    backend = SQLiteAuditBackend(":memory:")
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    pipe = Pipeline(trail=trail)
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


# --- Fail-closed on an unknown parent grant (v2.17.0) -----------------------
#
# The in-memory action -> grant map is bounded and per process. A child that
# names a parent the map does not hold (evicted, another process, invented)
# used to skip the check. The parent's grant now rides on its action_requested
# record, so the pipeline resolves it from the trail; a parent in neither place
# is denied unless the deployer chose unknown_parent="allow", and every decision
# on a delegated grant says whether the check ran.

import pytest  # noqa: E402

from vaara import pipeline as pipeline_mod  # noqa: E402


def _sqlite_trail(path):
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    return SQLiteAuditBackend(str(path)).load_trail()


def _decision(pipe, action_id):
    return [
        r for r in pipe.trail.get_action_trail(action_id)
        if r.event_type in (EventType.DECISION_MADE, EventType.ACTION_BLOCKED)
    ][-1]


def _child(pipe, parent_id, caps=BROADER, **kw):
    return pipe.intercept(
        agent_id="worker", tool_name="read_document",
        parameters={}, parent_action_id=parent_id, capabilities=caps, **kw,
    )


def test_parent_grant_is_recorded_on_its_request():
    pipe = Pipeline()
    p = _parent(pipe)
    req = pipe.trail.get_action_trail(p.action_id)[0]
    assert req.event_type == EventType.ACTION_REQUESTED
    assert req.data["capabilities"] == [
        {"arg": "amount", "op": "le", "value": "500"},
        {"arg": "vendor", "op": "in", "value": ["acme", "globex"]},
    ]


def test_request_without_grant_has_no_capabilities_key():
    pipe = Pipeline()
    p = pipe.intercept(agent_id="planner", tool_name="plan_task", parameters={})
    assert "capabilities" not in pipe.trail.get_action_trail(p.action_id)[0].data


def test_held_check_is_marked_on_the_decision():
    pipe = Pipeline()
    p = _parent(pipe)
    child = _child(pipe, p.action_id, caps=NARROWER)
    assert _decision(pipe, child.action_id).data["attenuation"] == {
        "status": "held", "source": "memory",
    }


def test_violation_is_marked_on_the_decision():
    pipe = Pipeline()
    p = _parent(pipe)
    child = _child(pipe, p.action_id)
    att = _decision(pipe, child.action_id).data["attenuation"]
    assert att["status"] == "violated"
    assert att["reason"].startswith("broadens:amount")


def test_root_grant_without_parent_carries_no_attenuation_mark():
    pipe = Pipeline()
    p = _parent(pipe)
    assert "attenuation" not in _decision(pipe, p.action_id).data


def test_broadening_child_denied_after_parent_evicted(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "_MAX_TRACKED_GRANTS", 2)
    pipe = Pipeline()
    p = _parent(pipe)
    for _ in range(3):  # push the parent grant out of the memory map
        _parent(pipe)
    assert p.action_id not in pipe._grants_for_action
    child = _child(pipe, p.action_id)
    assert child.allowed is False
    assert "privilege attenuation violation" in child.reason
    att = _decision(pipe, child.action_id).data["attenuation"]
    assert att["status"] == "violated" and att["source"] == "trail"


def test_narrowing_child_held_after_parent_evicted(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "_MAX_TRACKED_GRANTS", 1)
    pipe = Pipeline()
    p = _parent(pipe)
    _parent(pipe)
    child = _child(pipe, p.action_id, caps=NARROWER)
    assert "privilege attenuation" not in child.reason
    assert _decision(pipe, child.action_id).data["attenuation"] == {
        "status": "held", "source": "trail",
    }


def test_broadening_child_denied_across_two_pipelines_on_one_trail(tmp_path):
    db = tmp_path / "audit.db"
    first = Pipeline(trail=_sqlite_trail(db))
    p = _parent(first)
    # A fresh process: a new pipeline over the same database, empty memory.
    second = Pipeline(trail=_sqlite_trail(db))
    child = _child(second, p.action_id)
    assert child.allowed is False
    assert "privilege attenuation violation" in child.reason
    assert _decision(second, child.action_id).data["attenuation"]["source"] == "trail"


def test_invented_parent_is_denied_by_default():
    pipe = Pipeline()
    child = _child(pipe, "00000000-0000-4000-8000-000000000000", caps=NARROWER)
    assert child.allowed is False
    assert child.decision == "deny"
    assert "privilege attenuation violation (parent_unknown)" in child.reason
    d = _decision(pipe, child.action_id).data
    assert d["violation_type"] == "privilege_attenuation"
    assert d["unavailable_input"] == "parent_grant"
    assert d["attenuation"] == {"status": "unverified", "reason": "parent_unknown"}


def test_invented_parent_allowed_when_deployer_opts_in_but_marked():
    pipe = Pipeline(unknown_parent="allow")
    child = _child(pipe, "00000000-0000-4000-8000-000000000000", caps=NARROWER)
    assert "privilege attenuation" not in child.reason
    assert _decision(pipe, child.action_id).data["attenuation"] == {
        "status": "unverified", "reason": "parent_unknown",
    }


def test_unknown_parent_flag_rejects_other_words():
    with pytest.raises(ValueError):
        Pipeline(unknown_parent="maybe")


def test_unknown_parent_allow_is_in_the_policy_state():
    pipe = Pipeline(unknown_parent="allow")
    p = pipe.intercept(agent_id="planner", tool_name="plan_task", parameters={})
    assert _decision(pipe, p.action_id).data["policy_state"]["unknown_parent"] == "allow"


def test_parent_in_another_tenant_is_unknown():
    pipe = Pipeline()
    p = pipe.intercept(agent_id="planner", tool_name="plan_task", parameters={},
                       capabilities=PARENT_GRANT, tenant_id="tenant-a")
    pipe._grants_for_action.clear()
    child = _child(pipe, p.action_id, caps=NARROWER, tenant_id="tenant-b")
    assert child.allowed is False
    assert "parent_unknown" in child.reason


def test_parent_without_grant_is_marked_unverified_not_denied():
    pipe = Pipeline()
    p = pipe.intercept(agent_id="planner", tool_name="plan_task", parameters={})
    child = _child(pipe, p.action_id)
    assert "privilege attenuation" not in child.reason
    assert _decision(pipe, child.action_id).data["attenuation"] == {
        "status": "unverified", "reason": "parent_no_grant",
    }


def test_unreadable_parent_grant_is_denied(monkeypatch):
    pipe = Pipeline()
    p = _parent(pipe)
    pipe._grants_for_action.clear()
    req = pipe.trail.get_action_trail(p.action_id)[0]
    monkeypatch.setitem(req.data, "capabilities", {"_truncated": True})
    child = _child(pipe, p.action_id, caps=NARROWER)
    assert child.allowed is False
    assert "parent_grant_unreadable" in child.reason


def test_chain_still_verifies_with_recorded_grants(tmp_path):
    db = tmp_path / "audit.db"
    pipe = Pipeline(trail=_sqlite_trail(db))
    p = _parent(pipe)
    _child(pipe, p.action_id, caps=NARROWER)
    _child(pipe, "invented")
    assert pipe.trail.verify_chain() is None
    assert _sqlite_trail(db).verify_chain() is None
