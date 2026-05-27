"""v0.40 tenant_id end-to-end propagation tests."""

from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient

    from vaara.server import create_app
except ImportError:
    pytest.skip(
        "server extra not installed (pip install 'vaara[server]')",
        allow_module_level=True,
    )

from vaara.audit.trail import AuditRecord, AuditTrail, EventType
from vaara.taxonomy.actions import (
    ActionCategory,
    ActionRequest,
    ActionType,
    BlastRadius,
    Reversibility,
)


def _action_type(name: str = "t") -> ActionType:
    return ActionType(
        name=name,
        category=ActionCategory.DATA,
        reversibility=Reversibility.FULLY,
        blast_radius=BlastRadius.LOCAL,
    )


def _minimal_policy() -> dict:
    return {
        "version": 1,
        "thresholds_default": {"escalate": 0.5, "deny": 0.9},
    }


def test_score_body_tenant_id_lands_in_action_info():
    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/v1/score",
        json={"tool_name": "search", "agent_id": "a", "tenant_id": "tenant-a"},
    )
    assert resp.status_code == 200
    info = app.state.vaara.lookup_action(resp.json()["action_id"])
    assert info is not None
    assert info.tenant_id == "tenant-a"


def test_score_header_tenant_id_used_when_body_empty():
    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/v1/score",
        json={"tool_name": "search", "agent_id": "a"},
        headers={"X-Vaara-Tenant": "tenant-b"},
    )
    assert resp.status_code == 200
    info = app.state.vaara.lookup_action(resp.json()["action_id"])
    assert info.tenant_id == "tenant-b"


def test_score_body_tenant_wins_over_header():
    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/v1/score",
        json={"tool_name": "search", "agent_id": "a", "tenant_id": "body-tenant"},
        headers={"X-Vaara-Tenant": "header-tenant"},
    )
    info = app.state.vaara.lookup_action(resp.json()["action_id"])
    assert info.tenant_id == "body-tenant"


def test_audit_event_writes_with_tenant_from_action_lookup():
    app = create_app()
    client = TestClient(app)
    score = client.post(
        "/v1/score",
        json={"tool_name": "search", "agent_id": "a", "tenant_id": "tenant-c"},
    )
    action_id = score.json()["action_id"]
    resp = client.post(
        "/v1/audit/events",
        json={
            "event_type": "action_executed",
            "action_id": action_id,
            "agent_id": "a",
            "tool_name": "search",
            "payload": {"result": "ok"},
        },
    )
    assert resp.status_code == 201
    records = [
        r for r in app.state.vaara.audit._records if r.action_id == action_id
    ]
    assert any(r.tenant_id == "tenant-c" for r in records)


def test_audit_trail_tenant_propagates_to_followup_records():
    trail = AuditTrail()
    action_type = _action_type()
    req = ActionRequest(
        agent_id="a", tool_name="t", action_type=action_type,
        parameters={}, tenant_id="tenant-d",
    )
    action_id = trail.record_action_requested(req)
    trail.record_decision(
        action_id=action_id, agent_id="a", tool_name="t",
        decision="allow", reason="ok", risk_score=0.1,
    )
    trail.record_execution(
        action_id=action_id, agent_id="a", tool_name="t", result={"ok": True},
    )
    records = trail.get_action_trail(action_id)
    assert len(records) == 3
    assert all(r.tenant_id == "tenant-d" for r in records)


def test_audit_trail_tenant_map_evicts_under_pressure():
    trail = AuditTrail()
    trail._MAX_ACTION_TENANT_MAP = 100
    action_type = _action_type()
    last_ids: list[str] = []
    for i in range(150):
        req = ActionRequest(
            agent_id="a", tool_name="t", action_type=action_type,
            tenant_id=f"t{i}",
        )
        last_ids.append(trail.record_action_requested(req))
    assert len(trail._tenant_for_action) <= 100
    assert trail._tenant_for_action.get(last_ids[-1]) == "t149"
    assert last_ids[0] not in trail._tenant_for_action


def test_audit_trail_caps_tenant_id_length():
    trail = AuditTrail()
    action_type = _action_type()
    huge = "x" * 10_000
    req = ActionRequest(
        agent_id="a", tool_name="t", action_type=action_type,
        tenant_id=huge,
    )
    action_id = trail.record_action_requested(req)
    record = trail.get_action_trail(action_id)[0]
    assert len(record.tenant_id) <= trail._MAX_TENANT_ID_LEN


def test_audit_record_hash_excludes_tenant_id():
    """tenant_id is NOT part of compute_hash so pre-v0.40 chains re-verify."""
    rec_no = AuditRecord(
        record_id="r1", action_id="a1", event_type=EventType.ACTION_REQUESTED,
        timestamp=1.0, agent_id="a", tool_name="t",
    )
    rec_with = AuditRecord(
        record_id="r1", action_id="a1", event_type=EventType.ACTION_REQUESTED,
        timestamp=1.0, agent_id="a", tool_name="t", tenant_id="tenant-x",
    )
    assert rec_no.compute_hash() == rec_with.compute_hash()
