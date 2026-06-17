"""HTTP API reference-server tests.

Exercises the v1 contract: score, outcome, audit append, chain read, verify,
health, server identity.
"""

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


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_health(client):
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_server_info(client):
    r = client.get("/v1/server")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "vaara-reference-server"
    assert body["version"] == "1.0.1"
    assert body["capabilities"] == {
        "score": True, "audit": True, "outcome_feedback": True,
    }
    assert body["scorer"]["type"] == "AdaptiveScorer"
    assert 0 < body["scorer"]["alpha"] < 1
    assert body["scorer"]["threshold_allow"] < body["scorer"]["threshold_deny"]


def test_score_returns_assessment(client):
    r = client.post("/v1/score", json={
        "tool_name": "tx.transfer",
        "agent_id": "agent-007",
        "base_risk_score": 0.5,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] in ("allow", "escalate", "deny")
    assert 0 <= body["risk"]["point"] <= 1
    assert 0 <= body["risk"]["lower"] <= body["risk"]["upper"] <= 1
    assert body["thresholds"]["allow"] < body["thresholds"]["deny"]
    assert body["action_id"]
    assert isinstance(body["signals"], dict)
    assert isinstance(body["mwu_weights"], dict)


def test_score_validates_input(client):
    # Missing required fields.
    r = client.post("/v1/score", json={"tool_name": "x"})
    assert r.status_code == 422

    # Out-of-range confidence.
    r = client.post("/v1/score", json={
        "tool_name": "x", "agent_id": "a", "agent_confidence": 1.5,
    })
    assert r.status_code == 422


def test_score_outcome_roundtrip(client):
    r1 = client.post("/v1/score", json={
        "tool_name": "data.read", "agent_id": "a-1",
    })
    assert r1.status_code == 200
    action_id = r1.json()["action_id"]

    r2 = client.post("/v1/score/outcome", json={
        "action_id": action_id, "outcome_severity": 0.0,
    })
    assert r2.status_code == 204


def test_outcome_unknown_action_404(client):
    r = client.post("/v1/score/outcome", json={
        "action_id": "does-not-exist", "outcome_severity": 0.0,
    })
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "unknown_action"


def test_audit_append_and_read_chain(client):
    r1 = client.post("/v1/audit/events", json={
        "event_type": "action_requested",
        "action_id": "act-1",
        "agent_id": "a-1",
        "tool_name": "data.read",
        "payload": {"foo": "bar"},
    })
    assert r1.status_code == 201, r1.text
    e1 = r1.json()
    assert e1["chain_position"] == 0
    assert e1["previous_hash"] == ""
    assert len(e1["event_hash"]) == 64

    r2 = client.post("/v1/audit/events", json={
        "event_type": "decision_made",
        "action_id": "act-1",
        "payload": {"decision": "allow"},
    })
    e2 = r2.json()
    assert e2["chain_position"] == 1
    assert e2["previous_hash"] == e1["event_hash"]

    rc = client.get("/v1/audit/actions/act-1/chain")
    assert rc.status_code == 200
    chain = rc.json()
    assert chain["action_id"] == "act-1"
    assert len(chain["events"]) == 2
    assert chain["events"][0]["event_type"] == "action_requested"
    assert chain["events"][1]["event_type"] == "decision_made"


def test_audit_chain_unknown_action_404(client):
    r = client.get("/v1/audit/actions/no-such-action/chain")
    assert r.status_code == 404


def test_audit_chain_read_is_tenant_scoped(client):
    # Tenant A writes a chain for act-iso.
    r = client.post(
        "/v1/audit/events",
        json={
            "event_type": "action_requested",
            "action_id": "act-iso",
            "agent_id": "a-acme",
            "tool_name": "data.read",
            "payload": {"secret": "acme-only"},
        },
        headers={"X-Vaara-Tenant": "acme"},
    )
    assert r.status_code == 201, r.text

    # Tenant A can read its own chain.
    own = client.get(
        "/v1/audit/actions/act-iso/chain",
        headers={"X-Vaara-Tenant": "acme"},
    )
    assert own.status_code == 200
    assert len(own.json()["events"]) == 1
    assert own.json()["events"][0]["payload"]["secret"] == "acme-only"

    # Tenant B knows the action_id but must not read it — and gets 404, not
    # 403, so the response cannot confirm act-iso exists for another tenant.
    other = client.get(
        "/v1/audit/actions/act-iso/chain",
        headers={"X-Vaara-Tenant": "globex"},
    )
    assert other.status_code == 404
    assert "acme-only" not in other.text

    # A caller with no tenant header (single-tenant scope "") is likewise
    # walled off from a tenant-owned action.
    anon = client.get("/v1/audit/actions/act-iso/chain")
    assert anon.status_code == 404
    assert "acme-only" not in anon.text


def test_audit_event_bad_type_400(client):
    r = client.post("/v1/audit/events", json={
        "event_type": "not_a_real_event",
        "action_id": "x",
    })
    # Pydantic enum validation may return 422; underlying dispatcher would
    # return 400. Either is acceptable as a "bad input" signal.
    assert r.status_code in (400, 422)


def test_audit_verify_empty_chain(client):
    r = client.post("/v1/audit/verify", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["events_checked"] == 0


def test_audit_verify_after_events(client):
    client.post("/v1/audit/events", json={
        "event_type": "action_requested",
        "action_id": "act-v",
        "agent_id": "a",
        "tool_name": "t",
    })
    client.post("/v1/audit/events", json={
        "event_type": "decision_made",
        "action_id": "act-v",
    })
    r = client.post("/v1/audit/verify", json={})
    body = r.json()
    assert body["valid"] is True
    assert body["events_checked"] == 2
