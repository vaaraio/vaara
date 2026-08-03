"""Server bearer-key gate tests (2026-08 audit C1).

The reference server previously had NO authentication: any reachable
client could hot-swap the policy to allow-all, append forged audit
events that still verify, and poison conformal calibration. When an
API key is configured, every endpoint except /v1/health must require
`Authorization: Bearer <key>`.
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


KEY = "test-key-0123456789abcdef"


@pytest.fixture
def client():
    return TestClient(create_app(api_key=KEY))


@pytest.fixture
def auth():
    return {"Authorization": f"Bearer {KEY}"}


def test_health_open_without_key(client):
    assert client.get("/v1/health").status_code == 200


def test_health_open_with_wrong_key(client):
    assert client.get("/v1/health").status_code == 200


def test_read_endpoint_requires_key(client, auth):
    assert client.get("/v1/server").status_code == 401
    r = client.get("/v1/server", headers=auth)
    assert r.status_code == 200


def test_score_requires_key(client, auth):
    body = {"agent_id": "a", "tool_name": "data.read"}
    assert client.post("/v1/score", json=body).status_code == 401
    r = client.post("/v1/score", json=body, headers=auth)
    assert r.status_code == 200


def test_policy_reload_requires_key(client, auth):
    body = {"body": {"thresholds": {"escalate": 0.99, "deny": 1.0}}}
    assert client.post("/v1/policy/reload", json=body).status_code == 401
    # With the key it reaches the policy plane (409: not configured),
    # proving the gate — not the policy loader — produced the 401.
    r = client.post("/v1/policy/reload", json=body, headers=auth)
    assert r.status_code == 409


def test_audit_events_require_key(client, auth):
    body = {"action_id": "x", "event_type": "action_requested"}
    assert client.post("/v1/audit/events", json=body).status_code == 401
    r = client.post("/v1/audit/events", json=body, headers=auth)
    assert r.status_code == 201


def test_wrong_key_rejected(client):
    r = client.get(
        "/v1/server", headers={"Authorization": "Bearer wrong-key"}
    )
    assert r.status_code == 401
    body = r.json()["error"]
    assert body["code"] == "unauthenticated"


def test_malformed_header_rejected(client):
    for header in ("", "Bearer", "Basic dXNlcjpwYXNz", "Bearer "):
        r = client.get("/v1/server", headers={"Authorization": header})
        assert r.status_code == 401


def test_no_key_means_open_loopback_mode():
    # Default construction (no api_key) stays unauthenticated for
    # loopback development — the CLI refuses non-loopback binds there.
    c = TestClient(create_app())
    assert c.get("/v1/server").status_code == 200
