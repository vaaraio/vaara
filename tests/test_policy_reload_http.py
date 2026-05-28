"""HTTP /v1/policy/reload contract tests."""

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

from vaara.policy.controller import PolicyController
from vaara.policy.loader import from_dict


def _policy_dict(*, escalate, deny):
    return {
        "version": "0.1",
        "domains": ["eu_ai_act"],
        "action_classes": {
            "tx.transfer": {
                "category": "financial",
                "reversibility": "irreversible",
                "blast_radius": "local",
                "urgency": "timely",
                "regulatory": ["article_14"],
            }
        },
        "thresholds": {"default": {"escalate": escalate, "deny": deny}},
        "sequences": {
            "data_exfil": {
                "pattern": ["data.read", "data.export"],
                "risk_boost": 0.2,
                "window_seconds": 60,
                "regulatory": [],
            }
        },
        "escalation": {
            "routes": [{"operator_group": "on_call", "if": []}]
        },
    }


def _mutate(_p, *, escalate, deny):
    return _policy_dict(escalate=escalate, deny=deny)


@pytest.fixture
def base_policy():
    return from_dict(_policy_dict(escalate=0.55, deny=0.85))


@pytest.fixture
def client_with_controller(base_policy):
    controller = PolicyController(base_policy)
    app = create_app(policy_controller=controller)
    return TestClient(app), controller


@pytest.fixture
def client_without_controller():
    return TestClient(create_app())


def test_reload_with_body(client_with_controller, base_policy):
    client, _ = client_with_controller
    src = _mutate(base_policy, escalate=0.20, deny=0.95)

    r = client.post("/v1/policy/reload", json={"body": src})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == 2
    assert body["thresholds_default"]["escalate"] == 0.20
    assert body["thresholds_default"]["deny"] == 0.95
    assert body["sequence_count"] == len(base_policy.sequences)


def test_reload_rejects_both_path_and_body(client_with_controller):
    client, _ = client_with_controller
    r = client.post("/v1/policy/reload", json={"path": "x", "body": {}})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_reload_rejects_neither_path_nor_body(client_with_controller):
    client, _ = client_with_controller
    r = client.post("/v1/policy/reload", json={})
    assert r.status_code == 400


def test_reload_rejects_malformed_policy(client_with_controller, base_policy):
    client, _ = client_with_controller
    bad = _mutate(base_policy, escalate=0.9, deny=0.4)
    r = client.post("/v1/policy/reload", json={"body": bad})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "policy_invalid"


def test_reload_409_when_no_controller(client_without_controller):
    r = client_without_controller.post(
        "/v1/policy/reload", json={"body": {"version": "0.1"}}
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "policy_not_configured"


def test_reload_with_yaml_path(tmp_path, client_with_controller, base_policy):
    yaml = pytest.importorskip("yaml")

    client, _ = client_with_controller
    p = tmp_path / "policy.yaml"
    p.write_text(
        yaml.safe_dump(_mutate(base_policy, escalate=0.35, deny=0.75)),
        encoding="utf-8",
    )

    r = client.post(
        "/v1/policy/reload", json={"path": str(p), "format": "yaml"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["thresholds_default"]["escalate"] == 0.35


def test_reload_with_extra_field_rejected(client_with_controller):
    client, _ = client_with_controller
    r = client.post(
        "/v1/policy/reload",
        json={"path": "/dev/null", "stowaway_field": True},
    )
    # pydantic config extra="forbid" to 422 at schema level
    assert r.status_code == 422
