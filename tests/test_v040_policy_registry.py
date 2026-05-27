"""v0.40 PolicyRegistry per-tenant policy plane + /v1/policy/reload routing."""

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
from vaara.policy.registry import PolicyRegistry
from vaara.policy.schema import PolicyError


def _minimal_policy() -> dict:
    return {
        "version": "0.1",
        "thresholds_default": {"escalate": 0.5, "deny": 0.9},
    }


def test_policy_registry_reload_creates_new_tenant_slot():
    registry = PolicyRegistry()
    result = registry.reload("tenant-a", _minimal_policy())
    assert result.version == 1
    assert "tenant-a" in registry


def test_policy_registry_get_falls_back_to_default_slot():
    registry = PolicyRegistry()
    registry.reload("", _minimal_policy())
    assert registry.get("nonexistent") is not None
    assert registry.get_exact("nonexistent") is None


def test_policy_registry_load_directory(tmp_path):
    (tmp_path / "default.json").write_text(
        '{"version": "0.1", "thresholds_default": {"escalate": 0.4, "deny": 0.8}}'
    )
    (tmp_path / "tenant-a.json").write_text(
        '{"version": "0.1", "thresholds_default": {"escalate": 0.3, "deny": 0.7}}'
    )
    registry = PolicyRegistry()
    tenants = registry.load_directory(tmp_path)
    assert sorted(tenants) == ["", "tenant-a"]
    assert "" in registry
    assert "tenant-a" in registry


def test_policy_registry_load_directory_rejects_empty(tmp_path):
    registry = PolicyRegistry()
    with pytest.raises(PolicyError):
        registry.load_directory(tmp_path)


def test_policy_registry_accepts_policy_instance_on_reload():
    """Bulk-load path passes Policy directly to avoid re-parsing."""
    registry = PolicyRegistry()
    registry.reload("", _minimal_policy())
    policy_obj = from_dict(_minimal_policy())
    result = registry.reload("", policy_obj)
    assert result.version == 2


def test_policy_reload_per_tenant_via_body():
    registry = PolicyRegistry()
    registry.reload("", _minimal_policy())
    app = create_app(policy_registry=registry)
    client = TestClient(app)
    resp = client.post(
        "/v1/policy/reload",
        json={
            "body": {
                "version": "0.1",
                "thresholds_default": {"escalate": 0.2, "deny": 0.6},
            },
            "tenant_id": "tenant-x",
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["tenant_id"] == "tenant-x"
    assert "tenant-x" in registry


def test_policy_reload_per_tenant_via_header():
    registry = PolicyRegistry()
    registry.reload("", _minimal_policy())
    app = create_app(policy_registry=registry)
    client = TestClient(app)
    resp = client.post(
        "/v1/policy/reload",
        json={"body": _minimal_policy()},
        headers={"X-Vaara-Tenant": "tenant-y"},
    )
    assert resp.status_code == 200
    assert resp.json()["tenant_id"] == "tenant-y"


def test_policy_reload_back_compat_single_controller():
    controller = PolicyController(from_dict(_minimal_policy()))
    app = create_app(policy_controller=controller)
    client = TestClient(app)
    resp = client.post("/v1/policy/reload", json={"body": _minimal_policy()})
    assert resp.status_code == 200
    assert resp.json()["tenant_id"] == ""


def test_policy_reload_unconfigured_returns_409():
    app = create_app()
    client = TestClient(app)
    resp = client.post("/v1/policy/reload", json={"body": _minimal_policy()})
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "policy_not_configured"
