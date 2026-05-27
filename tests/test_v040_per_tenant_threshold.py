"""v0.40 per-tenant threshold dispatch at evaluate() time.

The scorer holds defaults bound from the default ("") slot via the
standard apply_policy listener path, and on every evaluate it looks up
the calling tenant's policy from the PolicyRegistry. A tenant with its
own thresholds gets those thresholds applied to THIS call; a tenant
with no policy of its own falls back to the scorer-bound defaults.
"""

from __future__ import annotations

from vaara.policy.controller import PolicyController
from vaara.policy.loader import from_dict
from vaara.policy.registry import PolicyRegistry
from vaara.scorer.adaptive import AdaptiveScorer


def _policy_dict(escalate: float, deny: float) -> dict:
    return {
        "version": "0.1",
        "thresholds": {"default": {"escalate": escalate, "deny": deny}},
    }


def _ctx(tenant_id: str = "", base_risk: float = 0.5) -> dict:
    return {
        "tool_name": "tool.test",
        "agent_id": "agent-test",
        "tenant_id": tenant_id,
        "base_risk_score": base_risk,
        "reversibility": "partially_reversible",
        "blast_radius": "local",
    }


def _registry_lookup(registry: PolicyRegistry):
    def _lookup(tid: str):
        ctrl = registry.get_exact(tid)
        return ctrl.policy if ctrl is not None else None
    return _lookup


def test_tenant_thresholds_override_default_for_that_call():
    registry = PolicyRegistry()
    registry.register("", PolicyController(from_dict(_policy_dict(0.4, 0.7))))
    registry.register(
        "tenant-a", PolicyController(from_dict(_policy_dict(0.05, 0.10)))
    )

    scorer = AdaptiveScorer()
    scorer.apply_policy(registry.get_exact("").policy)
    scorer.set_policy_lookup(_registry_lookup(registry))

    default_result = scorer.evaluate(_ctx(tenant_id="", base_risk=0.5))
    assert default_result["threshold_allow"] == 0.4
    assert default_result["threshold_deny"] == 0.7

    strict_result = scorer.evaluate(_ctx(tenant_id="tenant-a", base_risk=0.5))
    assert strict_result["threshold_allow"] == 0.05
    assert strict_result["threshold_deny"] == 0.10

    again = scorer.evaluate(_ctx(tenant_id="", base_risk=0.5))
    assert again["threshold_allow"] == 0.4
    assert again["threshold_deny"] == 0.7


def test_unknown_tenant_falls_back_to_scorer_defaults():
    registry = PolicyRegistry()
    registry.register("", PolicyController(from_dict(_policy_dict(0.4, 0.7))))

    scorer = AdaptiveScorer()
    scorer.apply_policy(registry.get_exact("").policy)
    scorer.set_policy_lookup(_registry_lookup(registry))

    result = scorer.evaluate(_ctx(tenant_id="ghost", base_risk=0.5))
    assert result["threshold_allow"] == 0.4
    assert result["threshold_deny"] == 0.7


def test_empty_tenant_id_skips_lookup_and_uses_defaults():
    calls = []

    def _lookup(tid: str):
        calls.append(tid)
        return None

    scorer = AdaptiveScorer(threshold_allow=0.4, threshold_deny=0.7)
    scorer.set_policy_lookup(_lookup)

    result = scorer.evaluate(_ctx(tenant_id="", base_risk=0.5))
    assert result["threshold_allow"] == 0.4
    assert result["threshold_deny"] == 0.7
    assert calls == []


def test_lookup_exception_falls_back_to_defaults():
    def _broken_lookup(tid: str):
        raise RuntimeError("registry unreachable")

    scorer = AdaptiveScorer(threshold_allow=0.4, threshold_deny=0.7)
    scorer.set_policy_lookup(_broken_lookup)

    result = scorer.evaluate(_ctx(tenant_id="tenant-a", base_risk=0.5))
    assert result["threshold_allow"] == 0.4
    assert result["threshold_deny"] == 0.7


def test_tenant_reload_visible_to_next_evaluate_without_listener():
    registry = PolicyRegistry()
    registry.register("", PolicyController(from_dict(_policy_dict(0.4, 0.7))))
    registry.register(
        "tenant-a", PolicyController(from_dict(_policy_dict(0.30, 0.50)))
    )

    scorer = AdaptiveScorer()
    scorer.apply_policy(registry.get_exact("").policy)
    scorer.set_policy_lookup(_registry_lookup(registry))

    first = scorer.evaluate(_ctx(tenant_id="tenant-a", base_risk=0.5))
    assert first["threshold_allow"] == 0.30

    registry.reload("tenant-a", _policy_dict(0.05, 0.10))
    after = scorer.evaluate(_ctx(tenant_id="tenant-a", base_risk=0.5))
    assert after["threshold_allow"] == 0.05
    assert after["threshold_deny"] == 0.10


def test_dry_run_evaluate_uses_per_tenant_thresholds():
    registry = PolicyRegistry()
    registry.register("", PolicyController(from_dict(_policy_dict(0.4, 0.7))))
    registry.register(
        "tenant-a", PolicyController(from_dict(_policy_dict(0.05, 0.10)))
    )

    scorer = AdaptiveScorer()
    scorer.apply_policy(registry.get_exact("").policy)
    scorer.set_policy_lookup(_registry_lookup(registry))

    default_dry = scorer.dry_run_evaluate(_ctx(tenant_id="", base_risk=0.5))
    strict_dry = scorer.dry_run_evaluate(_ctx(tenant_id="tenant-a", base_risk=0.5))

    assert default_dry["raw_result"]["threshold_allow"] == 0.4
    assert strict_dry["raw_result"]["threshold_allow"] == 0.05
    assert strict_dry["raw_result"]["threshold_deny"] == 0.10


def test_server_state_wires_lookup_from_registry():
    from vaara.server.state import ServerState

    registry = PolicyRegistry()
    registry.register("", PolicyController(from_dict(_policy_dict(0.4, 0.7))))
    registry.register(
        "tenant-a", PolicyController(from_dict(_policy_dict(0.05, 0.10)))
    )

    state = ServerState(policy_registry=registry)
    result = state.scorer.evaluate(_ctx(tenant_id="tenant-a", base_risk=0.5))
    assert result["threshold_allow"] == 0.05
    assert result["threshold_deny"] == 0.10


def test_score_response_surfaces_per_tenant_thresholds_over_http():
    """The /v1/score response's `thresholds` block reflects the
    per-call values, not the scorer's bound defaults. Regression test:
    smoke-test caught the response surfacing 0.4/0.7 for every tenant
    even though the decision itself was already dispatching correctly.
    """
    import pytest as _pytest
    try:
        from fastapi.testclient import TestClient
        from vaara.server import create_app
    except ImportError:
        _pytest.skip("server extra not installed (pip install 'vaara[server]')")
        return  # pytest.skip raises; the return keeps static analysers happy

    registry = PolicyRegistry()
    registry.register("", PolicyController(from_dict(_policy_dict(0.4, 0.7))))
    registry.register(
        "tenant-strict",
        PolicyController(from_dict(_policy_dict(0.05, 0.10))),
    )
    app = create_app(policy_registry=registry)
    client = TestClient(app)
    req = {"tool_name": "tool.read", "agent_id": "ag-1", "action_type": "data_read"}

    default_resp = client.post("/v1/score", json=req).json()
    assert default_resp["thresholds"]["allow"] == 0.4
    assert default_resp["thresholds"]["deny"] == 0.7

    strict_resp = client.post(
        "/v1/score", json=req, headers={"X-Vaara-Tenant": "tenant-strict"}
    ).json()
    assert strict_resp["thresholds"]["allow"] == 0.05
    assert strict_resp["thresholds"]["deny"] == 0.10

    unknown_resp = client.post(
        "/v1/score", json=req, headers={"X-Vaara-Tenant": "ghost"}
    ).json()
    assert unknown_resp["thresholds"]["allow"] == 0.4
    assert unknown_resp["thresholds"]["deny"] == 0.7


def test_non_policy_lookup_return_falls_back():
    def _bad_typed_lookup(tid: str):
        return {"not": "a Policy"}

    scorer = AdaptiveScorer(threshold_allow=0.4, threshold_deny=0.7)
    scorer.set_policy_lookup(_bad_typed_lookup)

    result = scorer.evaluate(_ctx(tenant_id="tenant-a", base_risk=0.5))
    assert result["threshold_allow"] == 0.4
    assert result["threshold_deny"] == 0.7
