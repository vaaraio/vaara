"""The record says which gates ran, in what order, and which never ran.

The ordering in ``_handle_tools_call`` is real: the operator perimeter filter
runs first, ``InterceptionPipeline.intercept`` takes the policy decision
second, and ``CredentialGateway.authorize`` checks the runtime-argument
binding third. Until now that sequence was a property of the source and
nothing else. A reader holding only the record could not tell a gate that ran
and allowed from a gate that never ran at all, which is the same defect as a
third state that never reaches the caller.

Two things are pinned here. The order, because the order is the claim. And
the three-way distinction on the credential gateway, because "no gateway in
this deployment" and "gateway present but this tool declares no constraints"
and "gateway ran and passed" are three different facts about enforcement and
collapsing them would be a false record of the strongest one.
"""

import json
from unittest.mock import MagicMock

import pytest


def _proxy(monkeypatch):
    from vaara.integrations import mcp_proxy
    from vaara.pipeline import InterceptionPipeline
    from vaara.audit.trail import AuditTrail

    trail = AuditTrail(on_record=lambda _r: None)
    pipeline = InterceptionPipeline(trail=trail)
    monkeypatch.setattr(
        "vaara.integrations._mcp_upstream.UpstreamMCPClient.__init__",
        lambda self, command, **kw: None,
    )
    p = mcp_proxy.VaaraMCPProxy(upstream_command=["echo"], pipeline=pipeline)

    upstream = MagicMock()
    upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1,
        "result": {"content": [{"type": "text", "text": "ok"}]},
    }
    p._upstream = upstream
    return p, pipeline, upstream


def _capture_gates(monkeypatch, p):
    """Record the ``gates`` list handed to every OVERT emission."""
    seen: list[list[str]] = []
    real = p._overt_emit

    def spy(**kw):
        extra = kw.get("extra") or {}
        if "gates" in extra:
            seen.append(list(extra["gates"]))
        return real(**kw)

    monkeypatch.setattr(p, "_overt_emit", spy)
    return seen


def _call(p, tool="read_file", args=None):
    return p._handle_request({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": args or {"path": "/tmp/x"}},
    })


def _payload(resp):
    return json.loads(resp["result"]["content"][0]["text"])


def _with_gateway(p, *, ok: bool, constrained: bool = True):
    verdict = MagicMock()
    verdict.ok = ok
    verdict.reason = "ok" if ok else "args digest mismatch"

    attest = MagicMock()
    attest.gateway = MagicMock()
    attest.gateway.authorize.return_value = verdict
    attest.is_constrained.return_value = constrained
    attest.emit_attestation.return_value = (MagicMock(), 1)
    attest.emit_grant.return_value = None
    p._attest = attest
    p._mint_credentials = True
    p._emit_authorization_receipts = False
    return p


def test_perimeter_filter_records_only_itself(monkeypatch):
    p, _pipeline, upstream = _proxy(monkeypatch)
    monkeypatch.setattr(p, "_tool_filtered", lambda name: True)
    gates = _capture_gates(monkeypatch, p)

    resp = _call(p)

    assert gates == [["operator_filter:filtered"]]
    assert _payload(resp)["gates"] == ["operator_filter:filtered"]
    upstream.request.assert_not_called()


def test_policy_denial_records_the_filter_it_passed_first(monkeypatch):
    p, pipeline, upstream = _proxy(monkeypatch)

    denied = MagicMock()
    denied.allowed = False
    denied.decision = "DENY"
    denied.reason = "blocked"
    denied.action_id = "a1"
    monkeypatch.setattr(pipeline, "intercept", lambda **kw: denied)
    gates = _capture_gates(monkeypatch, p)

    resp = _call(p)

    assert gates == [["operator_filter:pass", "policy:deny"]]
    assert _payload(resp)["gates"] == ["operator_filter:pass", "policy:deny"]
    upstream.request.assert_not_called()


def test_gateway_refusal_shows_it_ran_after_the_policy_allow(monkeypatch):
    """The ordering VATE case 3 pins, now visible in the record itself."""
    p, _pipeline, upstream = _proxy(monkeypatch)
    _with_gateway(p, ok=False)
    gates = _capture_gates(monkeypatch, p)

    resp = _call(p)

    assert gates, "no gate trail was emitted"
    trail = gates[-1]
    assert trail == [
        "operator_filter:pass",
        "policy:allow",
        "credential_gateway:refuse",
    ]
    # The claim is the order, so assert it as an order and not a set.
    assert trail.index("policy:allow") < trail.index("credential_gateway:refuse")
    assert "error" in resp
    upstream.request.assert_not_called()


def test_gateway_pass_is_recorded_and_reaches_upstream(monkeypatch):
    p, _pipeline, upstream = _proxy(monkeypatch)
    _with_gateway(p, ok=True)
    gates = _capture_gates(monkeypatch, p)

    _call(p)

    assert gates[-1] == [
        "operator_filter:pass",
        "policy:allow",
        "credential_gateway:pass",
    ]
    upstream.request.assert_called_once()


def test_unconstrained_tool_is_not_reported_as_having_passed(monkeypatch):
    """A gateway that was never consulted must not read as a gateway that allowed."""
    p, _pipeline, upstream = _proxy(monkeypatch)
    _with_gateway(p, ok=True, constrained=False)
    gates = _capture_gates(monkeypatch, p)

    _call(p)

    trail = gates[-1]
    assert trail == [
        "operator_filter:pass",
        "policy:allow",
        "credential_gateway:not_constrained",
    ]
    assert "credential_gateway:pass" not in trail
    p._attest.gateway.authorize.assert_not_called()


def test_deployment_without_a_gateway_says_so(monkeypatch):
    """Empty tool_constraints means no gateway and no enforcement anywhere.

    That is the shipped default, and a record that stayed silent about it
    would let a reader assume a binding check happened.
    """
    p, _pipeline, upstream = _proxy(monkeypatch)
    p._attest = None
    p._mint_credentials = False
    gates = _capture_gates(monkeypatch, p)

    _call(p)

    trail = gates[-1]
    assert trail == [
        "operator_filter:pass",
        "policy:allow",
        "credential_gateway:not_configured",
    ]
    upstream.request.assert_called_once()


@pytest.mark.parametrize("ok,expected", [
    (True, "credential_gateway:pass"),
    (False, "credential_gateway:refuse"),
])
def test_gateway_verdict_is_never_the_absent_marker(monkeypatch, ok, expected):
    """Whatever the gateway decides, it is distinct from not having run."""
    p, _pipeline, _upstream = _proxy(monkeypatch)
    _with_gateway(p, ok=ok)
    gates = _capture_gates(monkeypatch, p)

    _call(p)

    trail = gates[-1]
    assert expected in trail
    assert "credential_gateway:not_configured" not in trail
    assert "credential_gateway:not_constrained" not in trail
