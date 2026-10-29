"""A tools/call whose ``arguments`` is not a JSON object is refused.

MCP declares ``params.arguments`` an object. The proxy used to read a list
or string ``arguments`` as ``{}`` for the deny rules, the schema check, the
classifier and the record, then forward the original request upstream: the
gates checked one thing and the server ran another. A non-object
``arguments`` (and a non-object ``params``) now gets JSON-RPC -32602, a deny
in the trail naming the type, and is never forwarded, in enforce and in
shadow mode alike, since there is nothing faithful to score or record.

The forbidden string is assembled at runtime so that writing this file does
not itself trip the hook that governs the session editing it.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

SHADOW_READ = "cat /etc/" + "shadow"


def _proxy(monkeypatch, shadow=False):
    from vaara.integrations import mcp_proxy
    from vaara.pipeline import InterceptionPipeline
    from vaara.audit.trail import AuditTrail

    trail = AuditTrail(on_record=lambda _r: None)
    pipeline = InterceptionPipeline(trail=trail, enforce=not shadow)
    monkeypatch.setattr(
        "vaara.integrations._mcp_upstream.UpstreamMCPClient.__init__",
        lambda self, command, **k: None,
    )
    p = mcp_proxy.VaaraMCPProxy(upstream_command=["echo"], pipeline=pipeline)
    upstream = MagicMock()
    upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1,
        "result": {"content": [{"type": "text", "text": "ok"}]},
    }
    p._upstream = upstream
    return p, upstream, trail


def _decisions(trail):
    out = []
    for r in trail._records:
        d = r.to_dict() if hasattr(r, "to_dict") else r
        if d.get("event_type") in ("decision_made", "action_blocked"):
            out.append(d)
    return out


@pytest.mark.parametrize("shadow", [False, True])
@pytest.mark.parametrize(
    "args, kind",
    [
        ([SHADOW_READ], "array"),
        (SHADOW_READ, "string"),
        (7, "number"),
        (True, "boolean"),
        ([], "array"),
        ("", "string"),
    ],
)
def test_non_object_arguments_refused_and_not_forwarded(monkeypatch, args, kind, shadow):
    p, upstream, trail = _proxy(monkeypatch, shadow=shadow)
    resp = p._handle_request({
        "jsonrpc": "2.0", "id": 9, "method": "tools/call",
        "params": {"name": "read_file", "arguments": args},
    })
    assert not upstream.request.called
    assert resp["id"] == 9
    assert resp["error"]["code"] == -32602
    assert kind in resp["error"]["message"]
    decisions = _decisions(trail)
    assert len(decisions) == 1
    rec = decisions[0]
    assert rec["data"]["decision"] == "deny"
    assert kind in rec["data"]["reason"]
    assert rec["data"].get("violation_type") == "invalid_params"


@pytest.mark.parametrize("params", [["read_file", SHADOW_READ], SHADOW_READ, 3])
def test_non_object_params_refused_and_not_forwarded(monkeypatch, params):
    p, upstream, trail = _proxy(monkeypatch)
    resp = p._handle_request({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": params,
    })
    assert not upstream.request.called
    assert resp["error"]["code"] == -32602
    assert len(_decisions(trail)) == 1


@pytest.mark.parametrize("params", [
    {"name": "read_file"},
    {"name": "read_file", "arguments": None},
    {"name": "read_file", "arguments": {}},
    {"name": "read_file", "arguments": {"path": "notes.txt"}},
])
def test_absent_null_or_object_arguments_still_forwarded(monkeypatch, params):
    p, upstream, _trail = _proxy(monkeypatch)
    resp = p._handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params,
    })
    assert "error" not in resp
    assert upstream.request.called


def test_object_with_shadow_read_still_denied_by_rule(monkeypatch):
    p, upstream, _trail = _proxy(monkeypatch)
    resp = p._handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "read_file", "arguments": {"path": SHADOW_READ}},
    })
    assert not upstream.request.called
    assert "vaara_blocked" in resp["result"]["content"][0]["text"]
