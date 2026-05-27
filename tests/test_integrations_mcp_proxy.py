"""Tests for vaara.integrations.mcp_proxy.

Smoke tests covering the tools/call interception path with a mocked upstream
and a controlled pipeline. The subprocess machinery in ``_mcp_upstream`` has
its own integration shape and is covered separately when a real upstream
command is available locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest


@dataclass
class _StubInterceptResult:
    allowed: bool
    action_id: str = "stub-action-id"
    reason: str = ""
    decision: str = "ALLOW"


@pytest.fixture
def proxy(monkeypatch):
    """A VaaraMCPProxy with both UpstreamMCPClient and pipeline mocked.

    The proxy normally spawns a subprocess in __init__. We patch that out so
    tests can exercise the request-dispatch and interception logic in
    isolation, without any real MCP server on disk.
    """
    from vaara.integrations import mcp_proxy

    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    pipeline = MagicMock()
    p = mcp_proxy.VaaraMCPProxy(
        upstream_command=["echo"],
        pipeline=pipeline,
    )
    p._upstream = MagicMock()
    return p, pipeline


def _make_proxy(monkeypatch, **kwargs):
    from vaara.integrations import mcp_proxy

    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    pipeline = MagicMock()
    p = mcp_proxy.VaaraMCPProxy(upstream_command=["echo"], pipeline=pipeline, **kwargs)
    p._upstream = MagicMock()
    return p, pipeline


def test_blocked_tool_call_returns_mcp_tool_error(proxy):
    p, pipeline = proxy
    pipeline.intercept.return_value = _StubInterceptResult(
        allowed=False, reason="risk too high", decision="DENY", action_id="abc-123",
    )
    request = {
        "jsonrpc": "2.0", "id": 7, "method": "tools/call",
        "params": {"name": "sap.abap.write", "arguments": {"path": "/etc/something"}},
    }
    response = p._handle_tools_call(request)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 7
    assert response["result"]["isError"] is True
    text = response["result"]["content"][0]["text"]
    assert "risk too high" in text
    assert "vaara_blocked" in text
    p._upstream.request.assert_not_called()
    pipeline.report_outcome.assert_not_called()


def test_allowed_tool_call_forwards_and_reports_outcome(proxy):
    p, pipeline = proxy
    pipeline.intercept.return_value = _StubInterceptResult(allowed=True, action_id="xyz-999")
    upstream_response = {
        "jsonrpc": "2.0", "id": 9,
        "result": {"content": [{"type": "text", "text": "ok"}]},
    }
    p._upstream.request.return_value = upstream_response

    request = {
        "jsonrpc": "2.0", "id": 9, "method": "tools/call",
        "params": {"name": "sap.adt.read", "arguments": {"object": "ZCL_X"}},
    }
    response = p._handle_tools_call(request)
    assert response is upstream_response
    p._upstream.request.assert_called_once_with(request)
    pipeline.report_outcome.assert_called_once()
    kwargs = pipeline.report_outcome.call_args.kwargs
    assert kwargs["action_id"] == "xyz-999"
    assert kwargs["outcome_severity"] == 0.0


def test_upstream_error_response_yields_high_severity(proxy):
    p, pipeline = proxy
    pipeline.intercept.return_value = _StubInterceptResult(allowed=True, action_id="err-1")
    p._upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "upstream tool blew up"},
    }
    request = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "sap.adt.write", "arguments": {}},
    }
    p._handle_tools_call(request)
    assert pipeline.report_outcome.call_args.kwargs["outcome_severity"] == 1.0


def test_vaara_agent_id_override_is_stripped_before_forward(proxy):
    p, pipeline = proxy
    pipeline.intercept.return_value = _StubInterceptResult(allowed=True, action_id="strip-1")
    p._upstream.request.return_value = {"jsonrpc": "2.0", "id": 2, "result": {}}

    request = {
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {
            "name": "sap.adt.read",
            "arguments": {"_vaara_agent_id": "agent-007", "object": "ZCL_X"},
        },
    }
    p._handle_tools_call(request)
    intercept_kwargs = pipeline.intercept.call_args.kwargs
    assert intercept_kwargs["agent_id"] == "agent-007"
    # Forwarded arguments must not leak the Vaara-internal key.
    assert "_vaara_agent_id" not in intercept_kwargs["parameters"]
    forwarded = p._upstream.request.call_args.args[0]
    assert "_vaara_agent_id" not in forwarded["params"]["arguments"]


def test_non_intercepted_method_forwards_verbatim(proxy):
    p, pipeline = proxy
    upstream_response = {
        "jsonrpc": "2.0", "id": 3, "result": {"resources": []},
    }
    p._upstream.request.return_value = upstream_response
    request = {"jsonrpc": "2.0", "id": 3, "method": "resources/list"}
    response = p._handle_request(request)
    p._upstream.request.assert_called_once_with(request)
    pipeline.intercept.assert_not_called()
    assert response is upstream_response


def test_invalid_request_returns_minus_32600(proxy):
    p, _ = proxy
    response = p._handle_request("not a dict")
    assert response["error"]["code"] == -32600


def test_tools_list_denylist_drops_named_tools(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch, denylist={"delete_repository", "create_branch"})
    p._upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 11,
        "result": {"tools": [
            {"name": "search_repositories"},
            {"name": "create_branch"},
            {"name": "get_pull_request"},
            {"name": "delete_repository"},
        ]},
    }
    response = p._handle_request({"jsonrpc": "2.0", "id": 11, "method": "tools/list"})
    names = [t["name"] for t in response["result"]["tools"]]
    assert names == ["search_repositories", "get_pull_request"]
    pipeline.intercept.assert_not_called()


def test_tools_list_allowlist_restricts_to_listed_tools(monkeypatch):
    p, _ = _make_proxy(monkeypatch, allowlist={"search_repositories", "get_pull_request"})
    p._upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 12,
        "result": {"tools": [
            {"name": "search_repositories"},
            {"name": "create_branch"},
            {"name": "get_pull_request"},
        ]},
    }
    response = p._handle_request({"jsonrpc": "2.0", "id": 12, "method": "tools/list"})
    names = sorted(t["name"] for t in response["result"]["tools"])
    assert names == ["get_pull_request", "search_repositories"]


def test_tools_list_denylist_wins_when_overlapping_with_allowlist(monkeypatch):
    p, _ = _make_proxy(
        monkeypatch,
        allowlist={"search_repositories", "create_branch"},
        denylist={"create_branch"},
    )
    p._upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 13,
        "result": {"tools": [
            {"name": "search_repositories"},
            {"name": "create_branch"},
        ]},
    }
    response = p._handle_request({"jsonrpc": "2.0", "id": 13, "method": "tools/list"})
    names = [t["name"] for t in response["result"]["tools"]]
    assert names == ["search_repositories"]


def test_tools_list_no_policy_returns_upstream_response_unchanged(monkeypatch):
    p, _ = _make_proxy(monkeypatch)
    upstream_response = {
        "jsonrpc": "2.0", "id": 14,
        "result": {"tools": [{"name": "a"}, {"name": "b"}]},
    }
    p._upstream.request.return_value = upstream_response
    response = p._handle_request({"jsonrpc": "2.0", "id": 14, "method": "tools/list"})
    assert response is upstream_response


def test_tools_call_on_denylisted_tool_returns_filter_block(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch, denylist={"delete_repository"})
    request = {
        "jsonrpc": "2.0", "id": 15, "method": "tools/call",
        "params": {"name": "delete_repository", "arguments": {"owner": "vaaraio", "repo": "vaara"}},
    }
    response = p._handle_tools_call(request)
    assert response["result"]["isError"] is True
    text = response["result"]["content"][0]["text"]
    assert "Tool filtered by operator policy" in text
    assert "FILTERED" in text
    p._upstream.request.assert_not_called()
    pipeline.intercept.assert_not_called()
    pipeline.report_outcome.assert_not_called()


def test_tools_call_outside_allowlist_returns_filter_block(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch, allowlist={"search_repositories"})
    request = {
        "jsonrpc": "2.0", "id": 16, "method": "tools/call",
        "params": {"name": "create_branch", "arguments": {}},
    }
    response = p._handle_tools_call(request)
    assert response["result"]["isError"] is True
    assert "Tool filtered by operator policy" in response["result"]["content"][0]["text"]
    pipeline.intercept.assert_not_called()


def test_tools_call_inside_allowlist_still_runs_pipeline(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch, allowlist={"search_repositories"})
    pipeline.intercept.return_value = _StubInterceptResult(allowed=True, action_id="allow-1")
    p._upstream.request.return_value = {"jsonrpc": "2.0", "id": 17, "result": {}}
    request = {
        "jsonrpc": "2.0", "id": 17, "method": "tools/call",
        "params": {"name": "search_repositories", "arguments": {"q": "vaara"}},
    }
    p._handle_tools_call(request)
    pipeline.intercept.assert_called_once()
    p._upstream.request.assert_called_once_with(request)


def test_resources_list_denylist_drops_named_uris(monkeypatch):
    p, _ = _make_proxy(
        monkeypatch,
        resource_denylist={"file:///etc/secret", "file:///etc/shadow"},
    )
    p._upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 20,
        "result": {"resources": [
            {"uri": "file:///etc/secret", "name": "secret"},
            {"uri": "file:///etc/hosts", "name": "hosts"},
            {"uri": "file:///etc/shadow", "name": "shadow"},
            {"uri": "file:///var/log/app.log", "name": "applog"},
        ]},
    }
    response = p._handle_request({"jsonrpc": "2.0", "id": 20, "method": "resources/list"})
    uris = [r["uri"] for r in response["result"]["resources"]]
    assert uris == ["file:///etc/hosts", "file:///var/log/app.log"]


def test_resources_list_allowlist_restricts_to_listed_uris(monkeypatch):
    p, _ = _make_proxy(
        monkeypatch,
        resource_allowlist={"file:///etc/hosts"},
    )
    p._upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 21,
        "result": {"resources": [
            {"uri": "file:///etc/secret"},
            {"uri": "file:///etc/hosts"},
        ]},
    }
    response = p._handle_request({"jsonrpc": "2.0", "id": 21, "method": "resources/list"})
    uris = [r["uri"] for r in response["result"]["resources"]]
    assert uris == ["file:///etc/hosts"]


def test_resources_list_no_policy_returns_upstream_response_unchanged(monkeypatch):
    p, _ = _make_proxy(monkeypatch)
    upstream_response = {
        "jsonrpc": "2.0", "id": 22,
        "result": {"resources": [{"uri": "file:///a"}, {"uri": "file:///b"}]},
    }
    p._upstream.request.return_value = upstream_response
    response = p._handle_request({"jsonrpc": "2.0", "id": 22, "method": "resources/list"})
    assert response is upstream_response


def test_resources_read_on_denylisted_uri_returns_jsonrpc_error(monkeypatch):
    p, _ = _make_proxy(monkeypatch, resource_denylist={"file:///etc/secret"})
    request = {
        "jsonrpc": "2.0", "id": 23, "method": "resources/read",
        "params": {"uri": "file:///etc/secret"},
    }
    response = p._handle_request(request)
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 23
    assert response["error"]["code"] == -32000
    data = response["error"]["data"]
    assert data["vaara_blocked"] is True
    assert data["decision"] == "FILTERED"
    assert data["uri"] == "file:///etc/secret"
    p._upstream.request.assert_not_called()


def test_resources_read_outside_allowlist_returns_jsonrpc_error(monkeypatch):
    p, _ = _make_proxy(monkeypatch, resource_allowlist={"file:///etc/hosts"})
    request = {
        "jsonrpc": "2.0", "id": 24, "method": "resources/read",
        "params": {"uri": "file:///etc/secret"},
    }
    response = p._handle_request(request)
    assert response["error"]["code"] == -32000
    assert response["error"]["data"]["decision"] == "FILTERED"
    p._upstream.request.assert_not_called()


def test_resources_read_within_perimeter_audits_and_forwards(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch, resource_allowlist={"file:///etc/hosts"})
    upstream_response = {
        "jsonrpc": "2.0", "id": 25,
        "result": {"contents": [{"uri": "file:///etc/hosts", "text": "127.0.0.1 localhost"}]},
    }
    p._upstream.request.return_value = upstream_response
    request = {
        "jsonrpc": "2.0", "id": 25, "method": "resources/read",
        "params": {"uri": "file:///etc/hosts"},
    }
    response = p._handle_request(request)
    assert response is upstream_response
    # Risk scorer must NOT be invoked for resource reads (read-oriented surface).
    pipeline.intercept.assert_not_called()
    pipeline.report_outcome.assert_not_called()
    # Audit pair must be written via the trail directly.
    pipeline.trail.record_action_requested.assert_called_once()
    pipeline.trail.record_decision.assert_called_once()
    decision_kwargs = pipeline.trail.record_decision.call_args.kwargs
    assert decision_kwargs["decision"] == "allow"
    assert decision_kwargs["tool_name"] == "mcp.resource.read"


def test_prompts_list_denylist_drops_named_prompts(monkeypatch):
    p, _ = _make_proxy(monkeypatch, prompt_denylist={"jailbreak_template"})
    p._upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 26,
        "result": {"prompts": [
            {"name": "summarize"},
            {"name": "jailbreak_template"},
            {"name": "translate"},
        ]},
    }
    response = p._handle_request({"jsonrpc": "2.0", "id": 26, "method": "prompts/list"})
    names = [pr["name"] for pr in response["result"]["prompts"]]
    assert names == ["summarize", "translate"]


def test_prompts_list_allowlist_restricts_to_listed_prompts(monkeypatch):
    p, _ = _make_proxy(monkeypatch, prompt_allowlist={"summarize"})
    p._upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 27,
        "result": {"prompts": [{"name": "summarize"}, {"name": "translate"}]},
    }
    response = p._handle_request({"jsonrpc": "2.0", "id": 27, "method": "prompts/list"})
    names = [pr["name"] for pr in response["result"]["prompts"]]
    assert names == ["summarize"]


def test_prompts_get_on_denylisted_name_returns_jsonrpc_error(monkeypatch):
    p, _ = _make_proxy(monkeypatch, prompt_denylist={"jailbreak_template"})
    request = {
        "jsonrpc": "2.0", "id": 28, "method": "prompts/get",
        "params": {"name": "jailbreak_template", "arguments": {}},
    }
    response = p._handle_request(request)
    assert response["error"]["code"] == -32000
    data = response["error"]["data"]
    assert data["decision"] == "FILTERED"
    assert data["prompt"] == "jailbreak_template"
    p._upstream.request.assert_not_called()


def test_prompts_get_outside_allowlist_returns_jsonrpc_error(monkeypatch):
    p, _ = _make_proxy(monkeypatch, prompt_allowlist={"summarize"})
    request = {
        "jsonrpc": "2.0", "id": 29, "method": "prompts/get",
        "params": {"name": "translate", "arguments": {"text": "hi"}},
    }
    response = p._handle_request(request)
    assert response["error"]["code"] == -32000
    p._upstream.request.assert_not_called()


def test_prompts_get_within_perimeter_audits_and_forwards(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch, prompt_allowlist={"summarize"})
    upstream_response = {
        "jsonrpc": "2.0", "id": 30,
        "result": {"messages": [{"role": "user", "content": {"type": "text", "text": "..."}}]},
    }
    p._upstream.request.return_value = upstream_response
    request = {
        "jsonrpc": "2.0", "id": 30, "method": "prompts/get",
        "params": {"name": "summarize", "arguments": {"input": "hello world"}},
    }
    response = p._handle_request(request)
    assert response is upstream_response
    pipeline.intercept.assert_not_called()
    pipeline.trail.record_action_requested.assert_called_once()
    pipeline.trail.record_decision.assert_called_once()
    decision_kwargs = pipeline.trail.record_decision.call_args.kwargs
    assert decision_kwargs["decision"] == "allow"
    assert decision_kwargs["tool_name"] == "mcp.prompt.get"


def test_is_filtered_treats_non_string_names_as_filtered():
    """Defensive contract: if upstream returns a non-string in the name field
    (a misbehaving server returning a list, dict, None, or int), the perimeter
    must deny rather than crash on hash. Governance default for ambiguous
    input is deny. Flagged by CodeRabbit on PR #114.
    """
    from vaara.integrations.mcp_proxy import VaaraMCPProxy
    assert VaaraMCPProxy._is_filtered(None, None, set()) is True
    assert VaaraMCPProxy._is_filtered(["tool"], None, set()) is True
    assert VaaraMCPProxy._is_filtered({"name": "tool"}, None, set()) is True
    assert VaaraMCPProxy._is_filtered(42, None, set()) is True
    assert VaaraMCPProxy._is_filtered("tool", None, set()) is False


def test_uninterpreted_method_still_forwards_verbatim(monkeypatch):
    """Non-governed MCP methods (e.g. completion/complete, ping) pass through unchanged."""
    p, pipeline = _make_proxy(monkeypatch)
    upstream_response = {"jsonrpc": "2.0", "id": 31, "result": {}}
    p._upstream.request.return_value = upstream_response
    request = {"jsonrpc": "2.0", "id": 31, "method": "completion/complete"}
    response = p._handle_request(request)
    assert response is upstream_response
    pipeline.intercept.assert_not_called()


# --- streaming notification audit (v0.25.0) ---------------------------------

def test_progress_notification_records_perimeter_audit(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch)
    p._on_upstream_notification({
        "jsonrpc": "2.0", "method": "notifications/progress",
        "params": {"progressToken": "tok-1", "progress": 25, "total": 100},
    })
    classify = pipeline.registry.classify
    classify.assert_called_with("mcp.notification.progress", {
        "progressToken": "tok-1", "parent_action_id": "", "parent_tool": "",
    })
    pipeline.trail.record_action_requested.assert_called_once()
    pipeline.trail.record_decision.assert_called_once()


def test_progress_notification_correlates_to_inflight_tools_call(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch)
    pipeline.intercept.return_value = _StubInterceptResult(allowed=True, action_id="parent-act-9")
    progress_seen: list[dict] = []

    def upstream_request(req):
        if req["method"] == "tools/call":
            # Simulate a progress notification arriving while the upstream
            # works on the long-running call. The reader thread would invoke
            # this callback from a separate thread in production; calling it
            # synchronously here verifies the correlation logic.
            p._on_upstream_notification({
                "jsonrpc": "2.0", "method": "notifications/progress",
                "params": {"progressToken": "tok-stream", "progress": 50, "total": 100},
            })
            progress_seen.append(
                dict(p._inflight_progress),
            )
        return {"jsonrpc": "2.0", "id": req["id"], "result": {}}

    p._upstream.request.side_effect = upstream_request
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 42, "method": "tools/call",
        "params": {
            "name": "long_tool", "arguments": {},
            "_meta": {"progressToken": "tok-stream"},
        },
    })
    assert progress_seen == [{"tok-stream": ("parent-act-9", "mcp-proxy-client", "long_tool", "")}]
    # Map is cleaned up after the call returns.
    assert p._inflight_progress == {}
    # Audit was called for the progress event with the parent correlation.
    classify_calls = [c for c in pipeline.registry.classify.call_args_list
                      if c.args[0] == "mcp.notification.progress"]
    assert len(classify_calls) == 1
    params = classify_calls[0].args[1]
    assert params["parent_action_id"] == "parent-act-9"
    assert params["parent_tool"] == "long_tool"


def test_progress_notification_forwards_to_client(monkeypatch):
    p, _ = _make_proxy(monkeypatch)
    forwarded: list[dict] = []
    monkeypatch.setattr(p, "_write_to_client", forwarded.append)
    msg = {
        "jsonrpc": "2.0", "method": "notifications/progress",
        "params": {"progressToken": "tok-2", "progress": 1, "total": 1},
    }
    p._on_upstream_notification(msg)
    assert forwarded == [msg]


def test_message_notification_records_perimeter_audit(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch)
    p._on_upstream_notification({
        "jsonrpc": "2.0", "method": "notifications/message",
        "params": {"level": "info", "logger": "upstream", "data": "working"},
    })
    classify = pipeline.registry.classify
    classify.assert_called_with("mcp.notification.message", {
        "level": "info", "logger": "upstream",
    })


def test_non_governed_notification_still_forwards_without_audit(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch)
    forwarded: list[dict] = []
    monkeypatch.setattr(p, "_write_to_client", forwarded.append)
    msg = {
        "jsonrpc": "2.0", "method": "notifications/resources/list_changed",
        "params": {},
    }
    p._on_upstream_notification(msg)
    assert forwarded == [msg]
    pipeline.registry.classify.assert_not_called()


def test_audit_failure_in_notification_does_not_break_forwarding(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch)
    pipeline.registry.classify.side_effect = RuntimeError("audit blew up")
    forwarded: list[dict] = []
    monkeypatch.setattr(p, "_write_to_client", forwarded.append)
    msg = {
        "jsonrpc": "2.0", "method": "notifications/message",
        "params": {"level": "error"},
    }
    p._on_upstream_notification(msg)
    # Forward still happens — audit failure must not block streaming.
    assert forwarded == [msg]


def test_inflight_map_cleaned_when_tools_call_raises(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch)
    pipeline.intercept.return_value = _StubInterceptResult(allowed=True, action_id="x")
    from vaara.integrations._mcp_upstream import ProxyError
    p._upstream.request.side_effect = ProxyError("boom")
    with pytest.raises(ProxyError):
        p._handle_tools_call({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "t", "arguments": {},
                "_meta": {"progressToken": "tok-err"},
            },
        })
    assert p._inflight_progress == {}


def test_upstream_request_raises_proxy_error_when_reader_exits_without_response(monkeypatch):
    """Regression: reader-thread exit during request must raise ProxyError, not AssertionError.

    The reader sets pending.event on exit (so waiters do not hang) but does
    not populate pending.response. Without an explicit ProxyError the
    request() path's assertion either raises AssertionError (escapes the
    caller's ProxyError handler) or is optimized out under python -O
    (returns None silently). Flagged by CodeRabbit on PR #100.
    """
    from vaara.integrations import _mcp_upstream as up

    fake_proc = MagicMock()
    fake_proc.stdin = MagicMock()
    fake_proc.stdout = None
    monkeypatch.setattr(up.subprocess, "Popen", lambda *a, **k: fake_proc)

    client = up.UpstreamMCPClient(command=["dummy"])

    real_request_cls = up._UpstreamRequest

    def pre_signaled(*args, **kwargs):
        r = real_request_cls(*args, **kwargs)
        r.event.set()  # waiter unblocks immediately with response still None
        return r

    monkeypatch.setattr(up, "_UpstreamRequest", pre_signaled)

    with pytest.raises(up.ProxyError, match="closed before responding"):
        client.request({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    client.close()
