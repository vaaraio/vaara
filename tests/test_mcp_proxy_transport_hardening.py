# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Transport-level contract tests for the MCP proxy.

Every test here was written against a defect that a passing suite already
covered, because the existing tests asserted what the proxy assumed rather
than what the transport and the JSON-RPC 2.0 / MCP specs require:

* the Streamable HTTP transport never validated the ``Origin`` header, so a
  web page could drive a loopback-bound proxy (MCP requires the check);
* a non-string ``method`` raised out of the dispatcher, taking the stdio
  proxy process down with it;
* JSON-RPC batches were rejected although the transport advertises
  MCP 2025-03-26, where batching is part of the contract;
* ``--upstream 'name=cmd with args'`` produced a one-element argv that
  ``subprocess.Popen`` can never execute, though the docs show that form.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - server extra not installed
    pytest.skip(
        "server extra not installed (pip install 'vaara[server]')",
        allow_module_level=True,
    )

from vaara.integrations import mcp_proxy
from vaara.integrations.mcp_proxy import VaaraMCPProxy, _parse_upstream_specs

MCP_ACCEPT = "application/json, text/event-stream"


@pytest.fixture
def proxy(monkeypatch):
    """Single-upstream proxy whose upstream echoes a benign result."""
    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    pipeline = MagicMock()
    pipeline.intercept.return_value = MagicMock(
        allowed=True, action_id="act-1", decision="allow", reason="ok",
    )
    p = VaaraMCPProxy(upstream_command=["echo"], pipeline=pipeline)
    upstream = MagicMock()
    upstream.request.side_effect = lambda payload, *a, **kw: {
        "jsonrpc": "2.0", "id": payload.get("id"), "result": {"ok": True},
    }
    p._upstream = upstream
    return p


def _client(proxy) -> TestClient:
    return TestClient(proxy._build_http_app(), raise_server_exceptions=False)


def _post(client, body, **headers):
    merged = {"Accept": MCP_ACCEPT, **headers}
    return client.post("/mcp", content=json.dumps(body), headers=merged)


# ── Origin validation (DNS-rebinding / cross-site drive-by) ────────────────

def test_cross_origin_post_is_refused(proxy):
    """A browser page must not be able to drive a loopback-bound proxy.

    The MCP Streamable HTTP transport requires servers to validate Origin.
    Without it, a page on any site can issue a CORS-"simple" POST (no
    preflight) and the tool call executes; the attacker cannot read the
    reply, but the side effect already happened.
    """
    client = _client(proxy)
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "get_weather", "arguments": {}}}
    response = _post(client, body, Origin="https://evil.example")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "origin_not_allowed"
    assert not proxy._upstream.request.called


def test_cross_origin_simple_content_type_is_refused(proxy):
    """text/plain is the no-preflight bypass; the Origin gate still catches it."""
    client = _client(proxy)
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    response = _post(
        client, body,
        Origin="https://evil.example",
        **{"Content-Type": "text/plain;charset=UTF-8"},
    )
    assert response.status_code == 403
    assert not proxy._upstream.request.called


def test_cross_origin_sse_stream_is_refused(proxy):
    """GET /mcp is the other half of the transport and needs the same gate."""
    client = _client(proxy)
    response = client.get(
        "/mcp",
        headers={"Mcp-Session-Id": "abc123", "Origin": "https://evil.example"},
    )
    assert response.status_code == 403


def test_request_without_origin_is_allowed(proxy):
    """Native MCP clients (Claude Code, Cursor) send no Origin at all."""
    client = _client(proxy)
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    assert _post(client, body).status_code == 200


def test_explicitly_allowed_origin_passes(monkeypatch):
    """An operator who runs a browser client opts that origin in."""
    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    p = VaaraMCPProxy(
        upstream_command=["echo"],
        pipeline=MagicMock(),
        allowed_origins={"https://console.example"},
    )
    p._upstream = MagicMock()
    p._upstream.request.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}
    client = _client(p)
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    response = _post(client, body, Origin="https://console.example")
    assert response.status_code == 200


def test_allowed_origin_comparison_is_exact(monkeypatch):
    """A prefix or suffix match would let evil-console.example through."""
    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    p = VaaraMCPProxy(
        upstream_command=["echo"],
        pipeline=MagicMock(),
        allowed_origins={"https://console.example"},
    )
    p._upstream = MagicMock()
    client = _client(p)
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    for hostile in (
        "https://console.example.evil.test",
        "https://evil-console.example",
        "http://console.example",
        "null",
    ):
        assert _post(client, body, Origin=hostile).status_code == 403, hostile


def test_health_endpoint_stays_open_to_browsers(proxy):
    """/health carries no tenant data and load balancers poll it."""
    client = _client(proxy)
    response = client.get("/health", headers={"Origin": "https://evil.example"})
    assert response.status_code == 200


# ── malformed JSON-RPC must not kill the dispatcher ───────────────────────

@pytest.mark.parametrize("method", [123, ["tools/call"], {"a": 1}, None, 1.5])
def test_non_string_method_returns_invalid_request(proxy, method):
    """A non-string method used to raise straight out of _handle_request.

    On stdio that exception propagates through ``run()``, which has no
    guard, so one malformed line terminates the governance layer.
    """
    response = proxy._handle_request({"jsonrpc": "2.0", "id": 7, "method": method})
    assert response["error"]["code"] == -32600
    assert response["id"] == 7


def test_non_string_method_over_http_is_not_a_500(proxy):
    client = _client(proxy)
    response = _post(client, {"jsonrpc": "2.0", "id": 1, "method": 123})
    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32600


def test_stdio_loop_survives_a_malformed_method(proxy, monkeypatch, capsys):
    """The stdio read loop must answer and keep going, not die."""
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": 123}) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n",
    ]
    monkeypatch.setattr(mcp_proxy.sys, "stdin", iter(lines))
    proxy.run()
    written = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert written[0]["error"]["code"] == -32600
    assert written[1]["id"] == 2, "loop stopped after the malformed line"


def test_unexpected_handler_error_becomes_internal_error(proxy, monkeypatch, capsys):
    """Any unforeseen crash still owes the client a JSON-RPC reply."""
    def boom(_request):
        raise RuntimeError("unforeseen")

    monkeypatch.setattr(proxy, "_handle_request", boom)
    monkeypatch.setattr(
        mcp_proxy.sys, "stdin",
        iter([json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}) + "\n"]),
    )
    proxy.run()
    written = json.loads(capsys.readouterr().out.strip())
    assert written["error"]["code"] == -32603
    assert written["id"] == 3


# ── JSON-RPC batching (MCP 2025-03-26) ────────────────────────────────────

def test_batch_request_returns_a_batch_response(proxy):
    """2025-03-26 clients may batch; the proxy advertises that revision."""
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 2, "method": "prompts/list"},
    ]
    responses = proxy._handle_request(batch)
    assert isinstance(responses, list)
    assert [r["id"] for r in responses] == [1, 2]


def test_batch_over_http_returns_a_json_array(proxy):
    client = _client(proxy)
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    response = _post(client, batch)
    assert response.status_code == 200
    assert [item["id"] for item in response.json()] == [1, 2]


def test_batch_notifications_get_no_reply(proxy):
    """A batch of only notifications produces no response object at all."""
    batch = [
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/list"},
    ]
    responses = proxy._handle_request(batch)
    assert [r["id"] for r in responses] == [5]
    proxy._upstream.notify.assert_called_once()


def test_batch_of_only_notifications_over_http_is_202(proxy):
    client = _client(proxy)
    batch = [{"jsonrpc": "2.0", "method": "notifications/initialized"}]
    assert _post(client, batch).status_code == 202


def test_empty_batch_is_an_invalid_request(proxy):
    response = proxy._handle_request([])
    assert response["error"]["code"] == -32600


def test_batch_element_that_is_not_an_object_is_reported_per_element(proxy):
    batch = [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, "garbage"]
    responses = proxy._handle_request(batch)
    assert responses[0]["id"] == 1
    assert responses[1]["error"]["code"] == -32600


def test_batch_size_is_capped(proxy):
    """An unbounded batch is one request that fans out into thousands."""
    batch = [
        {"jsonrpc": "2.0", "id": i, "method": "tools/list"}
        for i in range(mcp_proxy._MCP_MAX_BATCH_SIZE + 1)
    ]
    response = proxy._handle_request(batch)
    assert response["error"]["code"] == -32600
    assert "batch" in response["error"]["message"].lower()


# ── upstream command parsing must produce an executable argv ──────────────

def test_named_upstream_with_arguments_is_split_into_argv():
    """docs/adapters.md shows this exact form; Popen needs a real argv.

    ``Popen(["npx -y @github/mcp-server"])`` looks for a file whose name is
    the whole string and raises FileNotFoundError, so the documented
    fan-out could never have started an upstream.
    """
    parsed = _parse_upstream_specs(
        ["github=npx -y @github/mcp-server", "sap=npx -y @sap/mdk-mcp-server"], [],
    )
    assert parsed == {
        "github": ["npx", "-y", "@github/mcp-server"],
        "sap": ["npx", "-y", "@sap/mdk-mcp-server"],
    }


def test_bare_command_with_arguments_is_split_into_argv():
    parsed = _parse_upstream_specs(["python -m foo --bar=baz"], [])
    assert parsed == {"default": ["python", "-m", "foo", "--bar=baz"]}


def test_quoted_argument_survives_splitting():
    parsed = _parse_upstream_specs(["srv=server --flag 'two words'"], [])
    assert parsed == {"srv": ["server", "--flag", "two words"]}


def test_plain_command_is_unchanged():
    assert _parse_upstream_specs(["echo"], []) == {"default": ["echo"]}


def test_legacy_upstream_args_still_append_to_the_first_slot():
    parsed = _parse_upstream_specs(["echo"], ["hello", "world"])
    assert parsed == {"default": ["echo", "hello", "world"]}


def test_duplicate_upstream_slot_is_an_error():
    """Silently keeping the last one hands the operator a smaller fleet."""
    with pytest.raises(SystemExit, match="duplicate"):
        _parse_upstream_specs(["a=cmd-a", "a=cmd-b"], [])


def test_two_bare_upstreams_collide_on_default_and_raise():
    with pytest.raises(SystemExit, match="duplicate"):
        _parse_upstream_specs(["cmd-a", "cmd-b"], [])


def test_unparseable_command_is_a_clean_error():
    with pytest.raises(SystemExit, match="invalid --upstream"):
        _parse_upstream_specs(["srv=server --flag 'unbalanced"], [])
