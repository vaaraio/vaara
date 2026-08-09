# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract tests for the Vaara MCP server's JSON-RPC surface.

Three gaps a passing suite left open:

* ``VAARA_API_KEY`` gated ``tools/call`` and nothing else, so the compliance
  report and scorer state stayed readable without the key;
* a ``tools/call`` sent without an ``id`` was executed and its decision
  thrown away, so the trail recorded a verdict the caller never saw;
* JSON-RPC batches were refused outright.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from vaara.integrations import mcp_server as mcp_server_mod
from vaara.integrations.mcp_server import VaaraMCPServer


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.delenv("VAARA_API_KEY", raising=False)
    pipeline = MagicMock()
    pipeline.status.return_value = {"calibrated": True}
    return VaaraMCPServer(pipeline=pipeline)


@pytest.fixture
def keyed_server(monkeypatch):
    monkeypatch.setenv("VAARA_API_KEY", "s3cret")
    pipeline = MagicMock()
    pipeline.status.return_value = {"calibrated": True}
    return VaaraMCPServer(pipeline=pipeline)


def _read(server, uri: str) -> dict:
    return server.handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "resources/read",
        "params": {"uri": uri},
    })


# ── API key must gate every data-bearing surface ──────────────────────────

def test_resources_read_requires_the_api_key(keyed_server):
    """vaara://status leaks scorer and calibration state without the key."""
    response = _read(keyed_server, "vaara://status")
    assert "error" in response
    assert response["error"]["code"] == -32602
    assert "auth" in response["error"]["message"].lower()


def test_compliance_resource_requires_the_api_key(keyed_server):
    """The compliance assessment is the most sensitive thing the server serves."""
    response = _read(keyed_server, "vaara://compliance")
    assert "error" in response


def test_resources_read_with_the_key_succeeds(keyed_server):
    response = keyed_server.handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "resources/read",
        "params": {"uri": "vaara://status", "_api_key": "s3cret"},
    })
    assert "result" in response, response


def test_resources_read_with_a_wrong_key_is_refused(keyed_server):
    response = keyed_server.handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "resources/read",
        "params": {"uri": "vaara://status", "_api_key": "wrong"},
    })
    assert "error" in response


def test_resources_read_is_open_when_no_key_is_configured(server):
    """Unset VAARA_API_KEY keeps the single-tenant stdio contract unchanged."""
    assert "result" in _read(server, "vaara://status")


def test_tools_call_still_requires_the_key(keyed_server):
    response = keyed_server.handle_request({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "vaara_check", "arguments": {"tool_name": "x"}},
    })
    assert response["error"]["code"] == -32602


# ── notifications must not execute governed tools ─────────────────────────

def test_tools_call_without_an_id_does_not_execute(server):
    """No id means no reply, so the caller can never see the decision.

    Running the interception anyway writes a verdict to the trail that
    nothing enforced: record and behaviour disagree.
    """
    server.handle_request({
        "jsonrpc": "2.0", "method": "tools/call",
        "params": {"name": "vaara_intercept", "arguments": {"tool_name": "tx.transfer"}},
    })
    server._pipeline.intercept.assert_not_called()


def test_real_notifications_are_still_accepted(server):
    response = server.handle_request({
        "jsonrpc": "2.0", "method": "notifications/initialized",
    })
    assert response is None or "error" not in response


def test_stdio_loop_writes_nothing_for_a_notification(server, monkeypatch, capsys):
    lines = [
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n",
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "ping"}) + "\n",
    ]
    monkeypatch.setattr(mcp_server_mod.sys, "stdin", iter(lines))
    server.run()
    written = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x]
    assert len(written) == 1
    assert written[0]["id"] == 2


# ── JSON-RPC batching ─────────────────────────────────────────────────────

def test_batch_returns_one_response_per_request(server):
    responses = server.handle_request([
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ])
    assert [r["id"] for r in responses] == [1, 2]


def test_batch_skips_notification_members(server):
    responses = server.handle_request([
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 9, "method": "ping"},
    ])
    assert [r["id"] for r in responses] == [9]


def test_empty_batch_is_invalid(server):
    assert server.handle_request([])["error"]["code"] == -32600


def test_batch_is_capped(server):
    batch = [
        {"jsonrpc": "2.0", "id": i, "method": "ping"}
        for i in range(mcp_server_mod._MCP_MAX_BATCH_SIZE + 1)
    ]
    assert server.handle_request(batch)["error"]["code"] == -32600


def test_stdio_loop_writes_a_batch_as_one_array(server, monkeypatch, capsys):
    batch = [
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "id": 2, "method": "ping"},
    ]
    monkeypatch.setattr(
        mcp_server_mod.sys, "stdin", iter([json.dumps(batch) + "\n"]),
    )
    server.run()
    written = json.loads(capsys.readouterr().out.strip())
    assert [item["id"] for item in written] == [1, 2]


def test_stdio_loop_writes_nothing_for_an_all_notification_batch(server, monkeypatch, capsys):
    batch = [{"jsonrpc": "2.0", "method": "notifications/initialized"}]
    monkeypatch.setattr(
        mcp_server_mod.sys, "stdin", iter([json.dumps(batch) + "\n"]),
    )
    server.run()
    assert capsys.readouterr().out.strip() == ""
