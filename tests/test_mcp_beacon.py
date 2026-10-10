# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The MCP beacon names the client behind a stdio ``initialize`` and records it."""

from __future__ import annotations

import io
import json
import os
import sqlite3
import sys

from vaara.integrations import _mcp_beacon as beacon
from vaara.integrations.mcp_server import VaaraMCPServer


def _tree(parents: dict[int, int], cmds: dict[int, str]):
    return dict(parent_of=lambda pid: parents.get(pid), cmd_of=lambda pid: cmds.get(pid, ""))


def _rows(db):
    with sqlite3.connect(db) as con:
        return [(a, json.loads(d)) for a, d in con.execute(
            "SELECT agent_id, data FROM audit_records WHERE event_type = 'agent_seen' ORDER BY seq")]


def _status(active: dict[str, bool]):
    return lambda agent: (active.get(agent, False), "detail")


def test_the_agent_is_found_above_its_launchers():
    tree = _tree({10: 11, 11: 12, 12: 13},
                 {10: "node /usr/lib/node_modules/npm/bin/npx-cli.js some-mcp",
                  11: "/bin/sh -c npx some-mcp",
                  12: "/opt/homebrew/bin/opencode",
                  13: "/sbin/launchd"})
    who = beacon.identify({"name": "something-else"}, 10, **tree)
    assert who == {"agent": "opencode", "name": "OpenCode", "pid": 12,
                   "command": "/opt/homebrew/bin/opencode", "source": "process"}


def test_vaara_wrappers_are_skipped():
    tree = _tree({20: 21}, {20: "/usr/local/bin/vaara-mcp-proxy --upstream x", 21: "claude"})
    assert beacon.identify(None, 20, **tree)["agent"] == "claude-code"


def test_client_info_names_it_when_the_process_table_cannot():
    tree = _tree({30: 31}, {30: "python -m vaara.integrations.mcp_server", 31: "electron"})
    who = beacon.identify({"name": "claude-code", "version": "2.1.0"}, 30, **tree)
    assert (who["agent"], who["source"], who["pid"]) == ("claude-code", "client_info", 30)
    state, detail = beacon.describe(who, {"name": "claude-code", "version": "2.1.0"},
                                    "the Vaara MCP server", _status({"claude-code": True}))
    assert state == "governed"
    assert "clientInfo claude-code 2.1.0" in detail
    assert "process table did not confirm" in detail


def test_unknown_client_is_ungoverned_under_its_own_name():
    who = beacon.identify({"name": "Cherry Studio"}, 40, **_tree({}, {40: "/Applications/Cherry"}))
    assert (who["agent"], who["name"]) == (None, "Cherry Studio")
    state, detail = beacon.describe(who, {"name": "Cherry Studio"}, "vaara-mcp-proxy")
    assert state == "ungoverned"
    assert detail.startswith("connected to vaara-mcp-proxy over stdio")


def test_client_names_map_to_adapters():
    names = {"claude-code": "claude-code", "claude-ai": "claude-desktop",
             "codex-mcp-client": "codex", "gemini-cli-mcp-client": "gemini",
             "cursor-vscode": "cursor", "opencode": "opencode",
             "GitHub Copilot": "copilot", "mcp-inspector": None}
    for name, agent in names.items():
        assert beacon.client_from_info({"name": name}) == agent, name
    assert beacon.client_from_info({"name": 5}) is None
    assert beacon.client_from_info("claude-code") is None


def test_ancestors_stop_at_init_and_cycles():
    assert beacon.ancestors(5, {5: 6, 6: 1}.get) == [5, 6]
    assert beacon.ancestors(5, {5: 6, 6: 5}.get) == [5, 6]
    assert len(beacon.ancestors(100, lambda p: p + 1, hops=3)) == 4


def test_parent_pid_of_this_process():
    assert beacon.parent_pid(os.getpid()) == os.getppid()


def test_records_once_on_the_agents_trail(tmp_path):
    db = tmp_path / "agents" / "audit.db"
    tree = _tree({50: 51}, {50: "uvx vaara-mcp", 51: "/usr/local/bin/codex"})
    b = beacon.Beacon("the Vaara MCP server", trail_path=db, background=False,
                      identify_fn=lambda info, pid: beacon.identify(info, pid, **tree),
                      status=_status({}), start_pid=lambda: 50)
    b.on_initialize({"clientInfo": {"name": "codex-mcp-client", "version": "0.50"}})
    b.on_initialize({"clientInfo": {"name": "codex-mcp-client"}})  # a second initialize
    rows = _rows(db)
    assert len(rows) == 1
    agent, data = rows[0]
    assert agent == "Codex"
    assert data["state"] == "reachable"
    assert data["pid"] == 51
    assert "clientInfo codex-mcp-client 0.50" in data["detail"]


def test_turned_off_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VAARA_BEACON", "0")
    db = tmp_path / "a.db"
    b = beacon.Beacon("x", trail_path=db, background=False, start_pid=lambda: 1)
    b.on_initialize({"clientInfo": {"name": "claude-code"}})
    assert not db.exists()


def test_a_failing_trail_never_reaches_the_client(tmp_path):
    def broken(_path):
        raise OSError("disk gone")
    b = beacon.Beacon("x", open_trail=broken, background=False, start_pid=lambda: 1,
                      identify_fn=lambda info, pid: {"agent": None, "name": "n", "pid": 1,
                                                     "command": "", "source": "unknown"})
    b.on_initialize({})  # logs, does not raise


def test_stdio_server_fires_the_beacon(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(beacon.Beacon, "on_initialize", lambda self, params: seen.append(
        (self.via, params["clientInfo"]["name"])))
    server = VaaraMCPServer(db_path=tmp_path / "trail.db")
    assert server.handle_request({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                                  "params": {"clientInfo": {"name": "x"}}})
    assert seen == []  # no stdio session, no parent to name
    msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
           "params": {"protocolVersion": "2024-11-05", "clientInfo": {"name": "opencode"}}}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(msg) + "\n"))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    server.run()
    server.close()
    assert seen == [("the Vaara MCP server", "opencode")]


def test_stdio_proxy_fires_the_beacon_and_forwards(monkeypatch):
    from unittest.mock import MagicMock

    from vaara.integrations import mcp_proxy

    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    p = mcp_proxy.VaaraMCPProxy(upstream_command=["echo"], pipeline=MagicMock())
    p._upstream = MagicMock()
    p._upstream.request.return_value = {"jsonrpc": "2.0", "id": 1, "result": {}}
    msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
           "params": {"clientInfo": {"name": "cursor-vscode"}}}
    assert p._handle_request(msg)["id"] == 1  # no stdio session yet, no beacon
    seen = []
    p._beacon = MagicMock(on_initialize=lambda params: seen.append(params))
    assert p._handle_request(msg)["id"] == 1
    assert seen == [msg["params"]]
    assert p._upstream.request.call_count == 2
