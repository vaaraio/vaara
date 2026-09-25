# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara scan`` finds agents Vaara does not govern and says so."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

from vaara.integrations import scan
from vaara.integrations.scan import Connection

ANTHROPIC_IP = "160.79.104.10"


def _status(active: dict[str, bool]):
    def status(agent: str) -> tuple[bool, str]:
        on = active.get(agent, False)
        return on, "every tool call, through its hooks" if on else "run `vaara init`"
    return status


def _cmds(table: dict[int, str]):
    return lambda pid: table.get(pid, "")


def test_processes_are_marked_by_adapter_and_route():
    conns = [
        Connection(10, "python", ANTHROPIC_IP, 443),          # a script, no adapter
        Connection(11, "claude", "127.0.0.1", 8790),           # through the llm-proxy
        Connection(12, "node", ANTHROPIC_IP, 443),             # codex, hooks not active
        Connection(13, "python", ANTHROPIC_IP, 443),           # Vaara's own proxy upstream
        Connection(14, "CherryStudio", "127.0.0.1", 11434),    # a desktop app on Ollama
        Connection(15, "ssh", "10.0.0.5", 22),                 # not a model at all
    ]
    cmds = {
        10: "/usr/bin/python3 agent.py",
        11: "claude",
        12: "node /usr/lib/node_modules/@openai/codex/bin/codex.js",
        13: "/x/bin/python /x/bin/vaara llm-proxy --upstream https://api.anthropic.com",
        14: "/Applications/Cherry Studio.app/Contents/MacOS/Cherry Studio",
        15: "ssh host",
    }
    found = scan.scan_processes(conns, {ANTHROPIC_IP: "api.anthropic.com"},
                                _cmds(cmds), _status({}))
    by_pid = {f.where: f for f in found}
    assert set(by_pid) == {"pid 10", "pid 11", "pid 12", "pid 14"}
    assert by_pid["pid 10"].state == "ungoverned"
    assert "api.anthropic.com" in by_pid["pid 10"].detail
    assert (by_pid["pid 11"].state, by_pid["pid 11"].name) == ("governed", "Claude Code")
    assert (by_pid["pid 12"].state, by_pid["pid 12"].name) == ("reachable", "Codex")
    assert by_pid["pid 14"].state == "ungoverned"
    assert "Ollama" in by_pid["pid 14"].detail


def test_an_adapted_agent_with_active_hooks_is_governed():
    conns = [Connection(20, "claude", ANTHROPIC_IP, 443)]
    found = scan.scan_processes(conns, {ANTHROPIC_IP: "api.anthropic.com"},
                                _cmds({20: "claude"}), _status({"claude-code": True}))
    assert [(f.state, f.name) for f in found] == [("governed", "Claude Code")]


def _write(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


def test_mcp_configs_anywhere(tmp_path):
    home = tmp_path
    naked = {"command": "npx", "args": ["some-server"]}
    routed = {"command": "vaara-mcp-proxy", "args": ["--upstream", "npx some-server"]}
    _write(home / ".claude.json", {"mcpServers": {"a": naked}})
    _write(home / "Library/Application Support/Claude/claude_desktop_config.json",
           {"mcpServers": {"a": naked, "b": routed}})
    _write(home / "code/proj/mcp.json", {"mcpServers": {"x": naked}})
    _write(home / "code/other/.mcp.json", {"mcpServers": {"x": naked}})
    _write(home / "code/safe/mcp_config.json", {"mcpServers": {"x": routed}})
    _write(home / ".config/Code/User/mcp.json", {"servers": {"y": naked}})
    _write(home / "code/node_modules/pkg/mcp.json", {"mcpServers": {"z": naked}})

    found = {f.where: f for f in scan.scan_mcp(home, status=_status({"claude-code": True}))}
    assert found["~/.claude.json"].state == "governed"
    assert found["~/code/other/.mcp.json"].state == "governed"   # Claude Code project scope
    assert found["~/Library/Application Support/Claude/claude_desktop_config.json"].state == "reachable"
    assert found["~/code/proj/mcp.json"].state == "ungoverned"
    assert found["~/code/safe/mcp_config.json"].state == "governed"
    assert found["~/.config/Code/User/mcp.json"].state == "ungoverned"
    assert not any("node_modules" in w for w in found)


def test_mcp_config_of_an_agent_without_active_hooks_is_reachable(tmp_path):
    _write(tmp_path / ".cursor/mcp.json", {"mcpServers": {"a": {"command": "npx"}}})
    [f] = scan.scan_mcp(tmp_path, status=_status({}))
    assert f.state == "reachable" and "Cursor" in f.detail


def _asar(path: Path, tree: dict) -> None:
    header = json.dumps(tree).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<IIII", 4, len(header) + 8, len(header) + 4, len(header)) + header)


def test_apps_bundling_an_sdk(tmp_path):
    cherry = tmp_path / "Cherry Studio.app"
    _asar(cherry / "Contents/Resources/app.asar", {"files": {"node_modules": {"files": {
        "@anthropic-ai": {"files": {"sdk": {"files": {}}}},
        "@modelcontextprotocol": {"files": {"sdk": {"files": {}}}},
        "left-pad": {"files": {}},
    }}}})
    unpacked = tmp_path / "Scribe.app"
    (unpacked / "Contents/Resources/app/node_modules/openai").mkdir(parents=True)
    (unpacked / "Contents/Resources/app/node_modules/openai/package.json").write_text("{}")
    cursor = tmp_path / "Cursor.app"
    _asar(cursor / "Contents/Resources/app.asar", {"files": {"node_modules": {"files": {
        "openai": {"files": {}}}}}})
    plain = tmp_path / "Notes.app"
    _asar(plain / "Contents/Resources/app.asar", {"files": {"node_modules": {"files": {}}}})

    found = {f.name: f for f in scan.scan_apps([cherry, unpacked, cursor, plain],
                                                status=_status({"cursor": True}))}
    assert set(found) == {"Cherry Studio", "Scribe", "Cursor"}
    assert found["Cherry Studio"].state == "ungoverned"
    assert "Anthropic SDK" in found["Cherry Studio"].detail
    assert "MCP SDK" in found["Cherry Studio"].detail
    assert "OpenAI SDK" in found["Scribe"].detail
    assert found["Cursor"].state == "governed"


def test_a_broken_asar_is_skipped(tmp_path):
    bad = tmp_path / "app.asar"
    bad.write_bytes(b"\x04\x00\x00\x00" + b"\xff" * 12 + b"not json")
    assert scan.asar_files(bad) is None


def test_proc_connections(tmp_path):
    proc = tmp_path
    (proc / "net").mkdir()
    # 160.79.104.10:443, little-endian hex, state 01 (ESTABLISHED), inode 4242.
    ip = "0A684FA0"
    (proc / "net/tcp").write_text(
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        f"   0: 0100007F:9C40 {ip}:01BB 01 00000000:00000000 00:00000000 00000000  1000        0 4242\n"
        f"   1: 0100007F:9C41 {ip}:01BB 0A 00000000:00000000 00:00000000 00000000  1000        0 4343\n")
    (proc / "net/tcp6").write_text("header\n")
    fd = proc / "77" / "fd"
    fd.mkdir(parents=True)
    os.symlink("socket:[4242]", fd / "3")
    os.symlink("socket:[4343]", fd / "4")
    assert scan._proc_connections(proc) == [Connection(77, "", ANTHROPIC_IP, 443)]


def test_cli_json(monkeypatch, capsys):
    from vaara.cli import main

    monkeypatch.setattr(scan, "run_scan", lambda **kw: [
        scan.Finding("ungoverned", "app", "Cherry Studio", "bundles Anthropic SDK", "/x")])
    assert main(["scan", "--json", "--fail-on-ungoverned"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out == [{"state": "ungoverned", "kind": "app", "name": "Cherry Studio",
                    "detail": "bundles Anthropic SDK", "where": "/x"}]
