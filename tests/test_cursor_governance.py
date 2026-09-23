"""Cursor governance: the mapping, the verdict Cursor reads, the imported path, init.

Cursor runs a ``preToolUse`` hook before every tool call and reads a
permission hook's verdict from stdout JSON, blocking when there is none. It
also imports the Claude Code hooks from ``~/.claude/settings.json`` by
default, so Vaara's hook can reach it by two routes. These tests hold the
translation of Cursor's tool names, the JSON Cursor needs on both allow and
deny, a Cursor call decided once whichever route it takes, and init writing
``~/.cursor/hooks.json`` with ``failClosed`` beside the operator's own hooks.

Payload field names for Cursor's file tools are not documented; the mapping
reads every name those tools are known to use. A live run on Cursor pins them.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.integrations import cursor
from vaara.integrations import init_governance as ig

pytest.importorskip("cryptography")


def _payload(tool: str, tool_input, **extra) -> dict:
    return {"conversation_id": "conv", "generation_id": "gen", "model": "m",
            "hook_event_name": "preToolUse", "cursor_version": "1.7.0",
            "workspace_roots": ["/w"], "tool_name": tool,
            "tool_input": tool_input, "tool_use_id": "t1", "cwd": "/w", **extra}


# --- mapping ----------------------------------------------------------------


def test_shell_maps_to_bash():
    [e] = cursor.to_hook_events(_payload("Shell", {"command": "ls", "working_directory": "/w"}))
    assert e["tool_name"] == "Bash"
    assert e["tool_input"]["command"] == "ls"
    assert e["session_id"] == "conv"


def test_mcp_tool_goes_to_the_classifier():
    [e] = cursor.to_hook_events(_payload("MCP:create_issue", {"title": "t"}))
    assert e["tool_name"] == "mcp__cursor__create_issue"


def test_mcp_params_sent_as_a_json_string_are_parsed():
    [e] = cursor.to_hook_events(_payload("MCP:run", json.dumps({"cmd": "x"})))
    assert e["tool_input"] == {"cmd": "x"}


@pytest.mark.parametrize("key", ["file_path", "path", "target_file", "filePath"])
def test_write_path_is_found_under_any_known_name(key):
    [e] = cursor.to_hook_events(_payload("Write", {key: "/a/b", "contents": "x"}))
    assert e["tool_name"] == "Write"
    assert e["tool_input"]["file_path"] == "/a/b"
    assert e["tool_input"]["content"] == "x"


def test_delete_is_checked_as_a_write_to_that_path():
    [e] = cursor.to_hook_events(_payload("Delete", {"path": "/home/u/.bashrc"}))
    assert e["tool_name"] == "Write"
    assert e["tool_input"]["file_path"] == "/home/u/.bashrc"


def test_render_allow_and_deny():
    out, code = cursor.render_pre(0, "")
    assert (json.loads(out), code) == ({"permission": "allow"}, 0)
    out, code = cursor.render_pre(2, "vaara-governance: BLOCKED Bash")
    assert code == 2
    assert json.loads(out)["permission"] == "deny"
    assert json.loads(out)["agent_message"] == "vaara-governance: BLOCKED Bash"


# --- the hook runner ---------------------------------------------------------


def _run_hook(args, event: dict, home: Path, extra_env: dict | None = None):
    (home / ".vaara").mkdir(parents=True, exist_ok=True)
    (home / ".vaara" / "config.json").write_text("{}")
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", ""),
           "VAARA_PLUGIN_SHADOW": "0", "PYTHONPATH": os.pathsep.join(sys.path),
           **(extra_env or {})}
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; from vaara.cli import main; sys.exit(main(sys.argv[1:]))",
         *args],
        input=json.dumps(event), capture_output=True, text=True, env=env, timeout=120)


def _rows(home: Path) -> list[tuple]:
    db = home / ".vaara" / "claude-code" / "audit.db"
    if not db.exists():
        return []
    return sqlite3.connect(db).execute(
        "SELECT agent_id, tool_name, event_type FROM audit_records").fetchall()


PRE = ["hook", "pre-tool-use", "--client", "cursor"]


def test_native_allow_prints_the_json_cursor_needs(tmp_path):
    proc = _run_hook(PRE, _payload("Shell", {"command": "ls -la"}), tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"permission": "allow"}
    assert ("cursor", "Bash", "action_requested") in _rows(tmp_path)


def test_native_deny_prints_the_reason_and_exits_2(tmp_path):
    proc = _run_hook(PRE, _payload("Shell", {"command": "cat /etc/shadow"}), tmp_path)
    assert proc.returncode == 2, proc.stderr
    verdict = json.loads(proc.stdout)
    assert verdict["permission"] == "deny"
    assert "etc_shadow_read" in verdict["agent_message"]


def test_native_allow_still_prints_when_governance_is_off(tmp_path):
    cfg = tmp_path / ".vaara" / "claude-code"
    cfg.mkdir(parents=True)
    (cfg / "config.json").write_text(json.dumps({"mode": "off"}))
    # A call the deny rules would stop: with governance off it must still
    # get an allow printed, or Cursor blocks it for want of JSON.
    proc = _run_hook(PRE, _payload("Shell", {"command": "cat /etc/shadow"}), tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"permission": "allow"}
    assert _rows(tmp_path) == []


def test_imported_hook_decides_a_cursor_call_when_no_native_hook(tmp_path):
    proc = _run_hook(["hook", "pre-tool-use"],
                     _payload("Shell", {"command": "cat /etc/shadow"}), tmp_path)
    assert proc.returncode == 2, proc.stderr
    assert proc.stdout.strip() == ""
    assert ("cursor", "Bash", "action_requested") in _rows(tmp_path)


def test_imported_hook_steps_aside_for_the_native_one(tmp_path):
    cursor.install_hooks("/opt/bin/vaara", tmp_path / ".cursor")
    proc = _run_hook(["hook", "pre-tool-use"],
                     _payload("Shell", {"command": "cat /etc/shadow"}), tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert _rows(tmp_path) == []


def test_post_failure_records_an_outcome(tmp_path):
    call = _payload("Shell", {"command": "false"})
    assert _run_hook(PRE, call, tmp_path).returncode == 0
    proc = _run_hook(["hook", "post-tool-use", "--client", "cursor"],
                     {**call, "hook_event_name": "postToolUseFailure"}, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert any(r[0] == "cursor" and "outcome" in r[2].lower() for r in _rows(tmp_path))


def test_writes_to_cursor_hooks_are_denied(tmp_path):
    target = str(tmp_path / ".cursor" / "hooks.json")
    proc = _run_hook(PRE, _payload("Write", {"file_path": target, "contents": "{}"}), tmp_path)
    assert proc.returncode == 2, proc.stderr
    assert "harness_config_write" in json.loads(proc.stdout)["agent_message"]


# --- init and ungovern --------------------------------------------------------


def test_install_keeps_other_hooks_and_is_idempotent(tmp_path):
    d = tmp_path / ".cursor"
    d.mkdir()
    (d / "hooks.json").write_text(json.dumps({"version": 1, "hooks": {
        "beforeShellExecution": [{"command": "./mine.sh"}],
        "preToolUse": [{"command": "./audit.sh"}]}}))
    assert cursor.install_hooks("/opt/bin/vaara", d) is True
    cfg = json.loads((d / "hooks.json").read_text())
    pre = cfg["hooks"]["preToolUse"]
    assert {"command": "./audit.sh"} in pre
    ours = [e for e in pre if "vaara hook" in e["command"]]
    assert ours == [{"command": "/opt/bin/vaara hook pre-tool-use --client cursor",
                     "timeout": cursor.HOOK_TIMEOUT, "failClosed": True}]
    assert cfg["hooks"]["beforeShellExecution"] == [{"command": "./mine.sh"}]
    assert "postToolUseFailure" in cfg["hooks"]
    assert cursor.install_hooks("/opt/bin/vaara", d) is False
    assert cursor.remove_hooks(d) is True
    cfg = json.loads((d / "hooks.json").read_text())
    assert cfg["hooks"] == {"beforeShellExecution": [{"command": "./mine.sh"}],
                            "preToolUse": [{"command": "./audit.sh"}]}


def test_init_writes_cursor_hooks_and_leaves_its_mcp_to_them(tmp_path, monkeypatch):
    d = tmp_path / ".cursor"
    d.mkdir()
    mcp = d / "mcp.json"
    mcp.write_text(json.dumps({"mcpServers": {"fs": {"command": "npx"}}}))
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [("Cursor", str(mcp))])
    monkeypatch.setattr(ig.shutil, "which", lambda name: "/usr/bin/" + name)
    report = ig.run_init(
        trail_db=tmp_path / "a.db", settings_path=tmp_path / "s.json",
        config_path=tmp_path / "c.json", vaara_bin="/opt/bin/vaara",
        proxy_bin="/usr/bin/vaara-mcp-proxy", cursor_dir=d,
        opencode_dir=tmp_path / "no-opencode")
    assert report.cursor_hooks == d / "hooks.json"
    assert cursor.native_hook_installed(d)
    assert "Cursor" not in report.mcp_rewritten
    assert "vaara-mcp-proxy" not in mcp.read_text()
    ung = ig.run_ungovern(settings_path=tmp_path / "s.json", service_home=tmp_path,
                          service_system="linux", service_runner=lambda c, **k: None,
                          opencode_dir=tmp_path / "no-opencode", cursor_dir=d)
    assert ung.cursor_removed is True
    assert not cursor.native_hook_installed(d)


def test_init_skips_cursor_when_it_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    monkeypatch.setattr(cursor.shutil, "which", lambda name: None)
    report = ig.run_init(
        trail_db=tmp_path / "a.db", settings_path=tmp_path / "s.json",
        config_path=tmp_path / "c.json", vaara_bin="/opt/bin/vaara",
        govern_mcp=False, cursor_dir=tmp_path / "absent",
        opencode_dir=tmp_path / "no-opencode")
    assert report.cursor_hooks is None
    assert not (tmp_path / "absent").exists()
