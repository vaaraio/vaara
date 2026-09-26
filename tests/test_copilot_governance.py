"""Copilot CLI governance: the mapping, the hook file, init and ungovern.

Copilot CLI runs ``preToolUse`` hooks from every file in
``~/.copilot/hooks/`` and blocks on exit 2 or a JSON deny. These tests hold
the translation of its tool calls (its own names to the Claude Code tools
the rules speak about, relative paths resolved against ``cwd``, ``toolArgs``
as an object or a JSON string), the runner's verdict on them, Vaara's own
hook file, and init saying when Copilot CLI will not run the hook.

The payload shapes are the ones Copilot CLI 1.0.88 sent its hooks in a run
of the real binary.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.deny_rules import load_deny_rules, match_deny_rule
from vaara.integrations import _hook_gate, copilot
from vaara.integrations import init_governance as ig
from vaara.integrations import scan

pytest.importorskip("cryptography")


def _payload(tool: str, args, **extra) -> dict:
    return {"sessionId": "s1", "timestamp": 1790397186408, "cwd": "/w",
            "toolName": tool, "toolArgs": args, **extra}


# --- mapping ----------------------------------------------------------------


@pytest.mark.parametrize("tool, args, name, field, value", [
    ("bash", {"command": "ls", "description": "x"}, "Bash", "command", "ls"),
    ("create", {"path": "ok.txt", "file_text": "x"}, "Write", "file_path", "/w/ok.txt"),
    ("create", {"path": "ok.txt", "file_text": "x"}, "Write", "content", "x"),
    ("edit", {"path": "a/b.py", "old_str": "a", "new_str": "b"}, "Edit", "file_path", "/w/a/b.py"),
    ("edit", {"path": "a/b.py", "old_str": "a", "new_str": "b"}, "Edit", "new_string", "b"),
    ("view", {"path": "/etc/hosts"}, "Read", "file_path", "/etc/hosts"),
    ("skill", {"skill": "deploy"}, "Skill", "skill", "deploy"),
])
def test_builtins_become_the_tools_the_rules_name(tool, args, name, field, value):
    [e] = copilot.to_hook_events(_payload(tool, args))
    assert e["tool_name"] == name
    assert e["tool_input"][field] == value
    assert e["session_id"] == "s1"


def test_tool_args_as_a_json_string_are_read():
    [e] = copilot.to_hook_events(_payload("bash", json.dumps({"command": "ls"})))
    assert (e["tool_name"], e["tool_input"]["command"]) == ("Bash", "ls")


def test_a_subagent_is_a_task():
    [e] = copilot.to_hook_events(_payload("task", {"prompt": "go", "agent_type": "explore"}))
    assert e["tool_name"] == "Task"


def test_other_builtins_keep_their_names_and_the_rest_go_to_the_classifier():
    [g] = copilot.to_hook_events(_payload("grep", {"pattern": "x"}))
    assert g["tool_name"] == "grep"
    [m] = copilot.to_hook_events(_payload("github-mcp-server-list_issues", {}))
    assert m["tool_name"] == "mcp__copilot__github-mcp-server-list_issues"


def test_outcomes_follow_the_result_and_the_failure_event():
    ok = {"resultType": "success", "textResultForLlm": "fine"}
    assert copilot.tool_response({"toolResult": ok}) == ok
    assert copilot.tool_response({"error": "Path does not exist"})["isError"] is True
    assert copilot.tool_response({"toolResult": {"resultType": "failure"}})["isError"] is True


def test_render_pre_carries_the_reason_as_json():
    assert copilot.render_pre(0, "ignored") == ("", 0)
    out, code = copilot.render_pre(2, "rule x")
    assert code == 2
    assert json.loads(out) == {"permissionDecision": "deny",
                               "permissionDecisionReason": "rule x"}


# --- the raw names, as the LLM proxy's tool gate sees them -----------------------


@pytest.mark.parametrize("tool, args", [
    ("bash", {"command": " ".join(["rm", "-rf", "/"])}),
    ("create", {"path": "/home/u/.bashrc", "file_text": "echo hi"}),
    ("edit", {"path": "/home/u/.ssh/authorized_keys", "new_str": "ssh-ed25519 AAAA"}),
    ("view", {"path": "/home/u/project/.env"}),
    ("task", {"prompt": "go"}),
])
def test_copilot_names_hit_the_rule_their_claude_code_twin_hits(tool, args):
    assert match_deny_rule(load_deny_rules(), tool, args) is not None


# --- the hook runner ---------------------------------------------------------


def _run_hook(args, event: dict, home: Path):
    (home / ".vaara").mkdir(parents=True, exist_ok=True)
    (home / ".vaara" / "config.json").write_text("{}")
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", ""),
           "VAARA_PLUGIN_SHADOW": "0", "PYTHONPATH": os.pathsep.join(sys.path)}
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


PRE = ["hook", "pre-tool-use", "--client", "copilot"]


def test_allowed_shell_is_recorded_as_copilot_with_nothing_on_stdout(tmp_path):
    proc = _run_hook(PRE, _payload("bash", {"command": "ls -la"}), tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""
    assert ("copilot", "Bash", "action_requested") in _rows(tmp_path)


def test_denied_shell_exits_2_with_a_json_deny(tmp_path):
    wipe = " ".join(["rm", "-rf", "/"])
    proc = _run_hook(PRE, _payload("bash", {"command": wipe}), tmp_path)
    assert proc.returncode == 2
    verdict = json.loads(proc.stdout)
    assert verdict["permissionDecision"] == "deny"
    assert "rm_rf_root" in verdict["permissionDecisionReason"]


@pytest.mark.parametrize("path", [
    ".github/hooks/off.json",
    ".github/copilot/settings.json",
])
def test_a_relative_write_into_copilot_hook_config_is_denied(tmp_path, path):
    proc = _run_hook(PRE, _payload(
        "create", {"path": path, "file_text": '{"disableAllHooks": true}'},
        cwd=str(tmp_path)), tmp_path)
    assert proc.returncode == 2
    assert "harness_config_write" in proc.stdout


def test_a_shell_rewrite_of_the_vaara_hook_file_is_denied(tmp_path):
    rm = " ".join(["rm", "-f", "~/.copilot/hooks/vaara.json"])
    proc = _run_hook(PRE, _payload("bash", {"command": rm}), tmp_path)
    assert proc.returncode == 2
    assert "harness_config_shell_write" in proc.stdout


def test_reading_the_hook_file_is_allowed(tmp_path):
    proc = _run_hook(PRE, _payload(
        "bash", {"command": "cat ~/.copilot/hooks/vaara.json"}), tmp_path)
    assert proc.returncode == 0, proc.stdout


def test_post_records_an_outcome(tmp_path):
    call = _payload("bash", {"command": "false"})
    assert _run_hook(PRE, call, tmp_path).returncode == 0
    proc = _run_hook(["hook", "post-tool-use", "--client", "copilot"],
                     {**call, "toolResult": {"resultType": "success",
                                             "textResultForLlm": "exit code 1"}}, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert any(r[0] == "copilot" and "outcome" in r[2].lower() for r in _rows(tmp_path))


# --- the hook file, init and ungovern ------------------------------------------


def test_install_writes_its_own_file_leaves_others_and_is_idempotent(tmp_path):
    d = tmp_path / ".copilot"
    (d / "hooks").mkdir(parents=True)
    other = d / "hooks" / "team.json"
    other.write_text('{"version": 1, "hooks": {}}')
    assert copilot.install_hooks("/opt/bin/vaara", d) is True
    cfg = json.loads((d / "hooks" / "vaara.json").read_text())
    [pre] = cfg["hooks"]["preToolUse"]
    assert pre["bash"] == _hook_gate.pre_command("/opt/bin/vaara", "copilot")
    assert pre["timeoutSec"] == _hook_gate.HOST_TIMEOUT
    assert cfg["hooks"]["postToolUse"] == cfg["hooks"]["postToolUseFailure"]
    assert copilot.install_hooks("/opt/bin/vaara", d) is False
    assert copilot.remove_hooks(d) is True
    assert not (d / "hooks" / "vaara.json").exists()
    assert other.read_text() == '{"version": 1, "hooks": {}}'


def test_remove_leaves_a_vaara_json_that_is_not_vaaras(tmp_path):
    d = tmp_path / ".copilot"
    (d / "hooks").mkdir(parents=True)
    (d / "hooks" / "vaara.json").write_text('{"version": 1, "hooks": {}}')
    assert copilot.remove_hooks(d) is False
    assert (d / "hooks" / "vaara.json").exists()


def test_hook_status(tmp_path):
    d = tmp_path / ".copilot"
    assert copilot.hook_status(d) == "missing"
    copilot.install_hooks("/opt/bin/vaara", d)
    assert copilot.hook_status(d) == "active"
    path = copilot.hooks_path(d)
    cfg = json.loads(path.read_text())
    path.write_text(json.dumps({**cfg, "disableAllHooks": True}))
    assert copilot.hook_status(d) == "disabled"
    path.write_text("{ not json")
    assert copilot.hook_status(d) == "unknown"


def _init(tmp_path, copilot_dir):
    return ig.run_init(
        trail_db=tmp_path / "a.db", settings_path=tmp_path / "s.json",
        config_path=tmp_path / "c.json", vaara_bin="/opt/bin/vaara",
        govern_mcp=False, copilot_dir=copilot_dir, gemini_dir=tmp_path / "no-gemini",
        codex_dir=tmp_path / "no-codex", cursor_dir=tmp_path / "no-cursor",
        opencode_dir=tmp_path / "no-opencode")


def test_init_writes_copilot_hooks_and_reports_it_governed(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    d = tmp_path / ".copilot"
    d.mkdir()
    report = _init(tmp_path, d)
    assert report.copilot_hooks == d / "hooks" / "vaara.json"
    assert report.copilot_changed is True
    assert report.copilot_status == "active"
    [row] = [r for r in ig.coverage(report) if r.name == "Copilot CLI"]
    assert row.state == "governed"
    ung = ig.run_ungovern(settings_path=tmp_path / "s.json", service_home=tmp_path,
                          service_system="linux", service_runner=lambda c, **k: None,
                          opencode_dir=tmp_path / "no-opencode",
                          cursor_dir=tmp_path / "no-cursor", codex_dir=tmp_path / "no-codex",
                          gemini_dir=tmp_path / "no-gemini", copilot_dir=d)
    assert ung.copilot_removed is True


def test_init_skips_copilot_when_it_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    monkeypatch.setattr(copilot.shutil, "which", lambda name: None)
    report = _init(tmp_path, tmp_path / "absent")
    assert report.copilot_hooks is None
    assert not (tmp_path / "absent").exists()


def test_scan_names_copilot_and_reads_its_hook_state(tmp_path, monkeypatch):
    assert scan._adapter_for("node /usr/lib/node_modules/@github/copilot/npm-loader.js") == (
        "copilot", "Copilot CLI")
    assert scan._adapter_for("/usr/local/bin/copilot -p go") == ("copilot", "Copilot CLI")
    monkeypatch.setenv("COPILOT_HOME", str(tmp_path / ".copilot"))
    assert scan.adapter_status("copilot")[0] is False
    copilot.install_hooks("/opt/bin/vaara")
    assert scan.adapter_status("copilot") == (True, "every tool call, through its hooks")
