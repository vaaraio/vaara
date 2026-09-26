"""Gemini CLI governance: the mapping, the hook state, init and ungovern.

Gemini CLI runs ``BeforeTool`` hooks from ``~/.gemini/settings.json`` and
blocks on exit 2 or a JSON deny. These tests hold the translation of its
tool calls (its own names to the Claude Code tools the rules speak about,
relative paths resolved against ``cwd``, MCP calls named from their
``mcp_context``), the runner's verdict on them, the settings file written
beside the operator's own settings, comments included, and init saying
when Gemini CLI will not run the hook.

The payload shape is the one Gemini CLI 0.61.0 sent in the end-to-end run
against the real binary.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.integrations import _hook_gate, gemini
from vaara.integrations import init_governance as ig

pytest.importorskip("cryptography")


def _payload(tool: str, tool_input, **extra) -> dict:
    return {"session_id": "s1", "transcript_path": "/x.jsonl", "cwd": "/w",
            "hook_event_name": "BeforeTool", "timestamp": "2026-09-24T09:00:46.545Z",
            "tool_name": tool, "tool_input": tool_input, **extra}


# --- mapping ----------------------------------------------------------------


@pytest.mark.parametrize("tool, args, name, field, value", [
    ("run_shell_command", {"command": "ls"}, "Bash", "command", "ls"),
    ("write_file", {"file_path": "ok.txt", "content": "x"}, "Write", "file_path", "/w/ok.txt"),
    ("replace", {"file_path": "a/b.py", "old_string": "a", "new_string": "b"},
     "Edit", "file_path", "/w/a/b.py"),
    ("read_file", {"file_path": "/etc/hosts"}, "Read", "file_path", "/etc/hosts"),
    ("web_fetch", {"prompt": "summarise https://x.test"}, "WebFetch", "url",
     "summarise https://x.test"),
    ("activate_skill", {"name": "deploy"}, "Skill", "skill", "deploy"),
])
def test_builtins_become_the_tools_the_rules_name(tool, args, name, field, value):
    [e] = gemini.to_hook_events(_payload(tool, args))
    assert e["tool_name"] == name
    assert e["tool_input"][field] == value
    assert e["session_id"] == "s1"


def test_read_many_files_is_one_read_per_path():
    events = gemini.to_hook_events(_payload("read_many_files", {"include": ["a", "/b"]}))
    assert [(e["tool_name"], e["tool_input"]["file_path"]) for e in events] == [
        ("Read", "/w/a"), ("Read", "/b")]


def test_other_builtins_keep_their_names():
    [e] = gemini.to_hook_events(_payload("glob", {"pattern": "*.py"}))
    assert e["tool_name"] == "glob"


def test_mcp_is_named_from_its_context():
    [e] = gemini.to_hook_events(_payload(
        "mcp_my_fs_read", {"p": 1},
        mcp_context={"server_name": "my_fs", "tool_name": "read", "command": "x"}))
    assert e["tool_name"] == "mcp__my_fs__read"


def test_unknown_tools_go_to_the_classifier():
    [m] = gemini.to_hook_events(_payload("mcp_srv_tool", {}))
    assert m["tool_name"] == "mcp__gemini__srv_tool"
    [x] = gemini.to_hook_events(_payload("codebase_investigator", {}))
    assert x["tool_name"] == "mcp__gemini__codebase_investigator"


def test_a_tool_error_is_reported_as_one():
    ok = {"llmContent": "fine", "returnDisplay": "fine"}
    assert gemini.tool_response({"tool_response": ok}) == ok
    bad = gemini.tool_response({"tool_response": {**ok, "error": {"message": "no"}}})
    assert bad["isError"] is True


def test_render_pre_carries_the_reason_as_json():
    assert gemini.render_pre(0, "ignored") == ("", 0)
    out, code = gemini.render_pre(2, "rule x")
    assert code == 2
    assert json.loads(out) == {"decision": "deny", "reason": "rule x"}


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


PRE = ["hook", "pre-tool-use", "--client", "gemini"]


def test_allowed_shell_is_recorded_as_gemini_with_nothing_on_stdout(tmp_path):
    proc = _run_hook(PRE, _payload("run_shell_command", {"command": "ls -la"}), tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""
    assert ("gemini", "Bash", "action_requested") in _rows(tmp_path)


def test_denied_shell_exits_2_with_a_json_deny(tmp_path):
    wipe = " ".join(["rm", "-rf", "/"])
    proc = _run_hook(PRE, _payload("run_shell_command", {"command": wipe}), tmp_path)
    assert proc.returncode == 2
    verdict = json.loads(proc.stdout)
    assert verdict["decision"] == "deny"
    assert "rm_rf_root" in verdict["reason"]


def test_a_relative_write_into_the_gemini_settings_is_denied(tmp_path):
    # The model writes the path relative to the project; Gemini CLI passes
    # it on as written. A project .gemini/settings.json can switch hooks off.
    proc = _run_hook(PRE, _payload(
        "write_file", {"file_path": ".gemini/settings.json",
                       "content": '{"hooksConfig": {"enabled": false}}'},
        cwd=str(tmp_path)), tmp_path)
    assert proc.returncode == 2
    assert "harness_config_write" in proc.stdout


def test_a_shell_write_to_the_gemini_settings_is_denied(tmp_path):
    sed = " ".join(["sed", "-i", "s/vaara//", "~/.gemini/settings.json"])
    proc = _run_hook(PRE, _payload("run_shell_command", {"command": sed}), tmp_path)
    assert proc.returncode == 2
    assert "harness_config_shell_write" in proc.stdout


def test_reading_the_gemini_settings_is_allowed(tmp_path):
    proc = _run_hook(PRE, _payload(
        "run_shell_command", {"command": "cat ~/.gemini/settings.json"}), tmp_path)
    assert proc.returncode == 0, proc.stdout


def test_post_records_an_outcome(tmp_path):
    call = _payload("run_shell_command", {"command": "false"})
    assert _run_hook(PRE, call, tmp_path).returncode == 0
    proc = _run_hook(["hook", "post-tool-use", "--client", "gemini"],
                     {**call, "hook_event_name": "AfterTool",
                      "tool_response": {"llmContent": "Exit Code: 1",
                                        "returnDisplay": ""}}, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert any(r[0] == "gemini" and "outcome" in r[2].lower() for r in _rows(tmp_path))


# --- settings, init and ungovern ----------------------------------------------


def test_install_keeps_other_settings_and_is_idempotent(tmp_path):
    d = tmp_path / ".gemini"
    d.mkdir()
    mine = {"matcher": "write_file", "hooks": [{"type": "command", "command": "./mine.sh"}]}
    before = {"theme": "Dracula", "hooks": {"BeforeTool": [mine]}}
    (d / "settings.json").write_text(json.dumps(before))
    assert gemini.install_hooks("/opt/bin/vaara", d) is True
    cfg = json.loads((d / "settings.json").read_text())
    assert cfg["theme"] == "Dracula"
    pre = cfg["hooks"]["BeforeTool"]
    assert pre[0] == mine
    assert pre[1] == {"matcher": ".*", "hooks": [{
        "type": "command", "name": gemini.HOOK_NAME,
        "command": _hook_gate.pre_command("/opt/bin/vaara", "gemini"),
        "timeout": gemini.HOOK_TIMEOUT_MS}]}
    assert "AfterTool" in cfg["hooks"]
    assert gemini.install_hooks("/opt/bin/vaara", d) is False
    assert gemini.remove_hooks(d) is True
    assert json.loads((d / "settings.json").read_text()) == before


def test_a_commented_settings_file_is_read_and_kept_as_a_backup(tmp_path):
    d = tmp_path / ".gemini"
    d.mkdir()
    text = ('{\n  // my theme\n  "theme": "a // not a comment",\n'
            '  /* block */ "general": {"vimMode": true}\n}\n')
    (d / "settings.json").write_text(text)
    assert gemini.install_hooks("/opt/bin/vaara", d) is True
    cfg = json.loads((d / "settings.json").read_text())
    assert cfg["theme"] == "a // not a comment"
    assert cfg["general"] == {"vimMode": True}
    assert (d / "settings.json.vaara-backup").read_text() == text


def test_a_settings_file_that_does_not_parse_is_left_alone(tmp_path):
    d = tmp_path / ".gemini"
    d.mkdir()
    (d / "settings.json").write_text("{ not json")
    with pytest.raises(ValueError):
        gemini.install_hooks("/opt/bin/vaara", d)
    assert (d / "settings.json").read_text() == "{ not json"
    assert gemini.hook_status(d) == "unknown"


def test_hook_status_follows_hooks_config(tmp_path):
    d = tmp_path / ".gemini"
    assert gemini.hook_status(d) == "missing"
    gemini.install_hooks("/opt/bin/vaara", d)
    assert gemini.hook_status(d) == "active"
    path = d / "settings.json"
    cfg = json.loads(path.read_text())
    path.write_text(json.dumps({**cfg, "hooksConfig": {"enabled": False}}))
    assert gemini.hook_status(d) == "disabled"
    path.write_text(json.dumps({**cfg, "hooksConfig": {"disabled": [gemini.HOOK_NAME]}}))
    assert gemini.hook_status(d) == "disabled"
    path.write_text(json.dumps({**cfg, "hooksConfig": {"disabled": ["other"]}}))
    assert gemini.hook_status(d) == "active"


def _init(tmp_path, gemini_dir):
    return ig.run_init(
        trail_db=tmp_path / "a.db", settings_path=tmp_path / "s.json",
        config_path=tmp_path / "c.json", vaara_bin="/opt/bin/vaara",
        govern_mcp=False, gemini_dir=gemini_dir, codex_dir=tmp_path / "no-codex",
        cursor_dir=tmp_path / "no-cursor", opencode_dir=tmp_path / "no-opencode")


def test_init_writes_gemini_hooks_and_reports_it_governed(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    d = tmp_path / ".gemini"
    d.mkdir()
    report = _init(tmp_path, d)
    assert report.gemini_settings == d / "settings.json"
    assert report.gemini_changed is True
    assert report.gemini_status == "active"
    [row] = [r for r in ig.coverage(report) if r.name == "Gemini CLI"]
    assert row.state == "governed"
    ung = ig.run_ungovern(settings_path=tmp_path / "s.json", service_home=tmp_path,
                          service_system="linux", service_runner=lambda c, **k: None,
                          opencode_dir=tmp_path / "no-opencode",
                          cursor_dir=tmp_path / "no-cursor",
                          codex_dir=tmp_path / "no-codex", gemini_dir=d)
    assert ung.gemini_removed is True


def test_init_says_so_when_gemini_hooks_are_switched_off(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    d = tmp_path / ".gemini"
    d.mkdir()
    (d / "settings.json").write_text(json.dumps({"hooksConfig": {"enabled": False}}))
    report = _init(tmp_path, d)
    [row] = [r for r in ig.coverage(report) if r.name == "Gemini CLI"]
    assert row.state == "NOT governed"
    assert "turned off" in row.detail


def test_init_warns_and_does_not_claim_a_settings_file_it_could_not_read(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    d = tmp_path / ".gemini"
    d.mkdir()
    (d / "settings.json").write_text("{ not json")
    report = _init(tmp_path, d)
    assert any("Gemini CLI hooks not written" in w for w in report.warnings)
    [row] = [r for r in ig.coverage(report) if r.name == "Gemini CLI"]
    assert row.state == "NOT governed"


def test_init_skips_gemini_when_it_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    monkeypatch.setattr(gemini.shutil, "which", lambda name: None)
    report = _init(tmp_path, tmp_path / "absent")
    assert report.gemini_settings is None
    assert not (tmp_path / "absent").exists()
