"""Codex governance: the mapping, the trust state, init and ungovern.

Codex runs ``PreToolUse`` hooks from ``~/.codex/hooks.json`` and blocks on
exit 2, but only once the user has trusted the hook, which Codex records as
a hash in ``config.toml``. These tests hold the translation of Codex's tool
calls (shell as ``Bash``, ``apply_patch`` as one file event per path),
the runner's verdict on them, the trust hash computed the way Codex
computes it, and init writing the hooks beside the operator's own and
saying when Codex will not run them yet.

The payloads are the ones Codex 0.156.1 sent in the end-to-end run against
the real binary; the hash vector is the one its app-server reported.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.integrations import codex
from vaara.integrations import init_governance as ig

pytest.importorskip("cryptography")

PATCH = ("*** Begin Patch\n*** Add File: a.txt\n+hello\n*** Update File: sub/b.txt\n"
         "@@\n-old\n+new\n*** End Patch\n")


def _payload(tool: str, tool_input, **extra) -> dict:
    return {"session_id": "s1", "turn_id": "t1", "transcript_path": "/x.jsonl",
            "cwd": "/w", "hook_event_name": "PreToolUse", "model": "m",
            "permission_mode": "bypassPermissions", "tool_name": tool,
            "tool_input": tool_input, "tool_use_id": "call_0", **extra}


# --- mapping ----------------------------------------------------------------


def test_shell_arrives_as_bash():
    [e] = codex.to_hook_events(_payload("Bash", {"command": "ls"}))
    assert e == {"tool_name": "Bash", "tool_input": {"command": "ls"}, "session_id": "s1"}


def test_patch_becomes_one_event_per_file_resolved_against_cwd():
    events = codex.to_hook_events(_payload("apply_patch", {"command": PATCH}))
    assert [(e["tool_name"], e["tool_input"]["file_path"]) for e in events] == [
        ("Write", "/w/a.txt"), ("Edit", "/w/sub/b.txt")]
    assert all(e["tool_input"]["content"] == PATCH for e in events)


def test_patch_without_file_headers_is_still_checked_as_an_edit():
    [e] = codex.to_hook_events(_payload("apply_patch", {"command": "garbage"}))
    assert e["tool_name"] == "Edit"


def test_view_image_is_a_read():
    [e] = codex.to_hook_events(_payload("view_image", {"path": "../.ssh/id_ed25519"}))
    assert e["tool_name"] == "Read"
    assert e["tool_input"]["file_path"] == "/.ssh/id_ed25519"


def test_mcp_keeps_its_name_and_other_tools_go_to_the_classifier():
    [m] = codex.to_hook_events(_payload("mcp__fs__read", {"p": 1}))
    assert m["tool_name"] == "mcp__fs__read"
    [x] = codex.to_hook_events(_payload("some_extension", {}))
    assert x["tool_name"] == "mcp__codex__some_extension"


# --- trust ------------------------------------------------------------------


def test_hook_hash_matches_what_codex_reported():
    # Codex 0.156.1 app-server hooks/list, currentHash for this handler.
    handler = {"type": "command",
               "command": "/workspace/.shared/audit/e2e/codex/probe_hook.sh",
               "timeout": 30}
    assert codex.hook_hash("pre_tool_use", None, handler) == (
        "sha256:82a7fdd308117dfdc897ef4cd49abd03c6adefd00260d1d2a10503e237d3bf61")


def _trust(home: Path, **state) -> None:
    hooks = json.loads((home / "hooks.json").read_text())["hooks"]["PreToolUse"]
    handler = hooks[0]["hooks"][0]
    key = f"{(home / 'hooks.json').resolve()}:pre_tool_use:0:0"
    lines = [f'[hooks.state."{key}"]']
    if state.get("hash", True):
        value = state.get("value") or codex.hook_hash("pre_tool_use", None, handler)
        lines.append(f'trusted_hash = "{value}"')
    if "enabled" in state:
        lines.append(f"enabled = {str(state['enabled']).lower()}")
    (home / "config.toml").write_text("\n".join(lines) + "\n")


def _no_toml_reader() -> bool:
    import importlib.util

    return not (importlib.util.find_spec("tomllib") or importlib.util.find_spec("tomli"))


# Python 3.10 reads config.toml through tomli, which the dev requirements pin.
@pytest.mark.skipif(_no_toml_reader(), reason="tomllib is Python 3.11+")
def test_trust_status_follows_config_toml(tmp_path):
    home = tmp_path / ".codex"
    assert codex.trust_status(home) == "missing"
    codex.install_hooks("/opt/bin/vaara", home)
    assert codex.trust_status(home) == "untrusted"
    _trust(home)
    assert codex.trust_status(home) == "trusted"
    _trust(home, value="sha256:" + "0" * 64)
    assert codex.trust_status(home) == "modified"
    _trust(home, enabled=False)
    assert codex.trust_status(home) == "disabled"


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


PRE = ["hook", "pre-tool-use", "--client", "codex"]


def test_allowed_shell_is_recorded_as_codex(tmp_path):
    proc = _run_hook(PRE, _payload("Bash", {"command": "ls -la"}), tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""
    assert ("codex", "Bash", "action_requested") in _rows(tmp_path)


def test_denied_shell_exits_2_with_the_reason_on_stderr(tmp_path):
    proc = _run_hook(PRE, _payload("Bash", {"command": "rm -rf /"}), tmp_path)
    assert proc.returncode == 2
    assert "rm_rf_root" in proc.stderr


def test_patch_touching_codex_hooks_is_denied_on_its_second_file(tmp_path):
    patch = ("*** Begin Patch\n*** Add File: notes.txt\n+x\n"
             f"*** Update File: {tmp_path}/.codex/hooks.json\n@@\n-a\n+b\n*** End Patch\n")
    proc = _run_hook(PRE, _payload("apply_patch", {"command": patch}), tmp_path)
    assert proc.returncode == 2
    assert "harness_config_write" in proc.stderr


def test_shell_write_to_codex_config_is_denied(tmp_path):
    proc = _run_hook(PRE, _payload(
        "Bash", {"command": "sed -i s/true/false/ ~/.codex/config.toml"}), tmp_path)
    assert proc.returncode == 2
    assert "harness_config_shell_write" in proc.stderr


@pytest.mark.parametrize("command", ["bash", "python3 -i", "cd /tmp && zsh -l", "ssh box"])
def test_bare_interactive_shell_is_denied(tmp_path, command):
    proc = _run_hook(PRE, _payload("Bash", {"command": command}), tmp_path)
    assert proc.returncode == 2
    assert "interactive_shell_session" in proc.stderr


@pytest.mark.parametrize("command", ["bash run.sh", "python3 -c 'print(1)'",
                                     "node --version", "ssh box uptime"])
def test_running_a_script_is_not_an_interactive_session(tmp_path, command):
    proc = _run_hook(PRE, _payload("Bash", {"command": command}), tmp_path)
    assert "interactive_shell_session" not in proc.stderr


def test_post_records_an_outcome(tmp_path):
    call = _payload("Bash", {"command": "false"})
    assert _run_hook(PRE, call, tmp_path).returncode == 0
    proc = _run_hook(["hook", "post-tool-use", "--client", "codex"],
                     {**call, "hook_event_name": "PostToolUse",
                      "tool_response": "exit 1"}, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert any(r[0] == "codex" and "outcome" in r[2].lower() for r in _rows(tmp_path))


# --- init and ungovern --------------------------------------------------------


def test_install_keeps_other_hooks_and_is_idempotent(tmp_path):
    d = tmp_path / ".codex"
    d.mkdir()
    mine = {"matcher": "Bash", "hooks": [{"type": "command", "command": "./mine.sh"}]}
    (d / "hooks.json").write_text(json.dumps({"hooks": {"PreToolUse": [mine]}}))
    assert codex.install_hooks("/opt/bin/vaara", d) is True
    cfg = json.loads((d / "hooks.json").read_text())
    pre = cfg["hooks"]["PreToolUse"]
    assert pre[0] == mine
    assert pre[1] == {"hooks": [{"type": "command",
                                 "command": "/opt/bin/vaara hook pre-tool-use --client codex",
                                 "timeout": codex.HOOK_TIMEOUT}]}
    assert "PostToolUse" in cfg["hooks"]
    assert codex.install_hooks("/opt/bin/vaara", d) is False
    assert codex.remove_hooks(d) is True
    assert json.loads((d / "hooks.json").read_text()) == {"hooks": {"PreToolUse": [mine]}}


def test_init_writes_codex_hooks_and_reports_them_untrusted(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    d = tmp_path / ".codex"
    d.mkdir()
    report = ig.run_init(
        trail_db=tmp_path / "a.db", settings_path=tmp_path / "s.json",
        config_path=tmp_path / "c.json", vaara_bin="/opt/bin/vaara",
        govern_mcp=False, codex_dir=d, cursor_dir=tmp_path / "no-cursor",
        opencode_dir=tmp_path / "no-opencode")
    assert report.codex_hooks == d / "hooks.json"
    assert report.codex_changed is True
    assert report.codex_trust in ("untrusted", "unknown")
    [row] = [r for r in ig.coverage(report) if r.name == "Codex"]
    assert row.state == "NOT governed"
    ung = ig.run_ungovern(settings_path=tmp_path / "s.json", service_home=tmp_path,
                          service_system="linux", service_runner=lambda c, **k: None,
                          opencode_dir=tmp_path / "no-opencode",
                          cursor_dir=tmp_path / "no-cursor", codex_dir=d)
    assert ung.codex_removed is True


def test_init_skips_codex_when_it_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(ig, "KNOWN_MCP_CLIENTS", [])
    monkeypatch.setattr(codex.shutil, "which", lambda name: None)
    report = ig.run_init(
        trail_db=tmp_path / "a.db", settings_path=tmp_path / "s.json",
        config_path=tmp_path / "c.json", vaara_bin="/opt/bin/vaara",
        govern_mcp=False, codex_dir=tmp_path / "absent",
        cursor_dir=tmp_path / "no-cursor", opencode_dir=tmp_path / "no-opencode")
    assert report.codex_hooks is None
    assert not (tmp_path / "absent").exists()


class TestCodexHomeElsewhere:
    """Codex reads its hooks from $CODEX_HOME; the harness rules follow it."""

    def _rules(self, monkeypatch, home):
        from vaara.deny_rules import load_deny_rules

        monkeypatch.setenv("CODEX_HOME", home)
        return load_deny_rules()

    def test_a_write_to_the_moved_hooks_file_is_refused(self, monkeypatch):
        from vaara.deny_rules import match_deny_rule

        rules = self._rules(monkeypatch, "/srv/agents/codexhome")
        target = "/srv/agents/codexhome/" + "hooks.json"
        assert match_deny_rule(rules, "Write", {"file_path": target})[0] == "harness_config_write"
        sed = " ".join(["sed", "-i", "s/a/b/", "/srv/agents/codexhome/" + "config.toml"])
        assert match_deny_rule(rules, "Bash", {"command": sed})[0] == "harness_config_shell_write"

    def test_reading_it_is_still_allowed(self, monkeypatch):
        from vaara.deny_rules import match_deny_rule

        rules = self._rules(monkeypatch, "/srv/agents/codexhome")
        read = "cat /srv/agents/codexhome/" + "hooks.json"
        assert match_deny_rule(rules, "Bash", {"command": read}) is None

    def test_the_default_home_leaves_the_rules_as_shipped(self, monkeypatch):
        from vaara.deny_rules import load_deny_rules

        monkeypatch.delenv("CODEX_HOME", raising=False)
        shipped = load_deny_rules()
        assert self._rules(monkeypatch, "/home/u/.codex") == shipped
