"""Tests for `vaara hook`: the in-package Claude Code hook runner.

The point of the runner: the plugin shells out to the vaara binary, so
any CLI install (pip, pipx, Homebrew) is a complete engine install and
the python3-environment split-brain cannot occur.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

_SHIM = (
    Path(__file__).parent.parent
    / "plugins" / "claude-code-vaara-governance" / "hooks" / "run.sh"
)


def _run_hook(args, event: dict, home: Path, extra_env: dict | None = None):
    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", ""),
        "VAARA_PLUGIN_SHADOW": "0",
        **(extra_env or {}),
    }
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; from vaara.cli import main; sys.exit(main(sys.argv[1:]))",
         *args],
        input=json.dumps(event), capture_output=True, text=True,
        env=env, timeout=120,
    )


def test_hook_pre_blocks_deny_pattern_with_bundled_rules(tmp_path):
    proc = _run_hook(
        ["hook", "pre-tool-use"],
        {"tool_name": "Bash", "tool_input": {"command": "cat /etc/shadow"}},
        tmp_path,
    )
    assert proc.returncode == 2, proc.stderr
    assert "BLOCKED" in proc.stderr


def test_hook_pre_allows_benign_bash(tmp_path):
    proc = _run_hook(
        ["hook", "pre-tool-use"],
        {"tool_name": "Bash", "tool_input": {"command": "ls -la"}},
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr


def test_hook_pre_shadow_records_but_allows(tmp_path):
    proc = _run_hook(
        ["hook", "pre-tool-use"],
        {"tool_name": "Bash", "tool_input": {"command": "cat /etc/shadow"}},
        tmp_path, extra_env={"VAARA_PLUGIN_SHADOW": "1"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "SHADOW deny" in proc.stderr


def test_hook_session_start_reports_and_discloses(tmp_path):
    cfg_dir = tmp_path / ".vaara" / "claude-code"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.json").write_text(json.dumps(
        {"article50_statement": "You are interacting with an AI assistant."}
    ))
    proc = _run_hook(
        ["hook", "session-start"], {"session_id": "sess-7"}, tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert "vaara-governance" in proc.stderr
    assert "article50_disclosure=recorded" in proc.stderr

    conn = sqlite3.connect(cfg_dir / "audit.db")
    rows = conn.execute(
        "SELECT data FROM audit_records WHERE tool_name = "
        "'vaara.article50.disclosure' AND event_type = 'action_requested'"
    ).fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0][0])["session_id"] == "sess-7"


def test_hook_post_tool_use_never_fails(tmp_path):
    proc = _run_hook(
        ["hook", "post-tool-use"],
        {"tool_name": "Bash", "tool_response": {"stderr": ""}},
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr


def test_shim_prefers_vaara_binary(tmp_path):
    """run.sh must route to `vaara hook` when the binary is on PATH."""
    bindir = Path(sys.executable).parent
    if not (bindir / "vaara").exists():
        pytest.skip("no vaara console script next to this interpreter")
    env = {
        "HOME": str(tmp_path),
        "PATH": f"{bindir}:/usr/bin:/bin",
        "VAARA_PLUGIN_SHADOW": "0",
    }
    proc = subprocess.run(
        ["sh", str(_SHIM), "pre-tool-use"],
        input=json.dumps(
            {"tool_name": "Bash", "tool_input": {"command": "cat /etc/shadow"}}
        ),
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert proc.returncode == 2, proc.stderr
    assert "BLOCKED" in proc.stderr


# ---------------------------------------------------------------------------
# Approvals loop: an escalate blocks on ~/.vaara/approvals until the app
# (played here by a responder thread) answers, or falls through on timeout.

def _escalate_config(home: Path) -> None:
    """Thresholds that make every scored mcp__ call an escalate."""
    cfg_dir = home / ".vaara" / "claude-code"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.json").write_text(json.dumps(
        {"thresholds": {"escalate": 0.0, "deny": 1.0}}
    ))


def _approval_responder(home: Path, decision: str):
    import threading
    import time as _time

    approvals = home / ".vaara" / "approvals"

    def responder():
        deadline = _time.monotonic() + 60
        while _time.monotonic() < deadline:
            for req in approvals.glob("*.request.json") if approvals.exists() else []:
                action_id = req.name.removesuffix(".request.json")
                (approvals / f"{action_id}.decision.json").write_text(
                    json.dumps({"decision": decision,
                                "decided_at": _time.time()})
                )
                return
            _time.sleep(0.05)

    thread = threading.Thread(target=responder, daemon=True)
    thread.start()
    return thread


def test_hook_pre_escalate_approved_by_human_allows(tmp_path):
    _escalate_config(tmp_path)
    thread = _approval_responder(tmp_path, "approve")
    proc = _run_hook(
        ["hook", "pre-tool-use"],
        {"tool_name": "mcp__files__read", "tool_input": {"path": "notes.txt"}},
        tmp_path,
    )
    thread.join(timeout=1)
    assert proc.returncode == 0, proc.stderr
    assert "APPROVED" in proc.stderr
    conn = sqlite3.connect(tmp_path / ".vaara" / "claude-code" / "audit.db")
    rows = conn.execute(
        "SELECT data FROM audit_records WHERE event_type = 'escalation_resolved'"
    ).fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0][0])["resolution"] == "allow"


def test_hook_pre_escalate_denied_by_human_blocks(tmp_path):
    _escalate_config(tmp_path)
    thread = _approval_responder(tmp_path, "deny")
    proc = _run_hook(
        ["hook", "pre-tool-use"],
        {"tool_name": "mcp__files__read", "tool_input": {"path": "notes.txt"}},
        tmp_path,
    )
    thread.join(timeout=1)
    assert proc.returncode == 2, proc.stderr
    assert "DENIED" in proc.stderr
    conn = sqlite3.connect(tmp_path / ".vaara" / "claude-code" / "audit.db")
    rows = conn.execute(
        "SELECT data FROM audit_records WHERE event_type = 'escalation_resolved'"
    ).fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0][0])["resolution"] == "deny"


def test_hook_pre_escalate_timeout_stays_blocked(tmp_path):
    _escalate_config(tmp_path)
    proc = _run_hook(
        ["hook", "pre-tool-use"],
        {"tool_name": "mcp__files__read", "tool_input": {"path": "notes.txt"}},
        tmp_path, extra_env={"VAARA_PLUGIN_APPROVALS_TIMEOUT": "0.3"},
    )
    # Escalate stays fail-closed: no human answer means no execution.
    assert proc.returncode == 2, proc.stderr
    assert "ESCALATE" in proc.stderr
    # No decision arrived, so nothing was resolved and no files linger.
    approvals = tmp_path / ".vaara" / "approvals"
    assert not approvals.exists() or list(approvals.iterdir()) == []


def test_hook_pre_escalate_approvals_disabled_blocks_as_before(tmp_path):
    _escalate_config(tmp_path)
    proc = _run_hook(
        ["hook", "pre-tool-use"],
        {"tool_name": "mcp__files__read", "tool_input": {"path": "notes.txt"}},
        tmp_path, extra_env={"VAARA_PLUGIN_APPROVALS": "0"},
    )
    # With the handshake off, prior behaviour is preserved: escalate blocks.
    assert proc.returncode == 2, proc.stderr
    assert "ESCALATE" in proc.stderr
    conn = sqlite3.connect(tmp_path / ".vaara" / "claude-code" / "audit.db")
    rows = conn.execute(
        "SELECT data FROM audit_records WHERE event_type = 'escalation_resolved'"
    ).fetchall()
    assert rows == []
