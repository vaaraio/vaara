"""A call a deny rule blocked is recorded as blocked.

The hook runner records rule hits through the pipeline with enforce off,
and the pipeline wrote the scorer's verdict. The scorer scores the tool
type, so a blocked ``Write`` into harness config went on the chain as
``decision_made: allow``: measured on a live trail 2026-09-23, 3,258
decisions and 2 blocks, while the hook had blocked calls all day. Every
view of the trail, the menu-bar light included, showed only allows.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.audit.trail import AuditTrail
from vaara.pipeline import InterceptionPipeline

pytest.importorskip("cryptography")


def _run(event: dict, home: Path):
    (home / ".vaara").mkdir(parents=True, exist_ok=True)
    (home / ".vaara" / "config.json").write_text("{}")
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", ""),
           "VAARA_PLUGIN_SHADOW": "0", "PYTHONPATH": os.pathsep.join(sys.path)}
    return subprocess.run(
        [sys.executable, "-c",
         "import sys; from vaara.cli import main; sys.exit(main(sys.argv[1:]))",
         "hook", "pre-tool-use"],
        input=json.dumps(event), capture_output=True, text=True, env=env, timeout=120)


def _decisions(home: Path) -> list[tuple[str, str, str]]:
    db = home / ".vaara" / "claude-code" / "audit.db"
    return [(t, e, json.loads(d).get("decision", ""))
            for t, e, d in sqlite3.connect(db).execute(
                "SELECT tool_name, event_type, data FROM audit_records "
                "WHERE event_type IN ('decision_made', 'action_blocked') ORDER BY seq")]


def test_rule_block_lands_as_action_blocked(tmp_path):
    target = str(tmp_path / ".claude" / "settings.json")
    proc = _run({"tool_name": "Write", "session_id": "s",
                 "tool_input": {"file_path": target, "content": "{}"}}, tmp_path)
    assert proc.returncode == 2
    assert _decisions(tmp_path) == [("Write", "action_blocked", "deny")]


def test_an_allowed_call_still_lands_as_allow(tmp_path):
    proc = _run({"tool_name": "Bash", "session_id": "s",
                 "tool_input": {"command": "ls"}}, tmp_path)
    assert proc.returncode == 0
    assert _decisions(tmp_path) == [("Bash", "decision_made", "allow")]


def test_policy_decision_overrides_the_score_and_keeps_it():
    trail = AuditTrail()
    result = InterceptionPipeline(trail=trail, enforce=False).intercept(
        agent_id="a", tool_name="Read", parameters={"file_path": "x"},
        policy_decision="deny", policy_reason="deny rule r: m")
    assert result.allowed is True  # enforce off: the rule, not the pipeline, blocks
    kinds = [str(getattr(r.event_type, "value", r.event_type)) for r in trail._records]
    assert "risk_scored" in kinds
    [blocked] = [r for r in trail._records
                 if str(getattr(r.event_type, "value", r.event_type)) == "action_blocked"]
    assert blocked.data["reason"] == "deny rule r: m"


def test_policy_decision_must_be_a_decision():
    with pytest.raises(ValueError):
        InterceptionPipeline(trail=AuditTrail()).intercept(
            agent_id="a", tool_name="Read", policy_decision="maybe")
