"""The shadow report: what would have been blocked, from the trail alone.

Shadow mode records true decisions while allowing everything, so the report
is a read over the audit DB: deny decisions land as action_blocked records,
escalates as decision_made records with decision=escalate.
"""

from __future__ import annotations

import json

from vaara.audit.shadow_report import render_text, shadow_report
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import AuditTrail


def _seed(db_path):
    backend = SQLiteAuditBackend(db_path)
    trail = AuditTrail(on_record=backend.write_record)
    trail.record_decision("a1", "agent-x", "read_file", "allow", "low risk", 0.05)
    trail.record_decision("a2", "agent-x", "shell_exec", "escalate", "mid risk", 0.45)
    trail.record_decision("a3", "agent-x", "shell_exec", "deny", "high risk", 0.92)
    trail.record_decision("a4", "agent-y", "delete_file", "deny", "high risk", 0.88)
    backend.close()


def test_shadow_report_counts(tmp_path):
    db = tmp_path / "audit.db"
    _seed(db)
    report = shadow_report(db, days=7)
    assert report["total_decisions"] == 4
    assert report["allowed"] == 1
    assert report["would_escalate"] == 1
    assert report["would_block"] == 2


def test_shadow_report_groups_blocked_by_tool(tmp_path):
    db = tmp_path / "audit.db"
    _seed(db)
    report = shadow_report(db, days=7)
    by_tool = {entry["tool_name"]: entry for entry in report["blocked"]}
    assert by_tool["shell_exec"]["count"] == 1
    assert by_tool["delete_file"]["count"] == 1
    assert by_tool["delete_file"]["max_risk"] == 0.88


def test_shadow_report_window_excludes_old_records(tmp_path):
    db = tmp_path / "audit.db"
    _seed(db)
    report = shadow_report(db, days=0)
    assert report["total_decisions"] == 0


def test_render_text_mentions_blocked_tools(tmp_path):
    db = tmp_path / "audit.db"
    _seed(db)
    text = render_text(shadow_report(db, days=7))
    assert "delete_file" in text
    assert "would have been blocked" in text.lower()


def test_report_is_json_serialisable(tmp_path):
    db = tmp_path / "audit.db"
    _seed(db)
    json.dumps(shadow_report(db, days=7))


def test_cli_trail_shadow_report_json(tmp_path, capsys):
    from vaara.cli import main

    db = tmp_path / "audit.db"
    _seed(db)
    rc = main(["trail", "shadow-report", "--db", str(db), "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["would_block"] == 2


def test_cli_trail_shadow_report_text_default(tmp_path, capsys):
    from vaara.cli import main

    db = tmp_path / "audit.db"
    _seed(db)
    rc = main(["trail", "shadow-report", "--db", str(db)])
    assert rc == 0
    assert "would have been blocked" in capsys.readouterr().out.lower()


def test_cli_trail_shadow_report_missing_db(tmp_path, capsys):
    from vaara.cli import main

    rc = main(["trail", "shadow-report", "--db", str(tmp_path / "nope.db")])
    assert rc == 2
    assert "not" in capsys.readouterr().err.lower()
