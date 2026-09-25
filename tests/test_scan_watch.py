# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara scan --watch`` records each agent that starts, once, with its state."""

from __future__ import annotations

import json
import plistlib
import sqlite3

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.integrations import scan, scan_watch


def _status(active: dict[str, bool]):
    return lambda agent: (active.get(agent, False), "detail")


def _rows(db):
    with sqlite3.connect(db) as con:
        return [(a, json.loads(d)) for a, d in con.execute(
            "SELECT agent_id, data FROM audit_records WHERE event_type = 'agent_seen' ORDER BY seq")]


def test_each_agent_is_recorded_once_with_its_state(tmp_path):
    db = tmp_path / "agents" / "audit.db"
    db.parent.mkdir()
    trail = SQLiteAuditBackend(db).load_trail()
    table = [(100, "claude"), (101, "node /x/@openai/codex/bin/codex.js"),
             (102, "/usr/bin/vaara hook pre-tool-use"), (103, "bash")]
    cherry = scan.Finding("ungoverned", "process", "Cherry Studio",
                          "talks to an address of api.anthropic.com", "pid 104")
    w = scan_watch.Watcher(trail, table=lambda: table + [(104, "Cherry Studio")],
                           connected=lambda: [cherry],
                           status=_status({"claude-code": True}))
    first = w.tick()
    assert sorted(f.name for f in first) == ["Cherry Studio", "Claude Code", "Codex"]
    assert w.tick() == []  # nothing new on the second pass

    rows = {agent: data for agent, data in _rows(db)}
    assert rows["Claude Code"]["state"] == "governed"
    assert rows["Codex"]["state"] == "reachable"
    assert rows["Cherry Studio"]["state"] == "ungoverned"
    assert rows["Cherry Studio"]["pid"] == 104
    assert trail.verify_chain() is None  # None: no break found


def test_a_reused_pid_is_recorded_again(tmp_path):
    trail = SQLiteAuditBackend(tmp_path / "a.db").load_trail()
    tables = iter([[(7, "claude")], [(8, "bash")], [(7, "codex")]])
    w = scan_watch.Watcher(trail, table=lambda: next(tables), connected=lambda: [],
                           status=_status({}))
    assert [f.name for f in w.tick()] == ["Claude Code"]
    assert w.tick() == []
    assert [f.name for f in w.tick()] == ["Codex"]


def test_run_passes(tmp_path):
    trail = SQLiteAuditBackend(tmp_path / "a.db").load_trail()
    slept = []
    w = scan_watch.Watcher(trail, table=lambda: [], connected=lambda: [], status=_status({}))
    w.run(5, passes=3, sleep=slept.append)
    assert slept == [5, 5]


def test_login_service_files(tmp_path):
    calls = []
    path, _msg = scan_watch.install("/opt/vaara/bin/vaara", interval=20, home=tmp_path,
                                    system="darwin", runner=lambda cmd, **kw: calls.append(cmd))
    plist = plistlib.loads(path.read_bytes())
    assert plist["Label"] == "io.vaara.scan-watch"
    assert plist["ProgramArguments"] == [
        "/opt/vaara/bin/vaara", "scan", "--watch", "--trail",
        str(tmp_path / ".vaara/trail/agents/audit.db"), "--interval", "20"]
    assert ["launchctl", "load", "-w", str(path)] in calls
    assert scan_watch.uninstall(home=tmp_path, system="darwin", runner=lambda *a, **k: None)
    assert not path.exists()

    path, _msg = scan_watch.install("/usr/bin/vaara", home=tmp_path, system="linux",
                                    runner=lambda cmd, **kw: None)
    assert "ExecStart=/usr/bin/vaara scan --watch" in path.read_text()
    assert scan_watch.install("x", home=tmp_path, system="win32")[0] is None
