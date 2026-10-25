"""A trail is read back as it was written.

The backend once carried a read-time substitution of agent ids, keyed on a
``gdpr_redactions`` row. Nothing in the tree called it, and a presented
record whose agent id differs from the stored one cannot be recomputed, so
every verifier reported a hash mismatch on a file nobody had touched. The
substitution is gone, and a new trail no longer creates the table. A file an
older version wrote still carries it and opens unchanged: the row is never
read and never dropped.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.pipeline import InterceptionPipeline


def _build(path: Path) -> None:
    backend = SQLiteAuditBackend(path)
    pipeline = InterceptionPipeline(trail=backend.load_trail())
    for i in range(3):
        pipeline.intercept(agent_id="agent-7", tool_name="file.read",
                           parameters={"path": f"/srv/{i}.txt"})
    backend.close()


def _tables(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()


def test_backend_has_no_redaction_api():
    assert not hasattr(SQLiteAuditBackend, "redact_agent_pii")
    assert not hasattr(SQLiteAuditBackend, "list_redactions")


def test_a_new_trail_has_no_redaction_table(tmp_path):
    path = tmp_path / "trail.db"
    _build(path)
    assert "gdpr_redactions" not in _tables(path)


def test_a_redaction_row_left_by_an_older_version_changes_nothing(tmp_path):
    path = tmp_path / "trail.db"
    _build(path)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS gdpr_redactions ("
        "original_id TEXT PRIMARY KEY, replacement TEXT NOT NULL, "
        "redacted_at REAL NOT NULL)")
    conn.execute(
        "INSERT INTO gdpr_redactions (original_id, replacement, redacted_at) "
        "VALUES (?, ?, ?)", ("agent-7", "[REDACTED]", time.time()))
    conn.commit()
    conn.close()

    backend = SQLiteAuditBackend(path)
    try:
        trail = backend.load_trail()
        records = trail.get_agent_records("agent-7")
        assert records, "the trail was written"
        assert {r.agent_id for r in records} == {"agent-7"}
        assert trail.verify_chain() is None
    finally:
        backend.close()

    assert "gdpr_redactions" in _tables(path), "an older file keeps what it had"
    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT original_id FROM gdpr_redactions").fetchall()
    conn.close()
    assert rows == [("agent-7",)]
