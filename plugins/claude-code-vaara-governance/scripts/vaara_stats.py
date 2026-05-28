#!/usr/bin/env python3
"""Vaara governance stats for the Claude Code audit trail.

Reads ~/.vaara/claude-code/audit.db (or VAARA_PLUGIN_AUDIT_DB) and prints
a summary: total records, counts by event type, top tools, last 5 actions.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path


def _audit_db_path() -> Path:
    override = os.environ.get("VAARA_PLUGIN_AUDIT_DB")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".vaara" / "claude-code" / "audit.db"


def main() -> int:
    db_path = _audit_db_path()
    if not db_path.exists():
        print(f"vaara-stats: audit DB not found at {db_path}", file=sys.stderr)
        print("SessionStart will create it on the next Claude Code restart.", file=sys.stderr)
        return 1

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    tables = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
    if "audit_records" not in tables:
        print(f"vaara-stats: unexpected schema (tables: {sorted(tables)})", file=sys.stderr)
        return 2

    total = cur.execute("select count(*) from audit_records").fetchone()[0]
    print(f"audit_db: {db_path}")
    print(f"records:  {total}")
    if total == 0:
        print("(no records yet, make a tool call to populate)")
        con.close()
        return 0

    print()
    print("by event_type:")
    by_type = cur.execute(
        "select event_type, count(*) from audit_records group by event_type order by 2 desc"
    ).fetchall()
    for et, n in by_type:
        print(f"  {et:24s} {n}")

    print()
    print("top tools:")
    by_tool = cur.execute(
        "select tool_name, count(*) from audit_records "
        "where tool_name != '' group by tool_name order by 2 desc limit 10"
    ).fetchall()
    for tool, n in by_tool:
        print(f"  {tool:24s} {n}")

    print()
    print("last 5 records:")
    rows = cur.execute(
        "select event_type, agent_id, tool_name, timestamp "
        "from audit_records order by timestamp desc limit 5"
    ).fetchall()
    for et, agent, tool, ts in rows:
        print(f"  [{ts:.3f}] {et:20s} agent={agent} tool={tool}")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
