#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vaara governance stats for the Claude Code audit trail.

Reads the trail the hooks write to, resolved the way the hooks resolve it:
VAARA_PLUGIN_AUDIT_DB, then the ``audit_db`` key in the plugin config (which
``vaara init`` points at ~/.vaara/trail/audit.db), then
~/.vaara/claude-code/audit.db. Prints total records, counts by event type,
top tools and the last 5 actions.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


def _audit_db_path() -> Path:
    """The hooks' own resolution, so stats read the trail the hooks write.

    This used to stop at the environment override and the legacy default, so
    after ``vaara init`` pointed the hooks at the unified trail, stats kept
    reading the old file and reported it missing or stale.
    """
    hooks = Path(__file__).resolve().parent.parent / "hooks"
    sys.path.insert(0, str(hooks))
    try:
        import _config  # type: ignore[import-not-found]
    finally:
        sys.path.remove(str(hooks))
    return _config.audit_db_path(_config.load_config())


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
