"""The shadow report: what would have been blocked, from the trail alone.

Shadow mode (``InterceptionPipeline(enforce=False)``, or ``--shadow`` on the
MCP proxy) records the true decision for every call while allowing all of
them, so an operator can run Vaara in front of live traffic without breaking
anything. This module answers the question that run exists to answer: over
the last N days, what would enforcement have done?

The report reads the audit DB directly and needs nothing beyond the standard
library. In an enforcing deployment the same numbers describe what actually
was blocked, since the trail records decisions identically in both modes.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

_DECISION_QUERY = """
SELECT event_type, tool_name, agent_id, data
FROM audit_records
WHERE timestamp >= ?
  AND (event_type = 'action_blocked'
       OR (event_type = 'decision_made'))
"""


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def shadow_report(db_path: Union[str, Path], days: int = 7) -> dict[str, Any]:
    """Summarise decisions over the trailing window, grouped by tool.

    Returns a JSON-serialisable dict. ``would_block`` counts deny decisions
    (stored as ``action_blocked`` records regardless of enforcement mode),
    ``would_escalate`` counts escalate decisions.
    """
    now = time.time()
    cutoff = now - days * 86400.0
    uri = f"file:{Path(db_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(_DECISION_QUERY, (cutoff,)).fetchall()
    finally:
        conn.close()

    allowed = 0
    escalated: dict[str, dict[str, Any]] = {}
    blocked: dict[str, dict[str, Any]] = {}

    for event_type, tool_name, agent_id, data_json in rows:
        try:
            data = json.loads(data_json)
        except (TypeError, ValueError):
            data = {}
        decision = data.get("decision", "")
        risk = data.get("risk_score")
        reason = data.get("reason", "")

        if event_type == "action_blocked":
            bucket = blocked
        elif decision == "escalate":
            bucket = escalated
        else:
            allowed += 1
            continue

        entry = bucket.setdefault(tool_name, {
            "tool_name": tool_name,
            "count": 0,
            "agents": set(),
            "max_risk": None,
            "sample_reason": reason,
        })
        entry["count"] += 1
        entry["agents"].add(agent_id)
        if isinstance(risk, (int, float)):
            entry["max_risk"] = risk if entry["max_risk"] is None else max(entry["max_risk"], risk)

    def _finalise(bucket: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for entry in sorted(bucket.values(), key=lambda e: (-e["count"], e["tool_name"])):
            entry["agents"] = sorted(entry["agents"])
            out.append(entry)
        return out

    blocked_list = _finalise(blocked)
    escalated_list = _finalise(escalated)
    would_block = sum(e["count"] for e in blocked_list)
    would_escalate = sum(e["count"] for e in escalated_list)

    return {
        "window_days": days,
        "since": _iso(cutoff),
        "until": _iso(now),
        "total_decisions": allowed + would_block + would_escalate,
        "allowed": allowed,
        "would_escalate": would_escalate,
        "would_block": would_block,
        "blocked": blocked_list,
        "escalated": escalated_list,
    }


def render_text(report: dict[str, Any]) -> str:
    """Plain-text rendering of a shadow report for terminals and emails."""
    lines = [
        f"Vaara shadow report, last {report['window_days']} day(s)",
        f"  window   {report['since']} .. {report['until']}",
        f"  decisions {report['total_decisions']}: "
        f"{report['allowed']} allowed, "
        f"{report['would_escalate']} would have escalated, "
        f"{report['would_block']} would have been blocked",
    ]
    if report["blocked"]:
        lines.append("")
        lines.append("Would have been blocked:")
        for entry in report["blocked"]:
            risk = f" max_risk={entry['max_risk']:.2f}" if entry["max_risk"] is not None else ""
            agents = ", ".join(entry["agents"])
            lines.append(f"  {entry['count']:4d}x {entry['tool_name']}{risk}  agents: {agents}")
            if entry["sample_reason"]:
                lines.append(f"        reason: {entry['sample_reason']}")
    if report["escalated"]:
        lines.append("")
        lines.append("Would have escalated to human review:")
        for entry in report["escalated"]:
            risk = f" max_risk={entry['max_risk']:.2f}" if entry["max_risk"] is not None else ""
            lines.append(f"  {entry['count']:4d}x {entry['tool_name']}{risk}")
    if not report["blocked"] and not report["escalated"]:
        lines.append("")
        lines.append("Nothing would have been blocked or escalated in this window.")
    return "\n".join(lines)
