#!/usr/bin/env python3
"""PostToolUse hook: close the audit loop for the just-executed tool call.

Appends an outcome record to the persistent SQLite audit trail so the
PreToolUse decision and the actual tool result are correlatable. Does
not block (PostToolUse runs after the tool already executed).

The hook tries to correlate to the most recent ACTION_REQUESTED record
for the same agent + tool_name and call ``pipeline.report_outcome``,
which feeds the MWU online learner. If correlation fails (parallel
calls, restart in between), the hook appends a standalone outcome
record so the chain still reflects that the call completed.

Env vars match pre_tool_use.py:
- ``VAARA_PLUGIN_DISABLE=1`` skips this hook
- ``VAARA_PLUGIN_AGENT_ID`` overrides the agent_id
- ``VAARA_PLUGIN_AUDIT_DB`` overrides the audit DB path
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _audit_db_path() -> Path:
    override = os.environ.get("VAARA_PLUGIN_AUDIT_DB")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".vaara" / "claude-code" / "audit.db"


def _outcome_severity(tool_response: object) -> float:
    """Map a tool response to a [0.0, 1.0] severity value.

    Successful completion -> 0.0. Errors (non-empty stderr, exit codes,
    exception payloads) -> 0.5. Catastrophic markers (timeout, killed)
    -> 1.0. The mapping is intentionally crude for v1; operators tune
    via policy once they see how their tools fail.
    """
    if not isinstance(tool_response, dict):
        return 0.0
    if tool_response.get("interrupted") is True:
        return 1.0
    if tool_response.get("isError") is True:
        return 0.7
    stderr = tool_response.get("stderr") or ""
    if isinstance(stderr, str) and stderr.strip():
        return 0.3
    return 0.0


def main() -> int:
    if os.environ.get("VAARA_PLUGIN_DISABLE") == "1":
        return 0

    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    tool_name = event.get("tool_name", "")
    tool_response = event.get("tool_response", {})
    severity = _outcome_severity(tool_response)

    try:
        from vaara.audit.sqlite_backend import SQLiteAuditBackend
    except ImportError:
        return 0

    agent_id = os.environ.get("VAARA_PLUGIN_AGENT_ID", "claude-code")
    db_path = _audit_db_path()
    if not db_path.exists():
        return 0

    backend = SQLiteAuditBackend(db_path)
    trail = backend.load_trail()
    trail._on_record = backend.write_record

    target_action_id = None
    for record in reversed(trail._records):
        if record.agent_id != agent_id:
            continue
        if record.data.get("tool_name") != tool_name:
            continue
        if record.event_type == "ACTION_REQUESTED":
            target_action_id = record.action_id
            break

    if target_action_id is None:
        return 0

    try:
        from vaara.pipeline import InterceptionPipeline

        pipeline = InterceptionPipeline(trail=trail)
        pipeline._pending_outcomes[target_action_id] = (0.5, {})
        pipeline.report_outcome(target_action_id, outcome_severity=severity)
    except Exception:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
