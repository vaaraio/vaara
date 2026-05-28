#!/usr/bin/env python3
"""PreToolUse hook: two-layer governance for the proposed tool call.

Layer 1 — regex deny patterns from ``policies/default_deny.json``. Applied
to Bash, WebFetch, and WebSearch input fields. A match short-circuits to
a hard deny — no ML, no classifier load. Operators replace the file via
``VAARA_PLUGIN_DENY_PATTERNS_FILE``.

Layer 2 — Vaara classifier. For ``mcp__*`` tools only. MCP tools carry
the structured taxonomy Vaara's adaptive scorer is trained for; the
classifier output is meaningful there. Bash, WebFetch, WebSearch DO NOT
route through the ML classifier (documented baseline 2026-05-28: the
classifier is not trained on shell command strings; output on raw bash
is noise).

Exit 0 on allow / escalate (escalate writes a warning to stderr). Exit
2 on deny with the block reason on stderr.

Env vars: VAARA_PLUGIN_DISABLE, VAARA_PLUGIN_SHADOW,
VAARA_PLUGIN_AGENT_ID, VAARA_PLUGIN_AUDIT_DB,
VAARA_PLUGIN_DENY_PATTERNS_FILE.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _deny_patterns import load_deny_rules, match_deny_rule  # noqa: E402


def _emit(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _audit_db_path() -> Path:
    override = os.environ.get("VAARA_PLUGIN_AUDIT_DB")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".vaara" / "claude-code" / "audit.db"


def _append_deny_audit(
    tool_name: str, tool_input: dict, rule_id: str, message: str, agent_id: str
) -> None:
    try:
        from vaara.audit.sqlite_backend import SQLiteAuditBackend
        from vaara.pipeline import InterceptionPipeline
    except ImportError:
        return
    db_path = _audit_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    backend = SQLiteAuditBackend(db_path)
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    pipeline = InterceptionPipeline(trail=trail, enforce=False)
    try:
        pipeline.intercept(
            agent_id=agent_id,
            tool_name=tool_name,
            parameters=tool_input,
            context={
                "vaara_governance_layer": "deny_pattern",
                "rule_id": rule_id,
                "rule_message": message,
            },
        )
    except Exception:
        pass


def _classify_mcp(
    tool_name: str, tool_input: dict, agent_id: str, session_id: str, shadow: bool
) -> int:
    try:
        from vaara.audit.sqlite_backend import SQLiteAuditBackend
        from vaara.pipeline import InterceptionPipeline
    except ImportError:
        _emit(
            "vaara-governance: vaara package not importable. "
            "Run `pip install vaara>=0.40.1`. Passing through this MCP call."
        )
        return 0

    db_path = _audit_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    backend = SQLiteAuditBackend(db_path)
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    pipeline = InterceptionPipeline(trail=trail, enforce=not shadow)

    try:
        result = pipeline.intercept(
            agent_id=agent_id,
            tool_name=tool_name,
            parameters=tool_input,
            session_id=session_id,
        )
    except Exception as exc:
        _emit(f"vaara-governance: classifier failed ({exc!r}); passing through.")
        return 0

    if result.allowed:
        if result.decision == "escalate":
            _emit(
                f"vaara-governance: ESCALATE on {tool_name} "
                f"(risk {result.risk_score:.2f}, action_id={result.action_id}). "
                f"Reason: {result.reason}"
            )
        return 0

    _emit(
        f"vaara-governance: BLOCKED {tool_name} "
        f"(risk {result.risk_score:.2f}, action_id={result.action_id}). "
        f"Reason: {result.reason}"
    )
    return 2


def main() -> int:
    if os.environ.get("VAARA_PLUGIN_DISABLE") == "1":
        return 0

    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    tool_name = event.get("tool_name", "")
    tool_input = event.get("tool_input", {}) or {}
    if not isinstance(tool_input, dict):
        tool_input = {"_raw": tool_input}
    session_id = event.get("session_id", "")
    agent_id = os.environ.get("VAARA_PLUGIN_AGENT_ID", "claude-code")
    shadow = os.environ.get("VAARA_PLUGIN_SHADOW") == "1"

    rules = load_deny_rules()
    match = match_deny_rule(rules, tool_name, tool_input)
    if match is not None:
        rule_id, message = match
        _append_deny_audit(tool_name, tool_input, rule_id, message, agent_id)
        if shadow:
            _emit(
                f"vaara-governance: SHADOW deny on {tool_name} "
                f"(rule={rule_id}): {message}"
            )
            return 0
        _emit(f"vaara-governance: BLOCKED {tool_name} (rule={rule_id}). {message}")
        return 2

    if not tool_name.startswith("mcp__"):
        return 0

    return _classify_mcp(tool_name, tool_input, agent_id, session_id, shadow)


if __name__ == "__main__":
    raise SystemExit(main())
