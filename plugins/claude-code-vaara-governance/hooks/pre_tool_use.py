#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PreToolUse hook: two-layer governance for the proposed tool call.

Layer 1 — regex deny patterns from ``policies/default_deny.json``. Applied
to shell and web input fields (Bash, WebFetch, WebSearch) and to file
mutation (Write, Edit, NotebookEdit), matching both the target path and
the content being written: an agent that cannot run ``curl | sh`` can
still write it to a file. A match short-circuits to a hard deny — no ML,
no classifier load. Operators replace the file via
``VAARA_PLUGIN_DENY_PATTERNS_FILE``.

Layer 2 — Vaara classifier. For ``mcp__*`` tools only. MCP tools carry
the structured taxonomy Vaara's adaptive scorer is trained for; the
classifier output is meaningful there. Bash, WebFetch, WebSearch DO NOT
route through the ML classifier (documented baseline 2026-05-28: the
classifier is not trained on shell command strings; output on raw bash
is noise).

Exit 0 on allow. Exit 2 on deny and on escalate alike, each with its own
reason on stderr: escalate holds the call for review rather than letting
it proceed unreviewed.

Env vars: VAARA_PLUGIN_DISABLE, VAARA_PLUGIN_SHADOW,
VAARA_PLUGIN_AGENT_ID, VAARA_PLUGIN_AUDIT_DB,
VAARA_PLUGIN_DENY_PATTERNS_FILE.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _config  # noqa: E402
from _deny_patterns import load_deny_rules, match_deny_rule  # noqa: E402
from _notify import notify as _raw_notify  # noqa: E402

CFG = _config.load_config()


def notify(verdict: str, tool_name: str, detail: str) -> None:
    if _config.notifications_enabled(CFG):
        _raw_notify(verdict, tool_name, detail)


def _emit(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _audit_db_path() -> Path:
    return _config.audit_db_path(CFG)


def _record_call(
    tool_name: str,
    tool_input: dict,
    agent_id: str,
    context: dict,
    session_id: str = "",
) -> None:
    """Write an ACTION_REQUESTED record for a call on the regex path.

    Every matched call is recorded, not only the denied ones. Recording
    denials alone left allowed shell, web and file calls out of the trail
    entirely, and PostToolUse then found no ACTION_REQUESTED to correlate
    against and wrote no outcome either. The audit trail therefore held
    MCP calls and blocks, and nothing else, while the plugin described it
    as covering shell and web too.

    ``enforce=False`` so this never changes the verdict: the decision on
    this path belongs to the deny rules, and scoring here is for the
    record only. Measured at roughly 5 ms to construct and 0.6 ms per
    call, which is the SQLite append, not the ML classifier: that stays
    on the ``mcp__*`` path.
    """
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
            context=context,
            session_id=session_id,
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
        # Fail CLOSED in protect mode: a gate that waves everything through
        # when its engine is missing is not a gate. Shadow mode records
        # nothing it can't score, so passing through there is honest; and
        # `"fail_open": true` in config.json is the documented escape hatch
        # for operators who prefer availability over enforcement.
        if shadow or _config.fail_open(CFG):
            _emit(
                "vaara-governance: vaara package not importable. "
                "Run `pip install vaara>=0.40.1`. Passing through this MCP call."
            )
            return 0
        _emit(
            "vaara-governance: BLOCKED (fail-closed): the vaara package is "
            "not importable, so this MCP call cannot be scored or recorded. "
            "Run `pip install vaara>=0.40.1`, or set \"fail_open\": true in "
            "~/.vaara/claude-code/config.json to pass through unscored."
        )
        notify("BLOCKED", tool_name, "vaara not installed; failing closed")
        return 2

    db_path = _audit_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    backend = SQLiteAuditBackend(db_path)
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    pipeline = InterceptionPipeline(trail=trail, enforce=not shadow)

    preset = _config.protection_preset(CFG)
    custom = _config.custom_thresholds(CFG)
    if preset or custom:
        try:
            from vaara.policy import from_dict
            from vaara.policy.modes import get_mode, to_policy_dict

            policy = to_policy_dict(get_mode(preset or "balanced"))
            if custom:
                escalate, deny = custom
                policy["thresholds"]["default"] = {
                    "escalate": escalate, "deny": deny,
                }
            pipeline.scorer.apply_policy(from_dict(policy))
        except Exception as exc:
            _emit(
                f"vaara-governance: policy (preset={preset!r}, "
                f"custom_thresholds={custom!r}) not applied ({exc}); "
                f"using default thresholds."
            )

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
        return 0

    # `allowed` is `decision == "allow"`, so escalate lands here, not in the
    # branch above. It used to be reported as a flat BLOCK, which told the
    # operator a policy denied the call when in truth it needed review.
    # Escalate still stops the call, because a gate that proceeds while
    # waiting for a human is not a gate.
    if result.decision == "escalate":
        _emit(
            f"vaara-governance: ESCALATE on {tool_name} — held pending review "
            f"(risk {result.risk_score:.2f}, action_id={result.action_id}). "
            f"Reason: {result.reason}"
        )
        notify(
            "ESCALATE", tool_name, f"risk {result.risk_score:.2f}: {result.reason}"
        )
        return 2

    _emit(
        f"vaara-governance: BLOCKED {tool_name} "
        f"(risk {result.risk_score:.2f}, action_id={result.action_id}). "
        f"Reason: {result.reason}"
    )
    notify("BLOCKED", tool_name, f"risk {result.risk_score:.2f}: {result.reason}")
    return 2


def main() -> int:
    if _config.plugin_disabled(CFG):
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
    agent_id = _config.agent_id(CFG)
    shadow = _config.shadow_mode(CFG)

    rules = load_deny_rules()
    match = match_deny_rule(rules, tool_name, tool_input)
    if match is not None:
        rule_id, message = match
        _record_call(
            tool_name, tool_input, agent_id,
            {
                "vaara_governance_layer": "deny_pattern",
                "rule_id": rule_id,
                "rule_message": message,
            },
            session_id,
        )
        if shadow:
            _emit(
                f"vaara-governance: SHADOW deny on {tool_name} "
                f"(rule={rule_id}): {message}"
            )
            notify("SHADOW deny", tool_name, message)
            return 0
        _emit(f"vaara-governance: BLOCKED {tool_name} (rule={rule_id}). {message}")
        notify("BLOCKED", tool_name, message)
        return 2

    if not tool_name.startswith("mcp__"):
        # Passed the deny rules. Record it anyway: an audit trail that
        # holds only the blocked calls cannot answer "what did the agent
        # do", which is the question the trail exists for. This also
        # gives PostToolUse an ACTION_REQUESTED to correlate its outcome
        # against, so the call and its result both land.
        _record_call(
            tool_name, tool_input, agent_id,
            {"vaara_governance_layer": "regex_pass"},
            session_id,
        )
        return 0

    return _classify_mcp(tool_name, tool_input, agent_id, session_id, shadow)


if __name__ == "__main__":
    raise SystemExit(main())
