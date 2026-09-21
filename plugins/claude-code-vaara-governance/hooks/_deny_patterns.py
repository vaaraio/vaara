# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layer-1 regex deny-pattern matching for the Vaara Claude Code plugin.

Loaded by ``pre_tool_use.py`` BEFORE the Vaara ML classifier. Matches
against tool input fields. Deny patterns are JSON, not YAML, to avoid
a pyyaml runtime dep on hook invocation.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


def _emit(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def deny_patterns_path() -> Path:
    override = os.environ.get("VAARA_PLUGIN_DENY_PATTERNS_FILE")
    if override:
        return Path(override).expanduser()
    plugin_root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if plugin_root:
        return Path(plugin_root) / "policies" / "default_deny.json"
    return Path(__file__).parent.parent / "policies" / "default_deny.json"


def load_deny_rules() -> list[dict]:
    path = deny_patterns_path()
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _emit(f"vaara-governance: deny-patterns load failed ({exc!r}); skipping layer 1.")
        return []
    return doc.get("rules", [])


def _rule_lifted(rule: dict) -> bool:
    """A rule names ``unless_env``; that variable set to 1 lifts it.

    This is how an operator exception is expressed: no subagents, unless
    this named job says so. The lift is still recorded by the caller.
    """
    name = rule.get("unless_env", "")
    return bool(name) and os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _field_text(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)):
        return json.dumps(value)
    return None


def match_deny_rule(
    rules: list[dict], tool_name: str, tool_input: dict
) -> tuple[str, str] | None:
    """Return (rule_id, message) for the first matching rule, else None.

    ``match_any`` rules fire on any call to a listed tool. Booleans and
    numbers in the input are matched as their JSON text, so a rule can
    say ``durable`` must not be ``true``.
    """
    for rule in rules:
        if tool_name not in rule.get("tools", []):
            continue
        if _rule_lifted(rule):
            continue
        if rule.get("match_any"):
            return rule.get("id", "unknown"), rule.get("message", "deny rule matched")
        pattern = rule.get("pattern", "")
        if not pattern:
            continue
        try:
            regex = re.compile(pattern)
        except re.error:
            continue
        for field in rule.get("fields", []):
            value = _field_text(tool_input.get(field, ""))
            if value is None:
                continue
            if regex.search(value):
                return rule.get("id", "unknown"), rule.get("message", "deny rule matched")
    return None
