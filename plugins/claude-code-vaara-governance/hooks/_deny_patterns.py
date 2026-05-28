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


def match_deny_rule(
    rules: list[dict], tool_name: str, tool_input: dict
) -> tuple[str, str] | None:
    """Return (rule_id, message) for the first matching rule, else None."""
    for rule in rules:
        if tool_name not in rule.get("tools", []):
            continue
        pattern = rule.get("pattern", "")
        if not pattern:
            continue
        try:
            regex = re.compile(pattern)
        except re.error:
            continue
        for field in rule.get("fields", []):
            value = tool_input.get(field, "")
            if not isinstance(value, str):
                continue
            if regex.search(value):
                return rule.get("id", "unknown"), rule.get("message", "deny rule matched")
    return None
