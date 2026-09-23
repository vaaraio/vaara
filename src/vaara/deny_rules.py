# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layer-1 deny rules, shared by every surface Vaara mediates.

One rule file, ``integrations/claude_code_deny.json``, governs the Claude
Code hook and the MCP proxy alike. The rules are regexes over tool input,
applied before any classifier, and a match is a hard deny with a named
rule on the record. Operators replace the file through
``VAARA_PLUGIN_DENY_PATTERNS_FILE``.

Two ways to apply the same rules:

- ``match_deny_rule``: by tool name. A rule lists the tools it governs and
  the input fields it reads. This is the Claude Code shape, where tool
  names are fixed (``Bash``, ``Write``, ``Agent``). Codex and Gemini CLI
  name the same operations differently; ``HARNESS_ALIASES`` translates
  their tool names and inputs into this shape before matching.
- ``match_deny_rule_any_field``: by content, ignoring the tool name. An
  MCP server names its tools whatever it likes, so the proxy cannot know
  that ``run_command`` is a shell or ``put_file`` is a write. Every rule
  with a pattern is run over every string argument instead. Rules marked
  ``match_any`` are tool-name policy and do not apply here.

Rule keys: ``id``, ``tools``, ``fields``, ``pattern``, ``message``,
optional ``match_any`` (fire on any call to the listed tools),
``unless_env`` (a variable that, set to 1, lifts the rule for a deliberate
exception) and ``any_field: false`` (the pattern reads one named field and
must not run over arbitrary arguments). Booleans and numbers in the input match as their JSON text.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

BUNDLED = Path(__file__).parent / "integrations" / "claude_code_deny.json"

_TRUE = ("1", "true", "yes")


def deny_rules_path(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        return Path(explicit).expanduser()
    override = os.environ.get("VAARA_PLUGIN_DENY_PATTERNS_FILE")
    if override:
        return Path(override).expanduser()
    plugin_root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if plugin_root:
        candidate = Path(plugin_root) / "policies" / "default_deny.json"
        if candidate.exists():
            return candidate
    return BUNDLED if BUNDLED.exists() else None


def load_deny_rules(explicit: Optional[str] = None) -> list[dict]:
    path = deny_rules_path(explicit)
    if path is None or not path.exists():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rules = doc.get("rules", [])
    return rules if isinstance(rules, list) else []


def rule_lifted(rule: dict) -> bool:
    """A rule names ``unless_env``; that variable set to 1 lifts it."""
    name = rule.get("unless_env", "")
    return bool(name) and os.environ.get(name, "").strip().lower() in _TRUE


def field_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)):
        return json.dumps(value)
    return None


_PATCH_PATH = re.compile(
    r"^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+?)\s*$", re.M)


def _codex_apply_patch(tool_input: dict) -> list[tuple[str, dict]]:
    """A Codex patch as one Write and one Edit per file it touches.

    Codex hands hooks the raw patch text under ``command``. The file rules
    read ``file_path`` and the written text, so each path named in the patch
    header is checked with the whole patch as its content.
    """
    patch = field_text(tool_input.get("command", "")) or ""
    paths = _PATCH_PATH.findall(patch) or [""]
    out: list[tuple[str, dict]] = []
    for path in paths:
        out.append(("Write", {"file_path": path, "content": patch}))
        out.append(("Edit", {"file_path": path, "new_string": patch}))
    return out


def _rename(target: str, **fields: str):
    """Translate to ``target``, copying input ``src`` to rule field ``dst``."""
    def translate(tool_input: dict) -> list[tuple[str, dict]]:
        if not fields:
            return [(target, tool_input)]
        return [(target, {dst: tool_input.get(src, "")
                          for dst, src in fields.items()})]
    return translate


def _gemini_read_many(tool_input: dict) -> list[tuple[str, dict]]:
    include = tool_input.get("include") or []
    if isinstance(include, str):
        include = [include]
    return [("Read", {"file_path": p}) for p in include if isinstance(p, str)]


#: Other harnesses' tool names, translated into the Claude Code tool and
#: input shape the rules are written in. Checked against the harnesses' own
#: sources: Codex ``codex-rs/core/src/tools`` (hooks already receive its shell
#: tools as ``Bash`` with ``command``), Gemini CLI
#: ``packages/core/src/tools/definitions/base-declarations.ts``.
HARNESS_ALIASES = {
    # Codex
    "apply_patch": _codex_apply_patch,
    "spawn_agent": _rename("Agent"),
    # Gemini CLI
    "run_shell_command": _rename("Bash", command="command"),
    "write_file": _rename("Write", file_path="file_path", content="content"),
    "replace": _rename("Edit", file_path="file_path",
                       new_string="new_string"),
    "read_file": _rename("Read", file_path="file_path"),
    "read_many_files": _gemini_read_many,
    "web_fetch": _rename("WebFetch", url="prompt"),
    "read_mcp_resource": _rename("ReadMcpResourceTool", uri="uri"),
    "activate_skill": _rename("Skill", skill="name"),
}


def _compiled(rule: dict) -> Optional[re.Pattern[str]]:
    pattern = rule.get("pattern", "")
    if not pattern:
        return None
    try:
        return re.compile(pattern)
    except re.error:
        return None


def match_deny_rule(
    rules: list[dict], tool_name: str, tool_input: dict
) -> Optional[tuple[str, str]]:
    """First (rule_id, message) whose tool list names ``tool_name``, else None.

    A tool no rule names, but which ``HARNESS_ALIASES`` knows, is matched
    as the Claude Code tool it translates to.
    """
    named = any(tool_name in rule.get("tools", []) for rule in rules)
    if not named and tool_name in HARNESS_ALIASES:
        for target, translated in HARNESS_ALIASES[tool_name](tool_input or {}):
            hit = _match_named(rules, target, translated)
            if hit is not None:
                return hit
        return None
    return _match_named(rules, tool_name, tool_input)


def _match_named(
    rules: list[dict], tool_name: str, tool_input: dict
) -> Optional[tuple[str, str]]:
    for rule in rules:
        if tool_name not in rule.get("tools", []):
            continue
        if rule_lifted(rule):
            continue
        if rule.get("match_any"):
            return rule.get("id", "unknown"), rule.get("message", "deny rule matched")
        regex = _compiled(rule)
        if regex is None:
            continue
        for field in rule.get("fields", []):
            value = field_text(tool_input.get(field, ""))
            if value is not None and regex.search(value):
                return rule.get("id", "unknown"), rule.get("message", "deny rule matched")
    return None


def _string_leaves(value: Any, depth: int = 0):
    """Every string in a JSON-shaped value, nested dicts and lists included."""
    if depth > 8:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _string_leaves(v, depth + 1)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _string_leaves(v, depth + 1)


def match_deny_rule_any_field(
    rules: list[dict], tool_input: dict
) -> Optional[tuple[str, str]]:
    """First (rule_id, message) whose pattern matches any string argument.

    Tool name is ignored. ``match_any`` rules are skipped: without a known
    tool name there is nothing for them to name.
    """
    leaves = list(_string_leaves(tool_input))
    if not leaves:
        return None
    for rule in rules:
        if rule.get("match_any") or rule.get("any_field") is False or rule_lifted(rule):
            continue
        regex = _compiled(rule)
        if regex is None:
            continue
        for text in leaves:
            if regex.search(text):
                return rule.get("id", "unknown"), rule.get("message", "deny rule matched")
    return None
