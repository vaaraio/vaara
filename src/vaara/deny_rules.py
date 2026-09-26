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
    return _with_codex_home(rules) if isinstance(rules, list) else []


#: The rules that protect harness configuration by path. Their patterns name
#: ``.codex/``; Codex reads its hooks from ``$CODEX_HOME`` when that is set.
_HARNESS_RULES = ("harness_config_write", "harness_config_shell_write",
                  "interpreter_config_write")


def _with_codex_home(rules: list) -> list:
    """Extend the harness rules to a ``$CODEX_HOME`` outside ``~/.codex``.

    Codex runs Vaara's hook from ``$CODEX_HOME/hooks.json``, and the hook
    process inherits that variable, so a Codex home anywhere else would
    otherwise leave the file that installs the gate unprotected.
    """
    home = os.environ.get("CODEX_HOME", "").rstrip("/")
    if not home or Path(home).name == ".codex":
        return rules
    # Each rule names the directory as `\.codex/` inside its own structure
    # (a write verb, an interpreter and a write call), so the directory is
    # widened in place and every other condition still applies.
    either = r"(?:\.codex/|" + re.escape(home + "/") + ")"
    out = []
    for rule in rules:
        if (isinstance(rule, dict) and rule.get("id") in _HARNESS_RULES
                and r"\.codex/" in str(rule.get("pattern", ""))):
            rule = {**rule, "pattern": rule["pattern"].replace(r"\.codex/", either)}
        out.append(rule)
    return out


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
#: ``packages/core/src/tools/definitions/base-declarations.ts``, and the tool
#: list Copilot CLI 1.0.88 sends its model.
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
    # Copilot CLI
    "bash": _rename("Bash", command="command"),
    "create": _rename("Write", file_path="path", content="file_text"),
    "edit": _rename("Edit", file_path="path", new_string="new_str"),
    "view": _rename("Read", file_path="path"),
    "task": _rename("Agent"),
    "skill": _rename("Skill", skill="skill"),
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


def _string_leaves(value: Any, depth: int = 0, key: str = ""):
    """Every string in a JSON-shaped value with the dict key it sits under,
    nested dicts and lists included. A list item keeps its list's key."""
    if depth > 8:
        return
    if isinstance(value, str):
        yield key, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _string_leaves(v, depth + 1, str(k))
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _string_leaves(v, depth + 1, key)


#: Rule fields that hold a file path. A rule reading only these is a path
#: rule: its pattern describes a path, not a command or written content.
_PATH_FIELDS = frozenset({"file_path", "notebook_path", "uri"})

_PATH_KEY = re.compile(r"path|file|dir|dest|target|uri|location", re.I)


def _path_shaped(key: str, text: str) -> bool:
    """An argument that is a path: under a path-like key, or one token."""
    return bool(_PATH_KEY.search(key)) or not any(c.isspace() for c in text)


#: Rule fields that hold a shell command. A rule reading only these is a
#: shell rule: its pattern describes a command line, not prose about one.
_SHELL_FIELDS = frozenset({"command"})

_COMMAND_KEY = re.compile(
    r"command|cmd|script|shell|exec|argv|^args?$|^code$|^input$", re.I)

#: A line read as a command line: env assignments, then a lowercase command
#: name, then arguments. A word ending in sentence punctuation or starting
#: with a capital marks prose ("The refused call was: ...").
_COMMAND_LINE = re.compile(
    r"\s*(?:[A-Za-z_]\w*=\S*\s+)*[a-z0-9_./~-]+"
    r"(?:\s+(?![A-Z][a-z])\S*[^\s:,.!?])*\s*")


def _shell_hit(regex: "re.Pattern[str]", key: str, text: str) -> bool:
    """A shell rule hit on an argument that is a command: under a
    command-like key, or a one-line string shaped like a command line
    up to the match."""
    if _COMMAND_KEY.search(key):
        return bool(regex.search(text))
    if "\n" in text.strip():
        return False
    m = regex.search(text)
    if not m:
        return False
    # Judge only the command the match sits in: the text after the last
    # shell separator before it.
    lead = re.split(r";|&&|\|\||\||\$\(|`", text[:m.start()])[-1]
    return lead.strip() == "" or bool(_COMMAND_LINE.fullmatch(lead))


def match_deny_rule_any_field(
    rules: list[dict], tool_input: dict
) -> Optional[tuple[str, str]]:
    """First (rule_id, message) whose pattern matches any string argument.

    Tool name is ignored. ``match_any`` rules are skipped: without a known
    tool name there is nothing for them to name. A path rule reads only
    path-shaped arguments: run over every string, it fired on a note that
    merely mentioned a harness config file (2026-09-23, a memory save refused
    as ``harness_config_write``). A shell rule reads only command-shaped
    arguments, for the same reason: a memory note describing a destructive
    command was refused as ``rm_rf_root`` (2026-09-24). Written-content rules
    read every string.
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
        fields = rule.get("fields") or []
        path_rule = bool(fields) and set(fields) <= _PATH_FIELDS
        shell_rule = bool(fields) and set(fields) <= _SHELL_FIELDS
        for key, text in leaves:
            if path_rule and not _path_shaped(key, text):
                continue
            if shell_rule:
                if _shell_hit(regex, key, text):
                    return rule.get("id", "unknown"), rule.get("message", "deny rule matched")
                continue
            if regex.search(text):
                return rule.get("id", "unknown"), rule.get("message", "deny rule matched")
    return None
