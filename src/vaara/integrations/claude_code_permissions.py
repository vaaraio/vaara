# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Claude Code's own deny rules, read so the hook does not ask in vain.

Claude Code evaluates ``permissions.deny`` in its settings files after the
PreToolUse hook, and a deny rule beats a hook's allow. On 2026-10-01 that
cost a human two approvals: both escalations were approved within seconds,
the hook printed an explicit allow, and Claude Code refused both calls on a
``Bash(rm *)`` rule in ``settings.local.json``. The question the human
answered never had a yes.

This module answers one narrow question: does a deny rule certainly refuse
this call? It reads the user and project settings and matches the rule
forms whose meaning is unambiguous: a bare tool name, an MCP server or
tool, and ``Bash(...)`` patterns against each subcommand of a compound
command. Path and domain specifiers are not matched. A miss is always
safe, because the caller then asks the human exactly as before.
"""

from __future__ import annotations

import fnmatch
import shlex
from pathlib import Path
from typing import Iterable, Optional

from vaara.integrations.hook_registration import _load, settings_paths

#: Shell operators that start a new subcommand. Claude Code checks each
#: subcommand of a compound command against the rules separately.
_OPERATORS = {"&&", "||", ";", "|", "&", ";;", "|&", "\n"}


def deny_rules(env: Optional[dict[str, str]] = None,
               cwd: Optional[str] = None) -> list[str]:
    """Every ``permissions.deny`` entry in the settings Claude Code reads."""
    paths = settings_paths(env)
    if cwd and not any(p.name == "settings.local.json" for p in paths):
        # Without CLAUDE_PROJECT_DIR, the event's cwd names the project.
        paths += [Path(cwd) / ".claude" / "settings.json",
                  Path(cwd) / ".claude" / "settings.local.json"]
    rules: list[str] = []
    for path in paths:
        permissions = _load(path).get("permissions")
        if not isinstance(permissions, dict):
            continue
        deny = permissions.get("deny")
        if isinstance(deny, list):
            rules += [r for r in deny if isinstance(r, str) and r.strip()]
    return rules


def subcommands(command: str) -> list[str]:
    """The simple commands of a shell line, quotes respected.

    An unparseable line (an unclosed quote) yields nothing, so no rule
    matches and the human is asked.
    """
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True,
                        punctuation_chars=True)
    lexer.whitespace_split = True
    parts: list[list[str]] = [[]]
    try:
        for token in lexer:
            if token in _OPERATORS or set(token) <= set("&|;"):
                parts.append([])
            else:
                parts[-1].append(token)
    except ValueError:
        return []
    return [" ".join(p) for p in parts if p]


def _bash_match(pattern: str, command: str) -> bool:
    if pattern.endswith(":*"):  # legacy prefix form
        prefix = pattern[:-2]
        return command == prefix or command.startswith(prefix + " ")
    return fnmatch.fnmatchcase(command, pattern)


def _matches(rule: str, tool_name: str, tool_input: dict) -> bool:
    rule = rule.strip()
    if "(" not in rule:
        if rule == tool_name:
            return True
        # mcp__server denies every tool of that server.
        return (rule.startswith("mcp__") and rule.count("__") == 1
                and tool_name.startswith(rule + "__"))
    name, _, spec = rule.partition("(")
    if not spec.endswith(")") or name != tool_name:
        return False
    spec = spec[:-1]
    if spec in ("", "*"):
        return True
    if tool_name != "Bash":
        return False  # path and domain specifiers: not matched, ask instead
    command = tool_input.get("command")
    if not isinstance(command, str):
        return False
    return any(_bash_match(spec, sub) for sub in subcommands(command))


def matching_deny_rule(tool_name: str, tool_input: dict,
                       rules: Iterable[str]) -> Optional[str]:
    """The first deny rule that certainly refuses this call, else None."""
    for rule in rules:
        try:
            if _matches(rule, tool_name, tool_input):
                return rule
        except Exception:
            continue
    return None
