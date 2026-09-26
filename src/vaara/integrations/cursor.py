# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cursor governance: the native hooks, and the mapping the hook runner needs.

Cursor runs a ``preToolUse`` hook before every agent tool call, shell, file
and MCP alike, and blocks the call on exit 2 or on ``"permission": "deny"``.
``vaara init`` writes ``vaara hook pre-tool-use --client cursor`` into
``~/.cursor/hooks.json`` with ``failClosed``: Cursor's default for a hook
that crashes or times out is to let the action through.

Cursor treats a permission hook's stdout as its verdict, and output that is
not the expected JSON blocks the action, so a silent allow blocks. The
runner prints ``{"permission": "allow"}`` or a deny with Vaara's reason
(:func:`render_pre`).

Cursor also imports hooks from ``~/.claude/settings.json`` by default, which
is where ``vaara init`` writes the Claude Code hooks, so the same binary can
be called twice for one Cursor call. :func:`is_cursor_event` lets the runner
recognise a Cursor payload whichever way it arrives. With the native hook in
place the imported one steps aside; without it, the imported one decides
with this mapping, so a machine whose ``hooks.json`` was never written is
still governed.

Cursor names tools ``Shell``, ``Read``, ``Write``, ``Grep``, ``Delete``,
``Task`` and ``MCP:<tool>``. :func:`to_hook_events` maps them onto the names
the deny rules use. Cursor documents the Shell arguments (``command``,
``working_directory``) and not the file tools', so the path and content are
taken from every field name those tools are known to use.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

AGENT_ID = "cursor"

#: Hook entries carry this in their command, as the Claude Code ones do.
_MARKER = "vaara hook "

#: Longer than the approval handshake's default 60 s wait, so an escalation
#: held for a human is answered before Cursor gives up on the hook.
HOOK_TIMEOUT = 90

_TOOLS: dict[str, str] = {
    "Shell": "Bash",
    "Read": "Read",
    "Write": "Write",
    "Edit": "Edit",
    # A delete is a write that leaves nothing: the rules that protect a file
    # from being rewritten protect it from being removed.
    "Delete": "Write",
    "Grep": "Grep",
    "Glob": "Glob",
    "Task": "Task",
    "WebFetch": "WebFetch",
    "WebSearch": "WebSearch",
}

_PATH_KEYS = ("file_path", "path", "target_file", "filePath", "file", "target_notebook")
_CONTENT_KEYS = ("content", "contents", "code_edit", "new_string", "text", "file_text")


def home_dir() -> Path:
    return Path.home() / ".cursor"


def hooks_path(directory: Optional[Path] = None) -> Path:
    return (directory or home_dir()) / "hooks.json"


def detected(directory: Optional[Path] = None) -> bool:
    """Cursor is installed here: its home directory or its binary exists."""
    return (directory or home_dir()).is_dir() or shutil.which("cursor") is not None


def is_cursor_event(event: dict) -> bool:
    """A payload Cursor sent. Every Cursor hook payload carries its version."""
    return isinstance(event, dict) and "cursor_version" in event


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _ours(entry: Any) -> bool:
    return isinstance(entry, dict) and _MARKER in str(entry.get("command", ""))


def native_hook_installed(directory: Optional[Path] = None) -> bool:
    hooks = _load(hooks_path(directory)).get("hooks")
    if not isinstance(hooks, dict):
        return False
    entries = hooks.get("preToolUse")
    return isinstance(entries, list) and any(_ours(e) for e in entries)


def _entries(vaara_bin: str) -> dict[str, list[dict]]:
    pre = f"{vaara_bin} hook pre-tool-use --client cursor"
    post = f"{vaara_bin} hook post-tool-use --client cursor"
    return {
        "preToolUse": [{"command": pre, "timeout": HOOK_TIMEOUT, "failClosed": True}],
        "postToolUse": [{"command": post, "timeout": 30}],
        "postToolUseFailure": [{"command": post, "timeout": 30}],
    }


def _strip(hooks: dict) -> dict:
    out: dict = {}
    for event, entries in hooks.items():
        if isinstance(entries, list):
            kept = [e for e in entries if not _ours(e)]
            if kept:
                out[event] = kept
        else:
            out[event] = entries
    return out


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.vaara-tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def install_hooks(vaara_bin: str, directory: Optional[Path] = None) -> bool:
    """Write Vaara's entries into ``hooks.json``, keeping every other hook.

    Re-running replaces Vaara's entries rather than adding a second set.
    True when the file changed.
    """
    path = hooks_path(directory)
    config = _load(path)
    before = json.dumps(config, sort_keys=True)
    hooks = config.get("hooks")
    hooks = _strip(hooks) if isinstance(hooks, dict) else {}
    for event, entries in _entries(vaara_bin).items():
        hooks.setdefault(event, []).extend(entries)
    config["version"] = config.get("version") or 1
    config["hooks"] = hooks
    if json.dumps(config, sort_keys=True) == before:
        return False
    _write(path, config)
    return True


def remove_hooks(directory: Optional[Path] = None) -> bool:
    path = hooks_path(directory)
    if not path.exists():
        return False
    config = _load(path)
    hooks = config.get("hooks")
    if not isinstance(hooks, dict):
        return False
    cleaned = _strip(hooks)
    if cleaned == hooks:
        return False
    config["hooks"] = cleaned
    _write(path, config)
    return True


def _first(args: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def to_hook_events(event: dict) -> list[dict]:
    """Translate a Cursor ``preToolUse`` or ``postToolUse`` payload."""
    raw_tool = event.get("tool_name")
    tool = raw_tool if isinstance(raw_tool, str) else ""
    args = event.get("tool_input")
    if isinstance(args, str):
        # beforeMCPExecution sends its params as a JSON string.
        try:
            args = json.loads(args)
        except ValueError:
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {"_raw": args} if args is not None else {}
    session = event.get("conversation_id") if isinstance(event.get("conversation_id"), str) else ""
    if tool.startswith("MCP:"):
        return [{"tool_name": f"mcp__cursor__{tool[4:] or 'unknown'}",
                 "tool_input": dict(args), "session_id": session}]
    name = _TOOLS.get(tool, tool or "unknown")
    tool_input = dict(args)
    path = _first(args, _PATH_KEYS)
    if path:
        tool_input["file_path"] = path
    content = _first(args, _CONTENT_KEYS)
    if content and name in ("Write", "Edit"):
        tool_input.setdefault("content", content)
        tool_input.setdefault("new_string", content)
    return [{"tool_name": name, "tool_input": tool_input, "session_id": session}]


def tool_response(event: dict) -> dict:
    """The outcome fields for a ``postToolUse`` or ``postToolUseFailure``."""
    if event.get("hook_event_name") == "postToolUseFailure":
        return {"isError": True}
    return {}


def render_pre(code: int, message: str) -> tuple[str, int]:
    """Cursor's verdict for a runner exit code: stdout JSON and exit code.

    Allow must be printed; Cursor blocks on a permission hook with no JSON.
    A block keeps exit 2 as well, which Cursor reads as deny on its own.
    """
    if code == 0:
        return json.dumps({"permission": "allow"}), 0
    reason = message or "Blocked by Vaara"
    return json.dumps({"permission": "deny", "user_message": reason,
                       "agent_message": reason}), 2
