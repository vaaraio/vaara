# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OpenCode governance: the plugin, and the mapping the hook runner needs.

OpenCode runs every tool call, its own and every MCP server's, through the
``tool.execute.before`` plugin hook, and a hook that throws stops the call
before it runs. The plugin this module installs (``opencode_plugin.js``)
hands each call to ``vaara hook pre-tool-use --client opencode`` and throws
when that exits non-zero. So one plugin puts the same gate in front of
OpenCode that the Claude Code hooks put in front of Claude Code: deny rules,
the classifier, approvals and the trail are the same code.

OpenCode names its tools and arguments differently (``bash``, ``edit``,
``filePath``, ``oldString``), and the deny rules are written against the
Claude Code names. :func:`to_hook_events` translates a call into those
names and keeps the original arguments beside the translated ones. A tool
it does not know is an MCP tool or one a plugin added, and goes to the
classifier as ``mcp__opencode__<tool>``, the path every MCP call takes in
the Claude Code hook.

OpenCode's MCP servers are not rewritten through ``vaara-mcp-proxy``: the
plugin already sees those calls, and gating them twice would record one
action as two.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional

#: Agent id on OpenCode's records. ``VAARA_PLUGIN_AGENT_ID`` overrides it,
#: as it does for Claude Code; the ``agent_id`` key in the Claude Code config
#: does not, since that file names the Claude Code agent.
AGENT_ID = "opencode"

#: OpenCode built-in tools and the Claude Code tool the deny rules know them
#: as. From the tool list of OpenCode 1.18.32's build agent, plus
#: ``apply_patch``, which OpenCode gives some models in place of edit and
#: write, and tools older versions shipped.
_BUILTIN: dict[str, str] = {
    "bash": "Bash",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "multiedit": "Edit",
    "apply_patch": "Edit",
    "patch": "Edit",
    "glob": "Glob",
    "grep": "Grep",
    "list": "LS",
    "webfetch": "WebFetch",
    "websearch": "WebSearch",
    "task": "Task",
    "skill": "Skill",
    "todowrite": "TodoWrite",
    "todoread": "TodoRead",
    "question": "AskUserQuestion",
    "lsp": "LSP",
    "invalid": "Invalid",
}

_PATCH_FILE = re.compile(
    r"^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+?)\s*$", re.M,
)

#: Where the plugin is installed. OpenCode loads every file in its global
#: plugin directory at startup.
PLUGIN_NAME = "vaara.js"
_PLUGIN_SOURCE = Path(__file__).with_name("opencode_plugin.js")


def config_dir() -> Path:
    """OpenCode's global config directory (``$XDG_CONFIG_HOME/opencode``)."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "opencode"


def plugin_path(directory: Optional[Path] = None) -> Path:
    return (directory or config_dir()) / "plugin" / PLUGIN_NAME


def detected(directory: Optional[Path] = None) -> bool:
    """OpenCode is installed here: its config directory or its binary exists."""
    return (directory or config_dir()).is_dir() or shutil.which("opencode") is not None


def install_plugin(vaara_bin: str, directory: Optional[Path] = None) -> bool:
    """Write the plugin, pinned to ``vaara_bin``. True when the file changed.

    The binary path is written into the file because OpenCode may start with
    a PATH that does not hold it, the same reason the Claude Code hooks carry
    an absolute path.
    """
    target = plugin_path(directory)
    text = _PLUGIN_SOURCE.read_text().replace("__VAARA_BIN__", vaara_bin.replace("\\", "\\\\").replace('"', '\\"'))
    if target.exists() and target.read_text() == text:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".vaara-tmp")
    tmp.write_text(text)
    os.replace(tmp, target)
    return True


def remove_plugin(directory: Optional[Path] = None) -> bool:
    target = plugin_path(directory)
    if not target.exists():
        return False
    target.unlink()
    return True


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _mapped_input(builtin: str, args: dict) -> dict:
    """The Claude Code argument names for one call, beside OpenCode's own."""
    out = dict(args)
    path = _str(args.get("filePath")) or _str(args.get("filepath")) or _str(args.get("path"))
    if builtin in ("Read", "Write", "Edit") and path:
        out["file_path"] = path
    if "oldString" in args:
        out["old_string"] = _str(args["oldString"])
    if "newString" in args:
        out["new_string"] = _str(args["newString"])
    edits = args.get("edits")
    if isinstance(edits, list):
        out["new_string"] = "\n".join(
            _str(e.get("newString")) for e in edits if isinstance(e, dict)
        )
    if "patchText" in args:
        out["content"] = out["new_string"] = _str(args["patchText"])
    if "subagent_type" in args and builtin == "Task":
        out.setdefault("subagent_type", args["subagent_type"])
    if builtin == "Skill" and "name" in args:
        out["skill"] = _str(args["name"])
    return out


def to_hook_events(event: dict) -> list[dict]:
    """Translate an OpenCode tool call into hook events.

    ``event`` is what the plugin sends: ``tool``, ``args``, ``sessionID``.
    The first event is the call as recorded. A patch that touches several
    files adds one event per further file, so a deny rule on a path sees
    each path on its own; only the first is recorded.
    """
    tool = _str(event.get("tool"))
    args = event.get("args")
    if not isinstance(args, dict):
        args = {"_raw": args} if args is not None else {}
    session = _str(event.get("sessionID"))
    builtin = _BUILTIN.get(tool)
    if builtin is None:
        name = tool or "unknown"
        return [{"tool_name": f"mcp__opencode__{name}", "tool_input": dict(args),
                 "session_id": session}]
    tool_input = _mapped_input(builtin, args)
    events = [{"tool_name": builtin, "tool_input": tool_input, "session_id": session}]
    paths = _PATCH_FILE.findall(_str(args.get("patchText")))
    if paths:
        tool_input["file_path"] = paths[0]
        for extra in paths[1:]:
            events.append({"tool_name": builtin,
                           "tool_input": {**tool_input, "file_path": extra},
                           "session_id": session})
    return events


def tool_response(output: Any) -> dict:
    """Map ``tool.execute.after``'s output onto the fields the outcome reads."""
    if not isinstance(output, dict):
        return {}
    metadata = output.get("metadata")
    exit_code = metadata.get("exit") if isinstance(metadata, dict) else None
    if isinstance(exit_code, int) and exit_code != 0:
        return {"stderr": f"exit {exit_code}"}
    return {}
