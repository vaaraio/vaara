# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GitHub Copilot CLI governance: the hooks, whether they run, and the mapping.

Copilot CLI loads every ``*.json`` file in ``~/.copilot/hooks/`` (or
``$COPILOT_HOME/hooks/``) when it starts and runs their ``preToolUse``
commands before each tool call. Vaara keeps its hooks in a file of its own
there, ``vaara.json``, so installing and removing them leaves every other
hook file as it was. ``vaara init`` writes ``vaara hook pre-tool-use
--client copilot`` into it.

A ``preToolUse`` command blocks the call on exit 2 or on a JSON
``"permissionDecision": "deny"`` on stdout. On exit 2 alone the model is
told only that the hook exited with code 2, so :func:`render_pre` prints
the JSON as well and the reason the model reads is Vaara's. A hook that
crashes or cannot be found also blocks the call. A hook that outlives its
``timeoutSec`` lets the call run, so the command is the gate in
:mod:`vaara.integrations._hook_gate`, whose deadline sits inside the
timeout and above the approval handshake's 60 s wait. Each of these was
checked against Copilot CLI 1.0.88.

``"disableAllHooks": true`` at the top of a hook file skips that file, so
:func:`hook_status` reads Vaara's own. The same key in ``config.json`` or
``settings.json`` under ``~/.copilot``, or in a repository's
``.github/copilot/settings.json``, left Vaara's hook running in 1.0.88.

A failed tool call reaches ``postToolUseFailure`` with an ``error`` and no
result, so Vaara registers its post-tool-use hook for both events.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
from pathlib import Path
from typing import Any, Optional

from vaara.integrations import _hook_gate

AGENT_ID = "copilot"

#: Vaara's hook file inside the hooks directory.
HOOKS_FILE = "vaara.json"
_MARKER = "vaara hook "

#: Longer than the gate's deadline, which is longer than the approval
#: handshake's default 60 s wait, in seconds.
HOOK_TIMEOUT = _hook_gate.HOST_TIMEOUT

#: Copilot CLI's own tools (1.0.88) that no deny rule translates. They are
#: recorded under their own names; any other name is a tool an MCP server
#: or an extension added and goes to the classifier.
BUILTIN_TOOLS = frozenset({
    "grep", "glob", "read_bash", "stop_bash", "list_bash", "sql",
    "session_store_sql", "read_agent", "list_agents", "write_agent",
    "fetch_copilot_cli_documentation",
})


def home_dir() -> Path:
    env = os.environ.get("COPILOT_HOME")
    return Path(env) if env else Path.home() / ".copilot"


def hooks_path(directory: Optional[Path] = None) -> Path:
    return (directory or home_dir()) / "hooks" / HOOKS_FILE


def detected(directory: Optional[Path] = None) -> bool:
    """Copilot CLI is installed here: its directory or its binary exists."""
    return (directory or home_dir()).is_dir() or shutil.which("copilot") is not None


def _config(vaara_bin: str) -> dict:
    post = f"{shlex.quote(vaara_bin)} hook post-tool-use --client copilot"
    after = [{"type": "command", "bash": post, "timeoutSec": 30}]
    return {"version": 1, "hooks": {
        "preToolUse": [{"type": "command",
                        "bash": _hook_gate.pre_command(vaara_bin, "copilot"),
                        "timeoutSec": HOOK_TIMEOUT}],
        "postToolUse": after,
        "postToolUseFailure": after,
    }}


def _load(path: Path) -> Optional[dict]:
    """The hook file, ``{}`` when there is none, None when it cannot be read."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def install_hooks(vaara_bin: str, directory: Optional[Path] = None) -> bool:
    """Write Vaara's hook file. True when it changed.

    Re-running rewrites the same file, so there is never a second set.
    """
    path = hooks_path(directory)
    config = _config(vaara_bin)
    if _load(path) == config:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".vaara-tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n")
    os.replace(tmp, path)
    return True


def _ours(config: dict) -> bool:
    groups = (config.get("hooks") or {}).get("preToolUse")
    return isinstance(groups, list) and any(
        isinstance(h, dict) and _MARKER in str(h.get("bash", "")) for h in groups)


def remove_hooks(directory: Optional[Path] = None) -> bool:
    """Delete Vaara's hook file, if it is Vaara's."""
    path = hooks_path(directory)
    config = _load(path)
    if not config or not _ours(config):
        return False
    path.unlink()
    return True


def hook_status(directory: Optional[Path] = None) -> str:
    """Whether Copilot CLI will run Vaara's ``preToolUse`` hook.

    ``active``, ``disabled`` (the file sets ``disableAllHooks``),
    ``missing`` (no Vaara hook) or ``unknown`` (the file could not be read).
    """
    config = _load(hooks_path(directory))
    if config is None:
        return "unknown"
    if not _ours(config):
        return "missing"
    if config.get("disableAllHooks") is True:
        return "disabled"
    return "active"


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _resolve(path: str, cwd: str) -> str:
    if not path or os.path.isabs(path) or not cwd:
        return path
    return os.path.normpath(os.path.join(cwd, path))


def _args(event: dict) -> dict:
    """``toolArgs`` as a dict. 1.0.88 sends an object; the docs say a JSON string."""
    args = event.get("toolArgs")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return {"_raw": args}
    if not isinstance(args, dict):
        return {"_raw": args} if args is not None else {}
    return dict(args)


def _event(tool: str, args: dict, session: str) -> dict:
    return {"tool_name": tool, "tool_input": args, "session_id": session}


def to_hook_events(event: dict) -> list[dict]:
    """Translate a Copilot CLI ``preToolUse`` or post-tool payload.

    Copilot CLI names its tools in its own words and hands relative paths
    over as the model wrote them. The ones the deny rules speak about
    become Claude Code's tool, the Copilot fields kept and the rule fields
    added, with each path resolved against the call's ``cwd``. ``task``
    starts a subagent and becomes ``Task``, the name the spawn rule reads.
    """
    tool = _str(event.get("toolName")) or "unknown"
    args = _args(event)
    session = _str(event.get("sessionId"))
    cwd = _str(event.get("cwd"))
    path = _resolve(_str(args.get("path")), cwd)

    if tool == "bash":
        return [_event("Bash", args, session)]
    if tool == "create":
        return [_event("Write", {**args, "file_path": path,
                                 "content": _str(args.get("file_text"))}, session)]
    if tool == "edit":
        return [_event("Edit", {**args, "file_path": path,
                                "old_string": _str(args.get("old_str")),
                                "new_string": _str(args.get("new_str"))}, session)]
    if tool == "view":
        return [_event("Read", {**args, "file_path": path}, session)]
    if tool == "skill":
        return [_event("Skill", args, session)]
    if tool == "task":
        return [_event("Task", args, session)]
    if tool in BUILTIN_TOOLS:
        return [_event(tool, args, session)]
    return [_event(f"mcp__copilot__{tool}", args, session)]


def tool_response(event: dict) -> dict:
    """The outcome fields for a ``postToolUse`` or ``postToolUseFailure``.

    A failure arrives with ``error`` and no result and is reported as
    ``isError``, which the runner scores. A shell command that ran and
    exited non-zero is a success to Copilot CLI, so it reads as one here.
    """
    if not isinstance(event, dict):
        return {}
    if event.get("error"):
        return {"error": event["error"], "isError": True}
    result = event.get("toolResult")
    if not isinstance(result, dict):
        return {}
    if result.get("resultType") not in (None, "success"):
        return {**result, "isError": True}
    return result


def render_pre(code: int, message: str) -> tuple[str, int]:
    """Copilot CLI's verdict for a runner exit code: stdout JSON and exit code.

    Exit 2 stays, so a block holds even if the JSON is not read. An allow
    prints nothing, which leaves Copilot CLI's own permission flow in place.
    """
    if code == 0:
        return "", 0
    return json.dumps({"permissionDecision": "deny",
                       "permissionDecisionReason": message or "Blocked by Vaara"}), 2
