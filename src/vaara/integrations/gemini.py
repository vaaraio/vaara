# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gemini CLI governance: the hooks, whether they run, and the mapping.

Gemini CLI runs ``BeforeTool`` command hooks from the user settings file
(``~/.gemini/settings.json``, or under ``$GEMINI_CLI_HOME``) before every
tool call, built-in or MCP, and blocks the call on exit 2 or on a JSON
``"decision": "deny"`` on stdout; :func:`render_pre` prints both, so the
reason the model reads is Vaara's and not whatever else reached stderr.
``vaara init`` writes ``vaara hook pre-tool-use --client gemini`` there.

User-level hooks need no review: Gemini CLI fingerprints and asks about
project hooks only. They stop running when ``hooksConfig.enabled`` is false
or the hook's name is listed in ``hooksConfig.disabled`` (``/hooks
disable``); :func:`hook_status` reads both, so init says which.

Gemini CLI has no fail-closed switch. A hook that crashes, exits with any
code but 0 and 2, or outlives its timeout lets the call through, so the
BeforeTool command is the gate in :mod:`vaara.integrations._hook_gate`:
any ending but Vaara's own verdict blocks the call, and the gate's
deadline sits inside the timeout and above the approval handshake's 60 s
wait. Gemini CLI's timeouts are in milliseconds.

The settings file may carry comments (Gemini CLI strips them before
parsing). Writing the hooks drops them, so the first rewrite of a file that
had any keeps the original beside it as ``settings.json.vaara-backup``.
A file that does not parse even then is left alone.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

from vaara.integrations import _hook_gate

AGENT_ID = "gemini"

#: The hook's ``name``: what ``/hooks`` lists and ``hooksConfig.disabled``
#: names. Entries are recognised by the command, as the other adapters do.
HOOK_NAME = "vaara-governance"
_MARKER = "vaara hook "

#: Longer than the gate's deadline, which is longer than the approval
#: handshake's default 60 s wait, in milliseconds.
HOOK_TIMEOUT_MS = _hook_gate.HOST_TIMEOUT * 1000

_EVENTS = (("BeforeTool", "pre-tool-use", HOOK_TIMEOUT_MS),
           ("AfterTool", "post-tool-use", 30_000))

#: Gemini CLI's own tools (docs/reference/tools.md, 0.61) that no deny rule
#: translates. They are recorded under their own names and pass the regex
#: layer as Claude Code's Glob or TodoWrite do; any other name is a tool an
#: extension or subagent added and goes to the classifier.
BUILTIN_TOOLS = frozenset({
    "glob", "grep_search", "search_file_content", "list_directory",
    "ask_user", "write_todos", "list_mcp_resources", "get_internal_docs",
    "enter_plan_mode", "exit_plan_mode", "complete_task", "update_topic",
    "google_web_search",
    "tracker_create_task", "tracker_update_task", "tracker_get_task",
    "tracker_list_tasks", "tracker_add_dependency", "tracker_visualize",
})


def home_dir() -> Path:
    env = os.environ.get("GEMINI_CLI_HOME")
    return (Path(env) if env else Path.home()) / ".gemini"


def settings_path(directory: Optional[Path] = None) -> Path:
    return (directory or home_dir()) / "settings.json"


def detected(directory: Optional[Path] = None) -> bool:
    """Gemini CLI is installed here: its directory or its binary exists."""
    return (directory or home_dir()).is_dir() or shutil.which("gemini") is not None


def _strip_comments(text: str) -> str:
    """``text`` without ``//`` and ``/* */`` comments outside strings."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _load(path: Path) -> tuple[Optional[dict], bool]:
    """The settings and whether the file had comments.

    None when the file exists and cannot be read as settings, so nobody
    overwrites a file it did not understand. A missing file is empty.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}, False
    except OSError:
        return None, False
    if not text.strip():
        return {}, False
    try:
        data = json.loads(text)
        commented = False
    except ValueError:
        try:
            data = json.loads(_strip_comments(text))
        except ValueError:
            return None, False
        commented = True
    return (data, commented) if isinstance(data, dict) else (None, False)


def _ours(handler: Any) -> bool:
    return isinstance(handler, dict) and _MARKER in str(handler.get("command", ""))


def _handler(vaara_bin: str, verb: str, timeout: int) -> dict:
    command = (_hook_gate.pre_command(vaara_bin, "gemini") if verb == "pre-tool-use"
               else f"{vaara_bin} hook {verb} --client gemini")
    return {"type": "command", "name": HOOK_NAME, "command": command,
            "timeout": timeout}


def _strip(hooks: dict) -> dict:
    """Every hook but Vaara's. A group left with no handler goes too."""
    out: dict = {}
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            out[event] = groups
            continue
        kept = []
        for group in groups:
            if isinstance(group, dict) and isinstance(group.get("hooks"), list):
                handlers = [h for h in group["hooks"] if not _ours(h)]
                if not handlers:
                    continue
                group = {**group, "hooks": handlers}
            kept.append(group)
        if kept:
            out[event] = kept
    return out


def _write(path: Path, obj: dict, commented: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = path.with_name(path.name + ".vaara-backup")
    if commented and not backup.exists():
        shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".vaara-tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def install_hooks(vaara_bin: str, directory: Optional[Path] = None) -> bool:
    """Write Vaara's hooks into ``settings.json``, keeping everything else.

    Re-running replaces Vaara's entries rather than adding a second set.
    True when the file changed. Raises ``ValueError`` when the file exists
    and does not parse, rather than replace it.
    """
    path = settings_path(directory)
    config, commented = _load(path)
    if config is None:
        raise ValueError(f"{path} is not a JSON object; hooks not written")
    before = json.dumps(config, sort_keys=True)
    hooks = config.get("hooks")
    hooks = _strip(hooks) if isinstance(hooks, dict) else {}
    for event, verb, timeout in _EVENTS:
        hooks.setdefault(event, []).append(
            {"matcher": ".*", "hooks": [_handler(vaara_bin, verb, timeout)]})
    config["hooks"] = hooks
    if json.dumps(config, sort_keys=True) == before and not commented:
        return False
    _write(path, config, commented)
    return True


def remove_hooks(directory: Optional[Path] = None) -> bool:
    path = settings_path(directory)
    if not path.exists():
        return False
    config, commented = _load(path)
    if config is None:
        return False
    hooks = config.get("hooks")
    if not isinstance(hooks, dict):
        return False
    cleaned = _strip(hooks)
    if cleaned == hooks:
        return False
    config["hooks"] = cleaned
    _write(path, config, commented)
    return True


def hook_status(directory: Optional[Path] = None) -> str:
    """Whether Gemini CLI will run Vaara's ``BeforeTool`` hook.

    ``active``, ``disabled`` (the hooks system is off, or ``/hooks
    disable`` named this hook), ``missing`` (no hook installed) or
    ``unknown`` (the settings could not be read).
    """
    config, _ = _load(settings_path(directory))
    if config is None:
        return "unknown"
    groups = (config.get("hooks") or {}).get("BeforeTool")
    handlers = [h for g in groups if isinstance(g, dict)
                for h in (g.get("hooks") or []) if _ours(h)] \
        if isinstance(groups, list) else []
    if not handlers:
        return "missing"
    switch = config.get("hooksConfig")
    switch = switch if isinstance(switch, dict) else {}
    if switch.get("enabled") is False:
        return "disabled"
    off = switch.get("disabled")
    off = off if isinstance(off, list) else []
    if all((h.get("name") or h.get("command")) in off for h in handlers):
        return "disabled"
    return "active"


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _resolve(path: str, cwd: str) -> str:
    if not path or os.path.isabs(path) or not cwd:
        return path
    return os.path.normpath(os.path.join(cwd, path))


def _event(tool: str, args: dict, session: str) -> dict:
    return {"tool_name": tool, "tool_input": args, "session_id": session}


def to_hook_events(event: dict) -> list[dict]:
    """Translate a Gemini CLI ``BeforeTool`` or ``AfterTool`` payload.

    Gemini CLI names its tools in its own words and hands relative paths
    over as the model wrote them. The ones the deny rules speak about
    become Claude Code's tool, the Gemini fields kept and the rule fields
    added, with each path resolved against the call's ``cwd``, so the path
    rules see the file the tool will touch. ``read_many_files`` becomes one
    ``Read`` per path it names; only the first is recorded. An MCP call
    becomes ``mcp__<server>__<tool>`` from its ``mcp_context``, the name
    Claude Code gives the same call.
    """
    tool = _str(event.get("tool_name")) or "unknown"
    args = event.get("tool_input")
    if not isinstance(args, dict):
        args = {"_raw": args} if args is not None else {}
    args = dict(args)
    session = _str(event.get("session_id"))
    cwd = _str(event.get("cwd"))

    def path(key: str = "file_path") -> str:
        return _resolve(_str(args.get(key)), cwd)

    if tool == "run_shell_command":
        return [_event("Bash", args, session)]
    if tool == "write_file":
        return [_event("Write", {**args, "file_path": path()}, session)]
    if tool == "replace":
        return [_event("Edit", {**args, "file_path": path()}, session)]
    if tool == "read_file":
        return [_event("Read", {**args, "file_path": path()}, session)]
    if tool == "read_many_files":
        include = args.get("include") or []
        include = [include] if isinstance(include, str) else include
        paths = [_resolve(p, cwd) for p in include if isinstance(p, str)] or [""]
        return [_event("Read", {**args, "file_path": p}, session) for p in paths]
    if tool == "web_fetch":
        return [_event("WebFetch", {**args, "url": _str(args.get("prompt"))}, session)]
    if tool == "read_mcp_resource":
        return [_event("ReadMcpResourceTool", args, session)]
    if tool == "activate_skill":
        return [_event("Skill", {**args, "skill": _str(args.get("name"))}, session)]
    if tool in BUILTIN_TOOLS:
        return [_event(tool, args, session)]
    mcp = event.get("mcp_context")
    if isinstance(mcp, dict) and _str(mcp.get("server_name")):
        name = _str(mcp.get("tool_name")) or tool
        return [_event(f"mcp__{mcp['server_name']}__{name}", args, session)]
    # A tool an extension, a subagent or an unrecognised MCP server added:
    # the classifier path, as for an MCP tool.
    rest = tool[len("mcp_"):] if tool.startswith("mcp_") else tool
    return [_event(f"mcp__gemini__{rest}", args, session)]


def tool_response(event: dict) -> dict:
    """The outcome fields for an ``AfterTool``.

    Gemini CLI sends ``llmContent``, ``returnDisplay`` and, when the tool
    failed, ``error``. A failure is reported as ``isError``, which the
    runner scores; a shell command that ran and exited non-zero is not a
    tool error to Gemini CLI, so it reads as a success here.
    """
    response = event.get("tool_response") if isinstance(event, dict) else None
    if not isinstance(response, dict):
        return {}
    if response.get("error"):
        return {**response, "isError": True}
    return response


def render_pre(code: int, message: str) -> tuple[str, int]:
    """Gemini CLI's verdict for a runner exit code: stdout JSON and exit code.

    On exit 2 with nothing on stdout, Gemini CLI hands the model all of
    stderr as the reason, including anything the runner logged before its
    verdict; the JSON carries the reason alone. Exit 2 stays, so a block
    holds even if the JSON is not read. An allow prints nothing.
    """
    if code == 0:
        return "", 0
    return json.dumps({"decision": "deny",
                       "reason": message or "Blocked by Vaara"}), 2
