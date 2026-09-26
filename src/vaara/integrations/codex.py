# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Codex CLI governance: the hooks, their trust state, and the mapping.

Codex runs ``PreToolUse`` command hooks from ``~/.codex/hooks.json`` (or
``$CODEX_HOME/hooks.json``) before a shell command, an ``apply_patch`` and
an MCP tool call, and blocks the call on exit 2 with the reason on stderr,
and also on a JSON decision on stdout, which :func:`render_pre` prints so
the reason Codex shows the model is Vaara's. ``vaara init`` writes
``vaara hook pre-tool-use --client codex`` there.

Codex does not run a hook until the user has trusted it. The interactive
CLI asks at startup ("Trust all and continue"), and ``/hooks`` reviews them
later; ``codex exec`` asks nothing and skips an untrusted hook without a
word. Trust is a hash of the hook's definition kept in ``config.toml``
under ``[hooks.state."<hooks.json>:<event>:<group>:<handler>"]``.
:func:`trust_status` recomputes that hash the way Codex does, so init can
say whether Codex will actually run the hook. Vaara does not write the
trust entry itself: that review is Codex's, and the user answers it.

Codex has no fail-closed switch. A hook that crashes or outlives its
timeout lets the call through, as in Claude Code, so the pre-tool-use
command is the gate in :mod:`vaara.integrations._hook_gate`: any ending
but Vaara's own verdict blocks the call, and the gate's deadline sits
inside the timeout and above the approval handshake's 60 s wait.

Codex reports its shell tools as ``Bash`` with ``tool_input.command``, as
Claude Code does. ``apply_patch`` carries the whole patch in
``tool_input.command``; :func:`to_hook_events` makes one file event per
path in it, resolved against the call's ``cwd``, so the path rules see
each file. Text typed into a running session through ``write_stdin`` runs
no hook at all (Codex's choice: it treats it as transport for a command
already checked), which is why the deny rules refuse to start a bare
interactive shell or interpreter.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from vaara.integrations import _hook_gate

AGENT_ID = "codex"

#: Hook entries carry this in their command, as the Claude Code ones do.
_MARKER = "vaara hook "

#: Longer than the gate's deadline, which is longer than the approval
#: handshake's default 60 s wait.
HOOK_TIMEOUT = _hook_gate.HOST_TIMEOUT

#: Codex's event names in hooks.json and the key labels in its trust state.
_EVENTS = (("PreToolUse", "pre_tool_use", "pre-tool-use", HOOK_TIMEOUT),
           ("PostToolUse", "post_tool_use", "post-tool-use", 30))

_PATCH_FILE = re.compile(
    r"^\*\*\* (Add File|Update File|Delete File|Move to): (.+?)\s*$", re.M,
)


def home_dir() -> Path:
    env = os.environ.get("CODEX_HOME")
    return Path(env) if env else Path.home() / ".codex"


def hooks_path(directory: Optional[Path] = None) -> Path:
    return (directory or home_dir()) / "hooks.json"


def detected(directory: Optional[Path] = None) -> bool:
    """Codex is installed here: its home directory or its binary exists."""
    return (directory or home_dir()).is_dir() or shutil.which("codex") is not None


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _ours(handler: Any) -> bool:
    return isinstance(handler, dict) and _MARKER in str(handler.get("command", ""))


def _handler(vaara_bin: str, verb: str, timeout: int) -> dict:
    command = (_hook_gate.pre_command(vaara_bin, "codex") if verb == "pre-tool-use"
               else f"{vaara_bin} hook {verb} --client codex")
    return {"type": "command", "command": command, "timeout": timeout}


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


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.vaara-tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def install_hooks(vaara_bin: str, directory: Optional[Path] = None) -> bool:
    """Write Vaara's hooks into ``hooks.json``, keeping every other hook.

    Re-running replaces Vaara's entries rather than adding a second set.
    True when the file changed.
    """
    path = hooks_path(directory)
    config = _load(path)
    before = json.dumps(config, sort_keys=True)
    hooks = config.get("hooks")
    hooks = _strip(hooks) if isinstance(hooks, dict) else {}
    for event, _label, verb, timeout in _EVENTS:
        hooks.setdefault(event, []).append({"hooks": [_handler(vaara_bin, verb, timeout)]})
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


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _canonical(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    return value


def hook_hash(event_label: str, matcher: Optional[str], handler: dict) -> str:
    """The hash Codex keeps as ``trusted_hash`` for one command handler.

    Codex serialises the event label, the group's matcher and the one
    normalised handler as TOML, so unset fields drop out, then hashes the
    key-sorted compact JSON of that (codex-rs hooks ``hook_hash`` and
    config ``version_for_toml``). Normalised means ``async`` is always
    present and the timeout defaults to 600.
    """
    normal: dict = {"type": "command", "command": handler.get("command", ""),
                    "timeout": max(int(handler.get("timeout") or 600), 1),
                    "async": bool(handler.get("async", False))}
    for key in ("commandWindows", "statusMessage", "additionalContextLimit"):
        if handler.get(key) is not None:
            normal[key] = handler[key]
    identity: dict = {"event_name": event_label, "hooks": [normal]}
    if matcher is not None:
        identity["matcher"] = matcher
    data = json.dumps(_canonical(identity), separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(data.encode()).hexdigest()


def _hook_states(directory: Optional[Path]) -> Optional[dict]:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return None
    path = (directory or home_dir()) / "config.toml"
    try:
        config = tomllib.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return None
    hooks = config.get("hooks")
    state = hooks.get("state") if isinstance(hooks, dict) else None
    return state if isinstance(state, dict) else {}


def trust_status(directory: Optional[Path] = None) -> str:
    """Whether Codex will run Vaara's PreToolUse hook.

    ``trusted``, ``untrusted`` (never reviewed), ``modified`` (trusted once,
    changed since, which Codex also skips), ``disabled`` (turned off in
    ``/hooks``), ``missing`` (no hook installed) or ``unknown`` (the config
    could not be read).
    """
    path = hooks_path(directory)
    groups = (_load(path).get("hooks") or {}).get("PreToolUse")
    if not isinstance(groups, list):
        return "missing"
    states = _hook_states(directory)
    if states is None:
        return "unknown"
    found = None
    for g, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        for h, handler in enumerate(group.get("hooks") or []):
            if not _ours(handler):
                continue
            key = f"{path.resolve()}:pre_tool_use:{g}:{h}"
            state = states.get(key) or states.get(f"{path}:pre_tool_use:{g}:{h}") or {}
            if state.get("enabled") is False:
                found = "disabled"
            elif not state.get("trusted_hash"):
                found = found or "untrusted"
            elif state["trusted_hash"] == hook_hash("pre_tool_use", group.get("matcher"), handler):
                return "trusted"
            else:
                found = found or "modified"
    return found or "missing"


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _resolve(path: str, cwd: str) -> str:
    if not path or os.path.isabs(path) or not cwd:
        return path
    return os.path.normpath(os.path.join(cwd, path))


def to_hook_events(event: dict) -> list[dict]:
    """Translate a Codex ``PreToolUse`` or ``PostToolUse`` payload.

    Shell and MCP calls already arrive in Claude Code's shape. A patch
    becomes one ``Edit`` event per file it touches (``Write`` for a file it
    adds), each carrying the patch text as the content; only the first is
    recorded. ``view_image`` reads a file, so the read rules see it as one.
    """
    tool = _str(event.get("tool_name")) or "unknown"
    args = event.get("tool_input")
    if not isinstance(args, dict):
        args = {"_raw": args} if args is not None else {}
    session = _str(event.get("session_id"))
    cwd = _str(event.get("cwd"))
    if tool == "apply_patch":
        patch = _str(args.get("command")) or _str(args.get("input"))
        files = _PATCH_FILE.findall(patch)
        base = {**args, "content": patch, "new_string": patch}
        if not files:
            return [{"tool_name": "Edit", "tool_input": base, "session_id": session}]
        return [{"tool_name": "Write" if kind == "Add File" else "Edit",
                 "tool_input": {**base, "file_path": _resolve(path, cwd)},
                 "session_id": session}
                for kind, path in files]
    if tool == "view_image":
        return [{"tool_name": "Read",
                 "tool_input": {**args, "file_path": _resolve(_str(args.get("path")), cwd)},
                 "session_id": session}]
    if tool == "Bash" or tool.startswith("mcp__"):
        return [{"tool_name": tool, "tool_input": dict(args), "session_id": session}]
    # Anything else is a tool Codex or an extension added: the classifier
    # path, as for an MCP tool.
    return [{"tool_name": f"mcp__codex__{tool}", "tool_input": dict(args),
             "session_id": session}]


def tool_response(event: dict) -> dict:
    """The outcome fields for a ``PostToolUse``.

    Codex sends an MCP call's result as a dict and a shell command's output
    as plain text, with no exit code, so a failed command reads here the
    same as one that worked.
    """
    response = event.get("tool_response") if isinstance(event, dict) else None
    return response if isinstance(response, dict) else {}


def render_pre(code: int, message: str) -> tuple[str, int]:
    """Codex's verdict for a runner exit code: stdout JSON and exit code.

    On a block Codex tells the model only the first line of stderr, and
    anything the runner logged before its verdict (a trail warning, say)
    would take that place. The JSON decision carries the reason itself.
    Exit 2 stays, so a block holds even if the JSON is not read. An allow
    prints nothing: Codex takes exit 0 with no output as success.
    """
    if code == 0:
        return "", 0
    reason = message or "Blocked by Vaara"
    return json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "deny",
        "permissionDecisionReason": reason}}), 2
