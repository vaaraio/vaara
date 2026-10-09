# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Who connected: the agent behind an MCP ``initialize``, governed or not.

On stdio the Vaara MCP server and ``vaara-mcp-proxy`` run as a child of the
client, so the client is an ancestor in the process table: read, not guessed.
Launchers such as ``npx``, ``uvx`` or a shell sit in between, so the walk goes
up a few levels to the first process Vaara has an adapter for. The client's
own ``clientInfo`` from ``initialize`` names it as well. That name is what the
client says about itself, so the record carries both and the process table
decides when both are present.

The beacon appends one ``agent_seen`` record to the trail ``vaara scan
--watch`` writes (``~/.vaara/trail/agents/audit.db``). The macOS app already
watches that trail and notifies when an agent that is not governed starts, so
an ungoverned client that connects to Vaara is announced the same way.

Stdio only: over HTTP there is no parent process to read. ``VAARA_BEACON=0``
turns it off.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from vaara.integrations import scan

logger = logging.getLogger(__name__)

# How far up the process tree to look for the agent. Two launchers deep
# (a shell running npx running node) is the worst seen in client configs.
_MAX_HOPS = 5

# clientInfo.name as each client sends it, mapped to scan's adapter ids.
_CLIENT_NAMES: tuple[tuple[str, str], ...] = (
    ("claude-code", r"^claude[- ]?code"),
    ("claude-desktop", r"^claude-ai$|^claude desktop"),
    ("codex", r"codex"),
    ("gemini", r"gemini"),
    ("copilot", r"copilot"),
    ("cursor", r"cursor"),
    ("opencode", r"opencode"),
)
_DISPLAY = {agent: name for agent, name, _ in scan._ADAPTED}


def enabled() -> bool:
    return os.environ.get("VAARA_BEACON", "1").strip().lower() not in ("0", "off", "false", "no")


def parent_pid(pid: int) -> Optional[int]:
    """The parent of ``pid`` from /proc, else ``ps``; None when unknown."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # The command name is in parentheses and may itself hold spaces.
        return int(stat.rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        pass  # no /proc (macOS) or the process is gone; ps answers either way
    try:
        out = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out) if out else None
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def ancestors(
    pid: int,
    parent_of: Callable[[int], Optional[int]] = parent_pid,
    hops: int = _MAX_HOPS,
) -> list[int]:
    """``pid`` and up to ``hops`` processes above it, nearest first."""
    chain = [pid]
    while len(chain) <= hops:
        up = parent_of(chain[-1])
        if not up or up <= 1 or up in chain:
            break
        chain.append(up)
    return chain


def client_from_info(client_info: Any) -> Optional[str]:
    """scan's adapter id for a ``clientInfo`` name, or None."""
    name = client_info.get("name") if isinstance(client_info, dict) else None
    if not isinstance(name, str):
        return None
    lowered = name.strip().lower()
    for agent, pattern in _CLIENT_NAMES:
        if re.search(pattern, lowered):
            return agent
    return None


def identify(
    client_info: Any,
    start_pid: int,
    *,
    parent_of: Callable[[int], Optional[int]] = parent_pid,
    cmd_of: Callable[[int], str] = scan.cmdline,
) -> dict:
    """Who is on the other end of this stdio session.

    Returns ``agent`` (scan's adapter id, or None), ``name``, ``pid`` and
    ``command`` of the process found, and ``source``: ``process`` when the
    process table named it, ``client_info`` when only the client's own claim
    did, ``unknown`` otherwise.
    """
    first: Optional[tuple[int, str]] = None
    for pid in ancestors(start_pid, parent_of):
        cmd = cmd_of(pid)
        if not cmd:
            continue
        if first is None:
            first = (pid, cmd)
        if re.search(r"(^|[/\s])vaara(-mcp-proxy)?(\s|$)", cmd):
            continue  # Vaara's own wrappers name the agent they serve
        adapted = scan._adapter_for(cmd)
        if adapted:
            return {"agent": adapted[0], "name": adapted[1], "pid": pid,
                    "command": cmd, "source": "process"}
    pid, cmd = first if first is not None else (start_pid, "")
    claimed = client_from_info(client_info)
    if claimed:
        return {"agent": claimed, "name": _DISPLAY.get(claimed, claimed), "pid": pid,
                "command": cmd, "source": "client_info"}
    info_name = client_info.get("name") if isinstance(client_info, dict) else None
    name = info_name if isinstance(info_name, str) and info_name.strip() else (
        os.path.basename(cmd.split()[0]) if cmd else "unknown MCP client")
    return {"agent": None, "name": name.strip(), "pid": pid, "command": cmd,
            "source": "unknown"}


def describe(
    who: dict,
    client_info: Any,
    via: str,
    status: Callable[[str], tuple[bool, str]] = scan.adapter_status,
) -> tuple[str, str]:
    """(state, detail) in scan's terms: governed, reachable or ungoverned."""
    parts = [f"connected to {via} over stdio"]
    if isinstance(client_info, dict) and isinstance(client_info.get("name"), str):
        version = client_info.get("version")
        parts.append("clientInfo " + client_info["name"]
                     + (f" {version}" if isinstance(version, str) and version else ""))
    if who["agent"] is None:
        parts.append("no Vaara adapter for this client, so its own tool calls run unchecked")
        return "ungoverned", "; ".join(parts)
    if who["source"] == "client_info":
        parts.append("named by clientInfo only, the process table did not confirm it")
    active, detail = status(who["agent"])
    if detail:
        parts.append(detail)
    return ("governed" if active else "reachable"), "; ".join(parts)


class Beacon:
    """Records the connecting client once per process, off the request path."""

    def __init__(
        self,
        via: str,
        *,
        trail_path: Optional[Path] = None,
        open_trail: Optional[Callable[[Path], Any]] = None,
        identify_fn: Callable[..., dict] = identify,
        status: Callable[[str], tuple[bool, str]] = scan.adapter_status,
        start_pid: Optional[Callable[[], int]] = None,
        background: bool = True,
    ) -> None:
        self.via = via
        self._trail_path = trail_path
        self._open_trail = open_trail
        self._identify = identify_fn
        self._status = status
        self._start_pid = start_pid or os.getppid
        self._background = background
        self._fired = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def on_initialize(self, params: Any) -> None:
        """Call from the stdio ``initialize`` handler. Never raises."""
        with self._lock:
            if self._fired or not enabled():
                return
            self._fired = True
        client_info = params.get("clientInfo") if isinstance(params, dict) else None
        if self._background:
            self._thread = threading.Thread(target=self._record, args=(client_info,),
                                            name="vaara-mcp-beacon", daemon=True)
            self._thread.start()
        else:
            self._record(client_info)

    def wait(self, timeout: float = 3.0) -> None:
        """Give a pending record time to land before the process exits.

        A client that sends ``initialize`` and closes stdin straight away
        ends the session before the daemon thread has written anything.
        """
        if self._thread is not None:
            self._thread.join(timeout)

    def _record(self, client_info: Any) -> None:
        try:
            who = self._identify(client_info, self._start_pid())
            state, detail = describe(who, client_info, self.via, self._status)
            from vaara.integrations import scan_watch
            path = self._trail_path or scan_watch.default_trail()
            trail = (self._open_trail or scan_watch.open_trail)(path)
            trail.record_agent_seen(agent=who["name"], state=state, detail=detail,
                                    pid=who["pid"], command=who["command"])
        except Exception:
            logger.exception("MCP beacon could not record the connecting client")
