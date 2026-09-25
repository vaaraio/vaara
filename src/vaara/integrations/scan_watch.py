# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara scan --watch``: record every AI agent that starts, governed or not.

``vaara scan`` answers once. This keeps answering. Every interval it reads the
process table for the agents Vaara has adapters for and the open connections
for anything talking to a model, and the first time it sees a process it
appends an ``agent_seen`` record to its own trail
(``~/.vaara/trail/agents/audit.db``): the agent, the pid, and whether Vaara
governs it. The macOS app watches that trail through ``sources.json`` and
notifies when an agent that is not governed starts.

The records go to a trail of their own so an engine older than this event type
never loads one: it would read the unknown type as a corrupt row.

``--install-watch`` runs it at login as a launchd agent (macOS) or a systemd
user unit (Linux); ``--uninstall-watch`` removes it.
"""

from __future__ import annotations

import plistlib
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from vaara.integrations import scan

LAUNCHD_LABEL = "io.vaara.scan-watch"
SYSTEMD_UNIT = "vaara-scan-watch.service"
DEFAULT_INTERVAL = 15.0
# Resolving the model hosts again this often catches a provider moving address.
_RESOLVE_EVERY = 600.0


def default_trail(home: Optional[Path] = None) -> Path:
    return (home or Path.home()) / ".vaara" / "trail" / "agents" / "audit.db"


def process_table() -> list[tuple[int, str]]:
    """(pid, command line) for every process, from ``ps``."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True,
                             text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid.isdigit():
            rows.append((int(pid), cmd.strip()))
    return rows


def launched_agents(
    table: Iterable[tuple[int, str]],
    status: Callable[[str], tuple[bool, str]] = scan.adapter_status,
) -> list[scan.Finding]:
    """Findings for adapted agents in the process table, connected or not."""
    found = []
    for pid, cmd in table:
        if re.search(r"(^|[/\s])vaara(\s|$)", cmd):
            continue  # Vaara's own hooks and proxies name the agent they serve
        adapted = scan._adapter_for(cmd)
        if adapted:
            active, detail = status(adapted[0])
            found.append(scan.Finding("governed" if active else "reachable", "process",
                                      adapted[1], detail, f"pid {pid}"))
    return found


class Watcher:
    """Appends one ``agent_seen`` record per agent process, the first time it is seen."""

    def __init__(
        self,
        trail: Any,
        *,
        table: Callable[[], list[tuple[int, str]]] = process_table,
        connected: Optional[Callable[[], list[scan.Finding]]] = None,
        status: Callable[[str], tuple[bool, str]] = scan.adapter_status,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.trail = trail
        self.table = table
        self.status = status
        self.clock = clock
        self._hosts: dict[str, str] = {}
        self._resolved_at: Optional[float] = None
        self.connected = connected or self._connected
        self.seen: dict[int, str] = {}

    def _connected(self) -> list[scan.Finding]:
        now = self.clock()
        if self._resolved_at is None or now - self._resolved_at > _RESOLVE_EVERY:
            self._hosts = scan.resolve_hosts()
            self._resolved_at = now
        return scan.scan_processes(ip_hosts=self._hosts, status=self.status)

    def tick(self) -> list[scan.Finding]:
        """One pass. Returns the findings recorded on this pass."""
        rows = self.table()
        alive = {pid for pid, _ in rows}
        by_pid: dict[int, scan.Finding] = {}
        # Connection findings first: they carry what the process talks to.
        for f in self.connected() + launched_agents(rows, self.status):
            pid = int(f.where.split()[-1]) if f.where.startswith("pid ") else -1
            by_pid.setdefault(pid, f)
        recorded = []
        for pid, f in sorted(by_pid.items()):
            if self.seen.get(pid) == f.name:
                continue
            self.seen[pid] = f.name
            self.trail.record_agent_seen(
                agent=f.name, state=f.state, detail=f.detail, pid=pid,
                command=next((c for p, c in rows if p == pid), ""),
            )
            recorded.append(f)
        # A pid that is gone can come back as a different process.
        for pid in list(self.seen):
            if alive and pid not in alive:
                del self.seen[pid]
        return recorded

    def run(self, interval: float = DEFAULT_INTERVAL, *, passes: Optional[int] = None,
            sleep: Callable[[float], None] = time.sleep) -> None:
        n = 0
        while passes is None or n < passes:
            self.tick()
            n += 1
            if passes is None or n < passes:
                sleep(interval)


def open_trail(path: Path) -> Any:
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteAuditBackend(path).load_trail()


# ---------------------------------------------------------------------------
# Running at login
# ---------------------------------------------------------------------------

def _argv(vaara_bin: str, trail: str, interval: float) -> list[str]:
    return [vaara_bin, "scan", "--watch", "--trail", trail, "--interval", f"{interval:g}"]


def render_launchd_plist(vaara_bin: str, trail: str, interval: float, log_dir: str) -> str:
    return plistlib.dumps({
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": _argv(vaara_bin, trail, interval),
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": f"{log_dir}/scan-watch.out.log",
        "StandardErrorPath": f"{log_dir}/scan-watch.err.log",
    }).decode()


def render_systemd_unit(vaara_bin: str, trail: str, interval: float) -> str:
    from vaara.integrations.proxy_service import _systemd_quote

    exec_start = " ".join(_systemd_quote(a) for a in _argv(vaara_bin, trail, interval))
    return (
        "[Unit]\nDescription=Vaara agent launch watch\n\n"
        f"[Service]\nExecStart={exec_start}\nRestart=on-failure\n\n"
        "[Install]\nWantedBy=default.target\n"
    )


def unit_path(system: str, home: Path) -> Optional[Path]:
    if system == "darwin":
        return home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
    if system == "linux":
        return home / ".config" / "systemd" / "user" / SYSTEMD_UNIT
    return None


def _run_quiet(runner: Callable[..., Any], cmd: list[str]) -> bool:
    try:
        runner(cmd, capture_output=True, check=False)
        return True
    except OSError:
        return False


def install(vaara_bin: str, *, interval: float = DEFAULT_INTERVAL, home: Optional[Path] = None,
            system: Optional[str] = None,
            runner: Callable[..., Any] = subprocess.run) -> tuple[Optional[Path], str]:
    """Write and start the login service. Returns (path, message)."""
    home = home or Path.home()
    system = system or sys.platform
    path = unit_path(system, home)
    if path is None:
        return None, f"no login service on {system!r}; run `vaara scan --watch` yourself"
    trail = str(default_trail(home))
    if system == "darwin":
        log_dir = home / ".vaara" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        text = render_launchd_plist(vaara_bin, trail, interval, str(log_dir))
        activate = [["launchctl", "unload", str(path)], ["launchctl", "load", "-w", str(path)]]
    else:
        text = render_systemd_unit(vaara_bin, trail, interval)
        activate = [["systemctl", "--user", "daemon-reload"],
                    ["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if not all(_run_quiet(runner, cmd) for cmd in activate):
        return path, f"wrote {path}; it starts at the next login"
    return path, f"installed and started {path}"


def uninstall(*, home: Optional[Path] = None, system: Optional[str] = None,
              runner: Callable[..., Any] = subprocess.run) -> bool:
    home = home or Path.home()
    system = system or sys.platform
    path = unit_path(system, home)
    if path is None or not path.exists():
        return False
    if system == "darwin":
        _run_quiet(runner, ["launchctl", "unload", str(path)])
    else:
        _run_quiet(runner, ["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT])
    path.unlink()
    return True
