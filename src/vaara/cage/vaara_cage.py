# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Vaara cage as a driver: ``vaara run`` under the Linux OS layer.

The cage is the ``vaara-agent`` AppArmor profile, a cgroup per launch, and
the fanotify guard deciding opens and execs in the operator's folders
(:mod:`vaara.oslayer`). Its policy is the OS-layer selection the operator
keeps with ``vaara os-layer``, not a file passed per launch, so ``policy``
must be None here.

The effective configuration is the rendered profile text; the guard
reports its digest in ``status``. ``vaara run`` hands the digest to the
tree it starts, and the deciding process confirms the cage by its own
AppArmor label.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from vaara.cage import BASIS_GUARD, CageState
from vaara.cage.driver import CageError, CageLaunch

NAME = "vaara-cage"


def apparmor_version() -> str:
    """``apparmor <parser version>``, or ``apparmor`` when the parser is absent."""
    parser = shutil.which("apparmor_parser") or (
        "/sbin/apparmor_parser" if os.path.exists("/sbin/apparmor_parser") else None
    )
    if parser is None:
        return "apparmor"
    try:
        out = subprocess.run([parser, "--version"], capture_output=True, text=True,
                             timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return "apparmor"
    for line in out.splitlines():
        words = line.split()
        if words and words[-1][:1].isdigit():
            return f"apparmor {words[-1]}"
    return "apparmor"


def profile_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


class VaaraCageDriver:
    name = NAME

    def __init__(self, socket_path: Optional[Path] = None,
                 vaara_argv: Optional[list[str]] = None) -> None:
        from vaara.oslayer.guard import SOCKET_PATH

        self._socket = Path(socket_path) if socket_path else SOCKET_PATH
        # How to start `vaara run`: the installed command, or this interpreter.
        self._vaara = vaara_argv or [sys.executable, "-c",
                                     "import sys; from vaara.cli import main; sys.exit(main())"]
        self._launches: dict[str, subprocess.Popen] = {}

    # ── Guard ────────────────────────────────────────────────────

    def _guard(self) -> dict[str, Any]:
        from vaara.oslayer.client import GuardError, request

        try:
            return request({"op": "status"}, socket_path=self._socket)
        except GuardError as exc:
            raise CageError(str(exc)) from None
        except OSError as exc:
            raise CageError(f"the guard did not answer: {exc}") from None

    def upstream_version(self) -> str:
        return apparmor_version()

    # ── Driver interface ─────────────────────────────────────────

    def start(self, agent: list[str], policy: Optional[Path] = None, *,
              name: Optional[str] = None) -> CageLaunch:
        if policy is not None:
            raise CageError("the Vaara cage takes its policy from the OS-layer selection "
                            "(vaara os-layer), not from a file")
        if not agent:
            raise CageError("start needs the agent's argv")
        state = self.enforcement_state()
        if not state.confirmed:
            raise CageError("the guard has no profile loaded, so the agent would run "
                            "unconfined; not starting it")
        launch_name = name or os.path.basename(agent[0])
        argv = [*self._vaara, "run", "--name", launch_name, "--", *agent]
        proc = subprocess.Popen(argv)
        self._launches[launch_name] = proc
        return CageLaunch(driver=NAME, name=launch_name, pid=proc.pid,
                          state=CageState(driver=NAME, upstream=state.upstream,
                                          config_digest=state.config_digest,
                                          confirmed=True, basis=BASIS_GUARD,
                                          name=launch_name))

    def stop(self, name: str) -> None:
        proc = self._launches.pop(name, None)
        if proc is None:
            raise CageError(f"no launch named {name!r} was started by this driver")
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def status(self, name: str) -> dict[str, Any]:
        status = self._guard()
        launches = [x for x in status.get("launches", []) if x.get("agent") == name]
        return {"guard": status, "launches": launches, "running": bool(launches)}

    def enforcement_state(self, name: Optional[str] = None) -> CageState:
        status = self._guard()
        loaded = bool(status.get("profile_loaded"))
        running = True
        if name:
            running = any(x.get("agent") == name for x in status.get("launches", []))
        return CageState(
            driver=NAME, upstream=self.upstream_version(),
            config_digest=str(status.get("profile_digest") or ""),
            confirmed=loaded and running, basis=BASIS_GUARD, name=name or "",
            detail={"profile": status.get("profile"), "folders": status.get("folders", []),
                    "apps": status.get("apps", []), "launches": status.get("launches", []),
                    "harden": bool(status.get("harden")), "egress": status.get("egress")},
        )

    def events(self, name: str, since: float = 0.0) -> Iterator[dict[str, Any]]:
        """The guard's own records for ``name``'s launches since ``since``."""
        from vaara.audit.sqlite_backend import SQLiteAuditBackend

        status = self._guard()
        trail = status.get("trail")
        if not trail or not os.path.exists(trail):
            return iter(())
        backend = SQLiteAuditBackend(trail)
        try:
            records = backend.query_time_range(since, time.time(), limit=10_000)
        finally:
            backend.close()

        def _gen() -> Iterator[dict[str, Any]]:
            for rec in records:
                if name and rec.agent_id != name:
                    continue
                yield {"ts": rec.timestamp, "source": "vaara-os-guard",
                       "event": rec.event_type.value, "tool": rec.tool_name,
                       "message": str((rec.data or {}).get("reason", "")),
                       "record_id": rec.record_id, "data": rec.data}

        return _gen()
