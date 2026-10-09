# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""nono as a cage driver: ``nono run --profile P --name N --detached -- agent``.

nono (Apache-2.0, from the Sigstore founders) is Landlock-first on Linux
with a static seccomp baseline for restricted networking by default, and
Seatbelt on macOS. Its policy is a profile, a name from its catalogue or a
path to a profile file; its supervisor keeps sessions that ``nono ps``,
``nono logs`` and ``nono stop`` address by name.

This driver passes the profile with ``--profile``. A profile given as a
file is digested by its bytes; a catalogue name is digested as the name,
since the resolved manifest lives in nono's own store. Inside, the
deciding process confirms the cage by its seccomp filter and
``no_new_privs`` under the default policy, or by ``no_new_privs`` alone
under ``--sandbox-policy landlock``, since Landlock requires it.

nono keeps its own audit sessions with a hash-chained event log
(``nono audit``); ``events`` returns the session's event log as nono
prints it with ``--json``.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from vaara.cage import BASIS_CONTROL, BASIS_DECLARED, CageState, environ_for
from vaara.cage._cli import Tool, digest_bytes, digest_file
from vaara.cage.driver import CageError, CageLaunch

NAME = "nono"


class NonoDriver:
    name = NAME

    def __init__(self, binary: Optional[str] = None, timeout: float = 120.0) -> None:
        self._tool = Tool("nono", binary, "NONO_BIN",
                          "install nono (brew install nono, or github.com/always-further/nono)",
                          timeout)

    def upstream_version(self) -> str:
        out = self._tool.run("--version", timeout=30).strip()
        m = re.search(r"(\d+\.\d+\.\d+\S*)", out)
        return f"nono {m.group(1)}" if m else "nono"

    def start(self, agent: list[str], policy: Optional[Path] = None, *,
              name: Optional[str] = None, allow: Optional[list[str]] = None) -> CageLaunch:
        if not agent:
            raise CageError("start needs the agent's argv")
        launch_name = name or re.sub(r"[^A-Za-z0-9_.-]", "-", os.path.basename(agent[0]))
        args = ["run", "--name", launch_name, "--detached"]
        digest = ""
        if policy is not None:
            policy_path = Path(policy)
            if policy_path.exists():
                digest = digest_file(policy_path)
                args += ["--profile", str(policy_path)]
            else:
                digest = digest_bytes(f"profile:{policy}".encode())
                args += ["--profile", str(policy)]
        for path in allow or ():
            args += ["--allow", path]
        state = CageState(driver=NAME, upstream=self.upstream_version(),
                          config_digest=digest, confirmed=False, basis=BASIS_DECLARED,
                          name=launch_name)
        env = dict(os.environ)
        env.update(environ_for(state))
        self._tool.run(*args, "--", *agent, env=env)
        return CageLaunch(driver=NAME, name=launch_name,
                          state=self.enforcement_state(launch_name),
                          detail={"declared": state.to_record()})

    def _session(self, name: str) -> Optional[dict[str, Any]]:
        sessions = self._tool.json("ps", "--json", "--all")
        if not isinstance(sessions, list):
            raise CageError("nono ps did not return a list")
        for session in sessions:
            if isinstance(session, dict) and (
                session.get("name") == name or session.get("session_id") == name
            ):
                return session
        return None

    def stop(self, name: str) -> None:
        self._tool.run("stop", name)

    def status(self, name: str) -> dict[str, Any]:
        session = self._session(name)
        if session is None:
            raise CageError(f"nono knows no session named {name!r}")
        return session

    def enforcement_state(self, name: Optional[str] = None) -> CageState:
        if not name:
            raise CageError("nono reports per session; give the launch name")
        session = self.status(name)
        status = str(session.get("status") or "")
        return CageState(
            driver=NAME, upstream=self.upstream_version(), config_digest="",
            confirmed=status.lower() == "running", basis=BASIS_CONTROL, name=name,
            detail={"session_id": session.get("session_id"), "status": status,
                    "profile": session.get("profile"), "network": session.get("network"),
                    "child_pid": session.get("child_pid"),
                    "exit_code": session.get("exit_code")},
        )

    def events(self, name: str, since: float = 0.0) -> Iterator[dict[str, Any]]:
        session = self._session(name)
        session_id = str(session.get("session_id")) if session else name
        out = self._tool.run("logs", session_id, "--json")

        def _gen() -> Iterator[dict[str, Any]]:
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    event = {"message": line}
                if not isinstance(event, dict):
                    event = {"message": str(event)}
                ts = event.get("ts") or event.get("timestamp") or time.time()
                yield {"ts": ts, "source": NAME, "message": str(event.get("message", event)),
                       "event": event}

        return _gen()
