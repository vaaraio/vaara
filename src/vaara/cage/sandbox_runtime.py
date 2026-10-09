# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Anthropic's sandbox-runtime as a cage driver: ``srt --settings S -- agent``.

sandbox-runtime (Apache-2.0, ``@anthropic-ai/sandbox-runtime``, command
``srt``) wraps a process in the OS's own sandbox: bubblewrap on Linux with
its own user, pid and network namespaces and a seccomp filter, Seatbelt
on macOS, and a proxy that filters the network by domain. Its policy is a
JSON settings file, ``~/.srt-settings.json`` by default.

This driver passes the settings file with ``--settings`` and digests its
bytes. The agent is a foreground child of ``srt``. Inside, the deciding
process confirms the cage on Linux by the init of its pid namespace being
bubblewrap, which ``--unshare-pid`` with a fresh ``/proc`` leaves in place.
``srt`` also sets ``SANDBOX_RUNTIME=1`` in the child's environment; that is
a declaration, not a kernel fact. On macOS the block stays at ``declared``.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterator, Optional

from vaara.cage import BASIS_DECLARED, CageState, environ_for
from vaara.cage._cli import ChildLaunch, ChildLaunches, Tool, digest_file
from vaara.cage.driver import CageError, CageLaunch

NAME = "sandbox-runtime"
DEFAULT_SETTINGS = Path.home() / ".srt-settings.json"


class SandboxRuntimeDriver:
    name = NAME

    def __init__(self, binary: Optional[str] = None, timeout: float = 120.0) -> None:
        self._tool = Tool("srt", binary, "SRT_BIN",
                          "install sandbox-runtime (npm i -g @anthropic-ai/sandbox-runtime)",
                          timeout)
        self._launches = ChildLaunches()

    def upstream_version(self) -> str:
        out = self._tool.run("--version", timeout=30).strip()
        m = re.search(r"(\d+\.\d+\.\d+\S*)", out)
        return f"sandbox-runtime {m.group(1)}" if m else "sandbox-runtime"

    def start(self, agent: list[str], policy: Optional[Path] = None, *,
              name: Optional[str] = None) -> CageLaunch:
        if not agent:
            raise CageError("start needs the agent's argv")
        launch_name = name or os.path.basename(agent[0])
        settings = Path(policy) if policy is not None else DEFAULT_SETTINGS
        if policy is not None or settings.exists():
            digest = digest_file(settings)
        else:
            digest = ""  # srt's built-in defaults; nothing on disk to digest
        state = CageState(driver=NAME, upstream=self.upstream_version(),
                          config_digest=digest, confirmed=False, basis=BASIS_DECLARED,
                          name=launch_name)
        env = dict(os.environ)
        env.update(environ_for(state))
        args: list[str] = []
        if policy is not None:
            args += ["--settings", str(settings)]
        proc = self._tool.spawn(*args, "--", *agent, env=env, stderr_to=subprocess.PIPE)
        launch = ChildLaunch(launch_name, proc, state)
        self._launches.add(launch)
        return CageLaunch(driver=NAME, name=launch_name, pid=proc.pid,
                          state=launch.enforcement_state())

    def stop(self, name: str) -> None:
        self._launches.pop(name).stop()

    def status(self, name: str) -> dict[str, Any]:
        return self._launches.get(name).status()

    def enforcement_state(self, name: Optional[str] = None) -> CageState:
        if not name:
            raise CageError("sandbox-runtime lives per launch; give the launch name")
        return self._launches.get(name).enforcement_state()

    def events(self, name: str, since: float = 0.0) -> Iterator[dict[str, Any]]:
        return self._launches.get(name).events(since)
