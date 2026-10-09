# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""gVisor as a cage driver: a container run with ``--runtime=runsc``.

gVisor (Apache-2.0, Google) is a user-space kernel: the agent's system
calls land in the Sentry, not the host kernel. It is installed as the OCI
runtime ``runsc`` and run through docker or podman, which is how this
driver runs it (:mod:`vaara.cage._engine`).

Inside, the deciding process confirms the cage by the kernel log: gVisor's
Sentry answers ``syslog`` with its own opening line, ``Starting gVisor``,
and a real kernel never does. ``runsc --version`` gives the upstream
label.
"""

from __future__ import annotations

import os
from typing import Optional

from vaara.cage._cli import Tool
from vaara.cage._engine import ContainerEngineDriver

NAME = "gvisor"


class GVisorDriver(ContainerEngineDriver):
    name = NAME
    runtime = "runsc"

    def __init__(self, engine: Optional[str] = None, runtime: Optional[str] = None,
                 runsc: Optional[str] = None, timeout: float = 300.0) -> None:
        super().__init__(engine, runtime, timeout)
        self._runsc = Tool("runsc", runsc or os.environ.get("RUNSC_BIN"), "RUNSC_BIN",
                           "install gVisor (gvisor.dev/docs/user_guide/install)", 30.0)

    def upstream_version(self) -> str:
        try:
            out = self._runsc.run("--version", timeout=30)
        except Exception:  # noqa: BLE001 - the engine may run runsc on a host we cannot see
            return "gvisor"
        for line in out.splitlines():
            if line.startswith("runsc version"):
                return "gvisor " + line[len("runsc version"):].strip()
        return "gvisor"
