# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kata Containers as a cage driver: a container run with the Kata runtime.

Kata (Apache-2.0, OpenInfra) runs each container in its own lightweight
VM with its own guest kernel. It is installed as a containerd shim
(``io.containerd.kata.v2``) and run through docker or podman with
``--runtime``, which is how this driver runs it (:mod:`vaara.cage._engine`).
The runtime name differs by install (``io.containerd.kata.v2``, ``kata``,
``kata-runtime``); pass ``runtime=`` or set ``VAARA_CAGE_KATA_RUNTIME``.

Inside, the deciding process confirms the cage by the hypervisor the CPU
reports underneath it: a VM-backed container shows one, a bare container
does not. ``kata-runtime --version`` gives the upstream label; Kata 4 ships
no ``kata-runtime``, so where it is not on ``PATH`` the shim's own
``containerd-shim-kata-v2 --version`` (``..., version: 4.3.1, ...``) does.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Optional

from vaara.cage._cli import Tool
from vaara.cage._engine import ContainerEngineDriver

NAME = "kata"
_SHIM = "containerd-shim-kata-v2"
_VERSION = re.compile(r"\bversion:\s*([0-9][^,\s]*)")


def _default_tool() -> str:
    """kata-runtime when it is on PATH (Kata 3), else the shim (Kata 4)."""
    if shutil.which("kata-runtime") or not shutil.which(_SHIM):
        return "kata-runtime"
    return _SHIM


class KataDriver(ContainerEngineDriver):
    name = NAME
    runtime = "io.containerd.kata.v2"

    def __init__(self, engine: Optional[str] = None, runtime: Optional[str] = None,
                 kata_runtime: Optional[str] = None, timeout: float = 300.0) -> None:
        super().__init__(engine, runtime or os.environ.get("VAARA_CAGE_KATA_RUNTIME"), timeout)
        self._kata = Tool("kata-runtime",
                          kata_runtime or os.environ.get("KATA_RUNTIME_BIN") or _default_tool(),
                          "KATA_RUNTIME_BIN",
                          "install Kata Containers (katacontainers.io)", 30.0)

    def upstream_version(self) -> str:
        try:
            out = self._kata.run("--version", timeout=30)
        except Exception:  # noqa: BLE001 - the shim may live on a host we cannot see
            return "kata"
        for line in out.splitlines():
            # "kata-runtime  : 3.x.y"
            if line.startswith("kata-runtime") and ":" in line:
                return "kata " + line.split(":", 1)[1].strip()
            # the shim: "Kata Containers containerd shim (Rust): id: ..., version: 4.3.1, ..."
            m = _VERSION.search(line)
            if m:
                return "kata " + m.group(1)
        return "kata"
