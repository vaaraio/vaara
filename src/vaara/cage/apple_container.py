# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apple's ``container`` as a cage driver: ``container run -d --name N ... IMAGE agent``.

apple/container (Apache-2.0) runs each Linux container in its own
lightweight virtual machine on a Mac with Apple silicon, through Apple's
Virtualization framework. Its CLI is ``container``; there is no policy
file, the confinement is the VM itself plus the run options (image,
read-only root, dropped capabilities).

This driver hands the cage declaration in as ``-e`` variables, digests the
request it made (image, options, agent command) as the effective
configuration, and reads ``container inspect N`` for state. Released
versions report ``status`` as a string; newer builds report an object with
``state``; both are read. Inside, the deciding process confirms the cage by
the hypervisor the CPU reports underneath it.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from vaara.cage import BASIS_CONTROL, BASIS_DECLARED, CageState, environ_for
from vaara.cage._cli import Tool, digest_json
from vaara.cage.driver import CageError, CageLaunch

NAME = "apple-container"


class AppleContainerDriver:
    name = NAME

    def __init__(self, binary: Optional[str] = None, timeout: float = 300.0) -> None:
        self._tool = Tool("container", binary, "APPLE_CONTAINER_BIN",
                          "install apple/container (github.com/apple/container/releases)",
                          timeout)

    def upstream_version(self) -> str:
        try:
            data = self._tool.json("system", "version", "--format", "json", timeout=30)
        except CageError:
            out = self._tool.run("--version", timeout=30)
            m = re.search(r"(\d+\.\d+\.\d+\S*)", out)
            return f"apple-container {m.group(1)}" if m else "apple-container"
        for entry in data if isinstance(data, list) else ():
            if isinstance(entry, dict) and entry.get("appName") == "container":
                return f"apple-container {entry.get('version', '')}".strip()
        return "apple-container"

    def start(self, agent: list[str], policy: Optional[Path] = None, *,
              name: Optional[str] = None, image: Optional[str] = None,
              read_only: bool = False, cap_drop: Optional[list[str]] = None) -> CageLaunch:
        if not agent:
            raise CageError("start needs the agent's argv")
        if policy is not None:
            raise CageError("the apple-container driver takes no policy file; the VM is "
                            "the boundary, set with --image and the run options")
        if not image:
            raise CageError("start needs an image (--image)")
        launch_name = name or re.sub(r"[^A-Za-z0-9_.-]", "-", os.path.basename(agent[0]))
        request = {"image": image, "read_only": read_only,
                   "cap_drop": list(cap_drop or ()), "agent": list(agent)}
        state = CageState(driver=NAME, upstream=self.upstream_version(),
                          config_digest=digest_json(request), confirmed=False,
                          basis=BASIS_DECLARED, name=launch_name)
        args = ["run", "-d", "--name", launch_name]
        if read_only:
            args.append("--read-only")
        for cap in cap_drop or ():
            args += ["--cap-drop", cap]
        for key, value in environ_for(state).items():
            args += ["-e", f"{key}={value}"]
        args += [image, *agent]
        self._tool.run(*args)
        return CageLaunch(driver=NAME, name=launch_name,
                          state=self.enforcement_state(launch_name),
                          detail={"declared": state.to_record(), "request": request})

    def stop(self, name: str) -> None:
        self._tool.run("stop", name)

    def remove(self, name: str) -> None:
        self._tool.run("delete", name)

    def status(self, name: str) -> dict[str, Any]:
        data = self._tool.json("inspect", name)
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            raise CageError(f"container inspect returned no object for {name!r}")
        config = data.get("configuration") or {}
        status = data.get("status")
        state = status.get("state") if isinstance(status, dict) else status
        image = config.get("image") or {}
        return {
            "name": name, "id": data.get("id") or config.get("id", ""),
            "status": str(state or ""),
            "runtime": config.get("runtimeHandler", ""),
            "image": image.get("reference", "") if isinstance(image, dict) else str(image),
            "read_only": config.get("readOnly"), "cap_drop": config.get("capDrop"),
            "active_digest": digest_json(config),
        }

    def enforcement_state(self, name: Optional[str] = None) -> CageState:
        if not name:
            raise CageError("apple-container reports per container; give the launch name")
        status = self.status(name)
        return CageState(
            driver=NAME, upstream=self.upstream_version(),
            config_digest=status["active_digest"],
            confirmed=status["status"].lower() == "running", basis=BASIS_CONTROL, name=name,
            detail={k: status[k] for k in ("status", "runtime", "image", "id")},
        )

    def events(self, name: str, since: float = 0.0) -> Iterator[dict[str, Any]]:
        out = self._tool.run("logs", name)

        def _gen() -> Iterator[dict[str, Any]]:
            now = time.time()
            for line in out.splitlines():
                if line.strip():
                    yield {"ts": now, "source": NAME, "message": line}

        return _gen()
