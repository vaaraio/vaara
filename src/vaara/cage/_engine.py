# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A cage that is an OCI runtime behind a container engine: gVisor and Kata.

Both are run the way everyone runs them, as the ``--runtime`` of docker or
podman. The driver creates the container with the runtime, the image and
the agent command, hands the cage declaration in as ``-e`` variables, and
reads the engine's ``inspect`` for state. The effective configuration the
record carries is ``sha256:`` over the request the driver made: runtime,
image, security options and the agent command, in canonical JSON. It is
handed to the container as ``VAARA_CAGE_DIGEST``, so every receipt from
inside carries it, and ``enforcement_state`` reads it back from the
container's environment so ``vaara cage status`` reports the same value
a receipt does. The engine's own view of the created container
(``HostConfig`` and ``Config``) is digested beside it as ``active_digest``,
in ``status`` and in the state's detail; a container Vaara did not launch
has no given digest and reports the engine's view as its own.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from vaara.cage import BASIS_DECLARED, BASIS_ENGINE, DIGEST_ENV, CageState, environ_for
from vaara.cage._cli import Tool, digest_json
from vaara.cage.driver import CageError, CageLaunch

_SINCE = re.compile(r"^(\d{4}-\d\d-\d\dT[^ ]+)\s(.*)$")


class ContainerEngineDriver:
    """Base for a runtime run through docker or podman."""

    name = "engine"
    runtime = ""          # the engine's --runtime value
    default_image = ""

    def __init__(self, engine: Optional[str] = None, runtime: Optional[str] = None,
                 timeout: float = 300.0) -> None:
        self._engine = Tool("engine", engine or os.environ.get("VAARA_CAGE_ENGINE") or "docker",
                            "VAARA_CAGE_ENGINE", "install docker or podman", timeout)
        if runtime:
            self.runtime = runtime

    # Subclasses say how the runtime itself reports its version.
    def upstream_version(self) -> str:  # pragma: no cover - overridden
        return self.name

    def start(self, agent: list[str], policy: Optional[Path] = None, *,
              name: Optional[str] = None, image: Optional[str] = None,
              security_opt: Optional[list[str]] = None) -> CageLaunch:
        if not agent:
            raise CageError("start needs the agent's argv")
        if policy is not None:
            raise CageError(f"the {self.name} driver takes its policy as --image and "
                            "--security-opt, not from a file; the runtime's own "
                            "configuration lives with the engine")
        image = image or self.default_image
        if not image:
            raise CageError("start needs an image (--image)")
        launch_name = name or re.sub(r"[^A-Za-z0-9_.-]", "-", os.path.basename(agent[0]))
        request = {"engine": self._engine.binary, "runtime": self.runtime, "image": image,
                   "security_opt": list(security_opt or ()), "agent": list(agent)}
        state = CageState(driver=self.name, upstream=self.upstream_version(),
                          config_digest=digest_json(request), confirmed=False,
                          basis=BASIS_DECLARED, name=launch_name)
        args = ["run", "-d", "--name", launch_name, f"--runtime={self.runtime}"]
        for opt in security_opt or ():
            args += ["--security-opt", opt]
        for key, value in environ_for(state).items():
            args += ["-e", f"{key}={value}"]
        args += [image, *agent]
        container_id = self._engine.run(*args).strip()
        return CageLaunch(driver=self.name, name=launch_name,
                          state=self.enforcement_state(launch_name),
                          detail={"container_id": container_id, "request": request})

    def _inspect(self, name: str) -> dict[str, Any]:
        data = self._engine.json("inspect", name)
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            raise CageError(f"{self._engine.name} inspect returned no object for {name!r}")
        return data

    def stop(self, name: str) -> None:
        self._engine.run("stop", name)

    def remove(self, name: str) -> None:
        self._engine.run("rm", "-f", name)

    def status(self, name: str) -> dict[str, Any]:
        data = self._inspect(name)
        state = data.get("State") or {}
        host = data.get("HostConfig") or {}
        config = data.get("Config") or {}
        given = next((str(e).split("=", 1)[1] for e in (config.get("Env") or ())
                      if str(e).startswith(DIGEST_ENV + "=")), "")
        return {
            "name": name, "id": data.get("Id", ""),
            "status": state.get("Status", ""), "running": bool(state.get("Running")),
            "exit_code": state.get("ExitCode"), "runtime": host.get("Runtime", ""),
            "image": config.get("Image", ""), "cmd": config.get("Cmd"),
            "config_digest": given,
            "active_digest": digest_json({"HostConfig": host, "Config": config}),
            "security_opt": host.get("SecurityOpt"),
        }

    def enforcement_state(self, name: Optional[str] = None) -> CageState:
        if not name:
            raise CageError(f"{self.name} reports per container; give the launch name")
        status = self.status(name)
        runtime_ok = (not self.runtime) or status["runtime"] == self.runtime
        return CageState(
            driver=self.name, upstream=self.upstream_version(),
            config_digest=status["config_digest"] or status["active_digest"],
            confirmed=bool(status["running"]) and runtime_ok, basis=BASIS_ENGINE, name=name,
            detail={k: status[k] for k in ("status", "runtime", "image", "exit_code", "id",
                                           "active_digest")},
        )

    def events(self, name: str, since: float = 0.0) -> Iterator[dict[str, Any]]:
        args = ["logs", "--timestamps"]
        if since:
            args += ["--since", str(int(since))]
        out = self._engine.run(*args, name)

        def _gen() -> Iterator[dict[str, Any]]:
            for line in out.splitlines():
                m = _SINCE.match(line)
                ts: Any = m.group(1) if m else time.time()
                yield {"ts": ts, "source": self.name,
                       "message": m.group(2) if m else line}

        return _gen()
