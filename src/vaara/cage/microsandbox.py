# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""microsandbox as a cage driver: ``msb run --conf C --name N --detach -- agent``.

microsandbox (Apache-2.0) boots each sandbox as a libkrun microVM from an
OCI image, with its own guest kernel and a per-sandbox network policy.
Its CLI is ``msb``; a sandbox's configuration is a YAML file (image,
memory, network allow list, scripts) passed with ``--conf``.

This driver passes the configuration with ``--conf`` and digests its
bytes, hands the cage declaration in as ``-e`` variables, and reads
``msb status N --format json`` for state. Inside, the deciding process
confirms the cage by the hypervisor the CPU reports underneath it.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from vaara.cage import BASIS_CONTROL, BASIS_DECLARED, CageState, environ_for
from vaara.cage._cli import Tool, digest_file
from vaara.cage.driver import CageError, CageLaunch

NAME = "microsandbox"


class MicrosandboxDriver:
    name = NAME

    def __init__(self, binary: Optional[str] = None, timeout: float = 300.0) -> None:
        self._tool = Tool("msb", binary, "MSB_BIN",
                          "install microsandbox (microsandbox.dev)", timeout)

    def upstream_version(self) -> str:
        out = self._tool.run("--version", timeout=30).strip()
        m = re.search(r"(\d+\.\d+\.\d+\S*)", out)
        return f"microsandbox {m.group(1)}" if m else "microsandbox"

    def start(self, agent: list[str], policy: Optional[Path] = None, *,
              name: Optional[str] = None, image: Optional[str] = None) -> CageLaunch:
        if not agent:
            raise CageError("start needs the agent's argv")
        if policy is None and not image:
            raise CageError("start needs a sandbox configuration (--policy sandbox.yaml) "
                            "or an image (--image)")
        launch_name = name or re.sub(r"[^A-Za-z0-9_.-]", "-", os.path.basename(agent[0]))
        digest = digest_file(Path(policy)) if policy is not None else ""
        state = CageState(driver=NAME, upstream=self.upstream_version(),
                          config_digest=digest, confirmed=False, basis=BASIS_DECLARED,
                          name=launch_name)
        args = ["run", "--name", launch_name, "--detach", "--no-tty"]
        if policy is not None:
            args += ["--conf", str(policy)]
        for key, value in environ_for(state).items():
            args += ["-e", f"{key}={value}"]
        if image:
            args.append(image)
        args += ["--", *agent]
        self._tool.run(*args)
        return CageLaunch(driver=NAME, name=launch_name,
                          state=self.enforcement_state(launch_name),
                          detail={"declared": state.to_record()})

    def stop(self, name: str) -> None:
        self._tool.run("stop", name)

    def remove(self, name: str) -> None:
        self._tool.run("rm", name)

    def status(self, name: str) -> dict[str, Any]:
        data = self._tool.json("status", name, "--format", "json")
        if isinstance(data, list):
            data = next((d for d in data if isinstance(d, dict) and d.get("name") == name),
                        data[0] if data else {})
        if not isinstance(data, dict):
            raise CageError("msb status did not return an object")
        return data

    def enforcement_state(self, name: Optional[str] = None) -> CageState:
        if not name:
            raise CageError("microsandbox reports per sandbox; give the launch name")
        data = self.status(name)
        status = str(data.get("status") or "")
        return CageState(
            driver=NAME, upstream=self.upstream_version(), config_digest="",
            confirmed=status.lower() == "running", basis=BASIS_CONTROL, name=name,
            detail={"status": status, "image": data.get("image"),
                    "command": data.get("command"), "cpus": data.get("cpus"),
                    "memory_mib": data.get("memory_mib")},
        )

    def events(self, name: str, since: float = 0.0) -> Iterator[dict[str, Any]]:
        out = self._tool.run("logs", name, "--json")

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
                yield {"ts": ts, "source": NAME,
                       "message": str(event.get("message") or event.get("line") or event),
                       "event": event}

        return _gen()
