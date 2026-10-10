# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Firecracker as a cage driver: a microVM booted from a configuration file.

Firecracker (Apache-2.0, AWS) boots a minimal VM from a kernel image and a
root filesystem, described by a JSON configuration (``boot-source``,
``drives``, ``machine-config``, optionally ``logger``), and is driven over
a unix-socket HTTP API. This driver runs
``firecracker --api-sock S --id N --config-file C`` and talks to the API
with the standard library.

A launcher cannot set environment variables inside a guest it only boots,
so the cage declaration goes on the kernel command line: the driver
appends ``vaara.cage=firecracker vaara.cage.digest=... vaara.cage.upstream=...
vaara.cage.name=N`` and ``vaara.agent=<argv as JSON>`` to ``boot_args``,
and :func:`vaara.cage.declared` reads ``/proc/cmdline`` when the
environment carries nothing. The guest's init is what starts the agent;
the driver gives it the argv, it does not run it. The record's
``config_digest`` is ``sha256:`` over the configuration file as given,
before the driver's additions. Inside, the deciding process confirms the
cage by the hypervisor the CPU reports underneath it.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from vaara.cage import BASIS_API, BASIS_DECLARED, CageState
from vaara.cage._cli import ChildLaunch, ChildLaunches, Tool, digest_file
from vaara.cage.driver import CageError, CageLaunch

NAME = "firecracker"


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float = 10.0) -> None:
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self) -> None:  # noqa: D401 - http.client API
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._path)
        self.sock = sock


def _api(path: str, method: str = "GET", body: Optional[dict] = None,
         *, socket_path: str, timeout: float = 10.0) -> Any:
    conn = _UnixHTTPConnection(socket_path, timeout)
    try:
        payload = json.dumps(body) if body is not None else None
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
    except OSError as exc:
        raise CageError(f"the Firecracker API at {socket_path} did not answer: {exc}") from None
    finally:
        conn.close()
    if resp.status >= 400:
        raise CageError(f"Firecracker API {method} {path}: {resp.status} "
                        f"{data.decode('utf-8', 'replace').strip()}")
    if not data:
        return None
    try:
        return json.loads(data)
    except ValueError:
        return data.decode("utf-8", "replace")


class FirecrackerDriver:
    name = NAME

    def __init__(self, binary: Optional[str] = None, timeout: float = 120.0,
                 run_dir: Optional[Path] = None) -> None:
        self._tool = Tool("firecracker", binary, "FIRECRACKER_BIN",
                          "install Firecracker (github.com/firecracker-microvm/firecracker)",
                          timeout)
        self._run_dir = Path(run_dir) if run_dir else Path(tempfile.gettempdir()) / "vaara-cage"
        self._launches = ChildLaunches()
        self._sockets: dict[str, str] = {}

    def upstream_version(self) -> str:
        out = self._tool.run("--version", timeout=30).strip()
        m = re.search(r"(\d+\.\d+\.\d+\S*)", out)
        return f"firecracker {m.group(1)}" if m else "firecracker"

    def _socket_for(self, name: str) -> str:
        if name in self._sockets:
            return self._sockets[name]
        path = self._run_dir / f"{name}.firecracker.sock"
        if not path.exists():
            raise CageError(f"no Firecracker launch named {name!r} is known here")
        return str(path)

    def start(self, agent: list[str], policy: Optional[Path] = None, *,
              name: Optional[str] = None) -> CageLaunch:
        if not agent:
            raise CageError("start needs the agent's argv")
        if policy is None:
            raise CageError("start needs the microVM configuration JSON (--policy)")
        launch_name = name or re.sub(r"[^A-Za-z0-9_.-]", "-", os.path.basename(agent[0]))
        try:
            config = json.loads(Path(policy).read_text(encoding="utf-8"))
        except OSError as exc:
            raise CageError(f"cannot read {policy}: {exc}") from None
        except ValueError:
            raise CageError(f"{policy}: the Firecracker configuration must be JSON") from None
        if not isinstance(config, dict) or not isinstance(config.get("boot-source"), dict):
            raise CageError(f"{policy}: no boot-source in the configuration")
        digest = digest_file(Path(policy))
        upstream = self.upstream_version()
        state = CageState(driver=NAME, upstream=upstream, config_digest=digest,
                          confirmed=False, basis=BASIS_DECLARED, name=launch_name)
        boot = config["boot-source"]
        tokens = [f"vaara.cage={NAME}", f"vaara.cage.digest={digest}",
                  f"vaara.cage.upstream={upstream.replace(' ', '_')}",
                  f"vaara.cage.name={launch_name}",
                  "vaara.agent=" + json.dumps(agent, separators=(",", ":"))]
        boot["boot_args"] = " ".join([str(boot.get("boot_args") or "").strip(), *tokens]).strip()

        self._run_dir.mkdir(parents=True, exist_ok=True)
        sock = self._run_dir / f"{launch_name}.firecracker.sock"
        if sock.exists():
            sock.unlink()
        effective = self._run_dir / f"{launch_name}.config.json"
        effective.write_text(json.dumps(config, indent=2), encoding="utf-8", newline="\n")
        proc = self._tool.spawn("--api-sock", str(sock), "--id", launch_name,
                                "--config-file", str(effective), stderr_to=subprocess.PIPE)
        self._sockets[launch_name] = str(sock)
        launch = ChildLaunch(launch_name, proc, state)
        self._launches.add(launch)
        deadline = time.time() + 15
        while time.time() < deadline and not sock.exists() and launch.alive():
            time.sleep(0.1)
        return CageLaunch(driver=NAME, name=launch_name, pid=proc.pid,
                          state=self.enforcement_state(launch_name),
                          detail={"api_sock": str(sock), "config": str(effective)})

    def stop(self, name: str) -> None:
        sock = self._socket_for(name)
        try:
            _api("/actions", "PUT", {"action_type": "SendCtrlAltDel"}, socket_path=sock)
        except CageError:
            pass
        launch = self._launches.pop(name) if name in self._launches.names() else None
        if launch is not None:
            launch.stop()
        self._sockets.pop(name, None)

    def status(self, name: str) -> dict[str, Any]:
        sock = self._socket_for(name)
        info = _api("/", socket_path=sock)
        if not isinstance(info, dict):
            raise CageError("the Firecracker API did not describe the instance")
        try:
            config = _api("/vm/config", socket_path=sock)
        except CageError:
            config = None
        return {"name": name, "id": info.get("id"), "state": info.get("state"),
                "vmm_version": info.get("vmm_version"), "app_name": info.get("app_name"),
                "config": config}

    def enforcement_state(self, name: Optional[str] = None) -> CageState:
        if not name:
            raise CageError("Firecracker reports per microVM; give the launch name")
        try:
            status = self.status(name)
        except CageError as exc:
            launch = self._launches.get(name)
            state = launch.enforcement_state()
            return CageState(driver=NAME, upstream=state.upstream,
                             config_digest=state.config_digest, confirmed=False,
                             basis=BASIS_DECLARED, name=name,
                             detail={"error": str(exc), "pid": launch.proc.pid})
        config_digest = ""
        if isinstance(status.get("config"), dict):
            from vaara.cage._cli import digest_json

            config_digest = digest_json(status["config"])
        return CageState(
            driver=NAME, upstream=f"firecracker {status.get('vmm_version') or ''}".strip(),
            config_digest=config_digest, confirmed=status.get("state") == "Running",
            basis=BASIS_API, name=name,
            detail={"state": status.get("state"), "id": status.get("id")},
        )

    def events(self, name: str, since: float = 0.0) -> Iterator[dict[str, Any]]:
        """Firecracker's own log when the configuration names one, then the
        process's stderr."""
        log_path = None
        try:
            status = self.status(name)
            logger = (status.get("config") or {}).get("logger") or {}
            log_path = logger.get("log_path")
        except CageError:
            pass

        def _gen() -> Iterator[dict[str, Any]]:
            if log_path and os.path.exists(log_path):
                try:
                    for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines():
                        yield {"ts": time.time(), "source": "firecracker-log", "message": line}
                except OSError:
                    pass
            if name in self._launches.names():
                yield from self._launches.get(name).events(since)

        return _gen()
