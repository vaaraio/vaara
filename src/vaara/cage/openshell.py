# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""NVIDIA OpenShell as a cage driver, through the ``openshell`` CLI.

OpenShell (Apache-2.0) confines an agent in a sandbox: Landlock, seccomp
with ``no_new_privs``, a per-sandbox supervisor that proxies all egress
under a YAML policy, credentials injected at the boundary. A gateway holds
the sandboxes and the CLI talks to the gateway. This driver shells out to
that CLI, pinned to whatever release the operator installed; nothing in
OpenShell is modified or vendored.

What the driver does with it:

- ``start``: ``openshell sandbox create --name N --policy P --detach -- agent``,
  with the four ``VAARA_CAGE*`` variables passed as ``--env`` so Vaara's
  hooks inside the sandbox know which cage they run in. The declared
  ``config_digest`` is ``sha256:`` of the policy YAML as submitted. The
  gateway may merge a global policy on top; ``status`` reports the digest
  of the policy the gateway holds as active beside the gateway's own
  ``policy_hash`` and version, so the two can be compared.
- ``enforcement_state``: ``openshell sandbox get N -o json``. Confirmed when
  the gateway reports the sandbox phase as ``Ready``.
- ``events``: ``openshell logs N --source sandbox --since ...``; the lines
  the supervisor writes in OCSF shorthand (``[ocsf]``) are returned with
  their fields split.
- ``stop``: ``openshell sandbox stop N``; ``delete`` removes the sandbox.

Inside the sandbox the deciding process confirms the cage by the kernel's
own word: a seccomp filter and ``no_new_privs`` on itself. That is what
the OpenShell sandbox sets on its main process and everything under it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from vaara.cage import BASIS_DECLARED, BASIS_GATEWAY, CageState, environ_for
from vaara.cage.driver import CageError, CageLaunch

NAME = "openshell"

# "[1234.567] [sandbox] [INFO ] [ocsf] CONNECT action=allow dst_host=x dst_port=443"
_LOG_LINE = re.compile(
    r"^\[(?P<ts>[^\]]*)\]\s+\[(?P<source>[^\]]*)\]\s+\[(?P<level>[^\]]*)\]\s+"
    r"\[(?P<target>[^\]]*)\]\s?(?P<message>.*)$"
)
_FIELD = re.compile(r"(\w+)=(\"[^\"]*\"|\S+)")


def _digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def parse_log_line(line: str) -> Optional[dict[str, Any]]:
    """One ``openshell logs`` line as a dict, or None for a line that is not one."""
    m = _LOG_LINE.match(line.rstrip("\n"))
    if not m:
        return None
    message = m.group("message")
    event: dict[str, Any] = {
        "ts": m.group("ts").strip(),
        "source": m.group("source").strip(),
        "level": m.group("level").strip(),
        "target": m.group("target").strip(),
        "message": message,
    }
    if event["target"] == "ocsf":
        head, _, rest = message.partition(" ")
        event["kind"] = head
        event["fields"] = {k: v.strip('"') for k, v in _FIELD.findall(rest)}
    return event


class OpenShellDriver:
    name = NAME

    def __init__(self, binary: Optional[str] = None, timeout: float = 120.0) -> None:
        self._binary = binary or os.environ.get("OPENSHELL_BIN") or "openshell"
        self._timeout = timeout

    # ── CLI ───────────────────────────────────────────────────────

    def _run(self, *args: str, timeout: Optional[float] = None) -> str:
        binary = shutil.which(self._binary) if "/" not in self._binary else self._binary
        if not binary or not os.path.exists(binary):
            raise CageError(f"{self._binary}: not found; install OpenShell from "
                            "github.com/NVIDIA/OpenShell and put it on PATH, "
                            "or set OPENSHELL_BIN")
        try:
            done = subprocess.run([binary, *args], capture_output=True, text=True,
                                  timeout=timeout or self._timeout)
        except subprocess.TimeoutExpired:
            raise CageError(f"openshell {args[0]} did not answer within "
                            f"{timeout or self._timeout:.0f}s") from None
        except OSError as exc:
            raise CageError(f"could not run {binary}: {exc}") from None
        if done.returncode != 0:
            err = (done.stderr or done.stdout).strip()
            raise CageError(f"openshell {' '.join(args[:2])} failed: {err or done.returncode}")
        return done.stdout

    def upstream_version(self) -> str:
        out = self._run("--version", timeout=15).strip()
        # clap prints "openshell 0.1.5"; keep it as the upstream label.
        return out.splitlines()[0] if out else "openshell"

    # ── Driver interface ─────────────────────────────────────────

    def start(self, agent: list[str], policy: Optional[Path] = None, *,
              name: Optional[str] = None, image: Optional[str] = None,
              providers: Optional[list[str]] = None) -> CageLaunch:
        if not agent:
            raise CageError("start needs the agent's argv")
        launch_name = name or re.sub(r"[^a-z0-9-]", "-", os.path.basename(agent[0]).lower())
        digest = ""
        if policy is not None:
            try:
                digest = _digest_bytes(Path(policy).read_bytes())
            except OSError as exc:
                raise CageError(f"cannot read policy {policy}: {exc}") from None
        upstream = self.upstream_version()
        declared = CageState(driver=NAME, upstream=upstream, config_digest=digest,
                             confirmed=False, basis=BASIS_DECLARED, name=launch_name)
        args = ["sandbox", "create", "--name", launch_name, "--detach", "--no-tty"]
        if policy is not None:
            args += ["--policy", str(policy)]
        if image:
            args += ["--from", image]
        for provider in providers or ():
            args += ["--provider", provider]
        for key, value in environ_for(declared).items():
            args += ["--env", f"{key}={value}"]
        args += ["--", *agent]
        self._run(*args)
        state = self.enforcement_state(launch_name)
        return CageLaunch(driver=NAME, name=launch_name, state=state,
                          detail={"declared": declared.to_record()})

    def stop(self, name: str) -> None:
        self._run("sandbox", "stop", name)

    def delete(self, name: str) -> None:
        self._run("sandbox", "delete", name)

    def status(self, name: str) -> dict[str, Any]:
        out = self._run("sandbox", "get", name, "-o", "json")
        try:
            data = json.loads(out)
        except ValueError:
            raise CageError("openshell sandbox get did not return JSON") from None
        if not isinstance(data, dict):
            raise CageError("openshell sandbox get returned something other than an object")
        try:
            active = self._run("sandbox", "get", name, "--policy-only")
        except CageError:
            active = ""
        data["active_policy_digest"] = _digest_bytes(active.encode()) if active else ""
        return data

    def enforcement_state(self, name: Optional[str] = None) -> CageState:
        if not name:
            raise CageError("OpenShell reports per sandbox; give the launch name")
        data = self.status(name)
        phase = str(data.get("phase") or "")
        admission = data.get("configuration_admission") or {}
        if not isinstance(admission, dict):
            admission = {}
        return CageState(
            driver=NAME, upstream=self.upstream_version(),
            config_digest=str(data.get("active_policy_digest") or ""),
            confirmed=phase == "Ready", basis=BASIS_GATEWAY, name=name,
            detail={"phase": phase,
                    "current_policy_version": data.get("current_policy_version"),
                    "policy_source": data.get("policy_source"),
                    "revision": data.get("revision"),
                    "policy_hash": admission.get("policy_hash", ""),
                    "exit_code": data.get("exit_code")},
        )

    def events(self, name: str, since: float = 0.0) -> Iterator[dict[str, Any]]:
        args = ["logs", name, "--source", "sandbox"]
        if since:
            args += ["--since", f"{max(1, int(time.time() - since))}s"]
        out = self._run(*args)

        def _gen() -> Iterator[dict[str, Any]]:
            for line in out.splitlines():
                event = parse_log_line(line)
                if event is not None:
                    yield event

        return _gen()
