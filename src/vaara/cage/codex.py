# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The OpenAI Codex sandbox as a cage driver: ``codex sandbox -- agent``.

Codex (Apache-2.0) ships the sandbox its own agent runs tools in, and
exposes it as a command: on Linux bubblewrap namespaces with Landlock,
seccomp and ``no_new_privs`` applied in-process; on macOS Seatbelt. The
sandbox state (writable roots, network) comes from the Codex config and
profiles, or is passed whole as JSON with ``--sandbox-state-json``.

This driver takes the policy as a JSON file holding that sandbox state and
passes it with ``--sandbox-state-json``, so the record's ``config_digest``
is ``sha256:`` over exactly the state Codex was given. Without a policy
file the sandbox runs under the Codex configuration stack (``-P`` names a
permission profile) and the digest is over the resolved profile name.

The agent is a foreground child: ``codex sandbox`` ends when the agent
ends. Inside, the deciding process confirms the cage by its seccomp filter
and ``no_new_privs`` on Linux. On macOS Codex sets ``CODEX_SANDBOX=seatbelt``
in the child's environment, which is a declaration, not a kernel fact, so
the block stays at ``declared`` there.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterator, Optional

from vaara.cage import BASIS_DECLARED, CageState, environ_for
from vaara.cage._cli import ChildLaunch, ChildLaunches, Tool, digest_bytes, digest_file
from vaara.cage.driver import CageError, CageLaunch

NAME = "codex"


class CodexSandboxDriver:
    name = NAME

    def __init__(self, binary: Optional[str] = None, timeout: float = 120.0) -> None:
        self._tool = Tool("codex", binary, "CODEX_BIN",
                          "install the Codex CLI (npm i -g @openai/codex)", timeout)
        self._launches = ChildLaunches()

    def upstream_version(self) -> str:
        out = self._tool.run("--version", timeout=30).strip()
        # "codex-cli 0.49.0" or "codex 0.49.0"
        m = re.search(r"(\d+\.\d+\.\d+\S*)", out)
        return f"codex {m.group(1)}" if m else "codex"

    def start(self, agent: list[str], policy: Optional[Path] = None, *,
              name: Optional[str] = None, permission_profile: Optional[str] = None,
              cwd: Optional[Path] = None) -> CageLaunch:
        if not agent:
            raise CageError("start needs the agent's argv")
        launch_name = name or os.path.basename(agent[0])
        args = ["sandbox"]
        if policy is not None:
            try:
                raw = Path(policy).read_bytes()
                json.loads(raw)
            except OSError as exc:
                raise CageError(f"cannot read policy {policy}: {exc}") from None
            except ValueError:
                raise CageError(f"{policy}: the Codex sandbox state must be JSON "
                                "(what codex/sandbox-state-meta returns)") from None
            digest = digest_bytes(raw)
            args += ["--sandbox-state-json", raw.decode("utf-8")]
        elif permission_profile:
            digest = digest_bytes(f"permission-profile:{permission_profile}".encode())
            args += ["--permission-profile", permission_profile]
            if cwd is not None:
                args += ["--cd", str(cwd)]
        else:
            digest = ""
        state = CageState(driver=NAME, upstream=self.upstream_version(),
                          config_digest=digest, confirmed=False, basis=BASIS_DECLARED,
                          name=launch_name)
        env = dict(os.environ)
        env.update(environ_for(state))
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
            raise CageError("the Codex sandbox lives per launch; give the launch name")
        return self._launches.get(name).enforcement_state()

    def events(self, name: str, since: float = 0.0) -> Iterator[dict[str, Any]]:
        return self._launches.get(name).events(since)


def digest_policy(path: Path) -> str:
    return digest_file(path)
