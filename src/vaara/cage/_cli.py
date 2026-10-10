# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What every driver that shells out shares: finding the cage's tool,
running it, digesting a policy file, and a launch held as a child process."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterator, Optional

from vaara.cage import BASIS_PROCESS, CageState
from vaara.cage.driver import CageError


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    try:
        return digest_bytes(Path(path).read_bytes())
    except OSError as exc:
        raise CageError(f"cannot read {path}: {exc}") from None


def digest_json(obj: Any) -> str:
    """``sha256:`` over the canonical JSON of ``obj`` (sorted keys, no spaces)."""
    return digest_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=False).encode())


class Tool:
    """One external command line tool, found once, run many times."""

    def __init__(self, name: str, binary: Optional[str], env_var: str,
                 hint: str, timeout: float = 120.0) -> None:
        self.name = name
        self._binary = binary or os.environ.get(env_var) or name
        self._hint = hint
        self._env_var = env_var
        self.timeout = timeout

    @property
    def binary(self) -> str:
        return self._binary

    def path(self) -> str:
        found = shutil.which(self._binary) if "/" not in self._binary else self._binary
        if not found or not os.path.exists(found):
            raise CageError(f"{self._binary}: not found; {self._hint}, or set {self._env_var}")
        return found

    def run(self, *args: str, timeout: Optional[float] = None,
            stdin: Optional[str] = None, env: Optional[dict[str, str]] = None) -> str:
        binary = self.path()
        try:
            done = subprocess.run([binary, *args], capture_output=True, text=True,
                                  timeout=timeout or self.timeout, input=stdin,
                                  env=env)
        except subprocess.TimeoutExpired:
            raise CageError(f"{self.name} {args[0] if args else ''} did not answer within "
                            f"{timeout or self.timeout:.0f}s") from None
        except OSError as exc:
            raise CageError(f"could not run {binary}: {exc}") from None
        if done.returncode != 0:
            err = (done.stderr or done.stdout).strip()
            raise CageError(f"{self.name} {' '.join(args[:2])} failed: "
                            f"{err or done.returncode}")
        return done.stdout

    def spawn(self, *args: str, env: Optional[dict[str, str]] = None,
              stderr_to: Optional[Any] = None) -> subprocess.Popen:
        binary = self.path()
        try:
            return subprocess.Popen([binary, *args], env=env, stderr=stderr_to)
        except OSError as exc:
            raise CageError(f"could not start {binary}: {exc}") from None

    def json(self, *args: str, timeout: Optional[float] = None) -> Any:
        out = self.run(*args, timeout=timeout)
        try:
            return json.loads(out)
        except ValueError:
            raise CageError(f"{self.name} {' '.join(args[:2])} did not return JSON") from None


class ChildLaunch:
    """A cage that lives as long as the child process the driver started:
    the agent wrapped by the cage's own binary, with the cage's stderr kept
    as the launch's events."""

    def __init__(self, name: str, proc: subprocess.Popen, state: CageState,
                 keep_lines: int = 2000) -> None:
        self.name = name
        self.proc = proc
        self.state = state
        self.started = time.time()
        self.lines: deque[tuple[float, str]] = deque(maxlen=keep_lines)
        self._reader: Optional[threading.Thread] = None
        if proc.stderr is not None:
            self._reader = threading.Thread(target=self._read, daemon=True,
                                            name=f"vaara-cage-{name}")
            self._reader.start()

    def _read(self) -> None:
        assert self.proc.stderr is not None
        for raw in self.proc.stderr:
            line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            self.lines.append((time.time(), line.rstrip("\n")))

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, grace: float = 10.0) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def status(self) -> dict[str, Any]:
        return {"name": self.name, "pid": self.proc.pid, "running": self.alive(),
                "exit_code": self.proc.poll(), "started": self.started,
                "cage": self.state.to_record()}

    def enforcement_state(self) -> CageState:
        return CageState(driver=self.state.driver, upstream=self.state.upstream,
                         config_digest=self.state.config_digest,
                         confirmed=self.alive(), basis=BASIS_PROCESS, name=self.name,
                         detail={"pid": self.proc.pid, "exit_code": self.proc.poll()})

    def events(self, since: float = 0.0) -> Iterator[dict[str, Any]]:
        for ts, line in list(self.lines):
            if ts >= since:
                yield {"ts": ts, "source": self.state.driver, "message": line}


class ChildLaunches:
    """The launches a process-backed driver holds, by name."""

    def __init__(self) -> None:
        self._launches: dict[str, ChildLaunch] = {}

    def add(self, launch: ChildLaunch) -> None:
        self._launches[launch.name] = launch

    def get(self, name: str) -> ChildLaunch:
        try:
            return self._launches[name]
        except KeyError:
            raise CageError(f"no launch named {name!r} was started by this driver") from None

    def pop(self, name: str) -> ChildLaunch:
        launch = self.get(name)
        del self._launches[name]
        return launch

    def names(self) -> list[str]:
        return list(self._launches)
