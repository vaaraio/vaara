# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One cgroup v2 per ``vaara run`` launch: the kill switch.

Every process a launch starts stays in its cgroup, so writing ``1`` to its
``cgroup.kill`` ends the whole tree at once, including processes that left
the session or were reparented. The guard creates the cgroup as root and
hands ``cgroup.kill`` to the operator, so ``vaara run`` can end its own
launch when the guard goes away.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Optional

CGROUP_ROOT = Path("/sys/fs/cgroup")
PARENT = "vaara"


def parent_dir(root: Path = CGROUP_ROOT) -> Path:
    return root / PARENT


def launch_dir(launch_id: str, root: Path = CGROUP_ROOT) -> Path:
    return parent_dir(root) / f"launch-{launch_id}"


def available(root: Path = CGROUP_ROOT) -> bool:
    """cgroup v2 is mounted and has ``cgroup.kill`` (Linux 5.14 and later)."""
    # The root cgroup has no cgroup.kill of its own; every child has one.
    if not (root / "cgroup.controllers").exists():
        return False
    try:
        for entry in root.iterdir():
            if entry.is_dir():
                return (entry / "cgroup.kill").exists()
    except OSError:
        return False
    return False


def create(launch_id: str, uid: int, gid: int, root: Path = CGROUP_ROOT) -> Path:
    path = launch_dir(launch_id, root)
    path.parent.mkdir(exist_ok=True)
    path.mkdir()
    os.chown(path / "cgroup.kill", uid, gid)
    return path


def add(path: Path, pid: int) -> None:
    (path / "cgroup.procs").write_text(f"{pid}\n")


def pids(path: Path) -> list[int]:
    try:
        return [int(p) for p in (path / "cgroup.procs").read_text().split()]
    except (OSError, ValueError):
        return []


def kill(path: Path) -> None:
    """End every process in ``path``. Falls back to signals without cgroup.kill."""
    try:
        (path / "cgroup.kill").write_text("1\n")
        return
    except OSError:
        pass
    for pid in pids(path):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def remove(path: Path, timeout: float = 2.0) -> bool:
    """Kill what is left in ``path`` and remove it. True when it is gone."""
    deadline = time.monotonic() + timeout
    while True:
        kill(path)
        try:
            path.rmdir()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)


def launch_of(pid: int) -> Optional[str]:
    """The launch id ``pid`` runs under, from ``/proc/<pid>/cgroup``."""
    try:
        return launch_in(Path(f"/proc/{pid}/cgroup").read_text())
    except OSError:
        return None


def launch_in(text: str) -> Optional[str]:
    """The launch id in the text of a ``/proc/<pid>/cgroup`` file, if any."""
    prefix = f"/{PARENT}/launch-"
    for line in text.splitlines():
        if line.startswith("0::"):
            rel = line[3:]
            if rel.startswith(prefix):
                return rel[len(prefix):].split("/", 1)[0]
    return None


def launches(root: Path = CGROUP_ROOT) -> list[Path]:
    try:
        return sorted(p for p in parent_dir(root).iterdir()
                      if p.is_dir() and p.name.startswith("launch-"))
    except OSError:
        return []
