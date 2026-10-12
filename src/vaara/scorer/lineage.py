# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Files an agent wrote after an outbound call, remembered across sessions.

The outbound-then-persist sequence catches the write: fetched content going
into a skill, memory, instruction or hook file. The harm comes later, when a
new session loads that file and acts on it (cross-session instruction
propagation, arXiv 2610.13227). A scorer's window ends with its session, so
the written path is kept here, and a later call that names it carries the
taint label (``agent.tainted_read``) into its own sequences.

A path is marked when the sequence matches, whatever the decision: a denied
write under enforcement never landed, but in shadow mode or after an approval
it did, and the store cannot tell which. A false taint costs a held call; a
missed one costs every session after it. ``clear`` removes a path once a
person has looked at the file.

The store is one JSON file, rewritten whole under a lock and replaced
atomically, so parallel hook processes cannot interleave half-writes. It is
capped (oldest marks drop first). An unreadable file reads as empty and is
rewritten on the next mark.
"""

from __future__ import annotations

import errno
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional, Union

DEFAULT_MAX_ENTRIES = 1000


def normalise(path: str) -> str:
    """One spelling per file: user expanded, separators and dot segments folded."""
    p = os.path.expanduser(path.replace("\\", "/"))
    if not p.startswith("/") and not (len(p) > 1 and p[1] == ":"):
        p = os.path.join(os.getcwd(), p)
    return os.path.normpath(p).replace("\\", "/")


@contextmanager
def _exclusive(lock_path: Path) -> Iterator[None]:
    with open(lock_path, "a+b") as fh:
        if sys.platform == "win32":
            import msvcrt

            fh.seek(0)
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EDEADLK):
                        raise
                    time.sleep(0.005)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class LineageStore:
    """Tainted agent-persistent files, keyed by normalised path."""

    def __init__(self, path: Union[str, Path, None] = None,
                 max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self.path = Path(path) if path is not None else Path(
            os.environ.get("VAARA_LINEAGE_PATH")
            or Path.home() / ".vaara" / "lineage.json")
        self.max_entries = max_entries

    def _read(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        tainted = data.get("tainted") if isinstance(data, dict) else None
        if not isinstance(tainted, dict):
            return {}
        return {k: v for k, v in tainted.items()
                if isinstance(k, str) and isinstance(v, dict)}

    def _write(self, tainted: dict[str, dict[str, Any]]) -> None:
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"version": 1, "tainted": tainted}, indent=1),
                       encoding="utf-8", newline="\n")
        os.replace(tmp, self.path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive(self.path.with_name(self.path.name + ".lock")):
            yield

    def mark(self, path: str, *, agent_id: str, session_id: str,
             pattern: str) -> None:
        """Remember ``path`` as written by ``pattern`` in this session."""
        key = normalise(path)
        with self._locked():
            tainted = self._read()
            tainted.pop(key, None)
            tainted[key] = {"agent_id": agent_id, "session_id": session_id,
                            "pattern": pattern, "marked_at": time.time()}
            while len(tainted) > self.max_entries:
                tainted.pop(next(iter(tainted)))
            self._write(tainted)

    def tainted(self, path: str) -> Optional[dict[str, Any]]:
        """The mark on ``path``, or None when it is clean."""
        return self._read().get(normalise(path))

    def clear(self, path: str) -> bool:
        """Drop the mark on ``path``. True when there was one."""
        key = normalise(path)
        with self._locked():
            tainted = self._read()
            if key not in tainted:
                return False
            del tainted[key]
            self._write(tainted)
            return True

    def entries(self) -> dict[str, dict[str, Any]]:
        return self._read()
