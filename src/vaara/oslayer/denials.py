# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The floor's refusals, as the kernel reports them.

AppArmor writes each refusal as an audit record. With auditd running the
record goes to ``/var/log/audit/audit.log``; without it the kernel prints it
to its own log, ``/dev/kmsg``. The guard follows both and keys each record on
its audit stamp, so a record seen twice is counted once.

    audit: type=1400 audit(1758850000.123:456): apparmor="DENIED"
      operation="open" class="file" profile="vaara-agent//tool"
      name="/home/me/.vaara/trail/audit.db" pid=4321 comm="rm"
      requested_mask="wd" denied_mask="wd" fsuid=1000 ouid=1000
"""

from __future__ import annotations

import errno
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

from vaara.oslayer.floor import is_agent_label

logger = logging.getLogger("vaara.os-guard")

KMSG = "/dev/kmsg"
AUDIT_LOG = "/var/log/audit/audit.log"

_FIELD = re.compile(r'(\w+)=("[^"]*"|\S+)')
_STAMP = re.compile(r"audit\((\d+\.\d+:\d+)\)")
_HEX = re.compile(r"^(?:[0-9A-F]{2})+$")
# The fields AppArmor writes either quoted or, when the value holds a space
# or a quote, hex-encoded. Every other field is a bare number or word.
_STRINGS = frozenset({"name", "comm", "profile", "peer", "target", "srcname", "exe"})


@dataclass(frozen=True)
class Denial:
    stamp: str
    operation: str
    profile: str
    target: str
    pid: int
    comm: str
    requested: str
    denied: str
    fields: dict = field(default_factory=dict, compare=False)


def _value(key: str, raw: str) -> str:
    if raw.startswith('"') and raw.endswith('"'):
        return raw[1:-1]
    if key in _STRINGS and _HEX.match(raw):
        try:
            return bytes.fromhex(raw).decode("utf-8", "replace")
        except ValueError:
            return raw
    return raw


def parse(line: str) -> Optional[Denial]:
    """A Denial when ``line`` is an AppArmor refusal by an agent profile."""
    if 'apparmor="DENIED"' not in line:
        return None
    stamp = _STAMP.search(line)
    fields = {k: _value(k, v) for k, v in _FIELD.findall(line)}
    profile = fields.get("profile", "")
    if not is_agent_label(profile):
        return None
    try:
        pid = int(fields.get("pid", "0"))
    except ValueError:
        pid = 0
    target = (fields.get("name") or fields.get("capname") or fields.get("peer")
              or fields.get("peer_addr") or "")
    return Denial(
        stamp=stamp.group(1) if stamp else "",
        operation=fields.get("operation", ""),
        profile=profile,
        target=target,
        pid=pid,
        comm=fields.get("comm", ""),
        requested=fields.get("requested_mask", fields.get("requested", "")),
        denied=fields.get("denied_mask", fields.get("denied", "")),
        fields=fields,
    )


def _follow_kmsg(stop: threading.Event) -> Iterator[str]:
    try:
        fd = os.open(KMSG, os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        return
    try:
        os.lseek(fd, 0, os.SEEK_END)
        while not stop.is_set():
            try:
                record = os.read(fd, 8192)
            except BlockingIOError:
                stop.wait(0.2)
                continue
            except OSError as exc:
                # EPIPE: records were overwritten before this reader got to
                # them. The next read continues from the oldest one left.
                if exc.errno == errno.EPIPE:
                    continue
                return
            text = record.decode("utf-8", "replace")
            yield text.split(";", 1)[1] if ";" in text else text
    finally:
        os.close(fd)


def _follow_file(path: str, stop: threading.Event) -> Iterator[str]:
    handle = None
    inode = None
    while not stop.is_set():
        if handle is None:
            try:
                handle = open(path, "r", errors="replace")
                inode = os.fstat(handle.fileno()).st_ino
                handle.seek(0, os.SEEK_END)
            except OSError:
                handle = None
                stop.wait(1.0)
                continue
        line = handle.readline()
        if line:
            yield line
            continue
        # Nothing new. A rotated log has a new inode at the same path.
        try:
            if os.stat(path).st_ino != inode:
                handle.close()
                handle = None
                continue
        except OSError:
            pass
        stop.wait(0.2)
    if handle is not None:
        handle.close()


class Follower:
    """Calls ``on_denial`` for each agent refusal the kernel reports from now on."""

    def __init__(self, on_denial: Callable[[Denial], None], *,
                 sources: tuple[str, ...] = (KMSG, AUDIT_LOG)) -> None:
        self._on_denial = on_denial
        self._sources = sources
        self._stop = threading.Event()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._seen_lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def _fresh(self, denial: Denial) -> bool:
        if not denial.stamp:
            return True
        with self._seen_lock:
            if denial.stamp in self._seen:
                return False
            self._seen[denial.stamp] = None
            while len(self._seen) > 4096:
                self._seen.popitem(last=False)
            return True

    def _run(self, source: str) -> None:
        lines = _follow_kmsg(self._stop) if source == KMSG else _follow_file(source, self._stop)
        for line in lines:
            denial = parse(line)
            if denial is not None and self._fresh(denial):
                try:
                    self._on_denial(denial)
                except Exception:
                    # A refusal that cannot be recorded is still a refusal;
                    # the kernel already enforced it. Say so and keep following.
                    logger.exception("could not record the floor refusal %s", denial.stamp)

    def start(self) -> None:
        for source in self._sources:
            t = threading.Thread(target=self._run, args=(source,), daemon=True,
                                 name=f"vaara-denials:{os.path.basename(source)}")
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        deadline = time.monotonic() + timeout
        for t in self._threads:
            t.join(max(0.0, deadline - time.monotonic()))
