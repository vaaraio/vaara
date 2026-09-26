# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fanotify permission events, through ctypes. Linux only, root only.

The kernel holds each open or exec under a marked directory until the
listener answers allow or deny. Opens are decided whole: the event does not
say whether the file is being opened to read or to write.

A listener must never open a path it watches itself, or it waits on its own
event. The guard answers its own pid's events with allow before anything
else, and reads only /proc and its own trail.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import struct
from dataclasses import dataclass
from typing import Iterator

FAN_CLOEXEC = 0x1
FAN_NONBLOCK = 0x2
FAN_CLASS_CONTENT = 0x4
FAN_UNLIMITED_QUEUE = 0x10
FAN_UNLIMITED_MARKS = 0x20

FAN_OPEN_PERM = 0x10000
FAN_OPEN_EXEC_PERM = 0x40000
FAN_EVENT_ON_CHILD = 0x08000000
FAN_ONDIR = 0x40000000
FAN_Q_OVERFLOW = 0x4000

FAN_MARK_ADD = 0x1
FAN_MARK_REMOVE = 0x2
FAN_MARK_FLUSH = 0x80

FAN_ALLOW = 0x01
FAN_DENY = 0x02
FAN_NOFD = -1
AT_FDCWD = -100

_META = struct.Struct("IBBHQii")
_RESPONSE = struct.Struct("iI")
WATCH_MASK = FAN_OPEN_PERM | FAN_OPEN_EXEC_PERM | FAN_EVENT_ON_CHILD | FAN_ONDIR

_libc = None


def _lib():
    global _libc
    if _libc is None:
        lib = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)
        lib.fanotify_init.argtypes = [ctypes.c_uint, ctypes.c_uint]
        lib.fanotify_init.restype = ctypes.c_int
        lib.fanotify_mark.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_uint64,
                                      ctypes.c_int, ctypes.c_char_p]
        lib.fanotify_mark.restype = ctypes.c_int
        _libc = lib
    return _libc


def _raise(what: str) -> None:
    err = ctypes.get_errno()
    raise OSError(err, f"{what}: {os.strerror(err)}")


@dataclass(frozen=True)
class Event:
    mask: int
    fd: int
    pid: int

    @property
    def is_exec(self) -> bool:
        return bool(self.mask & FAN_OPEN_EXEC_PERM)

    def path(self) -> str:
        try:
            return os.readlink(f"/proc/self/fd/{self.fd}")
        except OSError:
            return ""


class Group:
    """One fanotify group of permission events."""

    def __init__(self) -> None:
        fd = _lib().fanotify_init(
            FAN_CLOEXEC | FAN_CLASS_CONTENT | FAN_UNLIMITED_QUEUE | FAN_UNLIMITED_MARKS,
            os.O_RDONLY | os.O_LARGEFILE | os.O_CLOEXEC)
        if fd < 0:
            _raise("fanotify_init")
        self.fd = fd

    def mark(self, directory: str) -> None:
        """Hold every open and exec of ``directory`` and the files directly in it."""
        if _lib().fanotify_mark(self.fd, FAN_MARK_ADD, WATCH_MASK, AT_FDCWD,
                                os.fsencode(directory)) != 0:
            _raise(f"fanotify_mark {directory}")

    def unmark(self, directory: str) -> None:
        if _lib().fanotify_mark(self.fd, FAN_MARK_REMOVE, WATCH_MASK, AT_FDCWD,
                                os.fsencode(directory)) != 0:
            err = ctypes.get_errno()
            if err not in (errno.ENOENT, errno.EINVAL):
                _raise(f"fanotify_mark remove {directory}")

    def read(self, size: int = 64 * 1024) -> Iterator[Event]:
        """Block until events arrive, then yield each one."""
        buf = os.read(self.fd, size)
        yield from parse(buf)

    def respond(self, event: Event, allow: bool) -> None:
        try:
            os.write(self.fd, _RESPONSE.pack(event.fd, FAN_ALLOW if allow else FAN_DENY))
        finally:
            if event.fd >= 0:
                os.close(event.fd)

    def close(self) -> None:
        # Closing the group releases every mark, and the kernel allows every
        # event still waiting on an answer.
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def parse(buf: bytes) -> Iterator[Event]:
    off = 0
    while off + _META.size <= len(buf):
        event_len, _vers, _res, _meta_len, mask, fd, pid = _META.unpack_from(buf, off)
        if event_len < _META.size:
            break
        yield Event(mask=mask, fd=fd, pid=pid)
        off += event_len


def directories(root: str, *, limit: int = 20000) -> list[str]:
    """``root`` and every directory under it, without following symlinks."""
    out = []
    for current, dirs, _files in os.walk(root, followlinks=False):
        out.append(current)
        if len(out) >= limit:
            break
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(current, d))]
    return out
