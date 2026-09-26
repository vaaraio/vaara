# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fanotify permission events, through ctypes. Linux only, root only.

The kernel holds each open or exec through a marked mount until the listener
answers allow or deny. Opens are decided whole: the event does not say
whether the file is being opened to read or to write. Unlink, rename and link
raise no permission event at all.

A thread that opens a file on a marked mount waits for its own event to be
answered. The guard answers its own pid's events with allow before anything
else, and reads the group from more than one thread, so a reader that opens
a file by accident is answered by another.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import re
import struct
from dataclasses import dataclass
from typing import Iterator, Optional

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
FAN_MARK_MOUNT = 0x10
FAN_MARK_FLUSH = 0x80

FAN_ALLOW = 0x01
FAN_DENY = 0x02
FAN_NOFD = -1
AT_FDCWD = -100

_META = struct.Struct("IBBHQii")
_RESPONSE = struct.Struct("iI")
MOUNT_MASK = FAN_OPEN_PERM | FAN_OPEN_EXEC_PERM | FAN_ONDIR

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

    def __init__(self, *, nonblocking: bool = False) -> None:
        flags = FAN_CLOEXEC | FAN_CLASS_CONTENT | FAN_UNLIMITED_QUEUE | FAN_UNLIMITED_MARKS
        if nonblocking:
            flags |= FAN_NONBLOCK
        fd = _lib().fanotify_init(flags, os.O_RDONLY | os.O_LARGEFILE | os.O_CLOEXEC)
        if fd < 0:
            _raise("fanotify_init")
        self.fd = fd

    def mark_mount(self, mount_point: str) -> None:
        """Hold every open and exec through the mount at ``mount_point``.

        A directory mark misses a subdirectory made after it was set, and an
        agent can make one and work inside it. A mount mark sees the whole
        mount, so the guard filters by path instead.
        """
        if _lib().fanotify_mark(self.fd, FAN_MARK_ADD | FAN_MARK_MOUNT, MOUNT_MASK,
                                AT_FDCWD, os.fsencode(mount_point)) != 0:
            _raise(f"fanotify_mark mount {mount_point}")

    def unmark_mount(self, mount_point: str) -> None:
        if _lib().fanotify_mark(self.fd, FAN_MARK_REMOVE | FAN_MARK_MOUNT, MOUNT_MASK,
                                AT_FDCWD, os.fsencode(mount_point)) != 0:
            err = ctypes.get_errno()
            if err not in (errno.ENOENT, errno.EINVAL):
                _raise(f"fanotify_mark remove mount {mount_point}")

    def read(self, size: int = 64 * 1024) -> list[Event]:
        """The events waiting now. Blocks for one unless the group is nonblocking."""
        return list(parse(os.read(self.fd, size)))

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


def _unescape_mountinfo(field: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def mount_points(mountinfo: str = "/proc/self/mountinfo") -> list[str]:
    try:
        text = open(mountinfo).read()
    except OSError:
        return ["/"]
    return [_unescape_mountinfo(line.split()[4]) for line in text.splitlines()
            if len(line.split()) > 4]


def mount_point(path: str, points: Optional[list[str]] = None) -> str:
    """The mount ``path`` is on: the longest mount point it sits under."""
    best = "/"
    for point in mount_points() if points is None else points:
        if (path == point or path.startswith(point.rstrip("/") + "/")) and len(point) > len(best):
            best = point
    return best
