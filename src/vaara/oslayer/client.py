# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Talking to the guard over its unix socket, for vaara run and vaara os-layer.

One JSON line out, one JSON line back. A launch keeps its connection open
for as long as the agent runs: when either side closes it, the other ends
the launch.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

from vaara.oslayer.guard import SOCKET_PATH

NOT_RUNNING = "the OS guard is not running; start it with: sudo vaara os-guard"


class GuardError(RuntimeError):
    pass


class GuardUnavailable(GuardError):
    pass


class GuardRefused(GuardError):
    pass


def _open(path: Path, timeout: float) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(path))
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        sock.close()
        raise GuardUnavailable(NOT_RUNNING) from exc
    except OSError:
        sock.close()
        raise
    return sock


def _exchange(sock: socket.socket, payload: dict) -> dict:
    sock.sendall((json.dumps(payload) + "\n").encode())
    data = b""
    while b"\n" not in data:
        chunk = sock.recv(65536)
        if not chunk:
            break
        data += chunk
    if not data:
        raise GuardUnavailable("the guard closed the connection without answering")
    try:
        reply = json.loads(data.split(b"\n", 1)[0])
    except ValueError as exc:
        raise GuardError("the guard's answer is not JSON") from exc
    if not isinstance(reply, dict) or not reply.get("ok"):
        error = reply.get("error") if isinstance(reply, dict) else reply
        raise GuardRefused(f"the guard refused: {error}")
    return reply


def request(payload: dict, *, socket_path: Path = SOCKET_PATH, timeout: float = 10.0) -> dict:
    """One question to the guard, one answer."""
    with _open(Path(socket_path), timeout) as sock:
        return _exchange(sock, payload)


class LaunchHandle:
    """A launch the guard accepted. It lasts as long as this connection."""

    def __init__(self, sock: socket.socket, reply: dict) -> None:
        self._sock = sock
        self._closing = False
        self.launch_id = str(reply.get("launch", ""))
        self.cgroup = str(reply.get("cgroup", ""))

    def wait_closed(self) -> bool:
        """Block until the connection ends. True when the guard ended it."""
        self._sock.settimeout(None)
        try:
            while self._sock.recv(4096):
                pass
        except OSError:
            pass
        return not self._closing

    def close(self) -> None:
        self._closing = True
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()


def open_launch(payload: dict, *, socket_path: Path = SOCKET_PATH,
                timeout: float = 30.0) -> LaunchHandle:
    sock = _open(Path(socket_path), timeout)
    try:
        return LaunchHandle(sock, _exchange(sock, payload))
    except BaseException:
        sock.close()
        raise
