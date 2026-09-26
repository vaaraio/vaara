# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hook forwarding: a Vaara adapter's hooks inside a ``vaara run`` launch.

A harness with a Vaara adapter runs ``vaara hook <event>`` for every tool
call. Under ``vaara run`` that hook starts inside the agent's tree, where the
floor seals ``~/.vaara``: it cannot read its settings or write the trail, so
it cannot decide, and a hook that cannot decide blocks the call.

``vaara run`` stays outside the tree, so it answers for the hook. Before the
agent starts it binds a unix socket in the abstract namespace and puts its
name in ``VAARA_RUN_SOCKET``. ``vaara hook`` finds the variable and relays
the event instead of deciding it: its arguments and stdin go to ``vaara
run``, which runs the same ``vaara hook`` outside the floor and sends back
the exit status, stdout and stderr. The verdict is the one the hook gives
outside any launch.

- The socket has no path, so no process in the tree can remove it or bind a
  socket of its own in its place, and the name stays taken for as long as
  ``vaara run`` holds it.
- ``vaara run`` answers a caller only when it runs as the same user and
  sits in the launch's cgroup (or, on a launch without one, carries the
  ``vaara-agent`` profile).
- ``vaara hook`` sends only to the process ``VAARA_RUN_PID`` names.
- The hook outside the floor runs with ``vaara run``'s own environment.
  Nothing in the request can set a variable, so a caller cannot point it at
  another policy, trail or Python path.
- When the relay cannot reach ``vaara run`` it exits 125 with the reason on
  stderr. The gate in front of the hook turns that into a blocked call.

Any process in the tree can send an event of its own over the socket, in the
same way it could run the hook directly. It gets the verdict the hook gives
that event, and a record of it lands on the trail.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

SOCKET_ENV = "VAARA_RUN_SOCKET"
PID_ENV = "VAARA_RUN_PID"

#: Largest request read, stdin included. A Write of a large file carries the
#: whole file in the event.
MAX_REQUEST = 64 * 1024 * 1024

#: Exit status of a relay that could not reach ``vaara run``.
EXIT_UNREACHED = 125

_HOOK_MAIN = "import sys; from vaara.cli import main; sys.exit(main())"


def _deadline() -> int:
    from vaara.integrations._hook_gate import DEADLINE

    return DEADLINE


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _cgroup_of(pid: int) -> str:
    """The cgroup v2 path of ``pid``, as ``/proc/<pid>/cgroup`` gives it."""
    try:
        text = Path(f"/proc/{pid}/cgroup").read_text()
    except OSError:
        return ""
    for line in text.splitlines():
        if line.startswith("0::"):
            return line[3:]
    return ""


class HookServer:
    """``vaara run``'s side: answers the hooks of one launch."""

    def __init__(self, *, cgroup: str = "", cgroup_root: Path = Path("/sys/fs/cgroup"),
                 hook_cmd: Optional[list[str]] = None) -> None:
        self.name = f"vaara-run-{os.getpid()}-{secrets.token_hex(8)}"
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # A leading NUL puts the name in the abstract namespace.
        self._sock.bind("\0" + self.name)
        self._sock.listen(16)
        self._cgroup = ""
        self._cgroup_root = cgroup_root
        self.set_cgroup(cgroup)
        self._hook_cmd = hook_cmd or [sys.executable, "-c", _HOOK_MAIN]
        self._closed = threading.Event()

    def environ(self) -> dict[str, str]:
        """The variables the agent starts with."""
        return {SOCKET_ENV: self.name, PID_ENV: str(os.getpid())}

    def set_cgroup(self, path: str) -> None:
        """Answer only callers in the cgroup at ``path`` (a launch's directory)."""
        if not path:
            self._cgroup = ""
            return
        try:
            rel = Path(path).relative_to(self._cgroup_root)
        except ValueError:
            rel = Path(path)
        self._cgroup = "/" + str(rel).strip("/")

    def start(self) -> None:
        threading.Thread(target=self._serve, daemon=True, name="vaara-run-hooks").start()

    def close(self) -> None:
        self._closed.set()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._connection, args=(conn,), daemon=True,
                             name="vaara-run-hook").start()

    def allowed(self, pid: int, uid: int) -> bool:
        if uid != os.getuid():
            return False
        if not self._cgroup:
            # No cgroup for this launch: the caller has to carry the profile.
            from vaara.oslayer import floor

            return floor.is_agent_label(floor.label_of(pid))
        got = _cgroup_of(pid)
        return got == self._cgroup or got.startswith(self._cgroup + "/")

    def _connection(self, conn: socket.socket) -> None:
        from vaara.oslayer.guard import peer_credentials

        with conn:
            try:
                pid, uid, _ = peer_credentials(conn)
                if not self.allowed(pid, uid):
                    reply = {"ok": False, "error": "caller is not in this launch"}
                else:
                    reply = self.answer(_read(conn))
            except Exception as exc:  # noqa: BLE001 - the caller gets the reason
                reply = {"ok": False, "error": str(exc)}
            try:
                conn.sendall((json.dumps(reply) + "\n").encode())
            except OSError:
                pass

    def answer(self, request: dict) -> dict:
        """Run the hook outside the floor for one relayed event."""
        argv = request.get("argv")
        if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
            raise ValueError("argv must be a list of strings")
        stdin = base64.b64decode(request.get("stdin") or "")
        cwd = request.get("cwd")
        if not isinstance(cwd, str) or not os.path.isdir(cwd):
            cwd = None
        env = {k: v for k, v in os.environ.items() if k not in (SOCKET_ENV, PID_ENV)}
        limit = _deadline() - 2
        try:
            done = subprocess.run([*self._hook_cmd, "hook", *argv], input=stdin,
                                  capture_output=True, cwd=cwd, env=env, timeout=limit)
        except subprocess.TimeoutExpired:
            return {"ok": True, "rc": 2, "stdout": "",
                    "stderr": _b64(f"vaara-governance: BLOCKED (fail-closed): the Vaara hook "
                                   f"did not answer within {limit} s.\n".encode())}
        return {"ok": True, "rc": done.returncode, "stdout": _b64(done.stdout),
                "stderr": _b64(done.stderr)}


def _read(conn: socket.socket) -> dict:
    conn.settimeout(30.0)
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = conn.recv(1 << 16)
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if b"\n" in chunk:
            break
        if size > MAX_REQUEST:
            raise ValueError("request too large")
    data = b"".join(chunks).split(b"\n", 1)[0]
    request = json.loads(data or b"{}")
    if not isinstance(request, dict):
        raise ValueError("request is not a JSON object")
    return request


def relay(argv: list[str], *, stdin: Optional[bytes] = None) -> int:
    """``vaara hook``'s side inside a launch: forward the event, replay the answer."""
    from vaara.oslayer.client import (GuardError, GuardRefused, GuardUnavailable, _exchange,
                                      _open)
    from vaara.oslayer.guard import peer_credentials

    name = os.environ.get(SOCKET_ENV, "")
    try:
        want = int(os.environ.get(PID_ENV, ""))
    except ValueError:
        want = 0
    data = sys.stdin.buffer.read() if stdin is None else stdin
    request = {"argv": argv, "stdin": _b64(data), "cwd": os.getcwd()}
    reason = ""
    try:
        with _open("\0" + name, float(_deadline())) as sock:
            pid, _, _ = peer_credentials(sock)
            if not want or pid != want:
                reason = f"the socket in {SOCKET_ENV} is not held by vaara run (pid {want})"
            else:
                reply = _exchange(sock, request)
                sys.stdout.buffer.write(base64.b64decode(reply.get("stdout") or ""))
                sys.stdout.buffer.flush()
                sys.stderr.buffer.write(base64.b64decode(reply.get("stderr") or ""))
                sys.stderr.buffer.flush()
                return int(reply.get("rc", EXIT_UNREACHED))
    except GuardUnavailable:
        reason = "vaara run did not answer"
    except GuardRefused as exc:
        reason = str(exc).replace("the guard refused", "vaara run refused", 1)
    except (GuardError, OSError, ValueError) as exc:
        reason = str(exc) or type(exc).__name__
    sys.stderr.write(f"vaara-governance: the hook runs under vaara run and could not reach "
                     f"it: {reason}\n")
    return EXIT_UNREACHED
