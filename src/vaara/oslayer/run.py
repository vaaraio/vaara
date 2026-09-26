# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara run <agent> [args...]``: start an agent under the floor.

    vaara run copilot -p "tidy the README"
    vaara run --name reviewer claude

The agent starts confined by the ``vaara-agent`` profile, in a cgroup of its
own, and everything it starts inherits both. ``vaara run`` stays as the
agent's parent, unconfined, and holds the launch open with the guard:

- It refuses to start the agent when the guard is not running, or when
  AppArmor is not enabled, rather than start it unconfined.
- When the guard goes away, it ends the launch.
- When it goes away, the guard ends whatever the launch left running.

Its exit status is the agent's.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

from vaara.oslayer import cgroup, floor
from vaara.oslayer.guard import SOCKET_PATH

USAGE = "usage: vaara run [--name NAME] [--] <agent> [args...]"
HELP = f"""{USAGE}

Start an agent under the Linux OS layer: confined by the vaara-agent AppArmor
profile, in a cgroup the guard can end, with every open and exec in your ask
and record folders decided by the guard. Needs `sudo vaara os-guard` running.

  --name NAME   The agent's name in the trail (default: the program's name)

Everything after the agent's name is passed to the agent unchanged.
"""

# Exit statuses of our own, kept out of the range agents commonly use.
EXIT_REFUSED = 2
EXIT_GUARD_GONE = 125


class RunError(RuntimeError):
    pass


def parse_args(argv: list[str]) -> tuple[Optional[str], list[str]]:
    """(name, agent argv). Options end at the agent's name or at ``--``."""
    name: Optional[str] = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            i += 1
            break
        if arg in ("-h", "--help"):
            raise SystemExit(print(HELP) or 0)
        if arg == "--name":
            if i + 1 >= len(argv):
                raise RunError("--name needs a value")
            name = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--name="):
            name = arg.split("=", 1)[1]
            i += 1
            continue
        if arg.startswith("-"):
            raise RunError(f"unknown option {arg} (options go before the agent's name)")
        break
    agent_argv = argv[i:]
    if not agent_argv:
        raise RunError(USAGE)
    return name, agent_argv


def resolve(agent: str, path_env: Optional[str] = None) -> tuple[list[str], str]:
    """(argv prefix to exec, harness binary to register) for ``agent``.

    A ``#!`` script is started through its interpreter directly. The profile
    applies to whatever program the exec starts; started through the script,
    that program is ``env``, and the interpreter it then starts would land in
    the tool profile instead of the harness one.
    """
    found = agent if "/" in agent else shutil.which(agent, path=path_env)
    if not found or not os.path.isfile(found):
        raise RunError(f"{agent}: not found")
    if not os.access(found, os.X_OK):
        raise RunError(f"{found}: not executable")
    real = os.path.realpath(found)
    try:
        with open(real, "rb") as fh:
            head = fh.read(512)
    except OSError as exc:
        raise RunError(f"{found}: {exc.strerror}") from exc
    if not head.startswith(b"#!"):
        return [real], real
    line = head[2:].split(b"\n", 1)[0].decode("utf-8", "replace").strip()
    interpreter, _, rest = line.partition(" ")
    rest = rest.strip()
    if os.path.basename(interpreter) == "env":
        # The kernel hands env the rest of the line as one argument; only -S
        # splits it.
        if rest.startswith("-S"):
            words = shlex.split(rest[2:])
        else:
            words = [rest] if rest else []
        if not words or words[0].startswith("-"):
            raise RunError(f"{found}: cannot read the interpreter from its #! line")
        program = shutil.which(words[0], path=path_env)
        if not program:
            raise RunError(f"{found}: interpreter {words[0]} not found")
        return [os.path.realpath(program), *words[1:], found], real
    if not os.path.isfile(interpreter):
        raise RunError(f"{found}: interpreter {interpreter} not found")
    return [interpreter, *([rest] if rest else []), found], real


def _child(ready_fd: int, argv: list[str]) -> None:
    """In the forked child: wait for the guard's yes, then exec under the profile."""
    try:
        if os.read(ready_fd, 1) != b"g":
            os._exit(EXIT_REFUSED)
        os.close(ready_fd)
        # Refused by the kernel when the profile is not loaded, so the agent
        # never starts unconfined.
        attr = os.open(floor.exec_attr_path(), os.O_WRONLY)
        try:
            os.write(attr, f"exec {floor.PROFILE}".encode())
        finally:
            os.close(attr)
        # Python ignores these two; an ignored signal stays ignored across
        # exec, and the agent should start with the defaults.
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
        os.execv(argv[0], argv)
    except BaseException as exc:  # noqa: BLE001 - nothing may return from here
        try:
            os.write(2, f"vaara run: could not start {argv[0]} under the floor: {exc}\n".encode())
        finally:
            os._exit(126)


def _status(code: int) -> int:
    if os.WIFEXITED(code):
        return os.WEXITSTATUS(code)
    if os.WIFSIGNALED(code):
        return 128 + os.WTERMSIG(code)
    return 1


def run(name: Optional[str], agent_argv: list[str], *,
        socket_path: Path = SOCKET_PATH) -> int:
    if not sys.platform.startswith("linux"):
        raise RunError("vaara run governs agents on Linux")
    if not floor.apparmor_enabled():
        raise RunError("AppArmor is not enabled on this machine, so the agent would run "
                       "unconfined; not starting it")
    from vaara.oslayer.client import GuardRefused, GuardUnavailable, open_launch

    prefix, binary = resolve(agent_argv[0])
    argv = prefix + agent_argv[1:]
    agent = name or os.path.basename(agent_argv[0])

    ready_r, ready_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ready_w)
        _child(ready_r, argv)
    os.close(ready_r)

    try:
        launch = open_launch({"op": "launch", "pid": pid, "agent": agent, "binary": binary,
                              "argv": argv, "cwd": os.getcwd()}, socket_path=socket_path)
    except (GuardUnavailable, GuardRefused, OSError) as exc:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        if isinstance(exc, OSError):
            raise RunError(f"the guard did not answer: {exc}") from None
        raise RunError(str(exc)) from None

    guard_gone = threading.Event()

    def _watch() -> None:
        if not launch.wait_closed():
            return
        guard_gone.set()
        # The guard is gone, so nothing decides this launch's opens any
        # more. End it rather than let it run half-governed.
        cgroup.kill(Path(launch.cgroup))
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    threading.Thread(target=_watch, daemon=True, name="vaara-run-guard").start()

    # The agent shares the terminal; it gets ^C and ^\ itself and decides.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda s, _f: os.kill(pid, s))

    os.write(ready_w, b"g")
    os.close(ready_w)
    while True:
        try:
            _, code = os.waitpid(pid, 0)
            break
        except ChildProcessError:
            code = 0
            break
        except InterruptedError:
            continue
    launch.close()
    if guard_gone.is_set():
        print("vaara run: the OS guard stopped, so the agent was ended.", file=sys.stderr)
        return EXIT_GUARD_GONE
    return _status(code)


def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        name, agent_argv = parse_args(argv)
        return run(name, agent_argv)
    except RunError as exc:
        print(f"vaara run: {exc}", file=sys.stderr)
        return EXIT_REFUSED

