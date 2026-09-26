# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara os-guard``: the root half of the OS layer.

    sudo vaara os-guard [--user NAME]

On start the guard writes the floor as ``/etc/apparmor.d/vaara-agent`` and
loads it, then holds every open and exec in the operator's ask and record
folders until it has decided. It takes launches from ``vaara run`` over
``/run/vaara/os-guard.sock``, gives each one a cgroup, and writes every
decision, launch and floor refusal to its own trail,
``/var/lib/vaara/os-layer/audit.db``. The trail is root's and readable by
the operator's group; no process in an agent's tree can open it.

Stopping the guard ends every launch and every process running under the
profile. The profile stays loaded, so the floor holds without the guard.

What the guard decides and what it cannot see:

- It decides each open and each exec an agent makes in an ask or record
  folder. The kernel gives it no event for a delete, a rename or a link, so
  those happen in an ask or record folder undecided. A block folder is part
  of the floor and none of them happen there.
- It tells an agent from any other process by the AppArmor label, so a
  program that did not start under ``vaara run`` or an attached app is not
  an agent.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import itertools
import json
import logging
import os
import pwd
import secrets
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vaara.oslayer import cgroup, floor, selection
from vaara.oslayer.denials import Denial, Follower
from vaara.oslayer.fanotify import Event, Group, mount_point
from vaara.taxonomy.actions import (ActionCategory, ActionType, BlastRadius,
                                    RegulatoryDomain, Reversibility, UrgencyClass)

logger = logging.getLogger("vaara.os-guard")

SOCKET_PATH = Path("/run/vaara/os-guard.sock")
STATE_DIR = Path("/var/lib/vaara/os-layer")
TRAIL = STATE_DIR / "audit.db"
PROFILE_FILE = Path("/etc/apparmor.d") / floor.PROFILE

KNOWN_HARNESSES = ("claude", "codex", "gemini", "copilot", "cursor-agent", "opencode")
READERS = 2
WORKERS = 32
MAX_REQUEST = 64 * 1024
_DEDUPE = 100_000

OS_OPEN = ActionType(
    "os.open", ActionCategory.DATA, Reversibility.PARTIALLY, BlastRadius.LOCAL,
    UrgencyClass.DEFERRABLE, frozenset({RegulatoryDomain.GDPR}),
    "An agent process opens a file or folder in a folder the operator picked")
OS_EXEC = ActionType(
    "os.exec", ActionCategory.INFRASTRUCTURE, Reversibility.PARTIALLY, BlastRadius.LOCAL,
    UrgencyClass.TIMELY, frozenset(),
    "An agent process runs a program from a folder the operator picked")
OS_FLOOR = ActionType(
    "os.floor", ActionCategory.GOVERNANCE, Reversibility.FULLY, BlastRadius.SELF,
    UrgencyClass.IMMEDIATE, frozenset({RegulatoryDomain.EU_AI_ACT}),
    "The kernel refused an agent process something under the floor")
OS_LAUNCH = ActionType(
    "os.launch", ActionCategory.GOVERNANCE, Reversibility.FULLY, BlastRadius.LOCAL,
    UrgencyClass.TIMELY, frozenset({RegulatoryDomain.EU_AI_ACT}),
    "vaara run started an agent under the floor")
OS_ACTIONS = [OS_OPEN, OS_EXEC, OS_FLOOR, OS_LAUNCH]


class GuardError(RuntimeError):
    pass


@dataclass
class Launch:
    launch_id: str
    agent: str
    binary: str
    uid: int
    pid: int
    cgroup: Path
    started: float = field(default_factory=time.time)


def build_pipeline(trail_path: Path):
    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.pipeline import InterceptionPipeline
    from vaara.taxonomy.actions import create_default_registry

    registry = create_default_registry()
    for action in OS_ACTIONS:
        registry.register(action)
        registry.map_tool(action.name, action.name)
    backend = SQLiteAuditBackend(str(trail_path))
    # This process is root. Left alone, the backend would list the trail in
    # root's own ~/.vaara/sources.json, where the operator's app never looks.
    # The guard lists it in the operator's file instead (_register_trail).
    backend._registered = True
    return InterceptionPipeline(registry=registry, trail=backend.load_trail())


def _comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


def _exe(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return ""


def _status_field(pid: int, name: str) -> Optional[str]:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(name + ":"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def peer_credentials(conn: socket.socket) -> tuple[int, int, int]:
    """(pid, uid, gid) of the process at the other end of a unix socket."""
    raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    return struct.unpack("3i", raw)


def parser_path() -> Optional[str]:
    return shutil.which("apparmor_parser") or next(
        (p for p in ("/sbin/apparmor_parser", "/usr/sbin/apparmor_parser") if os.path.exists(p)),
        None)


class Guard:
    def __init__(self, operator: pwd.struct_passwd, *,
                 trail_path: Path = TRAIL,
                 socket_path: Path = SOCKET_PATH,
                 profile_file: Path = PROFILE_FILE,
                 cgroup_root: Path = cgroup.CGROUP_ROOT) -> None:
        self.uid = operator.pw_uid
        self.gid = operator.pw_gid
        self.user = operator.pw_name
        self.home = operator.pw_dir
        self.trail_path = Path(trail_path)
        self.socket_path = Path(socket_path)
        self.lock_path = self.socket_path.with_suffix(".lock")
        self.profile_file = Path(profile_file)
        self.cgroup_root = Path(cgroup_root)
        self.approvals_dir = Path(self.home) / ".vaara" / "approvals"
        self.pid = os.getpid()

        self._stop = threading.Event()
        self._lock_fd: Optional[int] = None
        self._selection = selection.Selection()
        self._selection_mtime: Optional[float] = None
        self._harness: dict[str, None] = {}
        self._mounts: set[str] = set()
        self._launches: dict[str, Launch] = {}
        self._launch_lock = threading.Lock()
        self._profile_lock = threading.Lock()

        self._group: Optional[Group] = None
        self._pipeline = None
        self._trail_lock = threading.Lock()
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=WORKERS, thread_name_prefix="vaara-decide")
        self._seq = itertools.count(1)
        self._held: dict[int, tuple[Event, float]] = {}
        self._held_lock = threading.Lock()
        self._recorded: OrderedDict[tuple, None] = OrderedDict()
        self._answers: OrderedDict[tuple, bool] = OrderedDict()
        self._asking: dict[tuple, concurrent.futures.Future] = {}
        self._open_requests: set[str] = set()
        self._cache_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._server: Optional[socket.socket] = None
        self._follower: Optional[Follower] = None

    # ── Start and stop ────────────────────────────────────────────

    def preflight(self) -> None:
        if not sys.platform.startswith("linux"):
            raise GuardError("the OS layer runs on Linux")
        if os.geteuid() != 0:
            raise GuardError("the guard runs as root: sudo vaara os-guard")
        if not floor.apparmor_enabled():
            raise GuardError("AppArmor is not enabled on this machine, so there is no floor "
                             "to load and no agent will be started")
        if parser_path() is None:
            raise GuardError("apparmor_parser was not found (package: apparmor)")
        if not cgroup.available(self.cgroup_root):
            raise GuardError("cgroup v2 with cgroup.kill (Linux 5.14 or later) is needed")

    def _take_lock(self) -> None:
        """One guard at a time: an exclusive lock held for the process's life."""
        self.lock_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise GuardError(f"another guard is already running ({self.lock_path})") from None
        os.ftruncate(fd, 0)
        os.write(fd, f"{self.pid}\n".encode())
        self._lock_fd = fd

    def start(self) -> None:
        self.preflight()
        self._take_lock()
        self._prepare_state()
        self._prepare_approvals()
        self._pipeline = build_pipeline(self.trail_path)
        self._register_trail()
        self._selection = self._read_selection()
        self._harness = dict.fromkeys(
            p for p in (shutil.which(n) for n in KNOWN_HARNESSES) if p)
        self._load_profile()

        self._group = Group(nonblocking=True)
        self._apply_marks()
        for i in range(READERS):
            self._spawn(self._reader, f"vaara-fanotify-{i}")
        self._follower = Follower(self._on_denial)
        self._follower.start()
        self._serve()
        self._spawn(self._housekeeping, "vaara-housekeeping")
        logger.info("guarding for %s: %d folder(s), %d app(s); trail %s",
                    self.user, len(self._selection.folders), len(self._selection.apps),
                    self.trail_path)

    def run(self) -> int:
        stop = threading.Event()

        def _signal(_signum, _frame):
            stop.set()

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, _signal)
        self.start()
        try:
            while not stop.wait(1.0) and not self._stop.is_set():
                pass
        finally:
            self.stop()
        return 0

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            try:
                self.socket_path.unlink()
            except OSError:
                pass
        ended = self.end_all_agents()
        if self._follower is not None:
            self._follower.stop()
        with self._held_lock:
            held = list(self._held.values())
            self._held.clear()
        for event, _deadline in held:
            self._respond(event, False)
        # A question whose agent is gone should not stay on the operator's
        # screen until it times out.
        for action_id in list(self._open_requests):
            try:
                (self.approvals_dir / f"{action_id}.request.json").unlink()
            except OSError:
                pass
        if self._group is not None:
            self._group.close()
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
        logger.info("stopped; ended %d agent process(es). The floor stays loaded.", ended)

    def end_all_agents(self) -> int:
        """Kill every launch cgroup and every process under the profile."""
        for path in cgroup.launches(self.cgroup_root):
            cgroup.remove(path)
        with self._launch_lock:
            self._launches.clear()
        return kill_labelled()

    def _spawn(self, target, name: str) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    def _prepare_state(self) -> None:
        self.trail_path.parent.mkdir(parents=True, exist_ok=True)
        os.chown(self.trail_path.parent, 0, self.gid)
        os.chmod(self.trail_path.parent, 0o750)

    def _prepare_approvals(self) -> None:
        """The operator's approvals directory and key, owned by the operator.

        The guard writes each request there as root. The operator's app reads
        it and signs a decision with the key, so both must stay the
        operator's: a directory or key the guard created as root would leave
        the app unable to answer.
        """
        from vaara.approvals import approval_key_path

        vaara_dir = self.approvals_dir.parent
        key = approval_key_path(self.approvals_dir)
        for d in (vaara_dir, self.approvals_dir, key.parent):
            if not d.exists():
                d.mkdir(mode=0o700)
                os.lchown(d, self.uid, self.gid)
        if not key.exists():
            fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as fh:
                os.fchown(fh.fileno(), self.uid, self.gid)
                fh.write(secrets.token_hex(32))

    def _register_trail(self) -> None:
        """List the trail in the operator's ~/.vaara/sources.json, as the operator."""
        code = "import sys; from vaara.audit import sources; sources.register(sys.argv[1])"
        env = {"HOME": self.home, "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        try:
            subprocess.run([sys.executable, "-c", code, str(self.trail_path)],
                           user=self.uid, group=self.gid, env=env, timeout=30,
                           check=False, capture_output=True)
        except (OSError, subprocess.SubprocessError):
            logger.warning("could not list %s in %s's sources.json", self.trail_path, self.user)

    # ── The floor ─────────────────────────────────────────────────

    def _read_selection(self) -> selection.Selection:
        path = selection.path_for(self.home)
        try:
            self._selection_mtime = path.stat().st_mtime
        except OSError:
            self._selection_mtime = None
        return selection.load(self.home)

    def render_profile(self) -> str:
        sel = self._selection
        return floor.render(
            [self.home, "/root"],
            block_folders=sel.folders_in("block"),
            watched_folders=sel.folders_in("ask") + sel.folders_in("record"),
            harness_binaries=list(self._harness),
            apps=sel.apps,
        )

    def _load_profile(self) -> None:
        with self._profile_lock:
            text = self.render_profile()
            tmp = self.profile_file.with_name(f".{self.profile_file.name}.tmp")
            tmp.write_text(text)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.profile_file)
            done = subprocess.run([parser_path(), "-r", str(self.profile_file)],
                                  capture_output=True, text=True, timeout=120)
            if done.returncode != 0:
                raise GuardError(f"apparmor_parser refused the profile: {done.stderr.strip()}")

    def _apply_marks(self) -> None:
        sel = self._selection
        wanted = {mount_point(p) for p in sel.folders_in("ask") + sel.folders_in("record")}
        for point in sorted(wanted - self._mounts):
            self._group.mark_mount(point)
        for point in sorted(self._mounts - wanted):
            self._group.unmark_mount(point)
        self._mounts = wanted

    def reload(self) -> None:
        """Pick up a changed selection: profile first, then marks, then decisions."""
        new = self._read_selection()
        old = self._selection
        self._selection = new
        try:
            self._load_profile()
        except GuardError:
            self._selection = old
            raise
        self._apply_marks()
        logger.info("selection reloaded: %d folder(s), %d app(s)",
                    len(new.folders), len(new.apps))

    def add_harness(self, binary: str) -> None:
        if not binary:
            return
        known = floor.harness_paths([binary])
        if all(b in self._harness for b in known):
            return
        for b in known:
            self._harness[b] = None
        self._load_profile()

    # ── Decisions ─────────────────────────────────────────────────

    def _reader(self) -> None:
        group = self._group
        poller = select.poll()
        poller.register(group.fd, select.POLLIN)
        while not self._stop.is_set():
            try:
                if not poller.poll(500):
                    continue
                events = group.read()
            except BlockingIOError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            for event in events:
                self._dispatch(event)

    def _dispatch(self, event: Event) -> None:
        """Answer ``event`` now, or hand it to a worker. Exactly one of the two."""
        if event.fd < 0:
            return
        if event.pid == self.pid:
            self._respond(event, True)
            return
        try:
            triage = self._triage(event)
        except Exception:
            logger.exception("open by pid %d could not be triaged; refused", event.pid)
            triage = False
        if isinstance(triage, bool):
            self._respond(event, triage)
            return
        path, folder, label = triage
        seq = next(self._seq)
        wait = self._selection.ask_timeout + 15 if folder.mode == "ask" else 30
        with self._held_lock:
            self._held[seq] = (event, time.monotonic() + wait)
        try:
            self._pool.submit(self._decide, seq, event, path, folder, label)
        except RuntimeError:
            # The pool is shutting down with the guard.
            self._answer(seq, False)

    def _triage(self, event: Event):
        """True or False to answer at once, or (path, folder, label) to decide."""
        path = event.path()
        folder = self._selection.match(path) if path else None
        if folder is None:
            return True
        label = floor.label_of(event.pid)
        if not floor.is_agent_label(label):
            # No label at all means the process is gone and nobody is waiting.
            return bool(label)
        return path, folder, label

    def _respond(self, event: Event, allow: bool) -> None:
        try:
            self._group.respond(event, allow)
        except OSError:
            pass

    def _answer(self, seq: int, allow: bool) -> None:
        with self._held_lock:
            held = self._held.pop(seq, None)
        if held is not None:
            self._respond(held[0], allow)

    def _decide(self, seq: int, event: Event, path: str, folder: selection.Folder,
                label: str) -> None:
        allow = False
        try:
            allow = self.verdict(event, path, folder, label)
        except Exception:
            # Fail closed: an open the guard could not decide does not happen.
            logger.exception("could not decide %s by pid %d; refused", path, event.pid)
        finally:
            self._answer(seq, allow)

    def who(self, pid: int) -> tuple[str, str]:
        """(agent id, session id) for ``pid``: its launch, or the attached app."""
        launch_id = cgroup.launch_of(pid)
        with self._launch_lock:
            launch = self._launches.get(launch_id) if launch_id else None
        if launch is not None:
            return launch.agent, f"launch-{launch.launch_id}"
        return f"app:{_comm(pid) or pid}", f"pid-{pid}"

    def verdict(self, event: Event, path: str, folder: selection.Folder, label: str) -> bool:
        tool = OS_EXEC.name if event.is_exec else OS_OPEN.name
        agent, session = self.who(event.pid)
        key = (session, tool, path)
        params = {"path": path, "folder": folder.path, "mode": folder.mode,
                  "pid": event.pid, "comm": _comm(event.pid), "program": _exe(event.pid),
                  "profile": label}
        if folder.mode == "block":
            self._record(agent, session, tool, params, "deny",
                         f"block folder {folder.path}", "os-layer:block")
            return False
        if folder.mode == "record":
            with self._cache_lock:
                if key in self._recorded:
                    return True
            self._record(agent, session, tool, params, "allow",
                         f"record folder {folder.path}", "os-layer:record")
            with self._cache_lock:
                self._recorded[key] = None
                while len(self._recorded) > _DEDUPE:
                    self._recorded.popitem(last=False)
            return True
        return self._ask(key, agent, session, tool, params, folder)

    def _ask(self, key: tuple, agent: str, session: str, tool: str, params: dict,
             folder: selection.Folder) -> bool:
        with self._cache_lock:
            if key in self._answers:
                return self._answers[key]
            future = self._asking.get(key)
            owner = future is None
            if owner:
                future = concurrent.futures.Future()
                self._asking[key] = future
        if not owner:
            # The same file, the same launch, already waiting on the operator.
            try:
                return bool(future.result(timeout=self._selection.ask_timeout + 10))
            except concurrent.futures.TimeoutError:
                return False
        answer = False
        try:
            answer = self._ask_operator(key, agent, session, tool, params, folder)
            return answer
        finally:
            future.set_result(answer)
            with self._cache_lock:
                self._asking.pop(key, None)

    def _ask_operator(self, key: tuple, agent: str, session: str, tool: str, params: dict,
                      folder: selection.Folder) -> bool:
        from vaara.approvals import request_approval

        verb = "run" if tool == OS_EXEC.name else "open"
        result = self._intercept(
            agent, session, tool, params, "escalate",
            f"ask folder {folder.path}: the operator decides", "os-layer:ask")
        detail = f"{agent} ({params['comm'] or params['pid']}) wants to {verb} {params['path']}"
        self._open_requests.add(result.action_id)
        try:
            human = request_approval(result.action_id, tool, detail,
                                     approvals_dir=self.approvals_dir,
                                     timeout=self._selection.ask_timeout)
        finally:
            self._open_requests.discard(result.action_id)
        if human not in ("approve", "deny"):
            return False
        with self._trail_lock:
            self._pipeline.resolve_escalation(
                result.action_id, "allow" if human == "approve" else "deny",
                reviewer="approvals-handshake",
                justification="human decision via ~/.vaara/approvals",
                approver="human", human_disposed=True)
        allowed = human == "approve"
        with self._cache_lock:
            self._answers[key] = allowed
            while len(self._answers) > _DEDUPE:
                self._answers.popitem(last=False)
        return allowed

    def _intercept(self, agent: str, session: str, tool: str, params: dict,
                   decision: str, reason: str, policy_id: str):
        with self._trail_lock:
            return self._pipeline.intercept(
                agent_id=agent, tool_name=tool, parameters=params, session_id=session,
                policy_decision=decision, policy_reason=reason, policy_id=policy_id)

    def _record(self, agent: str, session: str, tool: str, params: dict,
                decision: str, reason: str, policy_id: str) -> None:
        self._intercept(agent, session, tool, params, decision, reason, policy_id)

    def _on_denial(self, denial: Denial) -> None:
        agent, session = self.who(denial.pid) if denial.pid else ("unknown", "")
        self._record(agent, session, OS_FLOOR.name, {
            "operation": denial.operation, "target": denial.target,
            "requested": denial.requested, "denied": denial.denied,
            "pid": denial.pid, "comm": denial.comm, "profile": denial.profile,
            "stamp": denial.stamp,
        }, "deny", f"the floor refused {denial.operation} on {denial.target}",
            "os-layer:floor")

    # ── Launches ──────────────────────────────────────────────────

    def _serve(self) -> None:
        self.socket_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        # The operator's group reaches the socket; who is served is then
        # decided by the peer's credentials on each connection.
        if os.geteuid() == 0:
            os.chown(self.socket_path, 0, self.gid)
        os.chmod(self.socket_path, 0o660)
        server.listen(16)
        server.settimeout(0.5)
        self._server = server
        self._spawn(self._accept, "vaara-launches")

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._connection, args=(conn,), daemon=True,
                             name="vaara-launch").start()

    def _connection(self, conn: socket.socket) -> None:
        with conn:
            try:
                reply, launch = self.handle_request(conn)
            except Exception as exc:
                logger.exception("request failed")
                reply, launch = {"ok": False, "error": str(exc)}, None
            try:
                conn.sendall((json.dumps(reply) + "\n").encode())
            except OSError:
                if launch is not None:
                    self._end_launch(launch)
                return
            if launch is None:
                return
            # The launch lasts as long as this connection. When vaara run
            # goes away, for any reason, whatever it left behind goes too.
            conn.settimeout(None)
            try:
                while conn.recv(4096):
                    pass
            except OSError:
                pass
            self._end_launch(launch)

    def handle_request(self, conn: socket.socket) -> tuple[dict, Optional[Launch]]:
        peer_pid, peer_uid, peer_gid = peer_credentials(conn)
        if floor.is_agent_label(floor.label_of(peer_pid)):
            return {"ok": False, "error": "a process under the floor cannot talk to the guard"}, None
        if peer_uid not in (0, self.uid):
            return {"ok": False, "error": f"this guard serves {self.user} only"}, None
        request = _read_request(conn)
        op = request.get("op")
        if op == "status":
            return self.status(), None
        if op == "reload":
            self.reload()
            return {"ok": True}, None
        if op == "launch":
            return self.launch(request, peer_pid, peer_uid, peer_gid)
        return {"ok": False, "error": f"unknown op {op!r}"}, None

    def launch(self, request: dict, peer_pid: int, peer_uid: int,
               peer_gid: int) -> tuple[dict, Optional[Launch]]:
        try:
            child = int(request.get("pid"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "launch needs the pid of the waiting child"}, None
        if _status_field(child, "PPid") != str(peer_pid):
            return {"ok": False, "error": f"pid {child} is not a child of the caller"}, None
        uid_line = _status_field(child, "Uid") or ""
        if not uid_line or int(uid_line.split()[0]) != peer_uid:
            return {"ok": False, "error": f"pid {child} does not run as the caller"}, None
        if floor.label_of(child) not in ("", "unconfined"):
            return {"ok": False, "error": f"pid {child} is already confined"}, None
        binary = str(request.get("binary") or "")
        agent = str(request.get("agent") or os.path.basename(binary) or "agent")[:128]
        self.add_harness(binary)
        launch_id = secrets.token_hex(6)
        path = cgroup.create(launch_id, peer_uid, peer_gid, self.cgroup_root)
        try:
            cgroup.add(path, child)
        except OSError:
            cgroup.remove(path)
            raise
        launch = Launch(launch_id=launch_id, agent=agent, binary=binary, uid=peer_uid,
                        pid=child, cgroup=path)
        with self._launch_lock:
            self._launches[launch_id] = launch
        argv = [str(a)[:512] for a in (request.get("argv") or [])][:64]
        self._record(agent, f"launch-{launch_id}", OS_LAUNCH.name, {
            "binary": binary, "argv": argv, "cwd": str(request.get("cwd") or "")[:1024],
            "uid": peer_uid, "pid": child, "launch": launch_id,
        }, "allow", "vaara run started the agent under the floor", "os-layer:launch")
        return {"ok": True, "launch": launch_id, "cgroup": str(path),
                "profile": floor.PROFILE}, launch

    def _end_launch(self, launch: Launch) -> None:
        cgroup.remove(launch.cgroup)
        with self._launch_lock:
            self._launches.pop(launch.launch_id, None)
        session = f"launch-{launch.launch_id}"
        with self._cache_lock:
            for cache in (self._recorded, self._answers):
                for key in [k for k in cache if k[0] == session]:
                    del cache[key]

    def status(self) -> dict:
        sel = self._selection
        with self._launch_lock:
            launches = [{"launch": x.launch_id, "agent": x.agent, "pid": x.pid,
                         "binary": x.binary, "started": x.started}
                        for x in self._launches.values()]
        with self._held_lock:
            waiting = len(self._held)
        return {
            "ok": True, "pid": self.pid, "user": self.user,
            "profile": floor.PROFILE, "profile_loaded": floor.profile_loaded(),
            "trail": str(self.trail_path),
            "folders": [{"path": f.path, "mode": f.mode} for f in sel.folders],
            "apps": list(sel.apps), "ask_timeout": sel.ask_timeout,
            "launches": launches, "waiting": waiting,
        }

    # ── Housekeeping ──────────────────────────────────────────────

    def _housekeeping(self) -> None:
        last_listed = time.monotonic()
        while not self._stop.wait(1.0):
            self._sweep_held()
            try:
                mtime = selection.path_for(self.home).stat().st_mtime
            except OSError:
                mtime = None
            if mtime != self._selection_mtime:
                try:
                    self.reload()
                except Exception:
                    logger.exception("selection change not applied; the previous one stands")
            if time.monotonic() - last_listed > 3600:
                self._register_trail()
                last_listed = time.monotonic()

    def _sweep_held(self) -> None:
        """Refuse whatever has waited past its deadline. Nothing waits forever."""
        now = time.monotonic()
        with self._held_lock:
            late = [seq for seq, (_e, deadline) in self._held.items() if deadline <= now]
            events = [self._held.pop(seq)[0] for seq in late]
        for event in events:
            self._respond(event, False)


def _read_request(conn: socket.socket) -> dict:
    conn.settimeout(10.0)
    data = b""
    while b"\n" not in data:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
        if len(data) > MAX_REQUEST:
            raise GuardError("request too large")
    try:
        request = json.loads(data.split(b"\n", 1)[0] or b"{}")
    except ValueError as exc:
        raise GuardError("request is not JSON") from exc
    if not isinstance(request, dict):
        raise GuardError("request is not a JSON object")
    return request


def kill_labelled() -> int:
    """SIGKILL every process whose AppArmor label is the agent profile's."""
    killed = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if floor.is_agent_label(floor.label_of(pid)):
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except OSError:
                pass
    return killed


def unload(profile_file: Path = PROFILE_FILE) -> str:
    """Remove the profile from the kernel and from disk. The floor is gone after this."""
    parser = parser_path()
    if parser is None:
        raise GuardError("apparmor_parser was not found")
    if profile_file.exists():
        done = subprocess.run([parser, "-R", str(profile_file)], capture_output=True,
                              text=True, timeout=60)
        if done.returncode != 0 and floor.profile_loaded():
            raise GuardError(f"apparmor_parser could not remove the profile: "
                             f"{done.stderr.strip()}")
        profile_file.unlink()
    return floor.PROFILE


def operator_account(name: Optional[str]) -> pwd.struct_passwd:
    name = name or os.environ.get("SUDO_USER") or ""
    if not name:
        raise GuardError("name the operator: sudo vaara os-guard, or --user NAME under systemd")
    try:
        return pwd.getpwnam(name)
    except KeyError as exc:
        raise GuardError(f"no such user: {name}") from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vaara os-guard",
        description="The root half of the Linux OS layer: loads the floor, decides each "
                    "open and exec agents make in the folders you picked, takes launches "
                    "from vaara run, and keeps its own trail. Runs in the foreground.")
    p.add_argument("--user", default=None,
                   help="The operator whose folders, apps and approvals the guard serves "
                        "(default: the user who ran sudo)")
    p.add_argument("--unload", action="store_true",
                   help="Remove the vaara-agent profile from the kernel and from "
                        "/etc/apparmor.d, then exit. The floor no longer holds after this.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s vaara os-guard: %(message)s")
    try:
        if args.unload:
            if os.geteuid() != 0:
                raise GuardError("unloading the profile needs root: sudo vaara os-guard --unload")
            print(f"vaara os-guard: removed the {unload()} profile. The floor no longer holds.",
                  file=sys.stderr)
            return 0
        code = Guard(operator_account(args.user)).run()
    except GuardError as exc:
        print(f"vaara os-guard: {exc}", file=sys.stderr)
        return 2
    # Workers still waiting on a withdrawn question would hold the exit until
    # their timeout. Every record is already committed; leave now.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
