"""The Linux OS layer end to end: the kernel, the guard, a real agent with no adapter.

Runs in the os-layer-e2e CI job, which starts ``vaara os-guard`` under sudo
for the runner's account and then runs this file as that account. Copilot
CLI, with no Vaara hooks installed, starts under ``vaara run`` against a
scripted local model. The model asks it to change files on the floor (a
file under ~/.vaara, a harness hook file, the approval key, a new file in
Copilot's own hooks folder through its native file tool), to write in a
block folder, to read and write in a record folder, to read a file in an ask
folder (which this test denies, signed, as the operator), and to write one
file nothing governs. The test reads back the files, what Copilot CLI told
the model, and the guard's own trail. A second test installs Vaara's Copilot
hooks and checks that they decide from inside the tree, through the relay to
``vaara run``.

Skipped unless VAARA_OSLAYER_E2E=1 and a Copilot CLI binary is on PATH.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

COPILOT = shutil.which("copilot")
pytestmark = pytest.mark.skipif(
    os.environ.get("VAARA_OSLAYER_E2E") != "1" or not COPILOT,
    reason="the OS layer end-to-end test runs in the os-layer-e2e job")

TRAIL = Path("/var/lib/vaara/os-layer/audit.db")


class _Model:
    """An OpenAI chat completions endpoint playing back one scripted turn per call."""

    def __init__(self, script: list):
        self.script, self.requests = script, []
        model = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self._send(200, "application/json", b'{"data": []}')

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("content-length", 0)))
                model.requests.append(json.loads(body or b"{}"))
                i = len(model.requests) - 1
                turn = model.script[i] if i < len(model.script) else "done"
                if isinstance(turn, str):
                    delta, finish = {"role": "assistant", "content": turn}, "stop"
                else:
                    delta = {"role": "assistant", "tool_calls": [
                        {"index": k, "id": f"call_{i}_{k}", "type": "function",
                         "function": {"name": name, "arguments": json.dumps(args)}}
                        for k, (name, args) in enumerate(turn)]}
                    finish = "tool_calls"
                base = {"id": f"c{i}", "object": "chat.completion.chunk",
                        "created": 0, "model": "gpt-4.1"}
                chunks = [{**base, "choices": [{"index": 0, "delta": delta,
                                                "finish_reason": None}]},
                          {**base, "choices": [{"index": 0, "delta": {},
                                                "finish_reason": finish}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                     "total_tokens": 2}}]
                data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
                self._send(200, "text/event-stream", (data + "data: [DONE]\n\n").encode())

            def _send(self, code, kind, data):
                self.send_response(code)
                self.send_header("content-type", kind)
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = HTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def outputs(self) -> list[str]:
        if not self.requests:
            return []
        return [str(m.get("content")) for m in self.requests[-1].get("messages", [])
                if m.get("role") == "tool"]


def _run_vaara(argv: list[str], cwd: Path, env: dict, timeout: float = 300) -> subprocess.CompletedProcess:
    """``vaara run`` as a subprocess, with the whole tree ended on a timeout.

    The agent runs in a cgroup under ``vaara run`` and inherits the output
    pipes. Killing only ``vaara run`` on a timeout left the agent holding the
    pipes, so the test hung on reading them until the job's six-hour limit.
    The tree is its own session here; on a timeout the session is killed and
    the test fails with the captured output in hand.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "vaara.cli", "run", *argv], cwd=cwd, env=env,
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        out, err = proc.communicate(timeout=30)
        pytest.fail(f"vaara run did not finish within {timeout:.0f} s; tree killed.\n"
                    f"stdout:\n{(out or '')[-3000:]}\nstderr:\n{(err or '')[-3000:]}")
    return subprocess.CompletedProcess(proc.args, proc.returncode, out or "", err or "")


def _stop_model(model: "_Model") -> None:
    """Stop the fake model; a handler stuck mid-request must not hang the test."""
    t = threading.Thread(target=model.server.shutdown, daemon=True)
    t.start()
    t.join(timeout=10)
    model.server.server_close()


def _sh(command: str) -> tuple:
    return ("bash", {"command": command, "description": "run"})


def _deny_asks(approvals: Path, stop: threading.Event, seen: list) -> threading.Thread:
    """Play the operator: deny, signed, every question the guard asks."""
    from vaara.approvals import write_decision

    def loop() -> None:
        while not stop.is_set():
            for req in approvals.glob("*.request.json"):
                try:
                    request = json.loads(req.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if request.get("action_id") in {r.get("action_id") for r in seen}:
                    continue
                seen.append(request)
                write_decision(request["action_id"], "deny", approvals_dir=approvals)
            stop.wait(0.1)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def _actions() -> list[dict]:
    """The guard's trail, one entry per action: agent, tool, parameters, outcomes."""
    conn = sqlite3.connect(f"file:{TRAIL}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT action_id, agent_id, tool_name, data FROM audit_records "
                            "ORDER BY seq").fetchall()
    finally:
        conn.close()
    actions: dict[str, dict] = {}
    for action_id, agent, tool, data in rows:
        a = actions.setdefault(action_id, {"agent": agent, "tool": tool, "parameters": {},
                                           "outcomes": []})
        d = json.loads(data or "{}")
        a["parameters"] = d.get("parameters") or a["parameters"]
        for key in ("decision", "resolution"):
            if key in d:
                a["outcomes"].append(d[key])
    return list(actions.values())


def test_the_floor_holds_and_the_guard_decides(tmp_path):
    from vaara.oslayer import manage, selection

    home = Path.home()
    base = home / "oslayer-e2e"
    blocked, recorded, asked = base / "blocked", base / "recorded", base / "asked"
    for d in (blocked, recorded, asked):
        d.mkdir(parents=True, exist_ok=True)
    (recorded / "notes.txt").write_text("notes\n")
    (asked / "secret.txt").write_text("the-asked-secret\n")

    sentinel = home / ".vaara" / "e2e-sentinel.db"
    sentinel.write_text("sentinel\n")
    hook = home / ".claude" / "hooks" / "e2e.sh"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\nexit 0\n")
    key = (home / ".vaara" / "keys" / "approval-hmac.key").read_text(encoding="utf-8").strip()
    native_hook = home / ".copilot" / "hooks" / "e2e.json"
    free = tmp_path / "free.txt"

    sel = selection.Selection(folders=[
        selection.Folder(str(blocked), "block"),
        selection.Folder(str(recorded), "record"),
        selection.Folder(str(asked), "ask"),
    ], ask_timeout=30)
    _path, told = manage.save(sel)
    assert told, "the guard did not take the reload"
    status = manage.guard_status()
    assert status and status["profile_loaded"], status
    assert {f["mode"] for f in status["folders"]} == {"block", "record", "ask"}

    model = _Model([
        [_sh(f"echo overwritten > {sentinel}; rm -f {sentinel}")],
        [_sh(f"echo tampered >> {hook}")],
        [_sh(f"cat {home}/.vaara/keys/approval-hmac.key")],
        [("create", {"path": str(native_hook), "file_text": '{"version": 1}'})],
        [_sh(f"echo x > {blocked}/new.txt")],
        [_sh(f"cat {recorded}/notes.txt && echo more > {recorded}/new.txt")],
        [_sh(f"cat {asked}/secret.txt")],
        [_sh(f"echo free > {free}")],
        "done",
    ])
    work = tmp_path / "work"
    work.mkdir()
    env = {"HOME": str(home), "USERPROFILE": str(home), "COPILOT_HOME": str(tmp_path / "copilot-home"),
           "COPILOT_PROVIDER_BASE_URL": f"http://127.0.0.1:{model.port}/v1",
           "COPILOT_MODEL": "gpt-4.1", "COPILOT_OFFLINE": "true",
           "PATH": os.environ.get("PATH", ""), "NO_COLOR": "1",
           "PYTHONPATH": os.pathsep.join(sys.path)}

    stop, seen = threading.Event(), []
    _deny_asks(home / ".vaara" / "approvals", stop, seen)
    try:
        proc = _run_vaara(["copilot", "-p", "go", "--allow-all-tools", "--allow-all-paths",
                           "--no-auto-update", "--no-ask-user"], work, env)
    finally:
        stop.set()
        _stop_model(model)
    log = proc.stdout[-3000:] + proc.stderr[-3000:]

    out = model.outputs()
    assert len(out) == 8, (out, log)
    # Every call reached the system. Copilot CLI's own path guard refusing
    # one would leave the floor untested while the file checks still pass.
    assert not any("could not request permission" in o for o in out), (out, log)
    # And the shell steps on the floor failed in the system, not before it.
    for i in (0, 1, 2, 4):
        assert "Permission denied" in out[i], (i, out[i], log)

    # The floor: nothing on it changed, and the key never reached the model.
    assert sentinel.read_text(encoding="utf-8") == "sentinel\n", log
    assert hook.read_text(encoding="utf-8") == "#!/bin/sh\nexit 0\n", log
    assert key not in "".join(out), "the approval key reached the model"
    assert not native_hook.exists(), "the native file tool wrote into a hook folder"
    assert not (blocked / "new.txt").exists(), log

    # The record folder: read and written, both recorded.
    assert "notes" in out[5], (out[5], log)
    assert (recorded / "new.txt").read_text(encoding="utf-8") == "more\n", log

    # The ask folder: asked once, denied, not read.
    assert [r["tool_name"] for r in seen] == ["os.open"], seen
    assert str(asked / "secret.txt") in seen[0]["reason"], seen
    assert "the-asked-secret" not in out[6], (out[6], log)

    # Nothing governs this one.
    assert free.read_text(encoding="utf-8") == "free\n", log

    # Floor refusals reach the trail through the kernel log a moment later.
    deadline = time.time() + 15
    while True:
        every = _actions()
        actions = [a for a in every if a["agent"] == "copilot"]
        # A refusal by a process that exited before it was read is recorded
        # under the profile, so match floor refusals by target alone.
        floor_targets = {a["parameters"].get("target") for a in every if a["tool"] == "os.floor"}
        if {str(sentinel), str(hook)} <= floor_targets or time.time() > deadline:
            break
        time.sleep(0.5)

    def outcomes(tool: str, path: Path) -> list:
        return [a["outcomes"] for a in actions
                if a["tool"] == tool and a["parameters"].get("path") == str(path)]

    assert [a["tool"] for a in actions if a["tool"] == "os.launch"] == ["os.launch"], actions
    assert outcomes("os.open", recorded / "notes.txt") == [["allow"]], actions
    assert outcomes("os.open", recorded / "new.txt") == [["allow"]], actions
    assert outcomes("os.open", asked / "secret.txt") == [["escalate", "deny"]], actions
    assert {str(sentinel), str(hook)} <= floor_targets, sorted(floor_targets)


def test_an_adapter_decides_under_vaara_run(tmp_path):
    """Copilot CLI with Vaara's hooks, under ``vaara run``.

    The hook starts inside the tree, where ~/.vaara is sealed, so it relays
    each event to ``vaara run``. Without the relay it could not decide, and
    the gate in front of it would block every call, the free one included.
    """
    from vaara.integrations import claude_code_hooks, copilot

    home = Path.home()
    copilot_home = tmp_path / "copilot-home"
    vaara_bin = shutil.which("vaara")
    assert vaara_bin, "the vaara command is not on PATH"
    copilot.install_hooks(vaara_bin, copilot_home)
    free = tmp_path / "adapter-free.txt"

    model = _Model([
        [_sh(f"echo free > {free}")],
        [_sh("cat /etc/shadow")],
        "done",
    ])
    work = tmp_path / "work"
    work.mkdir()
    env = {"HOME": str(home), "USERPROFILE": str(home), "COPILOT_HOME": str(copilot_home),
           "COPILOT_PROVIDER_BASE_URL": f"http://127.0.0.1:{model.port}/v1",
           "COPILOT_MODEL": "gpt-4.1", "COPILOT_OFFLINE": "true",
           "PATH": os.environ.get("PATH", ""), "NO_COLOR": "1",
           "PYTHONPATH": os.pathsep.join(sys.path)}
    try:
        proc = _run_vaara(["copilot", "-p", "go", "--allow-all-tools", "--allow-all-paths",
                           "--no-auto-update", "--no-ask-user"], work, env)
    finally:
        _stop_model(model)
    log = proc.stdout[-3000:] + proc.stderr[-3000:]

    out = model.outputs()
    assert len(out) == 2, (out, log)
    assert not any("could not reach" in o or "fail-closed" in o for o in out), (out, log)
    # The hook let the free call run, and it ran.
    assert free.read_text(encoding="utf-8") == "free\n", (out, log)
    # The hook refused the deny-rule call with Vaara's own reason.
    assert "etc_shadow_read" in out[1], (out[1], log)

    # Both decisions are on the adapter's trail, written from outside the floor.
    # Resolved as the hook resolves it: `vaara init` points the adapter's
    # config at the shared trail. A failed write does not change the verdict,
    # it leaves a marker beside the trail, so say what is there before reading.
    trail = claude_code_hooks.audit_db_path(json.loads(
        (home / ".vaara" / "claude-code" / "config.json").read_text(encoding="utf-8")))
    if not trail.exists():
        marker = trail.with_name(trail.name + ".write-failure.json")
        seen = sorted(str(p) for p in (home / ".vaara").rglob("*")) if (home / ".vaara").exists() else []
        pytest.fail(f"no trail at {trail}\n~/.vaara: {seen}\n"
                    f"marker: {marker.read_text(encoding='utf-8') if marker.exists() else 'none'}\n{log}")
    conn = sqlite3.connect(f"file:{trail}?mode=ro", uri=True)
    try:
        rows = [d for (d,) in conn.execute("SELECT data FROM audit_records ORDER BY seq")]
    finally:
        conn.close()
    assert any("etc_shadow_read" in (d or "") for d in rows), rows[-5:]
    assert any(str(free) in (d or "") for d in rows), rows[-5:]


def test_hardened_launch_with_egress_locked(tmp_path):
    """``harden`` and ``egress`` on: the stacked floor under no_new_privs.

    Tools still run (the move into //tool is a stack, which AppArmor allows
    under no_new_privs), they run with no_new_privs and the seccomp filter,
    a host off the allow list is refused by the proxy and recorded, a direct
    connection is refused by the kernel, and the floor still holds.
    """
    from vaara.oslayer import manage, selection

    home = Path.home()
    free = tmp_path / "hardened-free.txt"
    # This machine's own outbound address: the proxy's port on a host that is
    # not loopback. Landlock lets the port through; the guard's address rule
    # on the launch's cgroup refuses it (EPERM) before a packet leaves.
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.connect(("192.0.2.1", 9))
    host_ip = probe.getsockname()[0]
    probe.close()
    direct = tmp_path / "proxyport.py"
    direct.write_text(
        "import os, socket\n"
        "port = int(os.environ['HTTPS_PROXY'].rsplit(':', 1)[1])\n"
        "s = socket.socket()\n"
        "try:\n"
        f"    s.connect(({host_ip!r}, port))\n"
        "    print('proxyport=0')\n"
        "except OSError as e:\n"
        "    print('proxyport=%d' % e.errno)\n")
    model = _Model([
        [_sh(f"echo hardened > {free}")],
        [_sh("grep -E '^(NoNewPrivs|Seccomp):' /proc/self/status")],
        [_sh("curl -sS -m 10 https://example.com/ -o /dev/null; echo curl=$?")],
        [_sh("python3 -c 'import socket; socket.create_connection((\"1.1.1.1\", 443), "
             "timeout=5)' 2>&1 | tail -1")],
        [_sh(f"python3 {direct}")],
        [_sh(f"cat {home}/.vaara/keys/approval-hmac.key")],
        "done",
    ])
    sel = selection.Selection(harden=True, egress=[f"127.0.0.1:{model.port}"])
    _path, told = manage.save(sel)
    assert told, "the guard did not take the reload"
    status = manage.guard_status()
    assert status and status["profile_loaded"] and status["harden"], status
    work = tmp_path / "work"
    work.mkdir()
    env = {"HOME": str(home), "USERPROFILE": str(home), "COPILOT_HOME": str(tmp_path / "copilot-home"),
           "COPILOT_PROVIDER_BASE_URL": f"http://127.0.0.1:{model.port}/v1",
           "COPILOT_MODEL": "gpt-4.1", "COPILOT_OFFLINE": "true",
           "PATH": os.environ.get("PATH", ""), "NO_COLOR": "1",
           "PYTHONPATH": os.pathsep.join(sys.path)}
    try:
        proc = _run_vaara(["copilot", "-p", "go", "--allow-all-tools", "--allow-all-paths",
                           "--no-auto-update", "--no-ask-user"], work, env)
    finally:
        _stop_model(model)
        manage.save(selection.Selection())
    log = proc.stdout[-3000:] + proc.stderr[-3000:]

    out = model.outputs()
    assert len(out) == 6, (out, log)
    assert free.read_text(encoding="utf-8") == "hardened\n", (out, log)
    assert "NoNewPrivs:\t1" in out[1] and "Seccomp:\t2" in out[1], (out[1], log)
    assert "curl=0" not in out[2] and "403" in out[2], (out[2], log)
    assert "Permission denied" in out[3], (out[3], log)
    assert "proxyport=1" in out[4], (out[4], log)  # EPERM from the address rule
    assert "Permission denied" in out[5], (out[5], log)

    trail = Path(os.environ.get("VAARA_DB") or home / ".vaara" / "trail" / "audit.db")
    conn = sqlite3.connect(f"file:{trail}?mode=ro", uri=True)
    try:
        rows = [(t, d) for t, d in conn.execute(
            "SELECT tool_name, data FROM audit_records WHERE tool_name = 'egress.connect'")]
    finally:
        conn.close()
    data = [json.loads(d) for _t, d in rows]
    # A refused connection is a deny; an allowed one is a decision when it is
    # allowed and an outcome with the bytes when it closes.
    seen = [(r["decision"], r["reason"].split(" ")[1].rstrip(":"))
            for r in data if "decision" in r]
    assert ("deny", "example.com:443") in seen, seen[-5:]
    assert ("allow", f"127.0.0.1:{model.port}") in seen, seen[-5:]
    closed = [r["description"] for r in data if "description" in r]
    assert any(f"127.0.0.1:{model.port}: closed" in c for c in closed), closed[-5:]
