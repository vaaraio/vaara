"""Hook forwarding under ``vaara run``: a hook inside the tree relays its event.

The relay and the server talk over a real abstract unix socket here. The
cgroup check is exercised on its own; the full path under the floor runs in
the os-layer-e2e CI job.
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

from vaara.oslayer import forward

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"),
                                reason="abstract unix sockets are Linux only")

# Stands in for `vaara hook`: echoes what it was given and exits 2.
_FAKE_HOOK = (
    "import json, os, sys; data = sys.stdin.buffer.read();"
    "print(json.dumps({'argv': sys.argv[1:], 'stdin': data.decode(), 'cwd': os.getcwd(),"
    " 'env': {k: v for k, v in os.environ.items() if k.startswith(('VAARA_RUN', 'FWD_'))}}));"
    "print('hook stderr', file=sys.stderr); sys.exit(2)"
)


@pytest.fixture
def server(monkeypatch):
    srv = forward.HookServer(hook_cmd=[sys.executable, "-c", _FAKE_HOOK])
    monkeypatch.setattr(srv, "allowed", lambda pid, uid: True)
    srv.start()
    for key, value in srv.environ().items():
        monkeypatch.setenv(key, value)
    yield srv
    srv.close()


def test_relay_replays_the_hooks_answer(server, capfdbinary, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = forward.relay(["pre-tool-use", "--client", "codex"], stdin=b'{"tool_name": "Bash"}')
    out, err = capfdbinary.readouterr()
    assert rc == 2
    got = json.loads(out)
    assert got["argv"] == ["hook", "pre-tool-use", "--client", "codex"]
    assert got["stdin"] == '{"tool_name": "Bash"}'
    assert got["cwd"] == str(tmp_path)
    assert b"hook stderr" in err


def test_hook_it_runs_does_not_relay_again(server, capfdbinary):
    forward.relay(["session-start"], stdin=b"")
    env = json.loads(capfdbinary.readouterr().out)["env"]
    assert "VAARA_RUN_SOCKET" not in env and "VAARA_RUN_PID" not in env


def test_request_cannot_set_variables(server):
    reply = server.answer({"argv": ["session-start"], "stdin": "",
                           "env": {"FWD_INJECTED": "1"}, "cwd": "/"})
    import base64

    env = json.loads(base64.b64decode(reply["stdout"]))["env"]
    assert "FWD_INJECTED" not in env


def test_relay_refuses_a_socket_held_by_another_pid(server, capfdbinary, monkeypatch):
    monkeypatch.setenv(forward.PID_ENV, str(os.getpid() + 1))
    rc = forward.relay(["pre-tool-use"], stdin=b"{}")
    err = capfdbinary.readouterr().err
    assert rc == forward.EXIT_UNREACHED
    assert b"not held by vaara run" in err


def test_relay_with_nothing_listening_exits_unreached(capfdbinary, monkeypatch):
    monkeypatch.setenv(forward.SOCKET_ENV, "vaara-run-test-nobody-here")
    monkeypatch.setenv(forward.PID_ENV, str(os.getpid()))
    rc = forward.relay(["pre-tool-use"], stdin=b"{}")
    assert rc == forward.EXIT_UNREACHED
    assert b"vaara run did not answer" in capfdbinary.readouterr().err


def test_caller_outside_the_launch_is_refused(capfdbinary, monkeypatch):
    srv = forward.HookServer(hook_cmd=[sys.executable, "-c", _FAKE_HOOK])
    srv.set_cgroup("/sys/fs/cgroup/vaara/launch-abc")
    monkeypatch.setattr(forward, "_cgroup_of", lambda pid: "/user.slice/session-1.scope")
    srv.start()
    try:
        for key, value in srv.environ().items():
            monkeypatch.setenv(key, value)
        rc = forward.relay(["pre-tool-use"], stdin=b"{}")
        assert rc == forward.EXIT_UNREACHED
        assert b"vaara run refused: caller is not in this launch" in capfdbinary.readouterr().err
    finally:
        srv.close()


def test_a_refusal_sent_before_the_request_is_read_still_arrives(capfdbinary, monkeypatch):
    # A refused caller is answered without its request being read. A request
    # larger than the socket buffer then meets a closed peer mid-send, and the
    # reason used to be lost to "Broken pipe" (seen about one run in six).
    srv = forward.HookServer(hook_cmd=[sys.executable, "-c", _FAKE_HOOK])
    srv.set_cgroup("/sys/fs/cgroup/vaara/launch-abc")
    monkeypatch.setattr(forward, "_cgroup_of", lambda pid: "/user.slice/session-1.scope")
    srv.start()
    try:
        for key, value in srv.environ().items():
            monkeypatch.setenv(key, value)
        rc = forward.relay(["pre-tool-use"], stdin=b"x" * (8 << 20))
        assert rc == forward.EXIT_UNREACHED
        assert b"vaara run refused: caller is not in this launch" in capfdbinary.readouterr().err
    finally:
        srv.close()


def test_launch_cgroup_membership(monkeypatch):
    srv = forward.HookServer()
    try:
        srv.set_cgroup("/sys/fs/cgroup/vaara/launch-abc")
        seen = {"pid": ""}
        monkeypatch.setattr(forward, "_cgroup_of", lambda pid: seen["pid"])
        uid = os.getuid()
        for path, ok in (("/vaara/launch-abc", True), ("/vaara/launch-abc/sub", True),
                         ("/vaara/launch-abcd", False), ("/vaara/launch-other", False),
                         ("", False)):
            seen["pid"] = path
            assert srv.allowed(1, uid) is ok, path
        seen["pid"] = "/vaara/launch-abc"
        assert srv.allowed(1, uid + 1) is False
    finally:
        srv.close()


def test_slow_hook_blocks_within_the_deadline(server, monkeypatch):
    monkeypatch.setattr(forward, "_deadline", lambda: 3)
    server._hook_cmd = [sys.executable, "-c", "import time; time.sleep(10)"]
    reply = server.answer({"argv": ["pre-tool-use"], "stdin": ""})
    import base64

    assert reply["rc"] == 2
    assert b"did not answer within 1 s" in base64.b64decode(reply["stderr"])


def test_vaara_hook_relays_under_vaara_run(monkeypatch):
    from vaara import cli

    seen = {}
    monkeypatch.setenv(forward.SOCKET_ENV, "vaara-run-x")
    monkeypatch.setattr(forward, "relay", lambda argv: seen.setdefault("argv", argv) and 7)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert cli.main(["hook", "pre-tool-use", "--client", "gemini"]) == 7
    assert seen["argv"] == ["pre-tool-use", "--client", "gemini"]


def test_run_hands_the_socket_to_the_agent():
    srv = forward.HookServer()
    try:
        env = srv.environ()
        assert env[forward.SOCKET_ENV] == srv.name
        assert env[forward.PID_ENV] == str(os.getpid())
    finally:
        srv.close()
