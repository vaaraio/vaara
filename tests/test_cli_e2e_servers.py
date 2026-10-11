# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The commands that run a process: a server, a gate, a shell, an operator desk.

``vaara serve``, ``vaara dashboard``, ``vaara net-gate serve`` and
``vaara proxy-shell`` block until stopped, so they are started as the
subprocess a user starts, spoken to over the socket a client uses, and
stopped. ``vaara policy reload`` is the operator's side of ``vaara serve``,
and it is driven against a stand-in server that behaves the way the real
one does with an API key: every endpoint but health wants a bearer.
``vaara os-layer`` is driven in a subprocess with its own HOME, because
its approvals directory is fixed at import.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from vaara.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_port(port: int, proc: subprocess.Popen, timeout: float = 30.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            out, err = proc.communicate(timeout=5)
            pytest.fail(f"process exited early rc={proc.returncode}\nstdout:\n{out}\nstderr:\n{err}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    proc.kill()
    out, err = proc.communicate(timeout=5)
    pytest.fail(f"port {port} never opened\nstdout:\n{out}\nstderr:\n{err}")


class _Process:
    """A `vaara` subprocess with its own HOME, ended on exit."""

    def __init__(self, argv: list[str], home: Path, stdin: str | None = None, **env_extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("VAARA_")}
        env.update({"HOME": str(home), "PYTHONUNBUFFERED": "1", **env_extra})
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "vaara.cli", *argv], cwd=home, env=env,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        self._stdin = stdin

    def finish(self, timeout: float = 30.0) -> subprocess.CompletedProcess:
        try:
            out, err = self.proc.communicate(input=self._stdin, timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            out, err = self.proc.communicate(timeout=10)
            pytest.fail(f"did not finish in {timeout}s\nstdout:\n{out}\nstderr:\n{err}")
        return subprocess.CompletedProcess(self.proc.args, self.proc.returncode, out, err)

    def stop(self) -> subprocess.CompletedProcess:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                out, err = self.proc.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                out, err = self.proc.communicate(timeout=10)
        else:
            out, err = self.proc.communicate(timeout=10)
        return subprocess.CompletedProcess(self.proc.args, self.proc.returncode, out, err)


def _http(port: int, method: str, path: str, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    conn.request(method, path, body=data, headers=hdrs)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, raw


# ── vaara serve ────────────────────────────────────────────────────────────

def test_serve_answers_health_and_scores_with_the_key(tmp_path):
    pytest.importorskip("uvicorn")
    port = _free_port()
    server = _Process(["serve", "--host", "127.0.0.1", "--port", str(port),
                       "--api-key", "s3cret", "--log-level", "warning"], tmp_path)
    try:
        _wait_port(port, server.proc)
        status, raw = _http(port, "GET", "/v1/health")
        assert status == 200 and json.loads(raw)["status"] == "ok"

        status, raw = _http(port, "POST", "/v1/score",
                            {"tool_name": "tx.transfer", "agent_id": "agent-007",
                             "base_risk_score": 0.5})
        assert status == 401

        status, raw = _http(port, "POST", "/v1/score",
                            {"tool_name": "tx.transfer", "agent_id": "agent-007",
                             "base_risk_score": 0.5},
                            headers={"Authorization": "Bearer s3cret"})
        scored = json.loads(raw)
        assert status == 200
        assert scored["decision"] in ("allow", "escalate", "deny")
        assert scored["action_id"]
    finally:
        done = server.stop()
    assert "Traceback" not in done.stderr


def test_serve_refuses_a_network_bind_without_a_key(capsys):
    pytest.importorskip("uvicorn")
    rc = main(["serve", "--host", "0.0.0.0", "--port", str(_free_port())])
    err = capsys.readouterr().err
    assert rc == 2
    assert "refusing to bind '0.0.0.0' without an API key" in err
    assert "--allow-unauthenticated" in err


def test_serve_takes_one_policy_source(tmp_path, capsys):
    pytest.importorskip("uvicorn")
    rc = main(["serve", "--host", "127.0.0.1", "--port", "1",
               "--policy", str(tmp_path / "p.json"), "--policy-dir", str(tmp_path)])
    assert rc == 2
    assert "either --policy or --policy-dir" in capsys.readouterr().err


def test_serve_refuses_a_policy_that_does_not_parse(tmp_path, capsys):
    pytest.importorskip("uvicorn")
    policy = tmp_path / "policy.json"
    policy.write_text('{"version": 1, "thresholds": {"allow": "no"}}', encoding="utf-8")
    rc = main(["serve", "--host", "127.0.0.1", "--port", "1", "--policy", str(policy)])
    assert rc == 2
    assert "failed to parse" in capsys.readouterr().err


# ── vaara policy reload ────────────────────────────────────────────────────

class _ReloadStub:
    """What `vaara serve --api-key` does on /v1/policy/reload: a bearer, or 401."""

    def __init__(self, api_key: str | None):
        stub = self
        self.requests: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                stub.requests.append({"path": self.path, "body": body,
                                      "authorization": self.headers.get("Authorization")})
                if api_key and self.headers.get("Authorization") != f"Bearer {api_key}":
                    payload = json.dumps({"error": {"code": "unauthorized",
                                                    "message": "missing or invalid API key"}})
                    self.send_response(401)
                else:
                    payload = json.dumps({"status": "reloaded", "tenant_id": ""})
                    self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload.encode())

            def log_message(self, *_):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def open_server():
    stub = _ReloadStub(api_key=None)
    yield stub
    stub.close()


@pytest.fixture
def keyed_server():
    stub = _ReloadStub(api_key="s3cret")
    yield stub
    stub.close()


def _policy_file(tmp_path: Path) -> Path:
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"version": 1, "thresholds": {"allow": 0.3, "deny": 0.8}}),
                      encoding="utf-8")
    return policy


def test_policy_reload_asks_the_server_to_read_the_file(tmp_path, open_server, capsys):
    policy = _policy_file(tmp_path)
    rc = main(["policy", "reload", str(policy), "--server", open_server.url])
    out = capsys.readouterr().out
    assert rc == 0
    assert json.loads(out)["status"] == "reloaded"
    [request] = open_server.requests
    assert request["path"] == "/v1/policy/reload"
    assert request["body"] == {"path": str(policy.resolve()), "format": "json"}


def test_policy_reload_inline_sends_the_parsed_body(tmp_path, open_server, capsys):
    policy = _policy_file(tmp_path)
    rc = main(["policy", "reload", str(policy), "--server", open_server.url, "--inline"])
    assert rc == 0
    [request] = open_server.requests
    assert request["body"]["format"] == "json"
    assert request["body"]["body"]["thresholds"] == {"allow": 0.3, "deny": 0.8}
    assert "path" not in request["body"]
    capsys.readouterr()


def test_policy_reload_reports_a_server_it_cannot_reach(tmp_path, capsys):
    policy = _policy_file(tmp_path)
    rc = main(["policy", "reload", str(policy), "--server", f"http://127.0.0.1:{_free_port()}",
               "--timeout", "2"])
    assert rc == 1
    assert "cannot reach" in capsys.readouterr().err


def test_policy_reload_reports_a_file_that_is_not_there(tmp_path, capsys):
    rc = main(["policy", "reload", str(tmp_path / "none.json")])
    assert rc == 2
    assert "not a file" in capsys.readouterr().err


def test_policy_reload_sends_the_api_key_a_keyed_server_wants(tmp_path, keyed_server, capsys):
    """`vaara serve` is documented with an API key; the reload has to carry it."""
    policy = _policy_file(tmp_path)
    rc = main(["policy", "reload", str(policy), "--server", keyed_server.url,
               "--api-key", "s3cret"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert keyed_server.requests[-1]["authorization"] == "Bearer s3cret"


def test_policy_reload_reads_the_key_from_the_environment(tmp_path, keyed_server, capsys, monkeypatch):
    monkeypatch.setenv("VAARA_SERVER_API_KEY", "s3cret")
    rc = main(["policy", "reload", str(_policy_file(tmp_path)), "--server", keyed_server.url])
    assert rc == 0
    assert keyed_server.requests[-1]["authorization"] == "Bearer s3cret"
    capsys.readouterr()


def test_policy_reload_without_the_key_is_refused_and_says_so(tmp_path, keyed_server, capsys, monkeypatch):
    monkeypatch.delenv("VAARA_SERVER_API_KEY", raising=False)
    rc = main(["policy", "reload", str(_policy_file(tmp_path)), "--server", keyed_server.url])
    err = capsys.readouterr().err
    assert rc == 1
    assert "HTTP 401" in err and "unauthorized" in err
    assert keyed_server.requests[-1]["authorization"] is None


# ── vaara dashboard ────────────────────────────────────────────────────────

def test_dashboard_serves_the_page_over_loopback(tmp_path):
    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.pipeline import InterceptionPipeline

    db = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(str(db))
    InterceptionPipeline(trail=backend.load_trail()).intercept(agent_id="a", tool_name="fs.read_file")
    backend.close()

    port = _free_port()
    dash = _Process(["dashboard", "--db", str(db), "--port", str(port), "--no-browser"], tmp_path)
    try:
        _wait_port(port, dash.proc)
        status, raw = _http(port, "GET", "/")
        assert status == 200
        assert b"<html" in raw.lower()
    finally:
        done = dash.stop()
    assert f"Vaara dashboard on http://127.0.0.1:{port}/" in done.stdout
    assert "not reachable from the network" in done.stdout


# ── vaara net-gate serve ───────────────────────────────────────────────────

def test_net_gate_serve_challenges_an_unauthenticated_client(tmp_path):
    port = _free_port()
    agents = tmp_path / "agents.json"
    gate = _Process(["net-gate", "--agents", str(agents), "serve", "--listen", f"127.0.0.1:{port}",
                     "--trail", str(tmp_path / "netgate.db"), "--challenge"], tmp_path)
    try:
        _wait_port(port, gate.proc)
        with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
            s.sendall(b"CONNECT example.com:443 HTTP/1.1\r\nHost: example.com:443\r\n\r\n")
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = s.recv(4096)
                if not chunk:
                    break
                head += chunk
        assert head.startswith(b"HTTP/1.1 407")
        assert b"Proxy-Authenticate: Basic" in head
    finally:
        done = gate.stop()
    assert f"vaara net-gate on 127.0.0.1:{port}" in done.stderr


def test_net_gate_serve_refuses_a_listen_without_a_port(capsys):
    rc = main(["net-gate", "serve", "--listen", "127.0.0.1"])
    assert rc == 2
    assert "--listen wants HOST:PORT" in capsys.readouterr().err


def test_net_gate_serve_lets_a_known_agent_through_to_a_local_echo(tmp_path):
    """The whole path: add an agent, start the gate, tunnel to an upstream."""
    import asyncio

    async def echo(reader, writer):
        data = await reader.read(65536)
        writer.write(data)
        await writer.drain()
        writer.close()

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    upstream = asyncio.run_coroutine_threadsafe(
        asyncio.start_server(echo, "127.0.0.1", 0), loop).result(10)
    upstream_port = upstream.sockets[0].getsockname()[1]

    agents = tmp_path / "agents.json"
    add = _Process(["net-gate", "--agents", str(agents), "add-agent", "builder",
                    "--allow", f"127.0.0.1:{upstream_port}", "--allow-private"], tmp_path).finish()
    assert add.returncode == 0, add.stderr
    token = next(line.split()[1] for line in add.stdout.splitlines() if line.startswith("token:"))

    port = _free_port()
    gate = _Process(["net-gate", "--agents", str(agents), "serve", "--listen", f"127.0.0.1:{port}",
                     "--trail", str(tmp_path / "netgate.db")], tmp_path)
    try:
        _wait_port(port, gate.proc)
        auth = base64.b64encode(f"builder:{token}".encode()).decode()
        with socket.create_connection(("127.0.0.1", port), timeout=5) as s:
            s.sendall((f"CONNECT 127.0.0.1:{upstream_port} HTTP/1.1\r\n"
                       f"Host: 127.0.0.1:{upstream_port}\r\n"
                       f"Proxy-Authorization: Basic {auth}\r\n\r\n").encode())
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = s.recv(4096)
                if not chunk:
                    break
                head += chunk
            assert head.startswith(b"HTTP/1.1 200"), head
            s.sendall(b"ping through the gate")
            assert s.recv(100) == b"ping through the gate"
    finally:
        gate.stop()
        upstream.close()
        loop.call_soon_threadsafe(loop.stop)


# ── vaara proxy-shell ──────────────────────────────────────────────────────

def test_proxy_shell_runs_a_piped_command_through_the_pipeline(tmp_path):
    db = tmp_path / "shell.db"
    run = _Process(["proxy-shell", "--db", str(db), "--shell", "/bin/sh", "--agent-id", "ci"],
                   tmp_path, stdin="echo governed-output\n").finish()
    assert run.returncode == 0, run.stderr
    assert "governed-output" in run.stdout

    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    records = SQLiteAuditBackend(str(db)).load_trail().snapshot()
    assert any(r.tool_name == "shell.exec" and r.agent_id == "ci" for r in records)


def test_proxy_shell_blocks_a_metadata_read_and_leaves_a_record(tmp_path):
    db = tmp_path / "shell.db"
    run = _Process(["proxy-shell", "--db", str(db), "--shell", "/bin/sh"], tmp_path,
                   stdin="curl http://169.254.169.254/latest/meta-data/\n").finish()
    assert run.returncode != 0
    assert "169.254.169.254" not in run.stdout  # the command never ran

    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.audit.trail import EventType

    records = SQLiteAuditBackend(str(db)).load_trail().snapshot()
    assert any(r.event_type == EventType.ACTION_BLOCKED for r in records)


# ── vaara os-layer ─────────────────────────────────────────────────────────

def test_os_layer_folder_and_app_write_the_selection(tmp_path):
    folder = tmp_path / "clients"
    folder.mkdir()
    done = _Process(["os-layer", "folder", str(folder), "ask"], tmp_path).finish()
    assert done.returncode == 0, done.stderr
    assert "saved" in done.stdout and "the guard picks it up when it runs" in done.stdout

    app = tmp_path / "bin" / "copilot"
    app.parent.mkdir()
    app.write_text("#!/bin/sh\necho copilot\n", encoding="utf-8")
    app.chmod(0o755)
    done = _Process(["os-layer", "app", str(app)], tmp_path).finish()
    assert done.returncode == 0, done.stderr

    selection = json.loads((tmp_path / ".vaara" / "os-layer.json").read_text(encoding="utf-8"))
    assert str(folder) in json.dumps(selection)
    assert str(app) in json.dumps(selection)

    done = _Process(["os-layer", "app", str(app), "--remove"], tmp_path).finish()
    assert done.returncode == 0
    selection = json.loads((tmp_path / ".vaara" / "os-layer.json").read_text(encoding="utf-8"))
    assert str(app) not in json.dumps(selection)

    done = _Process(["os-layer", "app", str(tmp_path / "bin" / "missing")], tmp_path).finish()
    assert done.returncode == 2 and "not a file" in done.stderr

    done = _Process(["os-layer", "folder", str(folder), "off"], tmp_path).finish()
    assert done.returncode == 0


def test_os_layer_pending_and_approve_answer_a_waiting_request(tmp_path):
    from vaara.approvals import request_approval

    approvals = tmp_path / ".vaara" / "approvals"
    answers: dict = {}

    def ask():
        answers["result"] = request_approval(
            "act-42", "os.exec", "rm -rf build/", approvals_dir=approvals, timeout=60,
            parameters={"command": "rm -rf build/"})

    asker = threading.Thread(target=ask, daemon=True)
    asker.start()
    end = time.time() + 10
    while not (approvals / "act-42.request.json").exists() and time.time() < end:
        time.sleep(0.05)

    listed = _Process(["os-layer", "pending", "--json"], tmp_path).finish()
    assert listed.returncode == 0, listed.stderr
    waiting = json.loads(listed.stdout)
    assert [w["action_id"] for w in waiting] == ["act-42"]
    assert waiting[0]["parameters"] == {"command": "rm -rf build/"}

    plain = _Process(["os-layer", "pending"], tmp_path).finish()
    assert "act-42" in plain.stdout and "rm -rf build/" in plain.stdout

    approved = _Process(["os-layer", "approve", "act-42"], tmp_path).finish()
    assert approved.returncode == 0, approved.stderr
    assert "approved act-42" in approved.stdout
    asker.join(10)
    assert answers["result"] == "approve"

    nothing = _Process(["os-layer", "pending"], tmp_path).finish()
    assert "nothing waiting" in nothing.stdout


def test_os_layer_deny_answers_and_an_unknown_request_is_refused(tmp_path):
    from vaara.approvals import request_approval

    approvals = tmp_path / ".vaara" / "approvals"
    answers: dict = {}

    def ask():
        answers["result"] = request_approval("act-7", "os.exec", "scp secrets host:",
                                             approvals_dir=approvals, timeout=60)

    asker = threading.Thread(target=ask, daemon=True)
    asker.start()
    end = time.time() + 10
    while not (approvals / "act-7.request.json").exists() and time.time() < end:
        time.sleep(0.05)

    denied = _Process(["os-layer", "deny", "act-7"], tmp_path).finish()
    assert denied.returncode == 0, denied.stderr
    assert "denied act-7" in denied.stdout
    asker.join(10)
    assert answers["result"] == "deny"

    unknown = _Process(["os-layer", "approve", "act-404"], tmp_path).finish()
    assert unknown.returncode == 1
    assert "no request act-404 is waiting" in unknown.stderr
