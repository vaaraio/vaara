# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The egress proxy: the allow list, the refused addresses, the record, and
a hardened child on the real kernel that reaches the world only through it."""

from __future__ import annotations

import http.server
import json
import socket
import subprocess
import sys
import textwrap
import threading
import urllib.request

import pytest

from vaara.oslayer import harden
from vaara.oslayer.egress import EgressProxy, parse_rule


def _resolver(table):
    def resolve(host, port, type=0):  # noqa: A002 - getaddrinfo's name
        addr = table[host]
        family = socket.AF_INET6 if ":" in addr else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (addr, port))]
    return resolve


@pytest.fixture
def upstream():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = f"hello from {self.path}".encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def test_rules():
    assert parse_rule("Example.com").matches("example.com", 443)
    assert parse_rule("example.com").matches("example.com", 80)
    assert not parse_rule("example.com").matches("example.com", 22)
    assert not parse_rule("example.com").matches("evil-example.com", 443)
    assert parse_rule("*.github.com").matches("api.github.com", 443)
    assert not parse_rule("*.github.com").matches("github.com", 443)
    assert parse_rule("db.internal:5432").matches("db.internal", 5432)
    assert not parse_rule("db.internal:5432").matches("db.internal", 443)
    assert parse_rule("[::1]:8080").matches("::1", 8080)
    for bad in ("", "a b", "http://x", "ex*ample.com", "x:0", "x:70000"):
        with pytest.raises(ValueError):
            parse_rule(bad)


def test_decide_refuses_metadata_and_loopback_behind_an_allowed_name():
    p = EgressProxy(["api.example.com", "meta.example.com", "lo.example.com"],
                    resolve=_resolver({"api.example.com": "93.184.216.34",
                                       "meta.example.com": "169.254.169.254",
                                       "lo.example.com": "127.0.0.1"}))
    assert p.decide("api.example.com", 443)[0]
    ok, reason, _ = p.decide("meta.example.com", 443)
    assert not ok and "link-local" in reason
    ok, reason, _ = p.decide("lo.example.com", 443)
    assert not ok and "loopback" in reason
    ok, reason, _ = p.decide("other.example.com", 443)
    assert not ok and "allow list" in reason


def test_decide_looks_through_mapped_addresses_and_refuses_ipv6_metadata():
    p = EgressProxy(["m4.example.com", "m6.example.com", "v6.example.com"],
                    resolve=_resolver({"m4.example.com": "::ffff:127.0.0.2",
                                       "m6.example.com": "fd00:ec2::254",
                                       "v6.example.com": "2001:db8::10"}))
    ok, reason, _ = p.decide("m4.example.com", 443)
    assert not ok and "loopback" in reason
    ok, reason, _ = p.decide("m6.example.com", 443)
    assert not ok and "metadata" in reason
    assert p.decide("v6.example.com", 443)[0]


def test_plain_http_through_the_proxy_is_recorded(upstream):
    seen = []
    p = EgressProxy([f"127.0.0.1:{upstream}"], record=seen.append)
    port = p.start()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler(
            {"http": f"http://127.0.0.1:{port}"}))
        body = opener.open(f"http://127.0.0.1:{upstream}/x", timeout=10).read()
        assert body == b"hello from /x"
        with pytest.raises(urllib.error.HTTPError) as refused:
            opener.open("http://not-allowed.example/", timeout=10)
        assert refused.value.code == 403
    finally:
        p.close()
    allowed = [e for e in seen if e["allowed"]]
    denied = [e for e in seen if not e["allowed"]]
    assert allowed[0]["host"] == "127.0.0.1" and allowed[0]["method"] == "GET"
    assert allowed[0]["bytes_down"] > 0
    assert denied[0]["host"] == "not-allowed.example"


def test_connect_tunnel(upstream):
    seen = []
    p = EgressProxy([f"127.0.0.1:{upstream}"], record=seen.append)
    port = p.start()
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        s.sendall(f"CONNECT 127.0.0.1:{upstream} HTTP/1.1\r\n\r\n".encode())
        assert s.recv(1024).startswith(b"HTTP/1.1 200")
        s.sendall(b"GET /tunnel HTTP/1.0\r\nHost: x\r\n\r\n")
        data = b""
        while chunk := s.recv(4096):
            data += chunk
        assert data.endswith(b"hello from /tunnel")
        s.close()
    finally:
        p.close()
    assert seen and seen[0]["method"] == "CONNECT" and seen[0]["allowed"]


@pytest.mark.skipif(not sys.platform.startswith("linux") or harden.landlock_abi() < 4
                    or harden.machine() not in harden.SYSCALLS,
                    reason="needs Linux with Landlock ABI 4")
def test_a_hardened_child_gets_out_only_through_the_proxy(upstream):
    seen = []
    p = EgressProxy([f"127.0.0.1:{upstream}"], record=seen.append)
    port = p.start()
    code = textwrap.dedent(f'''
        import json, os, socket, urllib.request
        from vaara.oslayer import harden
        harden.apply(egress_ports=[{port}])
        out = {{}}
        out["via_proxy"] = urllib.request.urlopen(
            "http://127.0.0.1:{upstream}/via", timeout=10).read().decode()
        try:
            socket.create_connection(("127.0.0.1", {upstream}), timeout=5)
            out["direct"] = "connected"
        except OSError as e:
            out["direct"] = e.errno
        print(json.dumps(out))
    ''')
    env = dict(p.environ())
    import os
    env.update({k: v for k, v in os.environ.items() if k not in env})
    try:
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              env=env, timeout=60)
    finally:
        p.close()
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["via_proxy"] == "hello from /via"
    assert out["direct"] == 13  # EACCES: Landlock refused the direct connection
    assert any(e["allowed"] and e["port"] == upstream for e in seen)


# ── The selection, the floor, the CLI and vaara run ────────────────────

def test_selection_keeps_harden_and_egress_and_writes_them_only_when_set(tmp_path):
    from vaara.oslayer import selection
    plain = selection.parse({"version": 1, "folders": [], "apps": []})
    assert not plain.hardened and plain.egress is None
    assert "harden" not in plain.to_json() and "egress" not in plain.to_json()
    sel = selection.parse({"harden": True, "egress": ["API.example.com", "bad host", "*.gh.io"]})
    assert sel.harden and sel.egress == ["api.example.com", "*.gh.io"]
    locked = selection.parse({"egress": []})
    assert locked.hardened and locked.egress == []
    folder = tmp_path / "notes"
    folder.mkdir()
    kept = selection.set_folder(sel, str(folder), "record")
    assert kept.harden and kept.egress == sel.egress
    with pytest.raises(selection.SelectionError):
        selection.set_egress(plain, ["http://nope"])
    assert selection.set_egress(sel, None).egress is None


def test_the_floor_stacks_the_tool_profile_only_when_hardened():
    from vaara.oslayer import floor
    plain = floor.render(["/home/op"], abi="abi <abi/4.0>,", install_paths=[])
    stacked = floor.render(["/home/op"], abi="abi <abi/4.0>,", install_paths=[], stacked=True)
    assert "  /** Cx -> tool," in plain and "&" not in plain
    assert "  /** Cix -> &tool," in stacked
    assert plain.replace("Cx -> tool", "Cix -> &tool") == stacked


def test_os_layer_cli_sets_and_shows_both(tmp_path, monkeypatch, capsys):
    from vaara.oslayer import manage, selection
    monkeypatch.setattr(manage, "_home", lambda: str(tmp_path))
    monkeypatch.setattr(manage, "_guard", lambda op: None)
    assert manage.main(["harden", "on"]) == 0
    assert manage.main(["egress", "api.example.com", "*.github.com"]) == 0
    sel = selection.load(str(tmp_path))
    assert sel.harden and sel.egress == ["api.example.com", "*.github.com"]
    capsys.readouterr()
    manage.main(["status"])
    out = capsys.readouterr().out
    assert "harden: on" in out and "api.example.com, *.github.com" in out
    assert manage.main(["egress"]) == 2
    assert manage.main(["egress", "--off"]) == 0
    assert selection.load(str(tmp_path)).egress is None
    assert manage.main(["egress", "--none"]) == 0
    assert selection.load(str(tmp_path)).egress == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="vaara run is Linux")
def test_vaara_run_starts_the_proxy_and_hardens_the_child(monkeypatch):
    import os

    from vaara.oslayer import run as run_mod
    monkeypatch.setattr(run_mod.floor, "apparmor_enabled", lambda: True)
    monkeypatch.setattr(run_mod.sys, "platform", "linux")
    seen = {}

    def fake_request(payload, *, socket_path=None, timeout=10.0):
        return {"ok": True, "profile_digest": "sha256:11", "harden": True,
                "egress": ["api.example.com"]}

    class StopHere(Exception):
        pass

    def fake_open_launch(payload, *, socket_path=None):
        seen["proxy"] = os.environ.get("HTTPS_PROXY")
        raise StopHere()

    def fake_child(ready_fd, argv, hardening=None):
        seen["hardening"] = hardening

    import vaara.oslayer.client as client
    monkeypatch.setattr(client, "request", fake_request)
    monkeypatch.setattr(client, "open_launch", fake_open_launch)
    monkeypatch.setattr("vaara.cage.vaara_cage.apparmor_version", lambda: "apparmor 4.0.1")
    monkeypatch.setattr(run_mod, "resolve", lambda agent: ([agent], agent))
    monkeypatch.setattr(run_mod, "_child", fake_child)
    monkeypatch.setattr(run_mod.os, "fork", lambda: 0)  # take the child's branch once
    monkeypatch.setattr(run_mod.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(run_mod.os, "waitpid", lambda pid, flags: (pid, 0))
    from unittest import mock
    with mock.patch.dict(os.environ), pytest.raises(StopHere):
        run_mod.run("reviewer", ["claude"])
    # The proxy reaches the agent's tree only, never this process's own
    # environment, which the unconfined hook inherits.
    assert seen["proxy"] is None
    url = seen["hardening"]["environ"]["HTTPS_PROXY"]
    port = int(url.rsplit(":", 1)[1])
    assert seen["hardening"]["egress_ports"] == [port]


def test_egress_records_land_on_the_trail(tmp_path, monkeypatch):
    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.audit.trail import EventType
    from vaara.oslayer.run import _egress_recorder
    db = tmp_path / "audit.db"
    monkeypatch.setenv("VAARA_DB", str(db))
    record = _egress_recorder("reviewer", {"driver": "vaara-cage", "confirmed": False})
    record({"method": "CONNECT", "host": "evil.example", "port": 443, "allowed": False,
            "reason": "not on the egress allow list"})
    record({"method": "CONNECT", "host": "api.example.com", "port": 443, "allowed": True,
            "reason": "allowed by api.example.com"})
    trail = SQLiteAuditBackend(str(db)).load_trail()
    # A deny is filed as a blocked action, an allow as a decision.
    blocked = trail.get_records_by_type(EventType.ACTION_BLOCKED)
    allowed = trail.get_records_by_type(EventType.DECISION_MADE)
    assert [d.data["decision"] for d in blocked + allowed] == ["deny", "allow"]
    assert blocked[0].tool_name == allowed[0].tool_name == "egress.connect"
    assert "evil.example:443" in blocked[0].data["reason"]
    assert blocked[0].data["policy_id"] == "os-layer.egress"
    assert allowed[0].data["cage"]["driver"] == "vaara-cage"


def _raw_upstream(reply_after_eof: bool):
    """A one-shot TCP server that records every byte it receives."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    got = bytearray()

    def serve():
        conn, _ = srv.accept()
        conn.settimeout(3)
        try:
            while True:
                data = conn.recv(65536)
                if not data:
                    break
                got.extend(data)
                if not reply_after_eof and b"\r\n\r\n" in got:
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nfirst")
        except socket.timeout:
            pass
        if reply_after_eof:
            conn.sendall(b"reply")
        conn.close()
        srv.close()

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    return srv.getsockname()[1], got, t


def _read_all(s: socket.socket) -> bytes:
    out = b""
    while True:
        data = s.recv(65536)
        if not data:
            return out
        out += data


def test_plain_http_carries_one_request_per_connection():
    """A second keep-alive request on the connection never reaches the first host."""
    up, got, t = _raw_upstream(reply_after_eof=False)
    p = EgressProxy([f"127.0.0.1:{up}"])
    port = p.start()
    try:
        c = socket.create_connection(("127.0.0.1", port), timeout=10)
        c.sendall(f"GET http://127.0.0.1:{up}/first HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
                  f"GET http://denied.example/second HTTP/1.1\r\nHost: denied.example\r\n\r\n"
                  .encode())
        reply = _read_all(c)
        c.close()
        t.join(timeout=5)
    finally:
        p.close()
    assert reply.endswith(b"first")
    assert b"/first" in got and b"Connection: close" in got
    assert b"/second" not in got and b"denied.example" not in got


def test_a_half_closed_tunnel_still_gets_its_reply():
    up, got, t = _raw_upstream(reply_after_eof=True)
    p = EgressProxy([f"127.0.0.1:{up}"])
    port = p.start()
    try:
        c = socket.create_connection(("127.0.0.1", port), timeout=10)
        c.sendall(f"CONNECT 127.0.0.1:{up} HTTP/1.1\r\n\r\n".encode())
        assert c.recv(4096).startswith(b"HTTP/1.1 200")
        c.sendall(b"request")
        c.shutdown(socket.SHUT_WR)
        reply = _read_all(c)
        c.close()
        t.join(timeout=5)
    finally:
        p.close()
    assert bytes(got) == b"request"
    assert reply == b"reply"
