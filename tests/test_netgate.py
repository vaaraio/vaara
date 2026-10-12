# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""vaara net-gate: who, mandate, policy, egress floor, the brake, the record.

Every test runs a real gateway on a loopback port and a real upstream server
behind it, and talks to the gateway over a socket the way an agent's HTTP
client does. The gateway runs on its own event loop in a thread.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import os
import socket
import threading
import time

import pytest

from vaara.netgate import NetGate, build_pipeline
from vaara.netgate._mandate import AllowRule, MandateStore
from vaara.netgate._wg import client_config, split_allowed_ips
from vaara.netgate.cli import main as netgate_main


# Harness


class _Loop:
    """An event loop in a thread, for the gateway and the upstream server."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro, timeout=10):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)


async def _echo(reader, writer):
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except ConnectionError:
        pass
    finally:
        writer.close()


async def _close_servers(*servers):
    # On the loop thread: Server.close() from the test thread races a
    # connection detaching here and wakes the waiters twice.
    for server in servers:
        server.close()
    for server in servers:
        await server.wait_closed()


async def _http_origin(reader, writer):
    head = await reader.readuntil(b"\r\n\r\n")
    body = json.dumps({"head": head.decode("latin-1")}).encode()
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode()
                 + b"\r\nConnection: close\r\n\r\n" + body)
    await writer.drain()
    writer.close()


@pytest.fixture
def env(tmp_path):
    loop = _Loop()
    echo = loop.run(asyncio.start_server(_echo, "127.0.0.1", 0))
    origin = loop.run(asyncio.start_server(_http_origin, "127.0.0.1", 0))
    echo_port = echo.sockets[0].getsockname()[1]
    origin_port = origin.sockets[0].getsockname()[1]
    store = MandateStore(tmp_path / "agents.json")
    pipeline = build_pipeline(tmp_path / "audit.db")
    gate = NetGate(store, pipeline, port=0)
    loop.run(gate.start())

    class Env:
        pass

    e = Env()
    e.loop, e.store, e.pipeline, e.gate = loop, store, pipeline, gate
    e.echo_port, e.origin_port, e.tmp = echo_port, origin_port, tmp_path
    yield e
    loop.run(gate.stop())
    loop.run(_close_servers(echo, origin))
    loop.close()


def _auth(agent, token):
    return "Proxy-Authorization: Basic " + base64.b64encode(f"{agent}:{token}".encode()).decode()


def _connect(gate_port, target, auth=None, timeout=5):
    s = socket.create_connection(("127.0.0.1", gate_port), timeout=timeout)
    req = f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n"
    if auth:
        req += auth + "\r\n"
    s.sendall((req + "\r\n").encode())
    return s


def _read_head(s):
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = s.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf


def _records(e, agent):
    trail = e.pipeline.trail
    return [json.dumps(r.to_dict(), default=str) for r in trail.get_agent_records(agent)]


def _wait_for(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


# Mandate rules


def test_allow_rule_patterns():
    assert AllowRule.parse("api.github.com").admits("api.github.com", 443)
    assert AllowRule.parse("api.github.com").admits("API.GitHub.com.", 80)
    assert not AllowRule.parse("api.github.com").admits("api.github.com", 22)
    sub = AllowRule.parse("*.pypi.org")
    assert sub.admits("files.pypi.org", 443)
    assert not sub.admits("pypi.org", 443)
    assert not sub.admits("evilpypi.org", 443)
    assert AllowRule.parse("db.internal:5432").admits("db.internal", 5432)
    assert AllowRule.parse("*").admits("anything.example", 443)
    for bad in ("", "a b", "x:99999", "x:abc", "*.*.x"):
        with pytest.raises(ValueError):
            AllowRule.parse(bad)


def test_store_keeps_only_the_token_hash(tmp_path):
    store = MandateStore(tmp_path / "agents.json")
    token = store.add("builder", ["api.github.com"])
    raw = (tmp_path / "agents.json").read_text()
    assert token not in raw
    assert store.authenticate("builder", token) is not None
    assert store.authenticate("builder", token + "x") is None
    assert store.authenticate("nobody", token) is None
    # POSIX only: Windows has no file modes, so chmod reports 666 there and the
    # file is protected by the ACL of the profile it sits in.
    if os.name != "nt":
        assert oct((tmp_path / "agents.json").stat().st_mode)[-3:] == "600"


# Split tunnel


def test_split_tunnel_full_and_cut():
    assert split_allowed_ips([]) == ["0.0.0.0/0", "::/0"]
    nets = [ipaddress.ip_network(n) for n in split_allowed_ips(["192.168.0.0/16", "10.0.0.5"])
            if ":" not in n]
    assert sum(n.num_addresses for n in nets) == 2**32 - 2**16 - 1
    assert not any(ipaddress.ip_address("192.168.4.4") in n for n in nets)
    assert not any(ipaddress.ip_address("10.0.0.5") in n for n in nets)
    assert any(ipaddress.ip_address("10.0.0.6") in n for n in nets)
    assert any(ipaddress.ip_address("1.1.1.1") in n for n in nets)


def test_wg_config_has_no_private_key():
    cfg = client_config(address="10.70.0.2/32", gateway_public_key="PUB",
                        endpoint="gw:51820", direct=["192.168.0.0/16"])
    assert "PrivateKey = <output of `wg genkey`" in cfg
    assert "Endpoint = gw:51820" in cfg
    assert "192.168.0.0/16" not in cfg.split("AllowedIPs = ")[1].splitlines()[0]


# The gateway


def test_allowed_tunnel_carries_bytes_and_is_recorded(env):
    token = env.store.add("builder", [f"127.0.0.1:{env.echo_port}"], allow_private=True)
    s = _connect(env.gate.port, f"127.0.0.1:{env.echo_port}", _auth("builder", token))
    assert _read_head(s).startswith(b"HTTP/1.1 200")
    s.sendall(b"hello through the gate")
    assert s.recv(100) == b"hello through the gate"
    s.close()
    assert _wait_for(lambda: any('"net.close"' in r for r in _records(env, "builder")))
    recs = "\n".join(_records(env, "builder"))
    assert '"net.connect"' in recs and '"allow"' in recs
    assert '"bytes_up": 22' in recs and '"bytes_down": 22' in recs


def test_agent_added_a_moment_ago_is_known_at_once(env):
    # Found in the end-to-end run: an agent added by another process and used
    # straight away was treated as a stranger until the watcher's next tick.
    token = MandateStore(env.tmp / "agents.json").add(
        "fresh", [f"127.0.0.1:{env.echo_port}"], allow_private=True)
    s = _connect(env.gate.port, f"127.0.0.1:{env.echo_port}", _auth("fresh", token))
    assert _read_head(s).startswith(b"HTTP/1.1 200")


def test_stranger_gets_silence(env):
    env.store.add("builder", ["*"])
    s = _connect(env.gate.port, "example.com:443", _auth("builder", "wrong-token"))
    assert s.recv(100) == b""
    s = _connect(env.gate.port, "example.com:443")
    assert s.recv(100) == b""
    assert _wait_for(lambda: any("net_gate:unauthenticated" in r
                                 for r in _records(env, "unauthenticated")))


def test_challenge_mode_answers_407(env, tmp_path):
    env.gate.challenge = True
    s = _connect(env.gate.port, "example.com:443")
    head = _read_head(s)
    assert head.startswith(b"HTTP/1.1 407")
    assert b"Proxy-Authenticate: Basic" in head


def test_outside_mandate_is_refused_and_recorded(env):
    token = env.store.add("builder", ["api.github.com"])
    s = _connect(env.gate.port, "evil.example:443", _auth("builder", token))
    head = _read_head(s)
    assert head.startswith(b"HTTP/1.1 403")
    assert b"outside agent builder's mandate" in head + s.recv(200)
    recs = "\n".join(_records(env, "builder"))
    assert "net_mandate:builder" in recs and '"deny"' in recs


def test_private_address_needs_allow_private(env):
    token = env.store.add("builder", [f"127.0.0.1:{env.echo_port}"], allow_private=False)
    s = _connect(env.gate.port, f"127.0.0.1:{env.echo_port}", _auth("builder", token))
    head = _read_head(s) + s.recv(300)
    assert head.startswith(b"HTTP/1.1 403")
    assert b"blocked address" in head
    assert _wait_for(lambda: any("egress floor" in r for r in _records(env, "builder")))


def test_revoked_agent_is_refused(env):
    token = env.store.add("builder", [f"127.0.0.1:{env.echo_port}"], allow_private=True)
    env.store.revoke("builder")
    s = _connect(env.gate.port, f"127.0.0.1:{env.echo_port}", _auth("builder", token))
    head = _read_head(s) + s.recv(300)
    assert head.startswith(b"HTTP/1.1 403") and b"revoked" in head
    assert any("net_gate:revoked" in r for r in _records(env, "builder"))


def test_revoke_ends_an_open_tunnel_mid_transfer(env):
    token = env.store.add("builder", [f"127.0.0.1:{env.echo_port}"], allow_private=True)
    other = env.store.add("other", [f"127.0.0.1:{env.echo_port}"], allow_private=True)
    s = _connect(env.gate.port, f"127.0.0.1:{env.echo_port}", _auth("builder", token))
    t = _connect(env.gate.port, f"127.0.0.1:{env.echo_port}", _auth("other", other))
    assert _read_head(s).startswith(b"HTTP/1.1 200")
    assert _read_head(t).startswith(b"HTTP/1.1 200")
    s.sendall(b"before")
    assert s.recv(100) == b"before"

    # Revoked through a second store on the same file, as `vaara net-gate
    # revoke` does from another process; the running gateway picks it up.
    MandateStore(env.tmp / "agents.json").revoke("builder")
    assert _wait_for(lambda: not any(c.agent_id == "builder" for c in env.gate.live.values()))
    s.settimeout(2)
    try:
        s.sendall(b"after")
        data = s.recv(100)
    except OSError:
        data = b""
    assert data == b""
    # The other agent is untouched.
    t.sendall(b"still here")
    assert t.recv(100) == b"still here"
    assert _wait_for(lambda: any('"reason": "revoked"' in r for r in _records(env, "builder")))


def test_plain_http_is_sent_in_origin_form_without_proxy_headers(env):
    token = env.store.add("builder", [f"127.0.0.1:{env.origin_port}"], allow_private=True)
    s = socket.create_connection(("127.0.0.1", env.gate.port), timeout=5)
    s.sendall((f"GET http://127.0.0.1:{env.origin_port}/v1/thing?x=1 HTTP/1.1\r\n"
               f"Host: 127.0.0.1:{env.origin_port}\r\n{_auth('builder', token)}\r\n"
               "Connection: keep-alive\r\n\r\n").encode())
    data = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
    head = json.loads(data.split(b"\r\n\r\n", 1)[1])["head"]
    assert head.startswith("GET /v1/thing?x=1 HTTP/1.1")
    assert "Proxy-Authorization" not in head
    assert "Connection: close" in head and "keep-alive" not in head


def test_policy_hold_is_refused_and_closed_as_policy(env, monkeypatch):
    token = env.store.add("builder", [f"127.0.0.1:{env.echo_port}"], allow_private=True)
    real = env.pipeline.intercept

    def hold(*a, **kw):
        result = real(*a, **kw)
        if kw.get("tool_name") == "net.connect" and "policy_decision" not in kw:
            result.decision = "escalate"
            result.reason = "test hold"
        return result

    resolved = []
    monkeypatch.setattr(env.pipeline, "intercept", hold)
    monkeypatch.setattr(env.pipeline, "resolve_escalation",
                        lambda *a, **kw: resolved.append(kw))
    s = _connect(env.gate.port, f"127.0.0.1:{env.echo_port}", _auth("builder", token))
    head = _read_head(s) + s.recv(300)
    assert head.startswith(b"HTTP/1.1 403") and b"policy escalate" in head
    assert resolved and resolved[0]["approver"] == "policy"


# The CLI


def test_cli_add_list_revoke_and_wg(tmp_path, capsys):
    agents = str(tmp_path / "agents.json")
    assert netgate_main(["--agents", agents, "add-agent", "builder",
                         "--allow", "api.github.com"]) == 0
    out = capsys.readouterr().out
    token = next(line.split()[1] for line in out.splitlines() if line.startswith("token:"))
    assert MandateStore(tmp_path / "agents.json").authenticate("builder", token)
    assert netgate_main(["--agents", agents, "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["allow"] == ["api.github.com"] and listed[0]["revoked"] == ""
    assert netgate_main(["--agents", agents, "revoke", "builder"]) == 0
    assert netgate_main(["--agents", agents, "revoke", "nobody"]) == 2
    capsys.readouterr()
    assert netgate_main(["wg-config", "--address", "10.70.0.2/32", "--gateway-key", "PUB",
                         "--endpoint", "gw:51820", "--direct", "192.168.0.0/16"]) == 0
    allowed = next(line for line in capsys.readouterr().out.splitlines()
                   if line.startswith("AllowedIPs = "))
    assert "192.168." not in allowed and "128.0.0.0/2" in allowed


def test_vaara_cli_forwards_net_gate(tmp_path, capsys):
    from vaara.cli import main as vaara_main

    assert vaara_main(["net-gate", "--agents", str(tmp_path / "a.json"), "list"]) == 0
    assert json.loads(capsys.readouterr().out) == []
