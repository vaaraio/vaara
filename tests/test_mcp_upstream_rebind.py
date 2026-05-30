"""DNS-rebind regression tests for the upstream egress floor.

The host-resolution floor (`assert_url_egress_allowed`) validates the IPs a
name resolves to, but on its own it then hands the *name* back to urllib,
which re-resolves at socket-connect time. A time-split rebind (public at the
check, a blocked address at the connect) would slip through that gap. The fix
validates and pins the IP at connect time and dials the IP literal, so the
address that passed the floor is the exact address the socket reaches.

These tests drive `socket.getaddrinfo`/`socket.create_connection` directly so
no real network is touched.
"""

from __future__ import annotations

import socket

import pytest

from vaara.integrations._egress_guard import EgressBlocked, pick_egress_ip
from vaara.integrations._mcp_upstream import ProxyError
from vaara.integrations._mcp_upstream_http import HttpUpstreamClient

PUBLIC = "93.184.216.34"
METADATA = "169.254.169.254"


def _addrinfo(ip: str, port):
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port or 0))]


def _sequence_resolver(*ips: str):
    """getaddrinfo stand-in yielding ips[i] on the i-th call, last repeats."""
    state = {"n": 0}

    def fake(host, port, *args, **kwargs):
        idx = min(state["n"], len(ips) - 1)
        state["n"] += 1
        return _addrinfo(ips[idx], port)

    return fake, state


def _recording_connect(targets: list[str]):
    def fake(address, *args, **kwargs):
        targets.append(address[0])
        raise ConnectionRefusedError("rebind-test: no real socket opened")

    return fake


# -- connector-level rebind defence -----------------------------------------


def test_rebind_after_preflight_is_blocked_at_connect(monkeypatch):
    """Public at the constructor check, metadata at connect: refused, no socket."""
    resolver, _ = _sequence_resolver(PUBLIC, METADATA)
    targets: list[str] = []
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    monkeypatch.setattr(socket, "create_connection", _recording_connect(targets))

    client = HttpUpstreamClient("http://rebind.test:8765/mcp", allow_private_hosts=False)
    with pytest.raises(ProxyError):
        client.request({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    # The floor refused the rebound address before any socket was opened.
    assert METADATA not in targets
    assert targets == []
    client.close()


def test_connect_dials_validated_ip_literal(monkeypatch):
    """The socket is opened to the validated IP literal, never the hostname."""
    resolver, _ = _sequence_resolver(PUBLIC)  # public on every resolution
    targets: list[str] = []
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    monkeypatch.setattr(socket, "create_connection", _recording_connect(targets))

    client = HttpUpstreamClient("http://pin.test:8765/mcp", allow_private_hosts=False)
    with pytest.raises(ProxyError):  # ConnectionRefusedError from the recorder
        client.request({"jsonrpc": "2.0", "id": 1, "method": "ping"})
    # Exactly one connect, to the literal we validated, not a re-resolved name.
    assert targets == [PUBLIC]
    client.close()


# -- pick_egress_ip unit behaviour ------------------------------------------


def test_pick_egress_ip_pins_first_public(monkeypatch):
    resolver, _ = _sequence_resolver(PUBLIC)
    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    assert pick_egress_ip("example.test", 443, allow_private=False) == PUBLIC


def test_pick_egress_ip_refuses_mixed_public_and_metadata(monkeypatch):
    """A single answer set containing a blocked address is refused outright."""

    def resolver(host, port, *args, **kwargs):
        return _addrinfo(PUBLIC, port) + _addrinfo(METADATA, port)

    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    with pytest.raises(EgressBlocked):
        pick_egress_ip("mixed.test", 443, allow_private=False)


def test_pick_egress_ip_returns_literals_without_resolving():
    assert pick_egress_ip("8.8.8.8", 443, allow_private=False) == "8.8.8.8"
    assert pick_egress_ip("127.0.0.1", 80, allow_private=True) == "127.0.0.1"
    with pytest.raises(EgressBlocked):  # loopback blocked without opt-in
        pick_egress_ip("127.0.0.1", 80, allow_private=False)
    with pytest.raises(EgressBlocked):  # metadata blocked even under opt-in
        pick_egress_ip("169.254.169.254", 80, allow_private=True)
