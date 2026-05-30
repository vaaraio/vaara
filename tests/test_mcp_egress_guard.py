"""SSRF egress floor for the remote MCP HTTP connector.

Asserts the connector refuses loopback / link-local / RFC1918 / ULA /
cloud-metadata upstream targets by default, re-checks every redirect hop, and
never carries the Authorization header to a cross-origin redirect target.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vaara.integrations._egress_guard import (
    EgressBlocked,
    _same_origin,
    assert_url_egress_allowed,
)
from vaara.integrations._mcp_upstream import ProxyError
from vaara.integrations._mcp_upstream_http import HttpUpstreamClient


# -- host-resolution floor --------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://[fe80::a9fe:a9fe]/latest/meta-data/",
        "http://2852039166/latest/meta-data/",  # dotless decimal
        "http://0xa9fea9fe/latest/meta-data/",  # dotless hex
    ],
)
def test_metadata_address_refused(url):
    with pytest.raises(EgressBlocked):
        assert_url_egress_allowed(url)


def test_metadata_refused_even_when_private_allowed():
    with pytest.raises(EgressBlocked):
        assert_url_egress_allowed(
            "http://169.254.169.254/latest/meta-data/", allow_private=True,
        )
    with pytest.raises(EgressBlocked):
        assert_url_egress_allowed("http://2852039166/", allow_private=True)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/mcp",
        "http://localhost/mcp",
        "http://10.0.0.5/mcp",
        "http://192.168.1.10/mcp",
        "http://172.16.0.1/mcp",
        "http://[::1]/mcp",
        "http://[fc00::1]/mcp",  # IPv6 ULA
        "http://[fe80::1]/mcp",  # IPv6 link-local
        "http://[::ffff:169.254.169.254]/mcp",  # IPv4-mapped metadata
    ],
)
def test_private_and_loopback_refused_by_default(url):
    with pytest.raises(EgressBlocked):
        assert_url_egress_allowed(url)


@pytest.mark.parametrize(
    "url",
    ["http://10.0.0.5/mcp", "http://127.0.0.1/mcp", "http://[fc00::1]/mcp"],
)
def test_private_allowed_with_opt_in(url):
    assert_url_egress_allowed(url, allow_private=True)


@pytest.mark.parametrize(
    "url",
    [
        "http://0.0.0.0/mcp",  # unspecified
        "http://[::]/mcp",  # unspecified IPv6
        "http://224.0.0.1/mcp",  # multicast
        "http://[ff02::1]/mcp",  # multicast IPv6
        "http://240.0.0.1/mcp",  # reserved
    ],
)
def test_never_routable_refused_even_with_opt_in(url):
    # allow_private trusts internal hosts, not the never-routable classes.
    # Opting in must not re-open 0.0.0.0, multicast, or reserved space.
    with pytest.raises(EgressBlocked):
        assert_url_egress_allowed(url, allow_private=True)


def test_public_host_allowed():
    # Literal public IP so the floor passes without a live DNS lookup.
    assert_url_egress_allowed("https://8.8.8.8/mcp")
    assert_url_egress_allowed("http://[2001:4860:4860::8888]/mcp")


def test_non_http_scheme_refused():
    with pytest.raises(EgressBlocked):
        assert_url_egress_allowed("file:///etc/passwd")
    with pytest.raises(EgressBlocked):
        assert_url_egress_allowed("gopher://10.0.0.1/")


def test_same_origin_logic():
    assert _same_origin("http://a.com/x", "http://a.com/y")
    assert _same_origin("http://a.com:80/x", "http://a.com/y")
    assert not _same_origin("http://a.com/x", "https://a.com/x")
    assert not _same_origin("http://a.com/x", "http://b.com/x")
    assert not _same_origin("http://a.com:80/x", "http://a.com:81/x")


# -- connector-level blocking ----------------------------------------------


def test_client_refuses_metadata_upstream():
    with pytest.raises(ProxyError):
        HttpUpstreamClient("http://169.254.169.254/mcp")


def test_client_refuses_private_upstream_by_default():
    with pytest.raises(ProxyError):
        HttpUpstreamClient("http://10.0.0.5/mcp")


def test_client_allows_private_with_opt_in():
    client = HttpUpstreamClient("http://10.0.0.5/mcp", allow_private_hosts=True)
    client.close()


# -- redirect handling ------------------------------------------------------


class _RedirectHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _do(self):
        cfg = self.server.cfg
        cfg["last_headers"] = dict(self.headers)
        location = cfg.get("location")
        if location is not None:
            # 302 so urllib follows it (it refuses to auto-follow 307 on POST);
            # the redirect target is reached as a GET, which is all these tests
            # need to observe the egress re-check and the cross-origin auth drop.
            self.send_response(302)
            self.send_header("Location", location)
            self.end_headers()
            return
        body = json.dumps(
            {"jsonrpc": "2.0", "id": cfg.get("want_id", 1), "result": {"ok": True}}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = _do
    do_GET = _do


@pytest.fixture
def redirect_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    httpd.daemon_threads = True
    httpd.cfg = {}
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/mcp"
    try:
        yield httpd, url
    finally:
        httpd.shutdown()


def test_redirect_to_private_ip_refused(redirect_server):
    """A redirect to the metadata address is refused even when the original
    host was opted in: the guard re-checks every hop."""
    httpd, url = redirect_server
    httpd.cfg["location"] = "http://169.254.169.254/latest/meta-data/"
    client = HttpUpstreamClient(url, allow_private_hosts=True)
    try:
        with pytest.raises(ProxyError):
            client.request({"jsonrpc": "2.0", "id": 5, "method": "tools/list"})
    finally:
        client.close()


def test_authorization_not_leaked_cross_origin(redirect_server):
    """A cross-origin redirect must not carry the upstream Authorization header
    to the new host."""
    target = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectHandler)
    target.daemon_threads = True
    target.cfg = {"want_id": 9}
    threading.Thread(target=target.serve_forever, daemon=True).start()
    target_url = f"http://127.0.0.1:{target.server_address[1]}/mcp"

    httpd, url = redirect_server
    httpd.cfg["location"] = target_url

    client = HttpUpstreamClient(
        url,
        headers={"Authorization": "Bearer secret-token"},
        allow_private_hosts=True,
    )
    try:
        reply = client.request({"jsonrpc": "2.0", "id": 9, "method": "tools/list"})
        assert reply["result"] == {"ok": True}
        landed = target.cfg.get("last_headers", {})
        assert not any(k.lower() == "authorization" for k in landed)
        first = httpd.cfg.get("last_headers", {})
        assert first.get("Authorization") == "Bearer secret-token"
    finally:
        client.close()
        target.shutdown()
