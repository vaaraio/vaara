"""HttpUpstreamClient against a real stdlib HTTP/SSE MCP server.

Exercises both reply transports (application/json and text/event-stream),
session-id + auth-header echo, the standing server-to-client SSE channel, and
the error path. The fake server runs on an ephemeral loopback port.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vaara.integrations._mcp_upstream import ProxyError
from vaara.integrations._mcp_upstream_http import HttpUpstreamClient


@pytest.fixture(autouse=True)
def _allow_loopback_upstream(monkeypatch):
    """The fake MCP servers in this module bind loopback, which the SSRF egress
    floor refuses by default. Opt in process-wide for the connector tests; the
    dedicated egress tests below construct with allow_private_hosts=False to
    assert the blocking path explicitly.
    """
    monkeypatch.setenv("VAARA_MCP_ALLOW_PRIVATE_UPSTREAM", "1")


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence the test server
        pass

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else None

    def _send_json(self, obj, *, status=200, session=None):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if session is not None:
            self.send_header("Mcp-Session-Id", session)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        cfg = self.server.cfg
        msg = self._read_body()
        cfg["last_post_headers"] = dict(self.headers)
        if isinstance(msg, dict) and "id" not in msg:  # notification
            cfg["notifications"].append(msg)
            self.send_response(202)
            self.end_headers()
            return
        if msg.get("method") == "initialize":
            self._send_json(
                {"jsonrpc": "2.0", "id": msg["id"],
                 "result": {"protocolVersion": "2025-06-18", "serverInfo": {"name": "fake"}}},
                session="sess-123",
            )
            return
        if cfg.get("http_error"):
            self._send_json({"error": "boom"}, status=cfg["http_error"])
            return
        if cfg.get("reply_mode") == "sse":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            note = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}}
            reply = {"jsonrpc": "2.0", "id": msg["id"], "result": {"ok": True}}
            self.wfile.write(b"data: " + json.dumps(note).encode() + b"\n\n")
            self.wfile.write(b"data: " + json.dumps(reply).encode() + b"\n\n")
            self.wfile.flush()
            return
        self._send_json({"jsonrpc": "2.0", "id": msg["id"], "result": {"ok": True}})

    def do_GET(self):
        cfg = self.server.cfg
        if cfg.get("no_push"):
            self.send_response(405)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        note = {"jsonrpc": "2.0", "method": "notifications/message", "params": {"hi": 1}}
        try:
            self.wfile.write(b"id: 1\ndata: " + json.dumps(note).encode() + b"\n\n")
            self.wfile.flush()
        except OSError:
            return
        while not cfg["stop"].is_set():  # hold the standing stream open
            time.sleep(0.02)


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.daemon_threads = True
    httpd.cfg = {"notifications": [], "stop": threading.Event()}
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/mcp"
    try:
        yield httpd, url
    finally:
        httpd.cfg["stop"].set()
        httpd.shutdown()


def _init(client, rid=1):
    return client.request({"jsonrpc": "2.0", "id": rid, "method": "initialize", "params": {}})


def test_initialize_captures_session_and_version(server):
    httpd, url = server
    client = HttpUpstreamClient(url)
    try:
        reply = _init(client)
        assert reply["result"]["protocolVersion"] == "2025-06-18"
        assert client._session_id == "sess-123"
        assert client._protocol_version == "2025-06-18"
    finally:
        client.close()


def test_session_and_auth_echoed_on_later_request(server):
    httpd, url = server
    client = HttpUpstreamClient(url, headers={"Authorization": "Bearer tok-xyz"})
    try:
        _init(client)
        reply = client.request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert reply["result"] == {"ok": True}
        sent = httpd.cfg["last_post_headers"]
        assert sent.get("Mcp-Session-Id") == "sess-123"
        assert sent.get("Mcp-Protocol-Version") == "2025-06-18"
        assert sent.get("Authorization") == "Bearer tok-xyz"
    finally:
        client.close()


def test_sse_reply_routes_notification_and_returns(server):
    httpd, url = server
    httpd.cfg["reply_mode"] = "sse"
    httpd.cfg["no_push"] = True  # isolate the inline-SSE path from the standing one
    seen = []
    client = HttpUpstreamClient(url, on_notification=seen.append)
    try:
        _init(client)
        reply = client.request({"jsonrpc": "2.0", "id": 7, "method": "tools/list"})
        assert reply == {"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}
        assert any(m.get("method") == "notifications/progress" for m in seen)
    finally:
        client.close()


def test_standing_stream_delivers_server_notification(server):
    httpd, url = server
    got = threading.Event()
    received = []

    def on_note(msg):
        received.append(msg)
        got.set()

    client = HttpUpstreamClient(url, on_notification=on_note)
    try:
        _init(client)  # starts the standing GET listener
        assert got.wait(timeout=5.0), "no server-initiated notification arrived"
        assert received[0]["method"] == "notifications/message"
    finally:
        client.close()


def test_http_error_raises_proxyerror(server):
    httpd, url = server
    httpd.cfg["http_error"] = 500
    httpd.cfg["no_push"] = True
    client = HttpUpstreamClient(url)
    try:
        _init(client)
        with pytest.raises(ProxyError, match="HTTP 500"):
            client.request({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    finally:
        client.close()


def test_request_after_close_raises(server):
    httpd, url = server
    client = HttpUpstreamClient(url)
    client.close()
    with pytest.raises(ProxyError, match="closed"):
        _init(client)


def test_proxy_fronts_remote_http_upstream(server):
    """End-to-end: VaaraMCPProxy routes a tools/call to a remote HTTP upstream."""
    from dataclasses import dataclass
    from unittest.mock import MagicMock

    from vaara.integrations.mcp_proxy import VaaraMCPProxy

    @dataclass
    class _Allow:
        allowed: bool = True
        action_id: str = "act-1"
        reason: str = ""
        decision: str = "ALLOW"

    httpd, url = server
    httpd.cfg["no_push"] = True  # keep this test to the request path
    pipeline = MagicMock()
    pipeline.intercept.return_value = _Allow()
    proxy = VaaraMCPProxy(upstream_urls={"default": url}, pipeline=pipeline)
    try:
        req = {
            "jsonrpc": "2.0", "id": 21, "method": "tools/call",
            "params": {"name": "remote.tool", "arguments": {}},
        }
        resp = proxy._handle_tools_call(req)
        assert resp["id"] == 21
        assert resp["result"] == {"ok": True}
        pipeline.report_outcome.assert_called_once()
    finally:
        proxy.close()
