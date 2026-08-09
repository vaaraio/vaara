# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Size limits on what a remote MCP upstream can send back.

The inbound side of the proxy caps a client message at 1 MiB before it
reaches json.loads. The outbound side had no cap at all: ``resp.read()``
consumed the whole reply, and the SSE reader looped until the upstream chose
to stop. ``--upstream-url`` points at a remote server over the network, which
is the direction the SSRF egress floor exists to distrust in the first place,
so a hostile or compromised upstream could exhaust the proxy's memory by
answering one JSON-RPC request with a body that never ends.
"""

from __future__ import annotations

import io
import json

import pytest

from vaara.integrations import _mcp_upstream_http as mod
from vaara.integrations._mcp_upstream import ProxyError


class _FakeResponse(io.BytesIO):
    """Enough of an http.client.HTTPResponse for the reply extractors."""

    def __init__(self, payload: bytes, content_type: str = "application/json"):
        super().__init__(payload)
        self.headers = {"Content-Type": content_type}


def _client() -> mod.HttpUpstreamClient:
    """A client with only the fields the reply extractors touch.

    Constructing the real thing would open a socket to an upstream; these
    tests are about what the extractors do with bytes that already arrived.
    """
    client = mod.HttpUpstreamClient.__new__(mod.HttpUpstreamClient)
    client._on_notification = None
    return client


def test_cap_is_bigger_than_the_inbound_cap():
    """Upstream replies are legitimately larger than the requests that ask."""
    from vaara.integrations.mcp_proxy import _MCP_HTTP_MAX_BODY_BYTES

    assert mod._MCP_UPSTREAM_MAX_BODY_BYTES > _MCP_HTTP_MAX_BODY_BYTES


def test_oversized_json_reply_is_refused():
    payload = b'{"jsonrpc":"2.0","id":1,"result":{"x":"' \
        + b"a" * (mod._MCP_UPSTREAM_MAX_BODY_BYTES + 10) + b'"}}'
    with pytest.raises(ProxyError, match="too large"):
        _client()._reply_from_json(_FakeResponse(payload))


def test_reply_at_the_limit_still_parses():
    body = {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}
    raw = json.dumps(body).encode()
    assert _client()._reply_from_json(_FakeResponse(raw)) == body


def test_empty_reply_still_reports_empty_not_oversized():
    with pytest.raises(ProxyError, match="empty"):
        _client()._reply_from_json(_FakeResponse(b""))


def test_non_json_reply_is_still_a_clean_error():
    with pytest.raises(ProxyError, match="non-JSON"):
        _client()._reply_from_json(_FakeResponse(b"<html>login</html>"))


def test_endless_sse_stream_is_cut_off():
    """A stream that never carries the reply must not run forever."""
    class _Endless:
        headers = {"Content-Type": "text/event-stream"}

        def __iter__(self):
            while True:
                yield b'data: {"jsonrpc":"2.0","method":"notifications/message"}\n'
                yield b"\n"

    with pytest.raises(ProxyError, match="too large|too many"):
        _client()._reply_from_sse(_Endless(), want_id=99)


def test_single_oversized_sse_event_is_refused():
    huge = b"x" * (mod._MCP_UPSTREAM_MAX_BODY_BYTES + 10)

    class _Fat:
        headers = {"Content-Type": "text/event-stream"}

        def __iter__(self):
            yield b"data: " + huge + b"\n"
            yield b"\n"

    with pytest.raises(ProxyError, match="too large"):
        _client()._reply_from_sse(_Fat(), want_id=1)


def test_a_normal_sse_reply_still_arrives():
    reply = {"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}

    class _Normal:
        headers = {"Content-Type": "text/event-stream"}

        def __iter__(self):
            yield b'data: {"jsonrpc":"2.0","method":"notifications/progress"}\n'
            yield b"\n"
            yield ("data: " + json.dumps(reply) + "\n").encode()
            yield b"\n"

    assert _client()._reply_from_sse(_Normal(), want_id=7) == reply
