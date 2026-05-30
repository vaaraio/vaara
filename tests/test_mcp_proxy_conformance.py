"""v0.45 Streamable HTTP transport conformance.

Three checks the proxy was missing: Mcp-Session-Id visible-ASCII charset
(alongside the existing length cap), MCP-Protocol-Version validation on
POST and GET, and POST Accept negotiation (the client must be able to
receive both application/json and text/event-stream).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vaara.integrations import mcp_proxy
from vaara.integrations.mcp_proxy import (
    VaaraMCPProxy,
    _accept_satisfies,
    _protocol_version_supported,
    _session_id_is_visible_ascii,
)

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

needs_server = pytest.mark.skipif(
    not HAS_FASTAPI,
    reason="server extra not installed (pip install 'vaara[server]')",
)

_NOTIFY = {"jsonrpc": "2.0", "method": "notifications/initialized"}
_PAYLOAD = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
_BOTH = "application/json, text/event-stream"


# ── pure predicate units (no server extra required) ────────────────────────

@pytest.mark.parametrize("value,ok", [
    ("", True),
    ("abc-123_XYZ.~", True),
    ("!~", True),                  # boundary code points 0x21 and 0x7E
    ("has space", False),          # 0x20 is below the visible range
    ("tab\there", False),          # 0x09
    ("del\x7f", False),            # 0x7F
    ("unié", False),          # non-ASCII
])
def test_session_id_visible_ascii(value, ok):
    assert _session_id_is_visible_ascii(value) is ok


@pytest.mark.parametrize("version,ok", [
    (None, True),
    ("", True),
    ("   ", True),
    ("2025-03-26", True),
    ("2025-06-18", True),
    ("2024-11-05", False),         # old two-endpoint transport, not this one
    ("garbage", False),
])
def test_protocol_version_supported(version, ok):
    assert _protocol_version_supported(version) is ok


@pytest.mark.parametrize("accept,media_type,ok", [
    (None, "application/json", True),
    ("", "application/json", True),
    ("*/*", "text/event-stream", True),
    ("application/*", "application/json", True),
    ("application/json", "application/json", True),
    ("application/json", "text/event-stream", False),
    (_BOTH, "text/event-stream", True),
    ("application/json;q=0.9, text/event-stream;q=0.1", "text/event-stream", True),
    ("text/html", "application/json", False),
])
def test_accept_satisfies(accept, media_type, ok):
    assert _accept_satisfies(accept, media_type) is ok


# ── endpoint behaviour ─────────────────────────────────────────────────────

@pytest.fixture
def http_proxy(monkeypatch):
    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    return VaaraMCPProxy(
        upstreams={"alpha": ["cmd-alpha"], "beta": ["cmd-beta"]},
        pipeline=MagicMock(),
    )


def _client(proxy):
    import unittest.mock as um

    with um.patch("uvicorn.run") as run_mock:
        captured: dict = {}
        run_mock.side_effect = lambda app, **kw: captured.__setitem__("app", app)
        proxy.run_http(host="127.0.0.1", port=0)
    return TestClient(captured["app"])


@needs_server
def test_post_accept_json_only_rejected(http_proxy):
    resp = _client(http_proxy).post(
        "/mcp", json=_NOTIFY,
        headers={"Accept": "application/json", "X-Vaara-Upstream": "alpha"},
    )
    assert resp.status_code == 406
    assert resp.json()["detail"]["error"]["code"] == "not_acceptable"


@needs_server
def test_post_accept_both_ok(http_proxy):
    resp = _client(http_proxy).post(
        "/mcp", json=_NOTIFY,
        headers={"Accept": _BOTH, "X-Vaara-Upstream": "alpha"},
    )
    assert resp.status_code == 202


@needs_server
def test_post_default_wildcard_accept_still_ok(http_proxy):
    # TestClient sends Accept: */* by default; the wildcard must satisfy.
    resp = _client(http_proxy).post(
        "/mcp", json=_NOTIFY, headers={"X-Vaara-Upstream": "alpha"},
    )
    assert resp.status_code == 202


@needs_server
def test_post_unsupported_protocol_version_rejected(http_proxy):
    resp = _client(http_proxy).post(
        "/mcp", json=_NOTIFY,
        headers={
            "Accept": _BOTH,
            "X-Vaara-Upstream": "alpha",
            "MCP-Protocol-Version": "1999-01-01",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["code"] == "unsupported_protocol_version"


@needs_server
def test_post_supported_protocol_version_ok(http_proxy):
    resp = _client(http_proxy).post(
        "/mcp", json=_NOTIFY,
        headers={
            "Accept": _BOTH,
            "X-Vaara-Upstream": "alpha",
            "MCP-Protocol-Version": "2025-06-18",
        },
    )
    assert resp.status_code == 202


@needs_server
def test_post_session_id_non_visible_ascii_rejected(http_proxy):
    # An embedded space (0x20) survives .strip() and is below the visible range.
    resp = _client(http_proxy).post(
        "/mcp", json=_PAYLOAD,
        headers={
            "Accept": _BOTH,
            "X-Vaara-Upstream": "alpha",
            "Mcp-Session-Id": "bad id",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["code"] == "session_id_invalid"


@needs_server
def test_post_session_id_too_long_rejected(http_proxy):
    resp = _client(http_proxy).post(
        "/mcp", json=_PAYLOAD,
        headers={
            "Accept": _BOTH,
            "X-Vaara-Upstream": "alpha",
            "Mcp-Session-Id": "a" * 129,
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["code"] == "session_id_too_long"


@needs_server
def test_get_unsupported_protocol_version_rejected(http_proxy):
    resp = _client(http_proxy).get(
        "/mcp",
        headers={
            "Mcp-Session-Id": "sess-1",
            "MCP-Protocol-Version": "1999-01-01",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["code"] == "unsupported_protocol_version"


@needs_server
def test_get_session_id_non_visible_ascii_rejected(http_proxy):
    resp = _client(http_proxy).get(
        "/mcp",
        headers={"Mcp-Session-Id": "bad id", "X-Vaara-Upstream": "alpha"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"]["code"] == "session_id_invalid"
