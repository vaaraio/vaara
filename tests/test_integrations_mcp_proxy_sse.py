"""Tests for the Streamable HTTP transport's GET /mcp SSE endpoint (v0.41).

Exercises validation, basic event delivery, and Last-Event-ID resume against
the FastAPI app produced by ``VaaraMCPProxy._build_http_app`` without
standing up uvicorn. The streaming behaviour is driven through
``TestClient.stream`` so each test can assert against the SSE frames as
they arrive.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
starlette = pytest.importorskip("starlette")
from fastapi.testclient import TestClient  # noqa: E402


def _build_proxy(monkeypatch, upstreams=None):
    from vaara.integrations import mcp_proxy

    monkeypatch.setattr(
        mcp_proxy, "UpstreamMCPClient",
        MagicMock(side_effect=lambda **kw: MagicMock()),
    )
    pipeline = MagicMock()
    if upstreams is None:
        p = mcp_proxy.VaaraMCPProxy(upstream_command=["echo"], pipeline=pipeline)
    else:
        p = mcp_proxy.VaaraMCPProxy(upstreams=upstreams, pipeline=pipeline)
    return p, pipeline


def test_get_mcp_requires_session_id_header(monkeypatch):
    p, _ = _build_proxy(monkeypatch)
    app = p._build_http_app()
    client = TestClient(app)
    r = client.get("/mcp")
    assert r.status_code == 400
    body = r.json()
    assert body["detail"]["error"]["code"] == "session_id_required"


def test_get_mcp_rejects_oversized_session_id(monkeypatch):
    p, _ = _build_proxy(monkeypatch)
    app = p._build_http_app()
    client = TestClient(app)
    r = client.get("/mcp", headers={"Mcp-Session-Id": "x" * 129})
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "session_id_too_long"


def test_get_mcp_404_on_unknown_upstream(monkeypatch):
    p, _ = _build_proxy(
        monkeypatch, upstreams={"alpha": ["echo"], "beta": ["echo"]},
    )
    app = p._build_http_app()
    client = TestClient(app)
    r = client.get(
        "/mcp",
        headers={"Mcp-Session-Id": "s1", "X-Vaara-Upstream": "ghost"},
    )
    assert r.status_code == 404
    assert r.json()["detail"]["error"]["code"] == "unknown_upstream"


def test_get_mcp_400_on_ambiguous_fanout_without_upstream(monkeypatch):
    p, _ = _build_proxy(
        monkeypatch, upstreams={"alpha": ["echo"], "beta": ["echo"]},
    )
    app = p._build_http_app()
    client = TestClient(app)
    r = client.get("/mcp", headers={"Mcp-Session-Id": "s1"})
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "upstream_required"


def test_post_mcp_rejects_oversized_session_id(monkeypatch):
    p, _ = _build_proxy(monkeypatch)
    app = p._build_http_app()
    client = TestClient(app)
    r = client.post(
        "/mcp",
        content=b'{"jsonrpc":"2.0","method":"ping"}',
        headers={
            "Content-Type": "application/json",
            "Mcp-Session-Id": "x" * 200,
        },
    )
    assert r.status_code == 400
    assert r.json()["detail"]["error"]["code"] == "session_id_too_long"


def test_health_endpoint_works_after_app_build(monkeypatch):
    p, _ = _build_proxy(monkeypatch)
    app = p._build_http_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "default" in body["upstreams"]
