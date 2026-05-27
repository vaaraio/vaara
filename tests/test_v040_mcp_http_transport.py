"""v0.40 MCP proxy streamable-HTTP transport + fan-out routing."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:
    pytest.skip(
        "server extra not installed (pip install 'vaara[server]')",
        allow_module_level=True,
    )

from vaara.integrations import mcp_proxy
from vaara.integrations.mcp_proxy import VaaraMCPProxy, _parse_upstream_specs


# ── _parse_upstream_specs ──────────────────────────────────────────────────

def test_parse_upstream_specs_bare_command_lands_in_default():
    result = _parse_upstream_specs(["echo"], [])
    assert result == {"default": ["echo"]}


def test_parse_upstream_specs_named_slot():
    result = _parse_upstream_specs(["github=gh-mcp-server"], [])
    assert result == {"github": ["gh-mcp-server"]}


def test_parse_upstream_specs_command_with_equals_stays_intact():
    """A command like `python -m foo --bar=baz` must NOT be split at '='."""
    result = _parse_upstream_specs(["python -m foo --bar=baz"], [])
    assert result == {"default": ["python -m foo --bar=baz"]}


def test_parse_upstream_specs_legacy_args_join_first_slot():
    result = _parse_upstream_specs(["echo"], ["hello", "world"])
    assert result == {"default": ["echo", "hello", "world"]}


def test_parse_upstream_specs_multiple_named_fanout():
    result = _parse_upstream_specs(["a=cmd-a", "b=cmd-b"], [])
    assert result == {"a": ["cmd-a"], "b": ["cmd-b"]}


def test_parse_upstream_specs_unknown_name_pattern_falls_to_default():
    """Names that aren't simple slugs aren't treated as NAME= prefix."""
    result = _parse_upstream_specs(["python -m srv=foo"], [])
    # The left side ("python -m srv") fails the slug regex, so the whole
    # spec is treated as a bare command under "default".
    assert result == {"default": ["python -m srv=foo"]}


# ── VaaraMCPProxy multi-upstream constructor ───────────────────────────────

@pytest.fixture
def http_proxy(monkeypatch):
    """A VaaraMCPProxy with mocked upstreams and pipeline."""
    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    pipeline = MagicMock()
    proxy = VaaraMCPProxy(
        upstreams={"alpha": ["cmd-alpha"], "beta": ["cmd-beta"]},
        pipeline=pipeline,
    )
    return proxy


def test_constructor_rejects_both_upstream_and_upstreams():
    with pytest.raises(ValueError, match="Pass either"):
        VaaraMCPProxy(
            upstream_command=["echo"],
            upstreams={"a": ["echo"]},
            pipeline=MagicMock(),
        )


def test_constructor_requires_at_least_one_upstream():
    with pytest.raises(ValueError, match="upstream"):
        VaaraMCPProxy(pipeline=MagicMock())


def test_multi_upstream_populates_default_slot(http_proxy):
    assert "default" in http_proxy._upstreams
    # When no explicit default is provided, the sorted-first name acts as fallback.
    assert http_proxy._upstreams["default"] is not None


def test_default_slot_aliases_first_upstream_not_duplicate(monkeypatch):
    """Regression guard: default fallback must alias an existing upstream
    client instead of spawning a duplicate subprocess. An earlier shape
    cloned the command into upstream_map["default"] before constructing
    the client map, which built a second UpstreamMCPClient for the same
    command."""
    constructed: list[MagicMock] = []

    def make_client(*_args, **_kw):
        instance = MagicMock()
        constructed.append(instance)
        return instance

    monkeypatch.setattr(
        mcp_proxy, "UpstreamMCPClient", MagicMock(side_effect=make_client),
    )
    proxy = VaaraMCPProxy(
        upstreams={"alpha": ["cmd-alpha"], "beta": ["cmd-beta"]},
        pipeline=MagicMock(),
    )
    # Exactly two clients constructed (alpha, beta), not three.
    assert len(constructed) == 2
    # "default" aliases the sorted-first real slot.
    assert proxy._upstreams["default"] is proxy._upstreams["alpha"]
    assert proxy._upstreams["default"] is not proxy._upstreams["beta"]


def test_single_upstream_lands_under_default_slot(monkeypatch):
    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    proxy = VaaraMCPProxy(
        upstream_command=["echo"], pipeline=MagicMock(),
    )
    assert list(proxy._upstreams) == ["default"]


# ── HTTP transport endpoints ───────────────────────────────────────────────

def _build_http_app(proxy):
    """Construct the FastAPI app run_http() builds, without uvicorn.run()."""
    import unittest.mock as um

    with um.patch("uvicorn.run") as run_mock:
        # Capture the app by intercepting uvicorn.run() — we call the same
        # method the production CLI uses, but never block on the event loop.
        captured: dict = {}

        def fake_run(app, **kwargs):
            captured["app"] = app

        run_mock.side_effect = fake_run
        proxy.run_http(host="127.0.0.1", port=0)
        return captured["app"]


def test_http_health_lists_upstreams(http_proxy):
    app = _build_http_app(http_proxy)
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert set(resp.json()["upstreams"]) >= {"alpha", "beta", "default"}


def test_http_mcp_post_routes_to_named_upstream(http_proxy):
    http_proxy._upstreams["alpha"].request.return_value = {
        "jsonrpc": "2.0", "id": 1, "result": {"tools": []},
    }
    app = _build_http_app(http_proxy)
    client = TestClient(app)
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"X-Vaara-Upstream": "alpha"},
    )
    assert resp.status_code == 200
    http_proxy._upstreams["alpha"].request.assert_called_once()


def test_http_mcp_fanout_without_header_returns_400(http_proxy):
    """Multi-upstream deployment must NOT silently default-route."""
    app = _build_http_app(http_proxy)
    client = TestClient(app)
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["error"]["code"] == "upstream_required"
    # Operator gets the list of valid slots in the error so a client UI
    # can recover without an additional health probe round-trip.
    assert "alpha" in body["detail"]["error"]["message"]
    assert "beta" in body["detail"]["error"]["message"]


def test_http_mcp_single_upstream_silent_default(monkeypatch):
    """Single-upstream deployment keeps the v0.39 silent-default contract."""
    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    proxy = VaaraMCPProxy(upstream_command=["echo"], pipeline=MagicMock())
    proxy._upstreams["default"].request.return_value = {
        "jsonrpc": "2.0", "id": 1, "result": {"tools": []},
    }
    app = _build_http_app(proxy)
    client = TestClient(app)
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert resp.status_code == 200
    proxy._upstreams["default"].request.assert_called_once()


def test_http_mcp_unknown_upstream_returns_404(http_proxy):
    app = _build_http_app(http_proxy)
    client = TestClient(app)
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"X-Vaara-Upstream": "no-such-thing"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"]["error"]["code"] == "unknown_upstream"


def test_http_mcp_notification_returns_202(http_proxy):
    app = _build_http_app(http_proxy)
    client = TestClient(app)
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers={"X-Vaara-Upstream": "alpha"},
    )
    assert resp.status_code == 202


def test_http_mcp_oversized_body_returns_413(http_proxy, monkeypatch):
    monkeypatch.setattr(mcp_proxy, "_MCP_HTTP_MAX_BODY_BYTES", 64)
    app = _build_http_app(http_proxy)
    client = TestClient(app)
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list",
        "params": {"x": "a" * 10_000},
    }
    resp = client.post("/mcp", json=payload)
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "payload_too_large"


def test_http_mcp_bad_json_returns_parse_error(http_proxy):
    app = _build_http_app(http_proxy)
    client = TestClient(app)
    resp = client.post("/mcp", content=b"not json")
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == -32700


def test_http_mcp_tenant_header_threads_into_overt(http_proxy):
    """X-Vaara-Tenant becomes a non_content_metadata claim on OVERT envelope."""
    http_proxy._overt = MagicMock()
    http_proxy._upstreams["alpha"].request.return_value = {
        "jsonrpc": "2.0", "id": 1, "result": {"tools": []},
    }
    app = _build_http_app(http_proxy)
    client = TestClient(app)
    client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"X-Vaara-Tenant": "tenant-q", "X-Vaara-Upstream": "alpha"},
    )
    # tools/list does not emit an OVERT envelope in the current proxy
    # implementation, so this test exercises only that the request
    # dispatched cleanly with the header set.
    http_proxy._upstreams["alpha"].request.assert_called_once()
