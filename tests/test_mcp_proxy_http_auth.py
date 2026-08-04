"""The HTTP transport must authenticate before it trusts scope headers.

``X-Vaara-Tenant`` picks which tenant's audit trail a call is attributed to
and ``X-Vaara-Upstream`` picks which upstream — and so which policy — governs
it. Read from an unauthenticated caller on a network-reachable bind, that is
tenant spoofing plus weakest-policy shopping by anyone who can reach the port.

Mirrors the guard `vaara serve` already has: a bearer gate on the app, and a
refusal to bind a non-loopback host without a key.
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


def _proxy(monkeypatch, **kw):
    from vaara.integrations import mcp_proxy
    from vaara.pipeline import InterceptionPipeline
    from vaara.audit.trail import AuditTrail

    monkeypatch.setattr(
        "vaara.integrations._mcp_upstream.UpstreamMCPClient.__init__",
        lambda self, command, **k: None,
    )
    trail = AuditTrail(on_record=lambda _r: None)
    pipeline = InterceptionPipeline(trail=trail)
    return mcp_proxy.VaaraMCPProxy(
        upstream_command=["echo"], pipeline=pipeline, **kw
    )


_MCP_HEADERS = {
    "accept": "application/json, text/event-stream",
    "content-type": "application/json",
}
_BODY = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}


def test_no_key_configured_means_open(monkeypatch):
    """Default (loopback dev) behaviour is unchanged."""
    monkeypatch.delenv("VAARA_PROXY_API_KEY", raising=False)
    p = _proxy(monkeypatch)
    assert p._api_key is None
    c = TestClient(p._build_http_app())
    assert c.get("/health").status_code == 200


def test_missing_key_is_401_and_scope_headers_are_not_honoured(monkeypatch):
    p = _proxy(monkeypatch, api_key="s3cret")
    c = TestClient(p._build_http_app())
    r = c.post(
        "/mcp",
        json=_BODY,
        headers={**_MCP_HEADERS, "X-Vaara-Tenant": "victim-tenant"},
    )
    assert r.status_code == 401, r.text
    assert r.json()["error"]["code"] == "unauthorized"


def test_wrong_key_is_401(monkeypatch):
    p = _proxy(monkeypatch, api_key="s3cret")
    c = TestClient(p._build_http_app())
    r = c.post(
        "/mcp",
        json=_BODY,
        headers={**_MCP_HEADERS, "Authorization": "Bearer wrong"},
    )
    assert r.status_code == 401


def test_health_stays_open_for_load_balancers(monkeypatch):
    p = _proxy(monkeypatch, api_key="s3cret")
    c = TestClient(p._build_http_app())
    assert c.get("/health").status_code == 200


def test_correct_key_passes_the_gate(monkeypatch):
    from unittest.mock import MagicMock

    p = _proxy(monkeypatch, api_key="s3cret")
    upstream = MagicMock()
    upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1, "result": {"tools": []},
    }
    p._upstream = upstream
    for name in list(p._upstreams):
        p._upstreams[name] = upstream

    c = TestClient(p._build_http_app())
    r = c.post(
        "/mcp",
        json=_BODY,
        headers={**_MCP_HEADERS, "Authorization": "Bearer s3cret"},
    )
    # Past the gate: whatever the handler does, it is not a 401.
    assert r.status_code != 401


def test_env_var_supplies_the_key(monkeypatch):
    monkeypatch.setenv("VAARA_PROXY_API_KEY", "from-env")
    p = _proxy(monkeypatch)
    assert p._api_key == "from-env"


def test_refuses_non_loopback_bind_without_a_key(monkeypatch):
    monkeypatch.delenv("VAARA_PROXY_API_KEY", raising=False)
    p = _proxy(monkeypatch)
    with pytest.raises(RuntimeError, match="refusing to bind"):
        p.run_http(host="0.0.0.0", port=59997)


def test_loopback_without_a_key_is_allowed(monkeypatch):
    """Local dev must not need a key. Guard the check, not uvicorn."""
    monkeypatch.delenv("VAARA_PROXY_API_KEY", raising=False)
    p = _proxy(monkeypatch)
    started = {}

    def fake_run(app, host, port, log_level):
        started["host"] = host

    monkeypatch.setattr("uvicorn.run", fake_run)
    p.run_http(host="127.0.0.1", port=59996)
    assert started["host"] == "127.0.0.1"
