"""HTTP transport must not serialise concurrent POSTs on the event loop.

`_handle_request` is a blocking sync call that waits on the upstream. Before
the to_thread offload it ran inline in the async endpoint, so two in-flight
POSTs serialised behind one another (real concurrency 1) and a slow upstream
stalled every other endpoint. These tests assert two concurrent POSTs overlap
in wall-clock, and that the per-request ContextVars (which select the upstream
and tag the tenant) survive the thread hop so each request still routes to its
own slot.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

try:
    import httpx
    from httpx import ASGITransport
except ImportError:
    pytest.skip(
        "server extra not installed (pip install 'vaara[server]')",
        allow_module_level=True,
    )

from vaara.integrations import mcp_proxy
from vaara.integrations.mcp_proxy import VaaraMCPProxy

_SLEEP = 0.30


def _build_app(proxy):
    import unittest.mock as um

    with um.patch("uvicorn.run") as run_mock:
        captured: dict = {}
        run_mock.side_effect = lambda app, **kw: captured.__setitem__("app", app)
        proxy.run_http(host="127.0.0.1", port=0)
        return captured["app"]


def _slow_upstream(reply_id):
    def _request(payload, *a, **kw):
        time.sleep(_SLEEP)
        return {"jsonrpc": "2.0", "id": payload["id"], "result": {"slot": reply_id}}

    client = MagicMock()
    client.request.side_effect = _request
    return client


async def _drive(app):
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        async def call(slot, rid):
            return await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": rid, "method": "tools/list"},
                headers={"X-Vaara-Upstream": slot},
            )

        start = time.perf_counter()
        r_alpha, r_beta = await asyncio.gather(call("alpha", 1), call("beta", 2))
        elapsed = time.perf_counter() - start
    return r_alpha, r_beta, elapsed


def test_concurrent_posts_overlap_and_keep_context(monkeypatch):
    monkeypatch.setattr(mcp_proxy, "UpstreamMCPClient", MagicMock())
    proxy = VaaraMCPProxy(
        upstreams={"alpha": ["cmd-a"], "beta": ["cmd-b"]},
        pipeline=MagicMock(),
    )
    # Each slot answers slowly and stamps its own name into the result, so an
    # overlapping pair that still routes correctly proves both concurrency and
    # that _REQUEST_UPSTREAM survived the to_thread hop.
    proxy._upstreams["alpha"] = _slow_upstream("alpha")
    proxy._upstreams["beta"] = _slow_upstream("beta")
    app = _build_app(proxy)

    # Best of three: the first drive warms the to_thread pool, and a loaded
    # CI runner can charge one attempt with worker-thread startup so the pair
    # serialises by jitter. Real overlap shows in the fastest attempt; the
    # wall-clock assertion runs against that one.
    attempts = [asyncio.run(_drive(app)) for _ in range(3)]
    r_alpha, r_beta, elapsed = min(attempts, key=lambda t: t[2])

    assert r_alpha.status_code == 200
    assert r_beta.status_code == 200
    # Context survived the hop: each request reached its own slot.
    assert r_alpha.json()["result"]["slot"] == "alpha"
    assert r_beta.json()["result"]["slot"] == "beta"
    # Overlap, not serialisation: two _SLEEP calls in well under 2*_SLEEP.
    assert elapsed < (2 * _SLEEP) - 0.05, (
        f"requests serialised: {elapsed:.3f}s for two {_SLEEP}s calls"
    )
