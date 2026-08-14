# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""The proxy answers its own liveness probe without calling the model.

Every path on the inference proxy is a catch-all that forwards upstream, so
before this endpoint existed a Kubernetes readiness probe had exactly two
options: poll a model-server path (which reports the MODEL's health, restarts
Vaara when the model is slow to load, and bills a request per probe) or use a
bare TCP check (which says a socket is open and nothing else).

``/healthz`` is answered locally. It is already in the origin guard's exempt
set, so a load balancer polling from another origin is not refused.
"""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

for _mod in ("httpx", "fastapi"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            f"inference-proxy deps not installed: no {_mod}",
            allow_module_level=True,
        )

import httpx  # noqa: E402

import vaara  # noqa: E402
from vaara.integrations._infer_proxy_app import build_app  # noqa: E402


class _CountingUpstream:
    """Mock upstream that fails the test if the probe ever reaches it."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(200, json={"upstream": "reached"})


def _get(app, path: str) -> httpx.Response:
    async def drive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://proxy",
        ) as client:
            return await client.get(path)

    return asyncio.run(drive())


def _build(enforce: bool = False, upstream_calls=None):
    counter = upstream_calls if upstream_calls is not None else _CountingUpstream()
    client = httpx.AsyncClient(transport=httpx.MockTransport(counter))
    pipeline = None
    if enforce:
        class _Enforcing:
            _enforce = True

        pipeline = _Enforcing()
    app = build_app(
        emitter=None, upstream="http://ollama", client=client, pipeline=pipeline,
    )
    return app, counter


def test_healthz_answers_without_touching_upstream():
    app, counter = _build()
    resp = _get(app, "/healthz")
    assert resp.status_code == 200
    assert counter.calls == 0, (
        "the liveness probe was forwarded to the model server"
    )


def test_healthz_reports_status_and_version():
    app, _ = _build()
    body = _get(app, "/healthz").json()
    assert body["status"] == "ok"
    assert body["version"] == vaara.__version__


def test_healthz_reports_observe_mode_by_default():
    app, _ = _build()
    assert _get(app, "/healthz").json()["mode"] == "observe"


def test_healthz_reports_enforce_mode_when_gating():
    app, _ = _build(enforce=True)
    assert _get(app, "/healthz").json()["mode"] == "enforce"


def test_healthz_does_not_disclose_the_upstream_url():
    """A probe endpoint reachable from the cluster names no internal host."""
    app, _ = _build()
    assert "ollama" not in _get(app, "/healthz").text


def test_other_paths_still_reach_the_upstream():
    """The local route must not shadow the passthrough it sits in front of."""
    app, counter = _build()
    resp = _get(app, "/api/tags")
    assert resp.status_code == 200
    assert resp.json() == {"upstream": "reached"}
    assert counter.calls == 1


def test_upstream_health_path_is_not_shadowed():
    """vLLM serves /health. Vaara claims /healthz so both stay reachable."""
    app, counter = _build()
    resp = _get(app, "/health")
    assert counter.calls == 1
    assert resp.json() == {"upstream": "reached"}


def test_healthz_ignores_a_foreign_origin_header():
    """The origin guard exempts health paths so load balancers are not refused."""
    app, _ = _build()

    async def drive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://proxy",
        ) as client:
            return await client.get(
                "/healthz", headers={"origin": "https://elsewhere.test"},
            )

    assert asyncio.run(drive()).status_code == 200
