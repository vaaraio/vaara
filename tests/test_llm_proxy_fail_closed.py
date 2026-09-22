"""The llm-proxy does not forward a request it could not record.

On 2026-09-22 the proxy forwarded 169 prompts across 68 minutes while every
one of them failed to reach the trail. The failure was counted and logged,
and the request went out regardless. These tests pin the replacement: the
proxy repairs the store and retries once, and when the record still cannot
be written it answers 503 and the upstream never sees the request.
``--fail-open`` is the stated way to forward anyway.

The upstream is a stub inside the process, and the store is made to fail by
replacing the backend's append. The on-disk repair itself is exercised
against real page damage in ``test_trail_repair.py``.
"""
from __future__ import annotations

import pytest

httpx = pytest.importorskip("httpx", reason="proxy deps not installed: no httpx")
pytest.importorskip("fastapi", reason="proxy deps not installed: no fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from vaara.audit.sqlite_backend import TrailRepair  # noqa: E402
from vaara.integrations._llm_proxy_app import build_app  # noqa: E402
from vaara.integrations.llm_proxy import _build_pipeline  # noqa: E402


@pytest.fixture
def pipeline(tmp_path):
    return _build_pipeline(tmp_path / "trail.db")


@pytest.fixture
def upstream(monkeypatch):
    calls: list[httpx.Request] = []
    real = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "hi"}]},
                              headers={"content-type": "application/json"})

    def fake(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        kwargs.pop("http2", None)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake)
    return calls


def _client(pipeline, **kw) -> TestClient:
    return TestClient(build_app(
        upstream="https://upstream.invalid", api_key="k", api_key_header="x-api-key",
        pipeline=pipeline, **kw,
    ))


def _post(client):
    return client.post("/v1/messages", json={
        "model": "claude-opus-5", "messages": [{"role": "user", "content": "hello"}],
    })


def _break_store(monkeypatch, pipeline, *, repair_result=None, heals=False):
    """Make every append fail; ``repair`` returns ``repair_result``.

    With ``heals`` the store starts working again once repair has run, the
    way a REINDEX clears index damage underneath a live connection.
    """
    backend = pipeline.trail._chain_backend()
    real_append = backend.append_record
    state = {"broken": True, "repairs": 0}

    def append(record, stamp):
        if state["broken"]:
            raise OSError("disk I/O error")
        return real_append(record, stamp)

    def repair():
        state["repairs"] += 1
        if heals:
            state["broken"] = False
        return repair_result

    monkeypatch.setattr(backend, "append_record", append)
    monkeypatch.setattr(backend, "repair", repair)
    return state


def test_a_request_that_cannot_be_recorded_is_not_forwarded(monkeypatch, pipeline, upstream):
    state = _break_store(monkeypatch, pipeline, repair_result=TrailRepair(
        db="x", method="failed", error="database disk image is malformed"))
    resp = _post(_client(pipeline))
    assert resp.status_code == 503
    body = resp.json()
    assert body["type"] == "vaara_trail_not_recording"
    assert "not forwarded" in body["error"]
    assert "vaara trail repair" in body["error"]
    assert upstream == []
    assert state["repairs"] == 1


def test_a_repair_that_works_lets_the_request_through_recorded(monkeypatch, pipeline, upstream):
    state = _break_store(monkeypatch, pipeline, heals=True,
                         repair_result=TrailRepair(db="x", method="reindex"))
    failures_before = pipeline.trail.persistence_failures
    resp = _post(_client(pipeline))
    assert resp.status_code == 200
    assert len(upstream) == 1
    assert state["repairs"] == 1
    # The retry landed: no failure beyond the ones that triggered the repair.
    after = pipeline.trail.persistence_failures
    resp2 = _post(_client(pipeline))
    assert resp2.status_code == 200
    assert pipeline.trail.persistence_failures == after
    assert after > failures_before


def test_repair_is_not_rerun_on_every_request(monkeypatch, pipeline, upstream):
    state = _break_store(monkeypatch, pipeline, repair_result=TrailRepair(
        db="x", method="failed", error="still broken"))
    client = _client(pipeline)
    assert _post(client).status_code == 503
    assert _post(client).status_code == 503
    assert _post(client).status_code == 503
    assert state["repairs"] == 1
    assert upstream == []


def test_fail_open_forwards_and_says_so(monkeypatch, pipeline, upstream, caplog):
    _break_store(monkeypatch, pipeline, repair_result=TrailRepair(
        db="x", method="failed", error="still broken"))
    with caplog.at_level("ERROR"):
        resp = _post(_client(pipeline, fail_open=True))
    assert resp.status_code == 200
    assert len(upstream) == 1
    assert "UNRECORDED" in caplog.text


def test_a_healthy_trail_never_repairs(monkeypatch, pipeline, upstream):
    backend = pipeline.trail._chain_backend()
    calls = []
    monkeypatch.setattr(backend, "repair", lambda: calls.append(1))
    client = _client(pipeline)
    for _ in range(3):
        assert _post(client).status_code == 200
    assert calls == []
    assert len(upstream) == 3
