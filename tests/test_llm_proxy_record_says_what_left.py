"""The prompt record describes the bytes that left, not the bytes that came in.

Sealing rewrites the request before it reaches the provider. A record built
from the pre-seal body would carry the secret the seal exists to keep out of
the trail copy, and a hash of bytes no provider ever received. The upstream
here is a stub in the process; nothing leaves the machine.
"""

from __future__ import annotations

import hashlib

import pytest

httpx = pytest.importorskip("httpx", reason="proxy deps not installed: no httpx")
pytest.importorskip("fastapi", reason="proxy deps not installed: no fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from vaara.audit.trail import EventType  # noqa: E402
from vaara.integrations._llm_proxy_app import build_app  # noqa: E402
from vaara.integrations.llm_proxy import _build_pipeline  # noqa: E402
from vaara.integrations.llm_seal import (  # noqa: E402
    SealRegistry,
    placeholder_for,
)

SECRET = "anti-note"
KEY = bytes(range(32))
TOKEN = placeholder_for(SECRET, KEY)


@pytest.fixture
def seen() -> dict:
    return {}


@pytest.fixture
def mount(monkeypatch, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        return httpx.Response(
            200, json={"content": [{"type": "text", "text": "ok"}]},
            headers={"content-type": "application/json"})

    real = httpx.AsyncClient

    def fake(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        kwargs.pop("http2", None)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake)


def _app(tmp_path, **kw):
    pipeline = _build_pipeline(tmp_path / "trail.db")
    client = TestClient(build_app(
        upstream="https://upstream.invalid", api_key="k",
        api_key_header="x-api-key", pipeline=pipeline, **kw))
    client.pipeline = pipeline
    return client


def _request_record(client):
    (rec,) = [r for r in client.pipeline.trail.get_records_by_type(
        EventType.ACTION_REQUESTED) if r.tool_name == "llm.prompt"]
    return rec.data["parameters"]


def _post(client, text):
    return client.post("/v1/messages", json={
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": text}]})


def test_full_record_holds_the_placeholder_not_the_secret(tmp_path, mount, seen):
    client = _app(tmp_path, mode="govern", audit_level="full",
                  seal_registry=SealRegistry({"concept": SECRET}, key=KEY))
    _post(client, f"explain the {SECRET} idea")
    assert SECRET not in seen["body"].decode()
    params = _request_record(client)
    flat = str(params["messages"])
    assert SECRET not in flat
    assert TOKEN in flat


def test_hash_record_is_the_hash_of_what_left(tmp_path, mount, seen):
    client = _app(tmp_path, audit_level="hash",
                  seal_registry=SealRegistry({"concept": SECRET}, key=KEY))
    _post(client, f"explain the {SECRET} idea")
    params = _request_record(client)
    assert params["prompt_hash"] == hashlib.sha256(seen["body"]).hexdigest()
    assert params["prompt_bytes"] == len(seen["body"])


def test_relay_full_record_is_the_hash_of_what_left(tmp_path, mount, seen):
    client = _app(tmp_path, mode="relay", audit_level="full",
                  seal_registry=SealRegistry({"concept": SECRET}, key=KEY))
    _post(client, f"explain the {SECRET} idea")
    params = _request_record(client)
    assert "messages" not in params
    assert params["prompt_hash"] == hashlib.sha256(seen["body"]).hexdigest()


def test_without_sealing_the_hash_is_of_the_raw_bytes(tmp_path, mount, seen):
    client = _app(tmp_path, audit_level="hash")
    _post(client, "plain request")
    params = _request_record(client)
    assert params["prompt_hash"] == hashlib.sha256(seen["body"]).hexdigest()
