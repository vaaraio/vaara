"""The provider's own token counts on the llm-proxy outcome record.

The envelope says how many bytes left. It cannot say what they cost: a
request whose prefix is unchanged is read from the provider's cache at a
fraction of the price of one that rewrote an early byte. The reply carries
that split, and the record now keeps it, so a prefix that stopped being
stable is visible on the first call instead of inferred later.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from vaara.audit.trail import EventType  # noqa: E402
from vaara.integrations._llm_proxy_app import build_app  # noqa: E402
from vaara.integrations.llm_proxy import _build_pipeline  # noqa: E402
from vaara.integrations.llm_usage import (  # noqa: E402
    StreamUsage,
    extract_usage,
)

import httpx  # noqa: E402


@pytest.fixture
def pipeline(tmp_path):
    return _build_pipeline(tmp_path / "audit.db")


def _records(pipeline, event: EventType) -> list:
    return [rec for rec in pipeline.trail.get_records_by_type(event)
            if rec.tool_name == "llm.prompt"]


def _mount(monkeypatch, handler):
    real = httpx.AsyncClient

    def fake(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        kwargs.pop("http2", None)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake)


def _app(pipeline):
    return build_app(
        upstream="https://upstream.invalid", api_key="k",
        api_key_header="x-api-key", pipeline=pipeline,
    )


def _post(client, **extra):
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    return client.post("/v1/messages", json=body)


class TestExtractUsage:

    def test_anthropic_shape(self):
        raw = json.dumps({"usage": {
            "input_tokens": 100, "output_tokens": 20,
            "cache_read_input_tokens": 900,
            "cache_creation_input_tokens": 0,
        }}).encode()
        usage = extract_usage(raw)
        assert usage["input_tokens"] == 100
        assert usage["cache_read_input_tokens"] == 900
        assert usage["cache_hit_pct"] == 90.0

    def test_openai_shape_subtracts_cached_from_prompt_tokens(self):
        raw = json.dumps({"usage": {
            "prompt_tokens": 1000, "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 800},
        }}).encode()
        usage = extract_usage(raw)
        assert usage["input_tokens"] == 200
        assert usage["cache_read_input_tokens"] == 800
        assert usage["output_tokens"] == 20
        assert usage["cache_hit_pct"] == 80.0

    def test_no_cache_fields_means_no_percentage(self):
        raw = json.dumps({"usage": {
            "input_tokens": 10, "output_tokens": 2}}).encode()
        usage = extract_usage(raw)
        assert usage["input_tokens"] == 10
        assert "cache_hit_pct" not in usage

    @pytest.mark.parametrize("raw", [
        b"", b"not json", b"[]", json.dumps({"no": "usage"}).encode(),
        json.dumps({"usage": "nonsense"}).encode(),
    ])
    def test_unreadable_reply_reports_nothing(self, raw):
        assert extract_usage(raw) == {}


class TestStreamUsage:

    def test_anthropic_start_and_delta_frames(self):
        acc = StreamUsage()
        acc.feed(b'data: {"type":"message_start","message":{"usage":'
                 b'{"input_tokens":5,"cache_read_input_tokens":95,'
                 b'"output_tokens":1}}}\n\n')
        acc.feed(b'data: {"type":"message_delta","usage":'
                 b'{"output_tokens":42}}\n\n')
        acc.feed(b"data: [DONE]\n\n")
        assert acc.usage["input_tokens"] == 5
        assert acc.usage["cache_read_input_tokens"] == 95
        assert acc.usage["output_tokens"] == 42
        assert acc.usage["cache_hit_pct"] == 95.0

    def test_frame_split_across_chunks(self):
        acc = StreamUsage()
        acc.feed(b'data: {"usage":{"input_tokens":7,')
        acc.feed(b'"output_tokens":3}}\n\n')
        assert acc.usage["input_tokens"] == 7
        assert acc.usage["output_tokens"] == 3

    def test_garbage_frames_are_skipped(self):
        acc = StreamUsage()
        acc.feed(b"event: ping\ndata: not json\n\ndata: []\n\n")
        assert acc.usage == {}


class TestOutcomeCarriesUsage:

    def test_buffered_reply(self, monkeypatch, pipeline):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "content": [],
                "usage": {"input_tokens": 50, "output_tokens": 5,
                          "cache_read_input_tokens": 450},
            }, headers={"content-type": "application/json"})

        _mount(monkeypatch, handler)
        _post(TestClient(_app(pipeline)))
        (out,) = _records(pipeline, EventType.OUTCOME_RECORDED)
        desc = json.loads(out.data["description"])
        assert desc["usage"]["input_tokens"] == 50
        assert desc["usage"]["cache_hit_pct"] == 90.0

    def test_streamed_reply(self, monkeypatch, pipeline):
        def handler(request: httpx.Request) -> httpx.Response:
            body = (b'data: {"type":"message_start","message":{"usage":'
                    b'{"input_tokens":10,"cache_read_input_tokens":90}}}\n\n'
                    b'data: {"type":"message_delta","usage":'
                    b'{"output_tokens":4}}\n\n')
            return httpx.Response(
                200, content=body,
                headers={"content-type": "text/event-stream"})

        _mount(monkeypatch, handler)
        _post(TestClient(_app(pipeline)), stream=True)
        (out,) = _records(pipeline, EventType.OUTCOME_RECORDED)
        desc = json.loads(out.data["description"])
        assert desc["status"] == "ok"
        assert desc["usage"]["output_tokens"] == 4
        assert desc["usage"]["cache_hit_pct"] == 90.0

    def test_reply_without_usage_adds_no_key(self, monkeypatch, pipeline):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"content": []},
                                  headers={"content-type": "application/json"})

        _mount(monkeypatch, handler)
        _post(TestClient(_app(pipeline)))
        (out,) = _records(pipeline, EventType.OUTCOME_RECORDED)
        desc = json.loads(out.data["description"])
        assert "usage" not in desc
