"""The OpenAI Responses API is governed like the two chat paths.

``POST /v1/responses`` used to take the pass-through, which since 1.95.0
records a digest and seals, but applied no model or rate policy and kept no
prompt record. Its body differs from chat: the system prompt is
``instructions``, the turn is ``input`` (a string or a list of items), text
blocks are ``input_text``, and its usage reports ``input_tokens`` with the
cached share inside it. These tests drive each of those through the real app.
"""
from __future__ import annotations

import json

import pytest

httpx = pytest.importorskip("httpx", reason="proxy deps not installed: no httpx")
pytest.importorskip("fastapi", reason="proxy deps not installed: no fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from vaara.audit.trail import EventType  # noqa: E402
from vaara.integrations._llm_proxy_app import build_app  # noqa: E402
from vaara.integrations._llm_proxy_shape import (  # noqa: E402
    extract_messages,
    flatten_messages,
)
from vaara.integrations.llm_envelope import measure_envelope  # noqa: E402
from vaara.integrations.llm_proxy import _build_pipeline  # noqa: E402
from vaara.integrations.llm_seal import SealRegistry  # noqa: E402
from vaara.integrations.llm_usage import StreamUsage, extract_usage  # noqa: E402

SECRET = "northern-lights"


@pytest.fixture
def pipeline(tmp_path):
    return _build_pipeline(tmp_path / "trail.db")


@pytest.fixture
def upstream(monkeypatch):
    seen: list[httpx.Request] = []
    replies: dict[str, httpx.Response] = {}
    real = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return replies.get("next") or httpx.Response(
            200, json={"output": [], "usage": {"input_tokens": 3,
                                               "output_tokens": 1}},
            headers={"content-type": "application/json"})

    def fake(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        kwargs.pop("http2", None)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake)
    return seen, replies


def _app(pipeline, **kw):
    return TestClient(build_app(
        upstream="https://upstream.invalid", api_key="k",
        api_key_header="authorization", pipeline=pipeline, **kw))


def _prompts(pipeline, event=EventType.ACTION_REQUESTED):
    return [r for r in pipeline.trail.get_records_by_type(event)
            if r.tool_name == "llm.prompt"]


BODY = {
    "model": "gpt-5",
    "instructions": "be brief",
    "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]},
        {"type": "function_call", "call_id": "c1", "name": "f",
         "arguments": "{\"x\": 1}"},
        {"type": "function_call_output", "call_id": "c1", "output": "42"},
    ],
}


class TestShape:

    def test_items_become_messages(self):
        msgs = extract_messages(BODY)
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
        assert "hello" in flatten_messages(msgs)
        assert "42" in flatten_messages(msgs)

    def test_string_input_is_the_user_turn(self):
        msgs = extract_messages({"model": "m", "input": "hi there"})
        assert msgs == [{"role": "user", "content": "hi there"}]

    def test_envelope_counts_instructions_as_system(self):
        env = measure_envelope({"model": "m", "instructions": "abc",
                                "input": "hello"}, b"x" * 10)
        assert env["bytes_system"] == 3
        assert env["bytes_last_user"] == 5
        assert env["message_count"] == 1


class TestUsage:

    def test_cached_share_is_taken_out_of_input_tokens(self):
        raw = json.dumps({"usage": {
            "input_tokens": 1000, "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 900}}}).encode()
        usage = extract_usage(raw)
        assert usage["input_tokens"] == 100
        assert usage["cache_read_input_tokens"] == 900
        assert usage["cache_hit_pct"] == 90.0

    def test_stream_completed_event_carries_usage(self):
        su = StreamUsage()
        su.feed(b'event: response.completed\n'
                b'data: {"type":"response.completed","response":{"usage":'
                b'{"input_tokens":10,"output_tokens":4}}}\n\n')
        assert su.usage["output_tokens"] == 4


class TestGoverned:

    def test_model_deny_applies(self, pipeline, upstream):
        seen, _ = upstream
        resp = _app(pipeline, model_deny=["gpt-*"]).post("/v1/responses", json=BODY)
        assert resp.status_code == 403
        assert seen == []

    def test_prompt_is_recorded_with_its_turns(self, pipeline, upstream):
        _app(pipeline, mode="govern", audit_level="full").post(
            "/v1/responses", json=BODY)
        (rec,) = _prompts(pipeline)
        params = rec.data["parameters"]
        assert params["model"] == "gpt-5"
        assert [m["role"] for m in params["messages"]] == [
            "system", "user", "assistant", "tool"]
        assert params["envelope"]["bytes_system"] == len("be brief")

    def test_sealed_on_the_way_out(self, pipeline, upstream):
        seen, _ = upstream
        _app(pipeline, seal_registry=SealRegistry({"c": SECRET})).post(
            "/v1/responses", json={"model": "m", "input": f"say {SECRET}"})
        assert SECRET.encode() not in seen[0].content
        (rec,) = _prompts(pipeline)
        assert rec.data["parameters"]["seal_count"] == 1

    def test_outcome_keeps_usage(self, pipeline, upstream):
        _, replies = upstream
        replies["next"] = httpx.Response(200, json={"output": [], "usage": {
            "input_tokens": 50, "output_tokens": 5,
            "input_tokens_details": {"cached_tokens": 40}}},
            headers={"content-type": "application/json"})
        _app(pipeline).post("/v1/responses", json=BODY)
        (out,) = _prompts(pipeline, EventType.OUTCOME_RECORDED)
        desc = json.loads(out.data["description"])
        assert desc["usage"]["input_tokens"] == 10
        assert desc["usage"]["cache_read_input_tokens"] == 40

    def test_retrieving_a_response_is_still_a_pass_through(self, pipeline, upstream):
        _app(pipeline).get("/v1/responses/resp_123")
        assert _prompts(pipeline) == []
        assert [r for r in pipeline.trail.get_records_by_type(
            EventType.ACTION_REQUESTED) if r.tool_name == "llm.passthrough"]
