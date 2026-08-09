# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The model-layer gate must fail closed when it cannot read the response.

``_infer_proxy_gate`` promises that "every tool call the model requests is
decided before the agent ever sees it". Three paths broke that promise by
staying silent instead of refusing:

* a buffered response that failed to parse left ``output`` at None, the gate
  saw nothing to decide, and the raw upstream body — tool calls included —
  was forwarded;
* the same for a streamed response the accumulator could not reconstruct;
* a tool call whose shape ``_parse_one`` did not recognise was skipped by the
  gate loop, so it shipped undecided.

All three are the same mistake: "I could not read it" was treated as "there
was nothing there". Only enforcing mode changes; observe mode must stay
byte-for-byte passthrough, which is its whole contract.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json

import pytest

for _mod in ("rfc8785", "cryptography", "httpx", "fastapi"):
    if importlib.util.find_spec(_mod) is None:  # pragma: no cover
        pytest.skip("proxy deps not installed", allow_module_level=True)

import httpx  # noqa: E402

from vaara.integrations._infer_proxy_app import build_app  # noqa: E402
from vaara.integrations._infer_proxy_gate import gate_tool_calls  # noqa: E402
from vaara.pipeline import InterceptionPipeline  # noqa: E402

MESSAGES = [{"role": "user", "content": "Do the thing."}]
GARBAGE = b"<!DOCTYPE html><html>upstream returned a login page</html>"


def _enforcing() -> InterceptionPipeline:
    from vaara.audit.trail import AuditTrail

    return InterceptionPipeline(enforce=True, trail=AuditTrail())


def _observing() -> InterceptionPipeline:
    from vaara.audit.trail import AuditTrail

    return InterceptionPipeline(enforce=False, trail=AuditTrail())


def _unparseable_upstream(request: httpx.Request) -> httpx.Response:
    """200 OK with a body no shape parser can read."""
    return httpx.Response(200, content=GARBAGE,
                          headers={"content-type": "text/html"})


def _unparseable_stream_upstream(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=GARBAGE,
                          headers={"content-type": "text/event-stream"})


def _drive(app, path: str, body: dict):
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://p") as c:
            resp = await c.post(path, json=body)
            return resp.status_code, await resp.aread()

    return asyncio.run(go())


# ── buffered ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,shape_model", [
    ("/v1/messages", "claude-sonnet-5"),
    ("/v1/chat/completions", "gpt-x"),
    ("/api/chat", "llama3"),
])
def test_unparseable_buffered_response_is_refused_when_enforcing(path, shape_model):
    app = build_app(
        emitter=None, upstream="http://up", pipeline=_enforcing(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(_unparseable_upstream)),
    )
    status, raw = _drive(app, path, {
        "model": shape_model, "messages": MESSAGES, "stream": False,
    })
    assert status == 200
    assert GARBAGE not in raw, "unreadable upstream body reached the agent"
    assert b"blocked by Vaara policy" in raw


def test_refusal_keeps_the_response_shape_the_client_expects():
    """An agent parsing an Anthropic reply must still get an Anthropic reply."""
    app = build_app(
        emitter=None, upstream="http://up", pipeline=_enforcing(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(_unparseable_upstream)),
    )
    _, raw = _drive(app, "/v1/messages", {
        "model": "claude-sonnet-5", "messages": MESSAGES, "stream": False,
    })
    doc = json.loads(raw)
    assert doc["type"] == "message"
    assert doc["role"] == "assistant"
    assert doc["stop_reason"] == "end_turn"
    assert any(b.get("type") == "text" for b in doc["content"])


def test_openai_refusal_shape():
    app = build_app(
        emitter=None, upstream="http://up", pipeline=_enforcing(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(_unparseable_upstream)),
    )
    _, raw = _drive(app, "/v1/chat/completions", {
        "model": "gpt-x", "messages": MESSAGES, "stream": False,
    })
    doc = json.loads(raw)
    assert doc["choices"][0]["finish_reason"] == "stop"
    assert "blocked by Vaara policy" in doc["choices"][0]["message"]["content"]


def test_ollama_refusal_shape():
    app = build_app(
        emitter=None, upstream="http://up", pipeline=_enforcing(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(_unparseable_upstream)),
    )
    _, raw = _drive(app, "/api/chat", {
        "model": "llama3", "messages": MESSAGES, "stream": False,
    })
    doc = json.loads(raw)
    assert doc["done"] is True
    assert "blocked by Vaara policy" in doc["message"]["content"]


def test_observe_mode_still_passes_an_unparseable_body_through():
    """Recording must never break passthrough — that contract is unchanged."""
    app = build_app(
        emitter=None, upstream="http://up", pipeline=_observing(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(_unparseable_upstream)),
    )
    status, raw = _drive(app, "/v1/messages", {
        "model": "claude-sonnet-5", "messages": MESSAGES, "stream": False,
    })
    assert status == 200
    assert raw == GARBAGE


def test_upstream_error_status_is_not_turned_into_a_refusal():
    """A 500 from the provider is the provider's answer, not a policy block."""
    def _err(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"upstream exploded")

    app = build_app(
        emitter=None, upstream="http://up", pipeline=_enforcing(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(_err)),
    )
    status, raw = _drive(app, "/v1/messages", {
        "model": "claude-sonnet-5", "messages": MESSAGES, "stream": False,
    })
    assert status == 500
    assert raw == b"upstream exploded"


# ── streamed ──────────────────────────────────────────────────────────────

def test_unparseable_stream_is_refused_when_enforcing():
    app = build_app(
        emitter=None, upstream="http://up", pipeline=_enforcing(),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(_unparseable_stream_upstream)),
    )
    status, raw = _drive(app, "/v1/messages", {
        "model": "claude-sonnet-5", "messages": MESSAGES, "stream": True,
    })
    assert status == 200
    assert GARBAGE not in raw
    assert b"blocked by Vaara policy" in raw


def test_observe_mode_streams_an_unparseable_body_through():
    app = build_app(
        emitter=None, upstream="http://up", pipeline=_observing(),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(_unparseable_stream_upstream)),
    )
    status, raw = _drive(app, "/v1/messages", {
        "model": "claude-sonnet-5", "messages": MESSAGES, "stream": True,
    })
    assert status == 200
    assert raw == GARBAGE


# ── the gate loop itself ──────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("bad", [
    {"id": "1", "type": "function"},                       # no function block
    {"id": "1", "type": "function", "function": {}},       # no name
    {"id": "1", "type": "function", "function": {"name": ""}},
    {"id": "1", "type": "function", "function": {"name": 42}},
    "not even an object",
    None,
])
def test_unrecognised_tool_call_shape_is_denied_not_skipped(bad):
    denials = _run(gate_tool_calls(
        _enforcing(), [bad], model_name="m",
    ))
    assert denials, "an undecidable tool call was allowed to ship"
    assert "could not be read" in denials[0]


def test_a_readable_call_alongside_an_unreadable_one_still_reports_both():
    good = {"id": "1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"p": "x"}'}}
    denials = _run(gate_tool_calls(
        _enforcing(), [good, {"junk": True}], model_name="m",
    ))
    assert any("could not be read" in d for d in denials)


def test_no_tool_calls_is_still_an_empty_denial_list():
    assert _run(gate_tool_calls(_enforcing(), [], model_name="m")) == []
    assert _run(gate_tool_calls(_enforcing(), None, model_name="m")) == []
