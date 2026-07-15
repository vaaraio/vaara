# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Offline tests for the inference proxy: emitter, shaping, and ASGI e2e.

No live ollama: the upstream is an ``httpx.MockTransport`` and
the proxy app is driven through ``httpx.ASGITransport`` with ``asyncio.run``
(no pytest-asyncio dependency).
"""

from __future__ import annotations

import asyncio
import importlib.util

import pytest

for _mod in ("rfc8785", "cryptography", "httpx", "fastapi"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "inference-proxy deps not installed (attestation extra + httpx/fastapi)",
            allow_module_level=True,
        )

import httpx  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation._inference_types import ModelDerived  # noqa: E402
from vaara.attestation._inference_verify import _verify_one  # noqa: E402
from vaara.integrations._infer_proxy_app import build_app  # noqa: E402
from vaara.integrations._infer_proxy_emit import InferenceAttestEmitter  # noqa: E402
from vaara.integrations._infer_proxy_shape import (  # noqa: E402
    StreamAccumulator,
    _coerce_eval_stats,
    extract_sampling,
)

MESSAGES = [{"role": "user", "content": "Summarize Q3."}]
MODEL = ModelDerived(
    model_ref="qwen3:30b-a3b",
    manifest_digest="sha256:" + "a" * 64,
    gguf_metadata_hash="sha256:" + "b" * 64,
    quantization="Q4_K_M",
    param_count="30B",
)


def _es256_emitter(tmp_path):
    key = ec.generate_private_key(ec.SECP256R1())
    return InferenceAttestEmitter(
        signing_key=key, alg="ES256", receipts_dir=tmp_path,
        secret_version="testv1",
    ), key.public_key()


# --- shaping ---------------------------------------------------------------


def test_coerce_eval_stats_drops_floats_and_bools():
    out = _coerce_eval_stats({"a": 5, "b": 1.5, "c": True, "d": None, "e": 7})
    assert out == {"a": 5, "e": 7}


def test_extract_sampling_openai_and_ollama():
    openai = extract_sampling({"temperature": 0.7, "model": "x", "top_p": 0.9}, False)
    assert openai == {"temperature": 0.7, "top_p": 0.9}
    ollama = extract_sampling({"options": {"seed": 42, "nope": 1}}, True)
    assert ollama == {"seed": 42}


def test_stream_accumulator_openai_sse():
    acc = StreamAccumulator(is_ollama=False)
    acc.feed(b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n')
    acc.feed(b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n')
    acc.feed(b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n')
    acc.feed(b"data: [DONE]\n\n")
    output, stats = acc.finalize()
    assert output["content"] == "Hello"
    assert stats == {"promptTokens": 3, "completionTokens": 2}


def test_stream_accumulator_ollama_ndjson():
    acc = StreamAccumulator(is_ollama=True)
    acc.feed(b'{"message":{"content":"Hi"},"done":false}\n')
    acc.feed(b'{"message":{"content":" there"},"done":true,"eval_count":4}\n')
    output, stats = acc.finalize()
    assert output["content"] == "Hi there"
    assert stats == {"evalCount": 4}


# --- emitter round-trip ----------------------------------------------------


def test_emitter_roundtrip_verifies(tmp_path):
    emitter, pub = _es256_emitter(tmp_path)
    att, counter = emitter.emit_attestation(
        model_ref="qwen3:30b-a3b", model_derived=MODEL,
        messages=MESSAGES, sampling={"temperature": 0.7},
    )
    out = {"content": "The Q3 report covers three regions."}
    emitter.emit_receipt(
        attestation=att, counter=counter, status="completed",
        output=out, eval_stats={"evalCount": 9},
    )
    att_doc = _read_json(tmp_path, "-infer-attest.json")
    receipt_doc = _read_json(tmp_path, "-infer-receipt.json")
    checks = _verify_one(receipt_doc, att_doc, pub)
    assert checks["ok"] is True
    assert checks["backLink"] is True
    assert checks["receiptSignature"] is True
    assert checks["attestationSignature"] is True
    assert checks["tier"] == "integrity"


def test_tampered_receipt_fails_signature(tmp_path):
    emitter, pub = _es256_emitter(tmp_path)
    att, counter = emitter.emit_attestation(
        model_ref="m", model_derived=MODEL, messages=MESSAGES, sampling={},
    )
    emitter.emit_receipt(
        attestation=att, counter=counter, status="completed",
        output={"content": "x"}, eval_stats=None,
    )
    receipt_doc = _read_json(tmp_path, "-infer-receipt.json")
    receipt_doc["outcomeDerived"]["status"] = "refused"  # flip after signing
    att_doc = _read_json(tmp_path, "-infer-attest.json")
    checks = _verify_one(receipt_doc, att_doc, pub)
    assert checks["receiptSignature"] is False
    assert checks["ok"] is False


# --- ASGI end-to-end against a mock ollama ---------------------------------


def _mock_ollama(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/show":
        return httpx.Response(200, json={
            "model_info": {"general.architecture": "qwen3"},
            "details": {"quantization_level": "Q4_K_M", "parameter_size": "30.5B"},
        })
    if path == "/api/tags":
        return httpx.Response(200, json={"models": [
            {"name": "qwen3:30b-a3b", "digest": "deadbeef"},
        ]})
    if path == "/v1/chat/completions":
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "hi there"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        })
    return httpx.Response(404, json={"error": "unmocked"})


def test_e2e_buffered_emits_valid_chain(tmp_path):
    emitter, pub = _es256_emitter(tmp_path)
    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_ollama))
    app = build_app(emitter=emitter, upstream="http://ollama", client=upstream_client)

    async def drive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
            return await c.post("/v1/chat/completions", json={
                "model": "qwen3:30b-a3b", "messages": MESSAGES,
                "temperature": 0.7, "stream": False,
            })

    resp = asyncio.run(drive())
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "hi there"

    att_doc = _read_json(tmp_path, "-infer-attest.json")
    receipt_doc = _read_json(tmp_path, "-infer-receipt.json")
    checks = _verify_one(receipt_doc, att_doc, pub)
    assert checks["ok"] is True
    # the model was resolved from the mock upstream, not the name-only fallback
    assert att_doc["modelDerived"]["manifestDigest"] == "sha256:deadbeef"
    assert att_doc["modelDerived"]["quantization"] == "Q4_K_M"
    assert receipt_doc["outcomeDerived"]["evalStats"]["promptTokens"] == 10


def _read_json(directory, suffix):
    import json
    matches = sorted(p for p in directory.iterdir() if p.name.endswith(suffix))
    assert matches, f"no file ending {suffix} in {directory}"
    return json.loads(matches[-1].read_text(encoding="utf-8"))


# --- Phase 1 observe: tool calls the model requests land in the trail ------


def _mock_ollama_with_tool_calls(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/show":
        return httpx.Response(200, json={
            "model_info": {"general.architecture": "qwen3"},
            "details": {"quantization_level": "Q4_K_M", "parameter_size": "30.5B"},
        })
    if path == "/api/tags":
        return httpx.Response(200, json={"models": [
            {"name": "qwen3:30b-a3b", "digest": "deadbeef"},
        ]})
    if path == "/v1/chat/completions":
        return httpx.Response(200, json={
            "choices": [{"message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "send_payment",
                                 "arguments": '{"to": "acme", "amount": 900}'},
                }],
            }}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        })
    return httpx.Response(404, json={"error": "unmocked"})


def _observing_pipeline():
    from vaara.pipeline import InterceptionPipeline

    pipeline = InterceptionPipeline(enforce=False)  # observe-only, never block
    return pipeline


def test_e2e_buffered_records_requested_tool_calls(tmp_path):
    emitter, _ = _es256_emitter(tmp_path)
    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_mock_ollama_with_tool_calls))
    pipeline = _observing_pipeline()
    app = build_app(emitter=emitter, upstream="http://ollama",
                    client=upstream_client, pipeline=pipeline)

    async def drive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
            return await c.post("/v1/chat/completions", json={
                "model": "qwen3:30b-a3b", "messages": MESSAGES, "stream": False,
            })

    resp = asyncio.run(drive())
    # Phase 1 is passthrough: the response reaches the caller unchanged.
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"
    # ...and the requested tool call is in the audit trail with its params.
    recorded = [
        r for r in pipeline.trail._records if r.tool_name == "send_payment"
    ]
    assert recorded, "tool call the model requested was never recorded"
    assert any(r.data.get("parameters", {}).get("amount") == 900
               for r in recorded if isinstance(r.data, dict))


def _mock_ollama_streaming_tool_calls(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/show":
        return httpx.Response(200, json={"model_info": {}, "details": {}})
    if path == "/api/tags":
        return httpx.Response(200, json={"models": []})
    if path == "/v1/chat/completions":
        body = (
            b'data: {"choices":[{"delta":{"tool_calls":[{"id":"call_9",'
            b'"type":"function","function":{"name":"delete_file",'
            b'"arguments":"{\\"path\\": \\"/etc\\"}"}}]}}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(200, content=body,
                              headers={"content-type": "text/event-stream"})
    return httpx.Response(404, json={"error": "unmocked"})


def test_e2e_streamed_records_requested_tool_calls(tmp_path):
    emitter, _ = _es256_emitter(tmp_path)
    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_mock_ollama_streaming_tool_calls))
    pipeline = _observing_pipeline()
    app = build_app(emitter=emitter, upstream="http://ollama",
                    client=upstream_client, pipeline=pipeline)

    async def drive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
            resp = await c.post("/v1/chat/completions", json={
                "model": "qwen3:30b-a3b", "messages": MESSAGES, "stream": True,
            })
            return resp.status_code, await resp.aread()

    status, raw = asyncio.run(drive())
    assert status == 200
    assert b"call_9" in raw  # stream reached the caller intact
    recorded = [r for r in pipeline.trail._records if r.tool_name == "delete_file"]
    assert recorded, "streamed tool call was never recorded"


def test_pipeline_recording_failure_never_breaks_passthrough(tmp_path):
    emitter, _ = _es256_emitter(tmp_path)
    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_mock_ollama_with_tool_calls))

    class ExplodingPipeline:
        def intercept(self, **kwargs):
            raise RuntimeError("trail on fire")

    app = build_app(emitter=emitter, upstream="http://ollama",
                    client=upstream_client, pipeline=ExplodingPipeline())

    async def drive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
            return await c.post("/v1/chat/completions", json={
                "model": "qwen3:30b-a3b", "messages": MESSAGES, "stream": False,
            })

    resp = asyncio.run(drive())
    assert resp.status_code == 200


def test_e2e_no_emitter_still_proxies_and_records(tmp_path):
    # `vaara proxy` observe mode: no signing key, no receipts — passthrough
    # plus trail recording must work on their own.
    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_mock_ollama_with_tool_calls))
    pipeline = _observing_pipeline()
    app = build_app(emitter=None, upstream="http://ollama",
                    client=upstream_client, pipeline=pipeline)

    async def drive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
            return await c.post("/v1/chat/completions", json={
                "model": "qwen3:30b-a3b", "messages": MESSAGES, "stream": False,
            })

    resp = asyncio.run(drive())
    assert resp.status_code == 200
    assert any(r.tool_name == "send_payment" for r in pipeline.trail._records)


# --- A2: Anthropic /v1/messages shape ---------------------------------------


def _mock_anthropic(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/show":
        return httpx.Response(404, json={})
    if path == "/api/tags":
        return httpx.Response(404, json={})
    if path == "/v1/messages":
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [
                {"type": "text", "text": "I'll check the weather."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "Helsinki"}},
            ],
            "usage": {"input_tokens": 12, "output_tokens": 30},
        })
    return httpx.Response(404, json={"error": "unmocked"})


def test_e2e_anthropic_buffered_records_tool_use(tmp_path):
    upstream_client = httpx.AsyncClient(transport=httpx.MockTransport(_mock_anthropic))
    pipeline = _observing_pipeline()
    app = build_app(emitter=None, upstream="http://anthropic",
                    client=upstream_client, pipeline=pipeline)

    async def drive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
            return await c.post("/v1/messages", json={
                "model": "claude-sonnet-5", "max_tokens": 100,
                "messages": MESSAGES, "stream": False,
            })

    resp = asyncio.run(drive())
    assert resp.status_code == 200
    assert resp.json()["content"][1]["id"] == "toolu_1"  # passthrough intact
    recorded = [r for r in pipeline.trail._records if r.tool_name == "get_weather"]
    assert recorded, "anthropic tool_use was never recorded"
    assert any(r.data.get("parameters", {}).get("city") == "Helsinki"
               for r in recorded if isinstance(r.data, dict))


def _mock_anthropic_stream(request: httpx.Request) -> httpx.Response:
    if request.url.path != "/v1/messages":
        return httpx.Response(404, json={})
    body = (
        b'event: message_start\n'
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}\n\n'
        b'event: content_block_start\n'
        b'data: {"type":"content_block_start","index":0,"content_block":'
        b'{"type":"tool_use","id":"toolu_9","name":"run_command","input":{}}}\n\n'
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","index":0,"delta":'
        b'{"type":"input_json_delta","partial_json":"{\\"cmd\\": \\"rm"}}\n\n'
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","index":0,"delta":'
        b'{"type":"input_json_delta","partial_json":" -rf /\\"}"}}\n\n'
        b'event: content_block_stop\n'
        b'data: {"type":"content_block_stop","index":0}\n\n'
        b'event: message_delta\n'
        b'data: {"type":"message_delta","delta":{},"usage":{"output_tokens":9}}\n\n'
        b'event: message_stop\n'
        b'data: {"type":"message_stop"}\n\n'
    )
    return httpx.Response(200, content=body,
                          headers={"content-type": "text/event-stream"})


def test_e2e_anthropic_streamed_records_tool_use(tmp_path):
    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_mock_anthropic_stream))
    pipeline = _observing_pipeline()
    app = build_app(emitter=None, upstream="http://anthropic",
                    client=upstream_client, pipeline=pipeline)

    async def drive():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as c:
            resp = await c.post("/v1/messages", json={
                "model": "claude-sonnet-5", "max_tokens": 100,
                "messages": MESSAGES, "stream": True,
            })
            return resp.status_code, await resp.aread()

    status, raw = asyncio.run(drive())
    assert status == 200
    assert b"toolu_9" in raw  # stream intact
    recorded = [r for r in pipeline.trail._records if r.tool_name == "run_command"]
    assert recorded, "streamed anthropic tool_use was never recorded"
    assert any(r.data.get("parameters", {}).get("cmd") == "rm -rf /"
               for r in recorded if isinstance(r.data, dict))
