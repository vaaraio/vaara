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
