# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Offline tests for the sovereign-governed Vaara console.

No live ollama or proxy: the console's upstream proxy is an
``httpx.MockTransport`` that emits a real signed pair as a side effect (the way
the real proxy would), and the console app is driven through
``httpx.ASGITransport`` with ``asyncio.run`` (no pytest-asyncio dependency).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json

import pytest

for _mod in ("rfc8785", "cryptography", "httpx", "fastapi"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "console deps not installed (attestation extra + httpx/fastapi)",
            allow_module_level=True,
        )

import httpx  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation._inference_types import ModelDerived  # noqa: E402
from vaara.integrations._infer_console_app import build_app  # noqa: E402
from vaara.integrations._infer_console_recall import MemoryRecall  # noqa: E402
from vaara.integrations._infer_proxy_emit import InferenceAttestEmitter  # noqa: E402

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
        signing_key=key, alg="ES256", receipts_dir=tmp_path, secret_version="testv1",
    ), key.public_key()


def _proxy_handler(emitter, content="hi there", stream=False):
    """Stand in for the signing proxy: emit a real pair, return ollama JSON."""

    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content or b"{}")
        att, counter = emitter.emit_attestation(
            model_ref=data.get("model", "m"), model_derived=MODEL,
            messages=data.get("messages"), sampling={},
        )
        emitter.emit_receipt(
            attestation=att, counter=counter, status="completed",
            output={"content": content}, eval_stats=None,
        )
        if stream:
            body = (
                json.dumps({"message": {"content": content}, "done": False}) + "\n"
                + json.dumps({"message": {"content": ""}, "done": True}) + "\n"
            ).encode()
            return httpx.Response(200, content=body)
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": content}, "done": True}
        )
    return handler


def _console(tmp_path, pub, *, stream=False, emitter=None, **kw):
    emitter = emitter or _es256_emitter(tmp_path)[0]
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_proxy_handler(emitter, stream=stream))
    )
    return build_app(
        proxy_url="http://proxy", receipts_dir=tmp_path,
        verifying_material=pub, client=client, **kw,
    )


def _drive(app, coro_fn):
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://console"
        ) as c:
            return await coro_fn(c)
    return asyncio.run(run())


# --- core: chat drives the proxy and verifies the emitted receipt ----------


def test_index_serves_page(tmp_path):
    emitter, pub = _es256_emitter(tmp_path)
    app = _console(tmp_path, pub, emitter=emitter)
    r = _drive(app, lambda c: c.get("/"))
    assert r.status_code == 200
    assert "Vaara console" in r.text
    # the official wordmark asset, inlined as a data URI (not a font fake)
    assert 'class="brandimg"' in r.text
    assert "data:image/png;base64," in r.text
    assert "__WORDMARK__" not in r.text  # placeholder fully substituted
    assert 'id="judge"' in r.text  # the cross-check judge dropdown


def test_buffered_chat_verifies_turn(tmp_path):
    emitter, pub = _es256_emitter(tmp_path)
    app = _console(tmp_path, pub, emitter=emitter)

    async def go(c):
        return await c.post("/api/chat", json={
            "model": "qwen3:30b-a3b", "messages": MESSAGES, "stream": False,
        })

    r = _drive(app, go)
    body = r.json()
    assert body["message"]["content"] == "hi there"
    assert body["turn"]["available"] is True
    assert body["turn"]["verdict"]["ok"] is True
    assert body["turn"]["verdict"]["receiptSignature"] is True
    assert body["turn"]["verdict"]["attestationSignature"] is True


def test_streaming_chat_then_latest_verdict(tmp_path):
    emitter, pub = _es256_emitter(tmp_path)
    app = _console(tmp_path, pub, emitter=emitter, stream=True)

    async def go(c):
        chunks = []
        async with c.stream("POST", "/api/chat", json={
            "model": "qwen3:30b-a3b", "messages": MESSAGES, "stream": True,
        }) as resp:
            async for b in resp.aiter_bytes():
                chunks.append(b)
        latest = await c.get("/api/turn/latest")
        return b"".join(chunks), latest.json()

    streamed, latest = _drive(app, go)
    assert b"hi there" in streamed
    assert latest["available"] is True
    assert latest["verdict"]["ok"] is True


def test_keyless_viewer_still_verifies_structurally(tmp_path):
    # No verifying key at all: the console is a viewer, so structural + back-link
    # checks still run and the turn is reported, proving the path never needs a key.
    emitter, _ = _es256_emitter(tmp_path)
    app = _console(tmp_path, None, emitter=emitter)

    async def go(c):
        return await c.post("/api/chat", json={"messages": MESSAGES, "stream": False})

    body = _drive(app, go).json()
    assert body["turn"]["available"] is True
    assert body["turn"]["verdict"]["ok"] is True
    assert "receiptSignature" not in body["turn"]["verdict"]  # keyless: not checked


# --- on-demand proof routes degrade honestly when unconfigured -------------


def test_verify_chain_unavailable_without_chain(tmp_path):
    emitter, pub = _es256_emitter(tmp_path)
    app = _console(tmp_path, pub, emitter=emitter)
    body = _drive(app, lambda c: c.post("/api/verify-chain")).json()
    assert body == {"available": False, "reason": "no TPM evidence chain configured"}


def test_crosscheck_unavailable_without_verifier(tmp_path):
    emitter, pub = _es256_emitter(tmp_path)
    app = _console(tmp_path, pub, emitter=emitter)
    body = _drive(app, lambda c: c.post("/api/crosscheck")).json()
    assert body == {"available": False, "reason": "no verifier identity configured"}


def test_latest_turn_empty_before_any_chat(tmp_path):
    emitter, pub = _es256_emitter(tmp_path)
    app = _console(tmp_path, pub, emitter=emitter)
    body = _drive(app, lambda c: c.get("/api/turn/latest")).json()
    assert body == {"available": False}


# --- model discovery -------------------------------------------------------


def _tags_handler(emitter, models):
    """Proxy stand-in: serve /api/tags for GET, sign chat for POST."""
    chat = _proxy_handler(emitter)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/api/tags"):
            return httpx.Response(200, json={"models": [{"name": m} for m in models]})
        return chat(request)

    return handler


def test_config_lists_models_recall_off(tmp_path):
    emitter, pub = _es256_emitter(tmp_path)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_tags_handler(emitter, ["qwen3:30b-a3b", "mistral"]))
    )
    app = build_app(
        proxy_url="http://proxy", receipts_dir=tmp_path,
        verifying_material=pub, client=client,
    )
    body = _drive(app, lambda c: c.get("/api/config")).json()
    assert body["models"] == ["qwen3:30b-a3b", "mistral"]
    assert body["recall"] is False


def test_config_models_empty_when_proxy_down(tmp_path):
    emitter, pub = _es256_emitter(tmp_path)

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("proxy down")

    client = httpx.AsyncClient(transport=httpx.MockTransport(boom))
    app = build_app(
        proxy_url="http://proxy", receipts_dir=tmp_path,
        verifying_material=pub, client=client,
    )
    body = _drive(app, lambda c: c.get("/api/config")).json()
    assert body["models"] == []
    assert body["recall"] is False
    assert body["crosscheck"] is False
    assert body["judgeDefault"] == ""


def test_config_recall_on_when_wired(tmp_path):
    emitter, pub = _es256_emitter(tmp_path)
    client = httpx.AsyncClient(transport=httpx.MockTransport(_tags_handler(emitter, [])))
    recall = MemoryRecall(tmp_path / "x.db", engine=lambda q, k: "")
    app = build_app(
        proxy_url="http://proxy", receipts_dir=tmp_path,
        verifying_material=pub, client=client, recall=recall,
    )
    body = _drive(app, lambda c: c.get("/api/config")).json()
    assert body["recall"] is True


# --- memory grounding ------------------------------------------------------


def _capture_handler(emitter, captured, content="hi there"):
    """Proxy stand-in that records the messages it was asked to sign."""

    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content or b"{}")
        captured.append(data.get("messages"))
        att, counter = emitter.emit_attestation(
            model_ref=data.get("model", "m"), model_derived=MODEL,
            messages=data.get("messages"), sampling={},
        )
        emitter.emit_receipt(
            attestation=att, counter=counter, status="completed",
            output={"content": content}, eval_stats=None,
        )
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": content}, "done": True}
        )

    return handler


def _grounding_app(tmp_path, captured, engine):
    emitter, pub = _es256_emitter(tmp_path)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_capture_handler(emitter, captured))
    )
    recall = MemoryRecall(tmp_path / "noindex.db", engine=engine)
    return build_app(
        proxy_url="http://proxy", receipts_dir=tmp_path,
        verifying_material=pub, client=client, recall=recall,
    )


def test_grounding_injects_reference_system_message(tmp_path):
    captured: list = []
    app = _grounding_app(
        tmp_path, captured, lambda q, k: f"[mem.md] note\nrecalled fact about {q}"
    )

    async def go(c):
        return await c.post("/api/chat", json={
            "model": "m", "messages": MESSAGES, "stream": False,
        })

    body = _drive(app, go).json()
    sent = captured[-1]
    assert sent[0]["role"] == "system"
    assert "recalled-memory" in sent[0]["content"]
    assert sent[1] == MESSAGES[0]
    assert body["turn"]["grounded"] == 1
    # the receipt is over the grounded prompt and still verifies
    assert body["turn"]["verdict"]["ok"] is True


def test_grounding_opt_out_per_turn(tmp_path):
    captured: list = []
    app = _grounding_app(tmp_path, captured, lambda q, k: "[mem.md] note\nfact")

    async def go(c):
        return await c.post("/api/chat", json={
            "model": "m", "messages": MESSAGES, "stream": False, "ground": False,
        })

    body = _drive(app, go).json()
    assert captured[-1] == MESSAGES
    assert body["turn"]["grounded"] == 0


def test_grounding_empty_recall_leaves_prompt_unchanged(tmp_path):
    captured: list = []
    app = _grounding_app(tmp_path, captured, lambda q, k: "(no matches for: x)")

    async def go(c):
        return await c.post("/api/chat", json={
            "model": "m", "messages": MESSAGES, "stream": False,
        })

    body = _drive(app, go).json()
    assert captured[-1] == MESSAGES
    assert body["turn"]["grounded"] == 0


def test_no_grounding_without_recall(tmp_path):
    emitter, pub = _es256_emitter(tmp_path)
    captured: list = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_capture_handler(emitter, captured))
    )
    app = build_app(
        proxy_url="http://proxy", receipts_dir=tmp_path,
        verifying_material=pub, client=client,
    )

    async def go(c):
        return await c.post("/api/chat", json={
            "model": "m", "messages": MESSAGES, "stream": False,
        })

    body = _drive(app, go).json()
    assert captured[-1] == MESSAGES
    assert body["turn"]["grounded"] == 0


# --- cross-check: judge chosen per request ---------------------------------


def _crosscheck_app(tmp_path, requested, *, judge_default=None, agreement="equivalent"):
    """Console wired with a signing cross-check identity and a stub judge factory.

    ``requested`` records every judge-model name the factory is asked to build, so
    a test can assert the per-request pick (or the default fallback) reached it.
    The stub judge always reports a verifier model whose weights pin differs from
    the subject's, so the diverse gate passes the way a real second model would.
    """
    emitter, pub = _es256_emitter(tmp_path)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_proxy_handler(emitter))
    )
    cc_key = ec.generate_private_key(ec.SECP256R1())
    crosscheck = {
        "judge_model": judge_default,
        "upstream": "http://up",
        "signing_material": cc_key,
        "alg": "ES256",
        "secret_version": "ccv1",
    }

    def factory(model):
        requested.append(model)

        def judge(*, messages, candidate_response):
            from vaara.attestation._inference_crosscheck import JudgeOutcome

            return JudgeOutcome(
                agreement=agreement,
                raw_judgment=f"VERDICT: {agreement}",
                model=ModelDerived(
                    model_ref=model,
                    manifest_digest="sha256:" + "c" * 64,
                    gguf_metadata_hash="sha256:" + "d" * 64,
                    quantization="Q4_K_M", param_count="3B",
                ),
            )

        return judge

    return build_app(
        proxy_url="http://proxy", receipts_dir=tmp_path, verifying_material=pub,
        client=client, crosscheck=crosscheck, judge_factory=factory,
    )


def _chat_then_crosscheck(app, payload):
    async def go(c):
        await c.post("/api/chat", json={
            "model": "qwen3:30b-a3b", "messages": MESSAGES, "stream": False,
        })
        return await c.post("/api/crosscheck", json=payload)

    return _drive(app, go).json()


def test_crosscheck_picks_judge_per_request(tmp_path):
    requested: list = []
    app = _crosscheck_app(tmp_path, requested)
    body = _chat_then_crosscheck(app, {"judge_model": "llama3.2:3b"})
    assert requested == ["llama3.2:3b"]  # the per-request pick reached the factory
    assert body["available"] is True
    assert body["agreement"] == "equivalent"
    assert body["diverse"] is True  # diverse gate intact: verifier != subject weights
    assert body["verifierModel"] == "llama3.2:3b"


def test_crosscheck_falls_back_to_default_judge(tmp_path):
    requested: list = []
    app = _crosscheck_app(tmp_path, requested, judge_default="qwen2.5:14b")
    body = _chat_then_crosscheck(app, {})  # no per-request judge -> default
    assert requested == ["qwen2.5:14b"]
    assert body["verifierModel"] == "qwen2.5:14b"


def test_crosscheck_no_judge_selected_is_unavailable(tmp_path):
    requested: list = []
    app = _crosscheck_app(tmp_path, requested)  # no default, none in body
    body = _chat_then_crosscheck(app, {})
    assert body == {"available": False, "reason": "no judge model selected"}
    assert requested == []  # the judge is never built without a model


def test_config_advertises_crosscheck_default(tmp_path):
    requested: list = []
    app = _crosscheck_app(tmp_path, requested, judge_default="qwen2.5:14b")
    body = _drive(app, lambda c: c.get("/api/config")).json()
    assert body["crosscheck"] is True
    assert body["judgeDefault"] == "qwen2.5:14b"
