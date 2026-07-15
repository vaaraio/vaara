"""A3 gate tests: the proxy blocks denied tool calls at the model layer.

With an enforcing pipeline, a denied tool_use never reaches the agent: the
response is rewritten so the tool call is gone and a policy-error text
explains why. Escalate blocks on the approvals handshake. Observe mode
(enforce=False) stays pure passthrough — covered in test_infer_proxy.py.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import threading
import time

import pytest

for _mod in ("rfc8785", "cryptography", "httpx", "fastapi"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip("proxy deps not installed", allow_module_level=True)

import httpx  # noqa: E402

from vaara.integrations._infer_proxy_app import build_app  # noqa: E402
from vaara.pipeline import InterceptionPipeline  # noqa: E402

MESSAGES = [{"role": "user", "content": "Do the thing."}]


def _pipeline(escalate: float, deny: float) -> InterceptionPipeline:
    from vaara.policy import from_dict
    from vaara.policy.modes import get_mode, to_policy_dict

    pipeline = InterceptionPipeline(enforce=True)
    policy = to_policy_dict(get_mode("balanced"))
    policy["thresholds"]["default"] = {"escalate": escalate, "deny": deny}
    pipeline.scorer.apply_policy(from_dict(policy))
    return pipeline


def _deny_all() -> InterceptionPipeline:
    return _pipeline(0.0005, 0.001)


def _escalate_all() -> InterceptionPipeline:
    return _pipeline(0.0, 1.0)


def _anthropic_upstream(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/messages":
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [
                {"type": "text", "text": "Deleting now."},
                {"type": "tool_use", "id": "toolu_1", "name": "delete_file",
                 "input": {"path": "/etc/passwd"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 9},
        })
    return httpx.Response(404, json={})


def _openai_upstream(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/chat/completions":
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "send_payment",
                                             "arguments": '{"amount": 900}'}}],
            }}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        })
    return httpx.Response(404, json={})


def _drive(app, path: str, body: dict):
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://p") as c:
            resp = await c.post(path, json=body)
            return resp.status_code, await resp.aread()

    return asyncio.run(go())


def test_anthropic_buffered_deny_rewrites_tool_use_away(tmp_path):
    app = build_app(
        emitter=None, upstream="http://up", pipeline=_deny_all(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(_anthropic_upstream)),
    )
    status, raw = _drive(app, "/v1/messages", {
        "model": "claude-sonnet-5", "messages": MESSAGES, "stream": False,
    })
    assert status == 200
    doc = json.loads(raw)
    kinds = [b.get("type") for b in doc["content"]]
    assert "tool_use" not in kinds, "denied tool_use leaked to the agent"
    text = " ".join(b.get("text", "") for b in doc["content"])
    assert "blocked by Vaara policy" in text
    assert doc["stop_reason"] == "end_turn"


def test_openai_buffered_deny_strips_tool_calls(tmp_path):
    app = build_app(
        emitter=None, upstream="http://up", pipeline=_deny_all(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(_openai_upstream)),
    )
    status, raw = _drive(app, "/v1/chat/completions", {
        "model": "gpt-x", "messages": MESSAGES, "stream": False,
    })
    assert status == 200
    doc = json.loads(raw)
    message = doc["choices"][0]["message"]
    assert not message.get("tool_calls")
    assert "blocked by Vaara policy" in (message.get("content") or "")
    assert doc["choices"][0]["finish_reason"] == "stop"


def test_allow_passes_response_unchanged(tmp_path):
    # Thresholds so high nothing denies: enforcing pipeline, allow verdicts.
    app = build_app(
        emitter=None, upstream="http://up", pipeline=_pipeline(0.98, 0.99),
        client=httpx.AsyncClient(transport=httpx.MockTransport(_anthropic_upstream)),
    )
    status, raw = _drive(app, "/v1/messages", {
        "model": "claude-sonnet-5", "messages": MESSAGES, "stream": False,
    })
    assert status == 200
    doc = json.loads(raw)
    assert doc["content"][1]["type"] == "tool_use"  # untouched
    assert doc["content"][1]["id"] == "toolu_1"


def _anthropic_stream_upstream(request: httpx.Request) -> httpx.Response:
    if request.url.path != "/v1/messages":
        return httpx.Response(404, json={})
    body = (
        b'event: content_block_start\n'
        b'data: {"type":"content_block_start","index":0,"content_block":'
        b'{"type":"tool_use","id":"toolu_2","name":"delete_file","input":{}}}\n\n'
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","index":0,"delta":'
        b'{"type":"input_json_delta","partial_json":"{\\"path\\": \\"/etc\\"}"}}\n\n'
        b'event: content_block_stop\n'
        b'data: {"type":"content_block_stop","index":0}\n\n'
        b'event: message_stop\n'
        b'data: {"type":"message_stop"}\n\n'
    )
    return httpx.Response(200, content=body,
                          headers={"content-type": "text/event-stream"})


def test_anthropic_streamed_deny_synthesizes_clean_stream(tmp_path):
    app = build_app(
        emitter=None, upstream="http://up", pipeline=_deny_all(),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(_anthropic_stream_upstream)),
    )
    status, raw = _drive(app, "/v1/messages", {
        "model": "claude-sonnet-5", "messages": MESSAGES, "stream": True,
    })
    assert status == 200
    assert b"toolu_2" not in raw, "denied tool_use bytes leaked into the stream"
    assert b"blocked by Vaara policy" in raw
    assert b"message_stop" in raw  # still a well-formed anthropic stream


def test_streamed_allow_replays_original_bytes(tmp_path):
    app = build_app(
        emitter=None, upstream="http://up", pipeline=_pipeline(0.98, 0.99),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(_anthropic_stream_upstream)),
    )
    status, raw = _drive(app, "/v1/messages", {
        "model": "claude-sonnet-5", "messages": MESSAGES, "stream": True,
    })
    assert status == 200
    assert b"toolu_2" in raw  # allowed stream reaches the agent intact


def test_escalate_approved_via_handshake_passes_through(tmp_path):
    approvals = tmp_path / "approvals"

    def responder():
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            reqs = list(approvals.glob("*.request.json")) if approvals.exists() else []
            if reqs:
                action_id = reqs[0].name.removesuffix(".request.json")
                (approvals / f"{action_id}.decision.json").write_text(
                    json.dumps({"decision": "approve", "decided_at": time.time()}))
                return
            time.sleep(0.02)

    thread = threading.Thread(target=responder, daemon=True)
    thread.start()
    app = build_app(
        emitter=None, upstream="http://up", pipeline=_escalate_all(),
        approvals_dir=approvals, approvals_timeout=10,
        client=httpx.AsyncClient(transport=httpx.MockTransport(_anthropic_upstream)),
    )
    status, raw = _drive(app, "/v1/messages", {
        "model": "claude-sonnet-5", "messages": MESSAGES, "stream": False,
    })
    thread.join(timeout=1)
    assert status == 200
    doc = json.loads(raw)
    assert doc["content"][1]["type"] == "tool_use"  # approved -> untouched


def test_escalate_without_approvals_dir_fails_closed(tmp_path):
    app = build_app(
        emitter=None, upstream="http://up", pipeline=_escalate_all(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(_anthropic_upstream)),
    )
    status, raw = _drive(app, "/v1/messages", {
        "model": "claude-sonnet-5", "messages": MESSAGES, "stream": False,
    })
    assert status == 200
    doc = json.loads(raw)
    assert "tool_use" not in [b.get("type") for b in doc["content"]]
    assert "blocked by Vaara policy" in " ".join(
        b.get("text", "") for b in doc["content"])


# --- allow-list bypass -------------------------------------------------------


def test_allowed_pattern_skips_gating(tmp_path):
    import asyncio
    from vaara.integrations._infer_proxy_gate import gate_tool_calls

    class DenyAllPipeline:
        _enforce = True
        def intercept(self, **kw):
            raise AssertionError("allow-listed call must not be gated")

    calls = [{"function": {"name": "read_file", "arguments": "{}"}}]
    denials = asyncio.run(
        gate_tool_calls(DenyAllPipeline(), calls, model_name="m",
                        allow_patterns=["read_*"]))
    assert denials == []


def test_unmatched_pattern_still_gated(tmp_path):
    import asyncio
    from vaara.integrations._infer_proxy_gate import gate_tool_calls

    class Pipe:
        _enforce = True
        def __init__(self): self.seen = []
        def intercept(self, **kw):
            self.seen.append(kw["tool_name"])
            class R: allowed = True; verdict = None
            return R()

    p = Pipe()
    calls = [{"function": {"name": "delete_repo", "arguments": "{}"}}]
    asyncio.run(
        gate_tool_calls(p, calls, model_name="m", allow_patterns=["read_*"]))
    assert p.seen == ["delete_repo"]
