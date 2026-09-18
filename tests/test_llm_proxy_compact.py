"""History compaction on the llm-proxy: old tool payloads leave once.

A coding agent resends its whole conversation on every call, so a file read
at 09:00 leaves the machine again at 09:01, 09:02 and every call after. With
compaction on, tool results and tool inputs older than the last N turns are
replaced by a size-and-digest stub. The last N turns stay verbatim, user text
is never touched, and the rewrite is deterministic so the same history always
compacts to the same bytes.
"""
from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from vaara.audit.trail import EventType  # noqa: E402
from vaara.integrations._llm_proxy_app import build_app  # noqa: E402
from vaara.integrations.llm_compact import (  # noqa: E402
    STUB_KEY,
    compact_messages,
)
from vaara.integrations.llm_proxy import _build_pipeline  # noqa: E402


def _turns(n_assistant: int, payload: str = "X" * 1000) -> list[dict]:
    """A conversation of n assistant turns, each with a tool_use and result."""
    msgs = [{"role": "user", "content": "start"}]
    for i in range(n_assistant):
        msgs.append({"role": "assistant", "content": [
            {"type": "text", "text": f"reading {i}"},
            {"type": "tool_use", "id": f"tu{i}", "name": "Read",
             "input": {"file_path": f"/f{i}", "blob": payload}},
        ]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"tu{i}",
             "content": f"contents of f{i}: {payload}"},
        ]})
    return msgs


class TestCompactMessages:

    def test_old_tool_results_become_stubs_recent_stay(self):
        msgs = _turns(5)
        out, stats = compact_messages(msgs, keep_turns=2)
        # turns 0,1,2 are old (5 turns, keep last 2 -> keep 3 and 4)
        old = out[2]["content"][0]
        assert old["type"] == "tool_result"
        assert old["tool_use_id"] == "tu0"
        assert STUB_KEY in old["content"]
        recent = out[-1]["content"][0]
        assert recent["content"].startswith("contents of f4")
        assert stats["compacted_blocks"] == 6      # 3 results + 3 inputs
        assert stats["bytes_before"] > stats["bytes_after"]

    def test_stub_carries_size_and_digest_not_content(self):
        msgs = _turns(3)
        out, _ = compact_messages(msgs, keep_turns=1)
        stub = out[2]["content"][0]["content"]
        original = "contents of f0: " + "X" * 1000
        assert str(len(original.encode())) in stub
        assert hashlib.sha256(original.encode()).hexdigest()[:16] in stub
        assert "X" * 20 not in stub

    def test_tool_use_input_is_stubbed_but_shape_kept(self):
        msgs = _turns(3)
        out, _ = compact_messages(msgs, keep_turns=1)
        tu = out[1]["content"][1]
        assert tu["type"] == "tool_use"
        assert tu["id"] == "tu0" and tu["name"] == "Read"
        assert isinstance(tu["input"], dict)
        assert STUB_KEY in tu["input"]
        assert "blob" not in tu["input"]

    def test_user_and_assistant_text_never_touched(self):
        msgs = _turns(4)
        msgs.insert(1, {"role": "user", "content": "my private idea sentence"})
        out, _ = compact_messages(msgs, keep_turns=1)
        assert out[1]["content"] == "my private idea sentence"
        assert out[2]["content"][0] == {"type": "text", "text": "reading 0"}

    def test_deterministic(self):
        msgs = _turns(6)
        a, _ = compact_messages(msgs, keep_turns=2)
        b, _ = compact_messages(msgs, keep_turns=2)
        assert json.dumps(a) == json.dumps(b)

    def test_keep_turns_zero_means_off(self):
        msgs = _turns(3)
        out, stats = compact_messages(msgs, keep_turns=0)
        assert out == msgs
        assert stats["compacted_blocks"] == 0

    def test_openai_tool_messages_are_stubbed(self):
        msgs = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read", "arguments": "{\"blob\": \"" + "Y" * 500 + "\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Y" * 900},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "ok"},
        ]
        out, stats = compact_messages(msgs, keep_turns=1)
        assert STUB_KEY in out[2]["content"]
        assert STUB_KEY in out[1]["tool_calls"][0]["function"]["arguments"]
        assert out[3]["content"] == "done"
        assert stats["compacted_blocks"] == 2


@pytest.fixture
def pipeline(tmp_path):
    return _build_pipeline(tmp_path / "trail.db")


def _mount(monkeypatch, seen: list):
    real = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]},
                              headers={"content-type": "application/json"})

    def fake(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        kwargs.pop("http2", None)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake)


def _records(pipeline, event):
    return [r for r in pipeline.trail.get_records_by_type(event)
            if r.tool_name == "llm.prompt"]


class TestProxyCompaction:

    def test_upstream_receives_compacted_body_and_record_says_so(
            self, monkeypatch, pipeline):
        seen: list = []
        _mount(monkeypatch, seen)
        app = build_app(
            upstream="https://upstream.invalid", api_key="k",
            api_key_header="x-api-key", pipeline=pipeline,
            compact_keep_turns=2)
        body = {"model": "claude-opus-5", "messages": _turns(5)}
        TestClient(app).post("/v1/messages", json=body)
        sent = json.loads(seen[0])
        assert STUB_KEY in sent["messages"][2]["content"][0]["content"]
        assert sent["messages"][-1]["content"][0]["content"].startswith(
            "contents of f4")
        (req,) = _records(pipeline, EventType.ACTION_REQUESTED)
        env = req.data["parameters"]["envelope"]
        assert env["bytes_before_compaction"] > env["bytes_out"]
        assert env["compacted_blocks"] == 6

    def test_off_by_default_forwards_bytes_unchanged(
            self, monkeypatch, pipeline):
        seen: list = []
        _mount(monkeypatch, seen)
        app = build_app(
            upstream="https://upstream.invalid", api_key="k",
            api_key_header="x-api-key", pipeline=pipeline)
        raw = json.dumps({"model": "claude-opus-5", "messages": _turns(3)}).encode()
        TestClient(app).post("/v1/messages", content=raw,
                             headers={"content-type": "application/json"})
        assert seen[0] == raw
        (req,) = _records(pipeline, EventType.ACTION_REQUESTED)
        env = req.data["parameters"]["envelope"]
        assert env["compacted_blocks"] == 0
        assert env["bytes_before_compaction"] == env["bytes_out"]

    def test_compaction_runs_before_sealing_and_measure(
            self, monkeypatch, pipeline):
        from vaara.integrations.llm_seal import SealRegistry
        seen: list = []
        _mount(monkeypatch, seen)
        app = build_app(
            upstream="https://upstream.invalid", api_key="k",
            api_key_header="x-api-key", pipeline=pipeline,
            compact_keep_turns=1, seal_registry=SealRegistry({"s": "SECRETWORD"}))
        msgs = _turns(3, payload="SECRETWORD")
        msgs.append({"role": "user", "content": "SECRETWORD again"})
        TestClient(app).post("/v1/messages", json={"model": "m", "messages": msgs})
        assert b"SECRETWORD" not in seen[0]
        (req,) = _records(pipeline, EventType.ACTION_REQUESTED)
        p = req.data["parameters"]
        # only the recent occurrences survive to be sealed; the old ones left
        # as digests, which is the point
        assert p["seal_count"] >= 1
        assert p["envelope"]["compacted_blocks"] == 4
