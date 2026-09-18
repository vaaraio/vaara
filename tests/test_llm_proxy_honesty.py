"""The llm-proxy trail must not claim what the proxy did not do.

Audit of 2026-09-16 found three ways the trail could be wrong about a request:

- Every ``llm.prompt`` record was silent on sealing. A trail that says a
  request went out cannot say whether the secrets went with it.
- The seal file was read once at startup. The monitor reads the file, the
  proxy holds a copy, and after an edit they disagree for the life of the
  process. A ``--seal-file`` that loaded nothing started anyway, unsealed.
- A client that disconnected mid-stream raised ``CancelledError`` or
  ``GeneratorExit``, neither an ``Exception``, so the forwarder recorded no
  outcome and the action stayed pending forever. 178 of them at the audit.

These tests pin the fixes. The upstream is a stub inside the process.
"""

from __future__ import annotations

import json
import time

import pytest

httpx = pytest.importorskip("httpx", reason="proxy deps not installed: no httpx")
pytest.importorskip("fastapi", reason="proxy deps not installed: no fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from vaara.audit.trail import EventType  # noqa: E402
from vaara.integrations._llm_proxy_app import (  # noqa: E402
    build_app,
    sweep_orphaned_outcomes,
)
from vaara.integrations.llm_proxy import _build_pipeline  # noqa: E402
from vaara.integrations.llm_seal import (  # noqa: E402
    SealRegistry,
    StreamUnsealer,
    placeholder_for,
)

SECRET = "northern-lights"
TOKEN = placeholder_for(SECRET)
STRANGER = "VAARA_SEAL_0123456789ab"   # a placeholder no registry here knows


@pytest.fixture
def pipeline(tmp_path):
    return _build_pipeline(tmp_path / "trail.db")


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


def _echo_handler(request: httpx.Request) -> httpx.Response:
    body = request.content.decode("utf-8")
    return httpx.Response(
        200,
        json={"content": [{"type": "text",
                           "text": f"you said {body} and also {STRANGER}"}]},
        headers={"content-type": "application/json"},
    )


def _post(client, text=f"explain {SECRET}"):
    return client.post("/v1/messages", json={
        "model": "claude-opus-5",
        "messages": [{"role": "user", "content": text}],
    })


class TestPromptRecordCarriesSealState:

    def test_sealed_request_records_active_and_count(
            self, monkeypatch, pipeline):
        _mount(monkeypatch, _echo_handler)
        app = build_app(
            upstream="https://upstream.invalid", api_key="k",
            api_key_header="x-api-key", pipeline=pipeline,
            seal_registry=SealRegistry({"concept": SECRET}),
        )
        _post(TestClient(app), f"{SECRET} twice: {SECRET}")
        (req,) = _records(pipeline, EventType.ACTION_REQUESTED)
        params = req.data["parameters"]
        assert params["seal_active"] is True
        assert params["seal_count"] == 2
        assert params["seal_fault"] is None

    def test_unsealed_request_says_so(self, monkeypatch, pipeline):
        _mount(monkeypatch, _echo_handler)
        app = build_app(
            upstream="https://upstream.invalid", api_key="k",
            api_key_header="x-api-key", pipeline=pipeline,
            seal_registry=SealRegistry(),
        )
        _post(TestClient(app))
        (req,) = _records(pipeline, EventType.ACTION_REQUESTED)
        params = req.data["parameters"]
        assert params["seal_active"] is False
        assert params["seal_count"] == 0

    def test_sealing_fault_is_recorded_not_hidden(
            self, monkeypatch, pipeline):
        _mount(monkeypatch, _echo_handler)
        reg = SealRegistry({"concept": SECRET})

        def boom(raw: bytes) -> bytes:
            raise RuntimeError("seal exploded")

        monkeypatch.setattr(reg, "seal_bytes", boom)
        app = build_app(
            upstream="https://upstream.invalid", api_key="k",
            api_key_header="x-api-key", pipeline=pipeline,
            seal_registry=reg,
        )
        resp = _post(TestClient(app))
        assert resp.status_code == 200          # still fails open
        (req,) = _records(pipeline, EventType.ACTION_REQUESTED)
        params = req.data["parameters"]
        assert params["seal_active"] is True
        assert params["seal_count"] == 0
        assert "RuntimeError" in params["seal_fault"]


class TestOutcomeCarriesUnsealState:

    def test_buffered_response_counts_restored_and_unmapped(
            self, monkeypatch, pipeline):
        _mount(monkeypatch, _echo_handler)
        app = build_app(
            upstream="https://upstream.invalid", api_key="k",
            api_key_header="x-api-key", pipeline=pipeline,
            seal_registry=SealRegistry({"concept": SECRET}),
        )
        _post(TestClient(app))
        (out,) = _records(pipeline, EventType.OUTCOME_RECORDED)
        desc = json.loads(out.data["description"])
        assert desc["status"] == "ok"
        assert desc["unseal_count"] == 1
        assert desc["unmapped_placeholders"] == [STRANGER]

    def test_streamed_response_counts_restored_and_unmapped(
            self, monkeypatch, pipeline):
        def handler(request: httpx.Request) -> httpx.Response:
            body = (f"data: {TOKEN} then {STRANGER}\n\n"
                    f"data: {TOKEN}\n\n").encode()
            return httpx.Response(
                200, content=body,
                headers={"content-type": "text/event-stream"})

        _mount(monkeypatch, handler)
        app = build_app(
            upstream="https://upstream.invalid", api_key="k",
            api_key_header="x-api-key", pipeline=pipeline,
            seal_registry=SealRegistry({"concept": SECRET}),
        )
        resp = TestClient(app).post("/v1/messages", json={
            "model": "m", "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.text.count(SECRET) == 2
        (out,) = _records(pipeline, EventType.OUTCOME_RECORDED)
        desc = json.loads(out.data["description"])
        assert desc["status"] == "ok"
        assert desc["unseal_count"] == 2
        assert desc["unmapped_placeholders"] == [STRANGER]


class TestStreamAbortIsRecorded:

    def test_generator_exit_closes_the_outcome(self, pipeline):
        """Starlette closes the body generator on client disconnect, which
        raises GeneratorExit inside it. That is a BaseException."""
        import asyncio

        from vaara.integrations._llm_proxy_app import _forward_stream

        result = pipeline.intercept(
            agent_id="a", tool_name="llm.prompt", parameters={"model": "m"})

        class _Upstream:
            async def aiter_bytes(self):
                yield b"data: one\n\n"
                yield b"data: two\n\n"

        async def run():
            gen = _forward_stream(_Upstream(), pipeline, result.action_id)
            first = await gen.__anext__()
            assert first
            await gen.aclose()

        asyncio.run(run())
        (out,) = _records(pipeline, EventType.OUTCOME_RECORDED)
        assert out.data["outcome_severity"] == 0.5
        assert json.loads(out.data["description"])["status"] == "aborted"

    def test_cancelled_error_closes_the_outcome(self, pipeline):
        import asyncio

        from vaara.integrations._llm_proxy_app import _forward_stream

        result = pipeline.intercept(
            agent_id="a", tool_name="llm.prompt", parameters={"model": "m"})

        class _Upstream:
            async def aiter_bytes(self):
                yield b"data: one\n\n"
                raise asyncio.CancelledError()

        async def run():
            gen = _forward_stream(_Upstream(), pipeline, result.action_id)
            await gen.__anext__()
            with pytest.raises(asyncio.CancelledError):
                await gen.__anext__()

        asyncio.run(run())
        (out,) = _records(pipeline, EventType.OUTCOME_RECORDED)
        assert json.loads(out.data["description"])["status"] == "aborted"


class TestOrphanSweep:

    def test_pending_prompts_older_than_the_process_are_closed(
            self, pipeline):
        old = pipeline.intercept(
            agent_id="a", tool_name="llm.prompt", parameters={"model": "m"})
        started = time.time() + 1.0
        new = pipeline.intercept(
            agent_id="a", tool_name="llm.prompt", parameters={"model": "m"})
        backend = pipeline.trail._backend
        backend.store_pending_outcome(new.action_id, "a", "llm.prompt",
                                      0.1, {})
        # The new one is re-stamped as created after the process started.
        backend._conn.execute(
            "UPDATE pending_outcomes SET created_at=? WHERE action_id=?",
            (started + 5, new.action_id))
        closed = sweep_orphaned_outcomes(pipeline, process_started=started)
        assert closed == 1
        outs = _records(pipeline, EventType.OUTCOME_RECORDED)
        assert [o.action_id for o in outs] == [old.action_id]
        assert json.loads(outs[0].data["description"])["status"] == "orphaned"

    def test_other_tools_pending_outcomes_are_left_alone(self, pipeline):
        pipeline.intercept(
            agent_id="a", tool_name="file.read", parameters={"path": "x"})
        assert sweep_orphaned_outcomes(
            pipeline, process_started=time.time() + 1) == 0


class TestSealFileReload:

    def test_registry_follows_the_file(self, tmp_path):
        f = tmp_path / "seal.json"
        f.write_text(json.dumps({"a": "one"}))
        reg = SealRegistry.from_file(f)
        assert len(reg) == 1
        f.write_text(json.dumps({"a": "one", "b": "two"}))
        import os
        os.utime(f, (time.time() + 10, time.time() + 10))
        assert reg.refresh() is True
        assert len(reg) == 2
        assert reg.refresh() is False

    def test_proxy_refreshes_per_request(self, monkeypatch, pipeline,
                                         tmp_path):
        import os
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = request.content.decode()
            return httpx.Response(200, json={"ok": True})

        _mount(monkeypatch, handler)
        f = tmp_path / "seal.json"
        f.write_text(json.dumps({"a": "one"}))
        reg = SealRegistry.from_file(f)
        app = build_app(
            upstream="https://upstream.invalid", api_key="k",
            api_key_header="x-api-key", pipeline=pipeline, seal_registry=reg)
        client = TestClient(app)
        _post(client, "one and two")
        assert "two" in seen["body"]
        f.write_text(json.dumps({"a": "one", "b": "two"}))
        os.utime(f, (time.time() + 10, time.time() + 10))
        _post(client, "one and two")
        assert "two" not in seen["body"]


class TestStreamUnsealerCounts:

    def test_counts_survive_a_split_placeholder(self):
        reg = SealRegistry({"c": SECRET})
        u = StreamUnsealer(reg)
        out = b""
        for ch in f"x {TOKEN} y {STRANGER} z".encode():
            out += u.feed(bytes([ch]))
        out += u.flush()
        assert out.decode() == f"x {SECRET} y {STRANGER} z"
        assert u.restored == 1
        assert u.unmapped == [STRANGER]


class TestEmptySealFileRefusesToStart:

    def test_refuses_without_allow_unsealed(self, tmp_path, capsys):
        from vaara.integrations.llm_proxy import main
        f = tmp_path / "seal.json"
        f.write_text("{}")
        rc = main(["--upstream", "https://x.test", "--auth-passthrough",
                   "--seal-file", str(f),
                   "--trail", str(tmp_path / "trail.db")])
        assert rc == 2
        assert "refusing to start unsealed" in capsys.readouterr().err

    def test_allow_unsealed_is_reachable(self):
        """The flag exists in the one parser `vaara llm-proxy` forwards to."""
        import inspect
        import re

        from vaara.integrations import llm_proxy
        assert "--allow-unsealed" in re.findall(
            r'add_argument\(\s*["\'](--[a-z0-9-]+)["\']',
            inspect.getsource(llm_proxy))


class TestRecordNamesWhoAndWhat:

    def test_agent_id_default_is_configurable(self, monkeypatch, pipeline):
        _mount(monkeypatch, _echo_handler)
        app = build_app(
            upstream="https://upstream.invalid", api_key="k",
            api_key_header="x-api-key", pipeline=pipeline,
            agent_id_default="claude-code@box", enforce=False,
            audit_level="hash",
        )
        _post(TestClient(app))
        (req,) = _records(pipeline, EventType.ACTION_REQUESTED)
        assert req.agent_id == "claude-code@box"
        params = req.data["parameters"]
        assert params["enforce"] is False
        assert params["audit_level"] == "hash"
        assert len(params["prompt_hash"]) == 64
        assert "messages" not in params

    def test_header_still_wins_over_the_default(self, monkeypatch, pipeline):
        _mount(monkeypatch, _echo_handler)
        app = build_app(
            upstream="https://upstream.invalid", api_key="k",
            api_key_header="x-api-key", pipeline=pipeline,
            agent_id_default="claude-code@box",
        )
        TestClient(app).post("/v1/messages", json={
            "model": "m", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-agent-id": "named"})
        (req,) = _records(pipeline, EventType.ACTION_REQUESTED)
        assert req.agent_id == "named"


class TestUpstreamTimeouts:

    def test_connect_and_write_are_bounded_read_is_not(self):
        """A dead upstream must fail, a slow generation must not."""
        import inspect
        import re

        from vaara.integrations import _llm_proxy_app
        src = inspect.getsource(_llm_proxy_app)
        m = re.search(r"httpx\.Timeout\((.*?)\)", src)
        assert m, "no httpx.Timeout construction found"
        args = m.group(1)
        assert "connect=10.0" in args
        assert "write=30.0" in args
        assert "read=None" in args
