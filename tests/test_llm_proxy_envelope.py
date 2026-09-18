"""The envelope meter and the marker watch on the llm-proxy record.

Every prompt record says how many bytes left, which part of the request they
came from, and which watched markers were inside them. Sizes and marker ids
are not content: a reader of the trail learns the shape of what left and
nothing of what it said.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from vaara.audit.trail import EventType  # noqa: E402
from vaara.integrations._llm_proxy_app import build_app  # noqa: E402
from vaara.integrations.llm_envelope import (  # noqa: E402
    MarkerWatch,
    measure_envelope,
)
from vaara.integrations.llm_proxy import _build_pipeline  # noqa: E402

import httpx  # noqa: E402


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
    return httpx.Response(
        200,
        json={"content": [{"type": "text", "text": "ok"}]},
        headers={"content-type": "application/json"},
    )


class TestMeasureEnvelope:

    def test_anthropic_shape_splits_system_tools_messages(self):
        body = {
            "model": "m",
            "system": [{"type": "text", "text": "S" * 100}],
            "tools": [{"name": "t", "description": "D" * 50}],
            "messages": [
                {"role": "user", "content": "first " + "a" * 20},
                {"role": "assistant", "content": "b" * 30},
                {"role": "user", "content": [{"type": "text", "text": "ask"}]},
            ],
        }
        out = json.dumps(body).encode()
        env = measure_envelope(body, out)
        assert env["bytes_out"] == len(out)
        assert env["bytes_system"] == 100
        assert env["bytes_tools"] == len(json.dumps(body["tools"]))
        assert env["bytes_messages"] == 26 + 30 + 3
        assert env["bytes_last_user"] == 3
        assert env["message_count"] == 3

    def test_openai_shape_counts_system_role_as_system(self):
        body = {
            "model": "m",
            "messages": [
                {"role": "system", "content": "S" * 40},
                {"role": "user", "content": "hello"},
            ],
        }
        env = measure_envelope(body, json.dumps(body).encode())
        assert env["bytes_system"] == 40
        assert env["bytes_messages"] == 5
        assert env["bytes_last_user"] == 5
        assert env["message_count"] == 1

    def test_string_system_prompt(self):
        body = {"model": "m", "system": "x" * 7, "messages": []}
        env = measure_envelope(body, b"{}")
        assert env["bytes_system"] == 7
        assert env["bytes_last_user"] == 0


class TestMarkerWatch:

    def test_reports_ids_present_never_strings(self, tmp_path):
        f = tmp_path / "markers.json"
        f.write_text(json.dumps({"A": "zebra-quartz-17", "B": "moon-ladle-4"}))
        w = MarkerWatch.from_file(str(f))
        found = w.present(b"the zebra-quartz-17 left twice zebra-quartz-17")
        assert found == ["A"]
        assert "zebra-quartz-17" not in json.dumps(found)

    def test_empty_watch_is_inactive(self):
        assert MarkerWatch().present(b"anything") == []
        assert not MarkerWatch().active

    def test_refresh_picks_up_new_markers(self, tmp_path):
        f = tmp_path / "markers.json"
        f.write_text(json.dumps({"A": "one"}))
        w = MarkerWatch.from_file(str(f))
        assert w.present(b"two") == []
        f.write_text(json.dumps({"A": "one", "B": "two"}))
        w.refresh()
        assert w.present(b"two") == ["B"]


@pytest.fixture
def pipeline(tmp_path):
    return _build_pipeline(tmp_path / "trail.db")


class TestRecordCarriesEnvelopeAndMarkers:

    def _app(self, pipeline, **kw):
        return build_app(
            upstream="https://upstream.invalid", api_key="k",
            api_key_header="x-api-key", pipeline=pipeline, **kw)

    def test_meta_level_record_carries_sizes(self, monkeypatch, pipeline):
        _mount(monkeypatch, _echo_handler)
        client = TestClient(self._app(pipeline, audit_level="meta"))
        client.post("/v1/messages", json={
            "model": "claude-opus-5",
            "system": "S" * 10,
            "messages": [{"role": "user", "content": "hello there"}],
        })
        (req,) = _records(pipeline, EventType.ACTION_REQUESTED)
        env = req.data["parameters"]["envelope"]
        assert env["bytes_out"] > 0
        assert env["bytes_system"] == 10
        assert env["bytes_last_user"] == 11
        # sizes only: the record never carries the text
        assert "hello there" not in json.dumps(req.data)

    def test_markers_present_recorded_by_id(self, monkeypatch, pipeline):
        _mount(monkeypatch, _echo_handler)
        watch = MarkerWatch({"A": "zebra-quartz-17", "B": "moon-ladle-4"})
        client = TestClient(self._app(pipeline, marker_watch=watch))
        client.post("/v1/messages", json={
            "model": "claude-opus-5",
            "messages": [{"role": "user",
                          "content": "please keep zebra-quartz-17 in mind"}],
        })
        (req,) = _records(pipeline, EventType.ACTION_REQUESTED)
        params = req.data["parameters"]
        assert params["markers_present"] == ["A"]
        assert "zebra-quartz-17" not in json.dumps(req.data)

    def test_no_watch_means_empty_list(self, monkeypatch, pipeline):
        _mount(monkeypatch, _echo_handler)
        client = TestClient(self._app(pipeline))
        client.post("/v1/messages", json={
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "x"}],
        })
        (req,) = _records(pipeline, EventType.ACTION_REQUESTED)
        assert req.data["parameters"]["markers_present"] == []

    def test_markers_are_checked_on_the_bytes_that_left(
            self, monkeypatch, pipeline):
        """A marker the seal replaced never left, so it is not present."""
        from vaara.integrations.llm_seal import SealRegistry
        _mount(monkeypatch, _echo_handler)
        watch = MarkerWatch({"A": "zebra-quartz-17"})
        seal = SealRegistry({"canary": "zebra-quartz-17"})
        client = TestClient(self._app(
            pipeline, marker_watch=watch, seal_registry=seal))
        client.post("/v1/messages", json={
            "model": "claude-opus-5",
            "messages": [{"role": "user", "content": "zebra-quartz-17"}],
        })
        (req,) = _records(pipeline, EventType.ACTION_REQUESTED)
        params = req.data["parameters"]
        assert params["seal_count"] == 1
        assert params["markers_present"] == []
