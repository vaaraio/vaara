"""`vaara llm-proxy --help` must say what the proxy does and no more.

Until 1.94.0 the help and the module docstring said the proxy "strips
secrets" and records "everything". It recorded two paths and forwarded every
other one with no record and no sealing, so a --seal-file secret sent to
/v1/responses left in the clear. 1.95.0 records every call and seals every
path.

These tests pin the text to the behaviour. The help names the two paths that
are recorded with their prompt, and the behaviour tests hold the rest: a
pass-through call is recorded by digest, never by content, and is sealed.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re

import pytest

httpx = pytest.importorskip("httpx", reason="proxy deps not installed: no httpx")
pytest.importorskip("fastapi", reason="proxy deps not installed: no fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from vaara import cli  # noqa: E402
from vaara.audit.trail import EventType  # noqa: E402
from vaara.integrations._llm_proxy_app import _CHAT_PATH_LIST, build_app  # noqa: E402
from vaara.integrations.llm_proxy import DESCRIPTION, _build_pipeline  # noqa: E402
from vaara.integrations.llm_seal import SealRegistry  # noqa: E402

SECRET = "northern-lights"


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _help() -> str:
    buf = io.StringIO()
    with contextlib.suppress(SystemExit):
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            cli.main(["llm-proxy", "--help"])
    return _flat(buf.getvalue())


class TestHelpNamesTheEdge:

    def test_help_prints_the_description(self):
        # argparse wraps at hyphens, so compare with all whitespace removed.
        assert re.sub(r"\s", "", DESCRIPTION) in re.sub(r"\s", "", _help())

    @pytest.mark.parametrize("path", _CHAT_PATH_LIST)
    def test_every_governed_path_is_named(self, path):
        assert path in DESCRIPTION

    def test_other_paths_are_declared_recorded_by_hash(self):
        assert "Every other call is recorded by method, path, size and " \
            "sha256, never by content" in DESCRIPTION
        assert "replaced on every path" in DESCRIPTION

    @pytest.mark.parametrize("claim", ["strip secrets", "strips secrets",
                                       "record everything",
                                       "Everything recorded"])
    def test_overclaims_are_gone(self, claim):
        from vaara.integrations import llm_proxy
        assert claim not in _help()
        assert claim not in (llm_proxy.__doc__ or "")


class TestBehaviourMatchesTheHelp:

    @pytest.fixture
    def sent(self, monkeypatch):
        seen: list[bytes] = []
        real = httpx.AsyncClient

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.content)
            return httpx.Response(200, json={"ok": True})

        def fake(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            kwargs.pop("http2", None)
            return real(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", fake)
        return seen

    @pytest.fixture
    def app_and_pipeline(self, tmp_path):
        pipeline = _build_pipeline(tmp_path / "trail.db")
        app = build_app(
            upstream="https://upstream.invalid", api_key="k",
            api_key_header="x-api-key", pipeline=pipeline,
            seal_registry=SealRegistry({"concept": SECRET}),
        )
        return app, pipeline

    def _passthrough_records(self, pipeline):
        return [r for r in pipeline.trail.get_records_by_type(
            EventType.ACTION_REQUESTED) if r.tool_name == "llm.passthrough"]

    def test_pass_through_is_recorded_by_digest(self, sent, app_and_pipeline):
        app, pipeline = app_and_pipeline
        body = b'{"input": "hello there"}'
        TestClient(app).post("/v1/responses", content=body,
                             headers={"content-type": "application/json"})
        assert len(sent) == 1
        (rec,) = self._passthrough_records(pipeline)
        params = rec.data["parameters"]
        assert params["method"] == "POST"
        assert params["path"] == "/v1/responses"
        assert params["bytes_out"] == len(sent[0])
        assert params["sha256_out"] == hashlib.sha256(sent[0]).hexdigest()
        assert "hello there" not in json.dumps(rec.data)

    def test_a_get_is_recorded_too(self, sent, app_and_pipeline):
        app, pipeline = app_and_pipeline
        TestClient(app).get("/v1/models")
        (rec,) = self._passthrough_records(pipeline)
        assert rec.data["parameters"]["method"] == "GET"

    def test_pass_through_is_sealed(self, sent, app_and_pipeline):
        app, pipeline = app_and_pipeline
        TestClient(app).post("/v1/responses", json={"input": SECRET})
        assert SECRET.encode() not in sent[0]
        (rec,) = self._passthrough_records(pipeline)
        assert rec.data["parameters"]["seal_active"] is True
        assert rec.data["parameters"]["seal_count"] == 1
        assert SECRET not in json.dumps(rec.data)
