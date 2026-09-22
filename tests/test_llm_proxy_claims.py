"""`vaara llm-proxy --help` must say what the proxy does and no more.

Until 1.94.0 the help and the module docstring said the proxy "strips
secrets" and records "everything". It records two paths, and it holds back
only the strings an operator lists in --seal-file. Every other path is
forwarded with no record, and --seal-file is not applied to it: a sealed
secret sent to /v1/responses leaves in the clear.

These tests pin the text to the behaviour in both directions. If the
pass-through starts recording or sealing, the behaviour tests fail and the
help has to be rewritten to claim the wider coverage.
"""

from __future__ import annotations

import contextlib
import io
import re

import pytest

httpx = pytest.importorskip("httpx", reason="proxy deps not installed: no httpx")
pytest.importorskip("fastapi", reason="proxy deps not installed: no fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from vaara import cli  # noqa: E402
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

    def test_other_paths_are_declared_unrecorded_and_unsealed(self):
        assert "Every other path is forwarded unrecorded and unsealed" \
            in DESCRIPTION

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

    def test_pass_through_writes_no_record(self, sent, app_and_pipeline):
        app, pipeline = app_and_pipeline
        before = pipeline.trail.size
        TestClient(app).post("/v1/responses", json={"input": "hello"})
        assert len(sent) == 1
        assert pipeline.trail.size == before

    def test_pass_through_is_not_sealed(self, sent, app_and_pipeline):
        app, _ = app_and_pipeline
        TestClient(app).post("/v1/responses", json={"input": SECRET})
        assert SECRET.encode() in sent[0]
