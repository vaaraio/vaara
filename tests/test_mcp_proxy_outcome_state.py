# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The MCP proxy's trail says what it saw of an allowed call, and no more.

An allowed call that raised on its way to the upstream used to look the same
three ways: the request never left the proxy, the request left and no reply
was read, and the server answered with a failure. The execution receipt said
``errored`` for all three, and the trail wrote no outcome at all for any of
them. The trail now closes the action with an outcome whose description names
which of the three it was. The receipt says ``unobserved`` for the one case
where the proxy cannot say whether the call ran (receipt -15, Table 4);
``not_sent`` and ``failed`` stay ``errored``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vaara.audit.trail import AuditTrail, EventType
from vaara.integrations import mcp_proxy
from vaara.integrations._mcp_upstream import (
    ProxyError,
    UpstreamFailed,
    UpstreamNotSent,
)
from vaara.pipeline import InterceptionPipeline

_CALL = {
    "jsonrpc": "2.0", "id": 1, "method": "tools/call",
    "params": {"name": "read_file", "arguments": {"path": "/tmp/x"}},
}


def _make_proxy(monkeypatch, **kwargs):
    trail = AuditTrail(on_record=lambda _r: None)
    pipeline = InterceptionPipeline(trail=trail)
    monkeypatch.setattr(
        "vaara.integrations._mcp_upstream.UpstreamMCPClient.__init__",
        lambda self, command, **kw: None,
    )
    p = mcp_proxy.VaaraMCPProxy(
        upstream_command=["echo"], pipeline=pipeline, **kwargs,
    )
    p._upstream = MagicMock()
    p._upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1,
        "result": {"content": [{"type": "text", "text": "ok"}]},
    }
    return p, pipeline


def _outcomes(pipeline):
    return pipeline.trail.get_records_by_type(EventType.OUTCOME_RECORDED)


@pytest.mark.parametrize("raised, status, severity", [
    (UpstreamNotSent("closed"), "not_sent", 0.5),
    (ProxyError("reply timed out"), "unobserved", 0.5),
    (UpstreamFailed("HTTP 500"), "failed", 1.0),
])
def test_a_call_that_raised_is_closed_with_the_state_the_proxy_can_name(
    monkeypatch, raised, status, severity,
):
    p, pipeline = _make_proxy(monkeypatch)
    p._upstream.request.side_effect = raised
    with pytest.raises(ProxyError):
        p._handle_tools_call(_CALL)
    (out,) = _outcomes(pipeline)
    assert json.loads(out.data["description"])["status"] == status
    assert out.data["outcome_severity"] == severity


def test_an_error_the_proxy_cannot_classify_is_unobserved(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch)
    p._upstream.request.side_effect = RuntimeError("reader thread died")
    with pytest.raises(RuntimeError):
        p._handle_tools_call(_CALL)
    (out,) = _outcomes(pipeline)
    assert json.loads(out.data["description"])["status"] == "unobserved"


def test_the_description_names_the_error_class_and_not_its_message(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch)
    p._upstream.request.side_effect = UpstreamFailed("HTTP 401: token sk-live-123")
    with pytest.raises(ProxyError):
        p._handle_tools_call(_CALL)
    (out,) = _outcomes(pipeline)
    assert "sk-live-123" not in out.data["description"]
    assert json.loads(out.data["description"])["error"] == "UpstreamFailed"


def test_an_error_reply_is_recorded_as_a_failure_the_proxy_saw(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch)
    p._upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1,
        "error": {"code": -32000, "message": "tool failed"},
    }
    p._handle_tools_call(_CALL)
    (out,) = _outcomes(pipeline)
    assert json.loads(out.data["description"])["status"] == "failed"
    assert out.data["outcome_severity"] == 1.0


def test_a_tool_error_result_is_recorded_as_a_failure_the_proxy_saw(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch)
    p._upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1, "result": {"isError": True, "content": []},
    }
    p._handle_tools_call(_CALL)
    (out,) = _outcomes(pipeline)
    assert json.loads(out.data["description"])["status"] == "failed"


def test_a_clean_reply_records_the_outcome_as_before(monkeypatch):
    p, pipeline = _make_proxy(monkeypatch)
    p._handle_tools_call(_CALL)
    (out,) = _outcomes(pipeline)
    assert out.data["outcome_severity"] == 0.0
    assert out.data["description"] == ""


# -- the receipt names an unobserved outcome --------------------------------

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")


@pytest.fixture
def receipts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "attest_receipts"
    d.mkdir()
    return d


@pytest.fixture
def emitter(tmp_path: Path, receipts_dir: Path):
    from vaara.integrations._mcp_attest import build_attest_emitter

    key = tmp_path / "attest.key"
    key.write_bytes(b"x" * 32)
    return build_attest_emitter(
        signing_key_path=key,
        receipts_dir=receipts_dir,
        upstream_commands={"default": ["echo"]},
    )


def _only_receipt(receipts_dir: Path) -> dict:
    (receipt,) = [
        json.loads(f.read_text(encoding="utf-8"))
        for f in sorted(receipts_dir.glob("*-receipt.json"))
    ]
    return receipt


@pytest.mark.parametrize("raised, status", [
    (UpstreamNotSent("closed"), "errored"),
    (ProxyError("reply timed out"), "unobserved"),
    (RuntimeError("reader thread died"), "unobserved"),
    (UpstreamFailed("HTTP 500"), "errored"),
])
def test_the_receipt_status_follows_what_the_proxy_saw(
    monkeypatch, emitter, receipts_dir, raised, status,
):
    p, _ = _make_proxy(monkeypatch, attest_emitter=emitter)
    p._upstream.request.side_effect = raised
    with pytest.raises(type(raised)):
        p._handle_tools_call(_CALL)
    od = _only_receipt(receipts_dir)["outcomeDerived"]
    assert od["status"] == status
    assert "resultCommitment" not in od


def test_a_clean_reply_still_reads_executed(monkeypatch, emitter, receipts_dir):
    p, _ = _make_proxy(monkeypatch, attest_emitter=emitter)
    p._handle_tools_call(_CALL)
    assert _only_receipt(receipts_dir)["outcomeDerived"]["status"] == "executed"


def test_an_error_reply_still_reads_errored(monkeypatch, emitter, receipts_dir):
    p, _ = _make_proxy(monkeypatch, attest_emitter=emitter)
    p._upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "no"},
    }
    p._handle_tools_call(_CALL)
    assert _only_receipt(receipts_dir)["outcomeDerived"]["status"] == "errored"
