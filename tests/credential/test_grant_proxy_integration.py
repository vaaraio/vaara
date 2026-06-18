"""The proxy mints + injects a bound credential on an allowed tools/call."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

KEY = b"x" * 32


@pytest.fixture
def receipts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "receipts"
    d.mkdir()
    return d


@pytest.fixture
def emitter(tmp_path: Path, receipts_dir: Path) -> Any:
    from vaara.integrations._mcp_attest import build_attest_emitter

    key = tmp_path / "attest.key"
    key.write_bytes(KEY)
    return build_attest_emitter(
        signing_key_path=key,
        receipts_dir=receipts_dir,
        upstream_commands={"default": ["echo"]},
    )


def _make_proxy(monkeypatch, *, emitter, mint: bool, **kwargs):
    from vaara.audit.trail import AuditTrail
    from vaara.integrations import mcp_proxy
    from vaara.pipeline import InterceptionPipeline

    trail = AuditTrail(on_record=lambda _r: None)
    pipeline = InterceptionPipeline(trail=trail)
    monkeypatch.setattr(
        "vaara.integrations._mcp_upstream.UpstreamMCPClient.__init__",
        lambda self, command, **kw: None,
    )
    p = mcp_proxy.VaaraMCPProxy(
        upstream_command=["echo"], pipeline=pipeline, attest_emitter=emitter, **kwargs
    )
    p._mint_credentials = mint
    mock_upstream = MagicMock()
    mock_upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1,
        "result": {"content": [{"type": "text", "text": "ok"}]},
    }
    p._upstream = mock_upstream
    return p


def _grants(receipts_dir: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(receipts_dir.glob("*-grant.json"))]


def _attests(receipts_dir: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(receipts_dir.glob("*-attest.json"))]


def test_allowed_call_mints_and_injects_grant(monkeypatch, emitter, receipts_dir):
    from vaara.attestation.receipt import attestation_digest
    from vaara.attestation.sep2787 import parse_attestation

    p = _make_proxy(monkeypatch, emitter=emitter, mint=True)
    request = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "read_file", "arguments": {"path": "/tmp/x"}},
    }
    p._handle_tools_call(request)

    grants = _grants(receipts_dir)
    assert len(grants) == 1
    grant = grants[0]

    # The grant binds the digest of the paired attestation.
    att = parse_attestation(_attests(receipts_dir)[0])
    assert grant["binding"]["attestationDigest"] == attestation_digest(att)

    # The credential is injected on the forwarded request payload.
    forwarded = p._upstream.request.call_args.args[0]
    assert forwarded["params"]["_meta"]["vaara/credential"] == grant


def test_mint_disabled_injects_nothing(monkeypatch, emitter, receipts_dir):
    p = _make_proxy(monkeypatch, emitter=emitter, mint=False)
    request = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "read_file", "arguments": {"path": "/tmp/x"}},
    }
    p._handle_tools_call(request)
    assert _grants(receipts_dir) == []
    forwarded = p._upstream.request.call_args.args[0]
    assert "_meta" not in forwarded.get("params", {})


def test_blocked_call_mints_no_grant(monkeypatch, emitter, receipts_dir):
    p = _make_proxy(monkeypatch, emitter=emitter, mint=True, denylist={"delete_db"})
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "delete_db", "arguments": {}},
    })
    assert _grants(receipts_dir) == []
