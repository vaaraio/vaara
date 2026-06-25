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


def _authzs(receipts_dir: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(receipts_dir.glob("*-authz.json"))]


def test_allowed_call_emits_recomputable_authorization_receipt(
    monkeypatch, emitter, receipts_dir
):
    """Proof-carrying enforcement: the allow-path mints a signed, recomputable
    authorization receipt an auditor confirms from the issuer key alone."""
    import hashlib

    from vaara.attestation._sep2787_canonical import canonical_json
    from vaara.attestation.decision import (
        parse_decision_record,
        verify_decision_signature,
    )

    p = _make_proxy(monkeypatch, emitter=emitter, mint=True)
    p._emit_authorization_receipts = True
    args = {"path": "/tmp/x"}
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "read_file", "arguments": args},
    })

    authzs = _authzs(receipts_dir)
    assert len(authzs) == 1
    bundle = authzs[0]
    record = parse_decision_record(bundle["record"])
    evidence = bundle["evidence"]

    # The broker's allow-proof: the decision is allow.
    assert record.decision_derived.decision == "allow"
    assert evidence["verdict"] == "allow"
    # Signed; verifies against the issuer key with zero trust in the producer.
    assert verify_decision_signature(record, verifying_material=KEY)
    # Content-addressed: the receipt pins the evidence by its JCS digest, and the
    # evidence bytes travel in the same file so the proof is self-contained.
    evidence_digest = "sha256:" + hashlib.sha256(canonical_json(evidence)).hexdigest()
    assert record.decision_derived.evidence_ref.digest == evidence_digest
    # Only the argument commitment travels; the raw arguments never do.
    assert args not in evidence.values()
    assert evidence["argsCommitment"] == (
        "sha256:" + hashlib.sha256(canonical_json(args)).hexdigest()
    )
    # Coverage: the decision names the observation boundary in the trace, so an
    # absent deny reads against a stated scope, not an assumed one. The server
    # fingerprint pins the exact capability surface the proxy decided over.
    coverage = evidence["coverage"]
    assert coverage["boundary"] == "vaara-mcp-proxy"
    assert coverage["scope"] == "calls-routed-through-chokepoint"
    assert coverage["serverFingerprint"] == emitter.fingerprint_for("default")
    # The coverage block is under the signature: it is part of what the digest
    # pins, so a reader cannot strip or rewrite the boundary without breaking it.
    assert verify_decision_signature(record, verifying_material=KEY)
    # The allow-proof pairs with exactly one grant and one attestation.
    assert len(_grants(receipts_dir)) == 1
    assert len(_attests(receipts_dir)) == 1


def test_authorization_receipt_off_by_default(monkeypatch, emitter, receipts_dir):
    """Credential minting on, proof flag off: a grant but no allow-proof."""
    p = _make_proxy(monkeypatch, emitter=emitter, mint=True)
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "read_file", "arguments": {"path": "/tmp/x"}},
    })
    assert len(_grants(receipts_dir)) == 1
    assert _authzs(receipts_dir) == []


def test_tool_constraints_minted_as_capabilities(monkeypatch, tmp_path, receipts_dir):
    """Capabilities from tool_constraints appear in the grant for the matching tool."""
    from vaara.integrations._mcp_attest import build_attest_emitter

    key = tmp_path / "attest.key"
    key.write_bytes(KEY)
    cfg = tmp_path / "constraints.json"
    cfg.write_text(json.dumps({
        "tools": {
            "read_file": [
                {"arg": "path", "op": "in", "value": ["/tmp", "/var/data"]},
            ]
        }
    }))
    em = build_attest_emitter(
        signing_key_path=key,
        receipts_dir=receipts_dir,
        upstream_commands={"default": ["echo"]},
        tool_constraints_path=cfg,
    )
    p = _make_proxy(monkeypatch, emitter=em, mint=True)
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "read_file", "arguments": {"path": "/tmp/x"}},
    })
    grants = _grants(receipts_dir)
    assert len(grants) == 1
    caps = grants[0].get("capabilities", [])
    assert caps == [{"arg": "path", "op": "in", "value": ["/tmp", "/var/data"]}]


def test_gateway_passes_constrained_tool_on_valid_grant(monkeypatch, tmp_path, receipts_dir):
    """Gateway built for constrained tool; valid grant passes, upstream is called."""
    from vaara.integrations._mcp_attest import build_attest_emitter

    key = tmp_path / "attest.key"
    key.write_bytes(KEY)
    cfg = tmp_path / "constraints.json"
    cfg.write_text(json.dumps({
        "tools": {"read_file": [{"arg": "path", "op": "in", "value": ["/tmp"]}]}
    }))
    em = build_attest_emitter(
        signing_key_path=key,
        receipts_dir=receipts_dir,
        upstream_commands={"default": ["echo"]},
        tool_constraints_path=cfg,
    )
    assert em.gateway is not None
    p = _make_proxy(monkeypatch, emitter=em, mint=True)
    resp = p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "read_file", "arguments": {"path": "/tmp"}},
    })
    # Gateway passed — upstream was called, no MCP error.
    assert p._upstream.request.called
    assert "error" not in resp


def test_gateway_blocks_constrained_tool_when_grant_missing(monkeypatch, tmp_path, receipts_dir):
    """Gateway fails closed: constrained tool with no credential gets MCP -32603."""
    from unittest.mock import patch

    from vaara.integrations._mcp_attest import build_attest_emitter

    key = tmp_path / "attest.key"
    key.write_bytes(KEY)
    cfg = tmp_path / "constraints.json"
    cfg.write_text(json.dumps({
        "tools": {"read_file": [{"arg": "path", "op": "in", "value": ["/tmp"]}]}
    }))
    em = build_attest_emitter(
        signing_key_path=key,
        receipts_dir=receipts_dir,
        upstream_commands={"default": ["echo"]},
        tool_constraints_path=cfg,
    )
    p = _make_proxy(monkeypatch, emitter=em, mint=True)
    # Simulate emit_grant failure so no credential is injected.
    with patch.object(em, "emit_grant", return_value=None):
        resp = p._handle_tools_call({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "read_file", "arguments": {"path": "/tmp/x"}},
        })
    assert "error" in resp
    assert resp["error"]["code"] == -32603
    assert "read_file" in resp["error"]["message"]
    assert not p._upstream.request.called


def test_unconstrained_tool_gets_no_capabilities(monkeypatch, tmp_path, receipts_dir):
    """A tool not in the constraints map gets an exact-args grant with no capabilities."""
    from vaara.integrations._mcp_attest import build_attest_emitter

    key = tmp_path / "attest.key"
    key.write_bytes(KEY)
    cfg = tmp_path / "constraints.json"
    cfg.write_text(json.dumps({"tools": {"other_tool": [{"arg": "x", "op": "eq", "value": "y"}]}}))
    em = build_attest_emitter(
        signing_key_path=key,
        receipts_dir=receipts_dir,
        upstream_commands={"default": ["echo"]},
        tool_constraints_path=cfg,
    )
    p = _make_proxy(monkeypatch, emitter=em, mint=True)
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "read_file", "arguments": {"path": "/tmp/x"}},
    })
    grants = _grants(receipts_dir)
    assert len(grants) == 1
    assert "capabilities" not in grants[0]
