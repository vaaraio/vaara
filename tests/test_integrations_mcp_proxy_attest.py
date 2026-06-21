"""Smoke tests for SEP-2787 attestation + receipt pairing in VaaraMCPProxy.

Mirrors the structure of test_integrations_mcp_proxy_overt.py. Uses HS256
(raw bytes key) for all tests to avoid PEM generation overhead in fixtures;
the signing stack is tested end-to-end in test_execution_receipt.py and the
SEP-2787 vector suite.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Attestation rides the SEP-2787 stack: RFC 8785 canonicalization (rfc8785)
# and the signing backend (cryptography), both in the `attestation` extra.
# Skip the whole module when the extra is absent, matching the sibling
# attestation suites; CI's base test job does not install it.
pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")


@pytest.fixture
def attest_key(tmp_path: Path) -> Path:
    p = tmp_path / "attest.key"
    p.write_bytes(b"x" * 32)
    return p


@pytest.fixture
def attest_receipts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "attest_receipts"
    d.mkdir()
    return d


@pytest.fixture
def emitter(attest_key: Path, attest_receipts_dir: Path) -> Any:
    from vaara.integrations._mcp_attest import build_attest_emitter
    return build_attest_emitter(
        signing_key_path=attest_key,
        receipts_dir=attest_receipts_dir,
        upstream_commands={"default": ["echo"]},
    )


def _make_proxy(monkeypatch, *, emitter, **kwargs):
    from vaara.integrations import mcp_proxy
    from vaara.pipeline import InterceptionPipeline
    from vaara.audit.trail import AuditTrail

    trail = AuditTrail(on_record=lambda _r: None)
    pipeline = InterceptionPipeline(trail=trail)
    monkeypatch.setattr(
        "vaara.integrations._mcp_upstream.UpstreamMCPClient.__init__",
        lambda self, command, **kw: None,
    )
    p = mcp_proxy.VaaraMCPProxy(
        upstream_command=["echo"],
        pipeline=pipeline,
        attest_emitter=emitter,
        **kwargs,
    )
    mock_upstream = MagicMock()
    mock_upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1,
        "result": {"content": [{"type": "text", "text": "ok"}]},
    }
    p._upstream = mock_upstream
    return p, pipeline


def _attests(receipts_dir: Path) -> list[dict]:
    return [
        json.loads(f.read_text())
        for f in sorted(receipts_dir.glob("*-attest.json"))
    ]


def _receipts(receipts_dir: Path) -> list[dict]:
    return [
        json.loads(f.read_text())
        for f in sorted(receipts_dir.glob("*-receipt.json"))
    ]


# ---------------------------------------------------------------------------
# Basic on/off
# ---------------------------------------------------------------------------

def test_attest_disabled_by_default(monkeypatch):
    from vaara.integrations import mcp_proxy
    from vaara.pipeline import InterceptionPipeline
    from vaara.audit.trail import AuditTrail
    monkeypatch.setattr(
        "vaara.integrations._mcp_upstream.UpstreamMCPClient.__init__",
        lambda self, command, **kw: None,
    )
    trail = AuditTrail(on_record=lambda _r: None)
    pipeline = InterceptionPipeline(trail=trail)
    p = mcp_proxy.VaaraMCPProxy(upstream_command=["echo"], pipeline=pipeline)
    assert p._attest is None


def test_allowed_tool_call_writes_attest_and_receipt_pair(
    monkeypatch, emitter, attest_receipts_dir
):
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": "read_file", "arguments": {"path": "/tmp/x"}},
    })
    attests = _attests(attest_receipts_dir)
    receipts = _receipts(attest_receipts_dir)
    assert len(attests) == 1
    assert len(receipts) == 1


def test_pair_files_share_nonce_prefix(monkeypatch, emitter, attest_receipts_dir):
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": "search", "arguments": {}},
    })
    attest_files = sorted(attest_receipts_dir.glob("*-attest.json"))
    receipt_files = sorted(attest_receipts_dir.glob("*-receipt.json"))
    assert len(attest_files) == 1 and len(receipt_files) == 1
    # Both filenames share the same {counter:010d}-{nonce[:8]} prefix.
    assert attest_files[0].name[:19] == receipt_files[0].name[:19]


# ---------------------------------------------------------------------------
# Cryptographic correctness
# ---------------------------------------------------------------------------

def test_attestation_signature_verifies(monkeypatch, emitter, attest_receipts_dir, attest_key):
    from vaara.attestation.sep2787 import verify_attestation, parse_attestation
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": "write_file", "arguments": {"path": "/tmp/y", "content": "hi"}},
    })
    raw = _attests(attest_receipts_dir)[0]
    attestation = parse_attestation(raw)
    signing_material = attest_key.read_bytes()
    assert verify_attestation(attestation, verifying_material=signing_material)


def test_receipt_back_link_valid(monkeypatch, emitter, attest_receipts_dir, attest_key):
    from vaara.attestation.sep2787 import parse_attestation
    from vaara.attestation.receipt import parse_receipt, verify_back_link
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": "list_dir", "arguments": {"path": "/tmp"}},
    })
    attestation = parse_attestation(_attests(attest_receipts_dir)[0])
    receipt = parse_receipt(_receipts(attest_receipts_dir)[0])
    result = verify_back_link(receipt, attestation=attestation)
    assert result.ok


def test_receipt_signature_verifies(monkeypatch, emitter, attest_receipts_dir, attest_key):
    from vaara.attestation.receipt import parse_receipt, verify_receipt_signature
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": "run_cmd", "arguments": {}},
    })
    receipt = parse_receipt(_receipts(attest_receipts_dir)[0])
    signing_material = attest_key.read_bytes()
    assert verify_receipt_signature(receipt, verifying_material=signing_material)


def test_successful_outcome_maps_to_executed(monkeypatch, emitter, attest_receipts_dir):
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": "ping", "arguments": {}},
    })
    receipt = _receipts(attest_receipts_dir)[0]
    assert receipt["outcomeDerived"]["status"] == "executed"


def test_upstream_error_maps_to_errored(monkeypatch, emitter, attest_receipts_dir):
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    p._upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "upstream error"},
    }
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": "bad_tool", "arguments": {}},
    })
    receipt = _receipts(attest_receipts_dir)[0]
    assert receipt["outcomeDerived"]["status"] == "errored"


def test_upstream_raise_still_writes_paired_errored_receipt(
    monkeypatch, emitter, attest_receipts_dir
):
    # Transport failure: the upstream raises rather than returning an error
    # response. The attestation is already written, so a paired errored
    # receipt must still be emitted instead of leaving an orphan.
    # _handle_tools_call re-raises; the proxy dispatch turns ProxyError into a
    # client error response one level up.
    from vaara.integrations._mcp_upstream import ProxyError
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    p._upstream.request.side_effect = ProxyError("upstream unavailable")
    with pytest.raises(ProxyError):
        p._handle_tools_call({
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": "flaky_tool", "arguments": {}},
        })
    attest_files = sorted(attest_receipts_dir.glob("*-attest.json"))
    receipt_files = sorted(attest_receipts_dir.glob("*-receipt.json"))
    assert len(attest_files) == 1
    assert len(receipt_files) == 1
    # Pair shares the counter+nonce prefix, and the outcome is errored.
    assert attest_files[0].name[:19] == receipt_files[0].name[:19]
    assert _receipts(attest_receipts_dir)[0]["outcomeDerived"]["status"] == "errored"


# ---------------------------------------------------------------------------
# Attestation content
# ---------------------------------------------------------------------------

def test_attestation_iss_is_vaara_mcp_proxy(monkeypatch, emitter, attest_receipts_dir):
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": "get", "arguments": {}},
    })
    attest = _attests(attest_receipts_dir)[0]
    assert attest["issuerAsserted"]["iss"] == "vaara-mcp-proxy"


def test_attestation_intent_derives_from_tool_name(monkeypatch, emitter, attest_receipts_dir):
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": "fetch_url", "arguments": {}},
    })
    attest = _attests(attest_receipts_dir)[0]
    assert attest["plannerDeclared"]["intent"] == "tools/call/fetch_url"


def test_intent_override_via_request_intent_contextvar(
    monkeypatch, emitter, attest_receipts_dir
):
    from vaara.integrations.mcp_proxy import _REQUEST_INTENT
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    tok = _REQUEST_INTENT.set("customer-service/lookup")
    try:
        p._handle_tools_call({
            "jsonrpc": "2.0", "id": 1,
            "method": "tools/call",
            "params": {"name": "lookup", "arguments": {}},
        })
    finally:
        _REQUEST_INTENT.reset(tok)
    attest = _attests(attest_receipts_dir)[0]
    assert attest["plannerDeclared"]["intent"] == "customer-service/lookup"


def test_perimeter_blocked_tool_call_writes_no_pair(
    monkeypatch, emitter, attest_receipts_dir
):
    # Perimeter denylist blocks before intercept; no attestation should be emitted.
    p, _ = _make_proxy(monkeypatch, emitter=emitter, denylist={"delete_db"})
    p._handle_tools_call({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": "delete_db", "arguments": {}},
    })
    assert len(_attests(attest_receipts_dir)) == 0
    assert len(_receipts(attest_receipts_dir)) == 0


# ---------------------------------------------------------------------------
# Manifest fingerprint
# ---------------------------------------------------------------------------

def test_manifest_fingerprint_upgrades_on_tools_list(
    monkeypatch, emitter, attest_receipts_dir
):
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    # Initially cmd-hash.
    assert emitter.fingerprint_for("default").startswith("cmd:sha256:")
    # Fake a tools/list response coming back through the proxy.
    p._upstream.request.return_value = {
        "jsonrpc": "2.0", "id": 1,
        "result": {"tools": [{"name": "tool_a"}, {"name": "tool_b"}]},
    }
    p._handle_tools_list({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert emitter.fingerprint_for("default").startswith("manifest:sha256:")


def test_manifest_fingerprint_idempotent(monkeypatch, emitter):
    p, _ = _make_proxy(monkeypatch, emitter=emitter)
    tools_response = {
        "jsonrpc": "2.0", "id": 1,
        "result": {"tools": [{"name": "t1"}]},
    }
    p._upstream.request.return_value = tools_response
    p._handle_tools_list({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    first_fp = emitter.fingerprint_for("default")
    p._handle_tools_list({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert emitter.fingerprint_for("default") == first_fp


# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------

def test_build_attest_emitter_rejects_non_p256_ec_key(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from vaara.integrations._mcp_attest import AttestConfigError, build_attest_emitter
    key = ec.generate_private_key(ec.SECP384R1())
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path = tmp_path / "p384.pem"
    key_path.write_bytes(pem)
    with pytest.raises(AttestConfigError, match="secp256r1"):
        build_attest_emitter(
            signing_key_path=key_path,
            receipts_dir=tmp_path / "r",
            upstream_commands={"default": ["echo"]},
        )


def test_build_attest_emitter_accepts_p256_ec_key(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from vaara.integrations._mcp_attest import build_attest_emitter
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    key_path = tmp_path / "p256.pem"
    key_path.write_bytes(pem)
    emitter = build_attest_emitter(
        signing_key_path=key_path,
        receipts_dir=tmp_path / "r",
        upstream_commands={"default": ["echo"]},
    )
    assert emitter.fingerprint_for("default").startswith("cmd:sha256:")


def test_named_single_upstream_fingerprints_under_default(attest_key, attest_receipts_dir):
    # End-to-end of the slot-keying fix: an emitter built the way main() now
    # builds it for a named single upstream returns a real cmd-hash under
    # "default", never the unknown-* placeholder.
    from vaara.integrations._mcp_attest import build_attest_emitter
    from vaara.integrations.mcp_proxy import _attest_upstreams_for_slots
    emitter = build_attest_emitter(
        signing_key_path=attest_key,
        receipts_dir=attest_receipts_dir,
        upstream_commands=_attest_upstreams_for_slots({"github": ["github-mcp-server"]}),
    )
    fp = emitter.fingerprint_for("default")
    assert fp.startswith("cmd:sha256:")
    assert "unknown" not in fp


# ---------------------------------------------------------------------------
# Config error handling
# ---------------------------------------------------------------------------

def test_build_attest_emitter_rejects_missing_key(tmp_path):
    from vaara.integrations._mcp_attest import AttestConfigError, build_attest_emitter
    with pytest.raises(AttestConfigError, match="not found"):
        build_attest_emitter(
            signing_key_path=tmp_path / "nonexistent.key",
            receipts_dir=tmp_path / "r",
            upstream_commands={"default": ["echo"]},
        )


def test_build_attest_emitter_rejects_short_key(tmp_path):
    from vaara.integrations._mcp_attest import AttestConfigError, build_attest_emitter
    short = tmp_path / "short.key"
    short.write_bytes(b"tooshort")
    with pytest.raises(AttestConfigError, match="at least 16 bytes"):
        build_attest_emitter(
            signing_key_path=short,
            receipts_dir=tmp_path / "r",
            upstream_commands={"default": ["echo"]},
        )


# ---------------------------------------------------------------------------
# Completeness sequence (contiguity)
# ---------------------------------------------------------------------------

def test_completeness_seq_is_monotonic_and_gap_free(emitter):
    seen = [emitter._next_completeness("vaara-mcp-proxy") for _ in range(5)]
    assert [s for s, _ in seen] == [0, 1, 2, 3, 4]
    assert [c for _, c in seen] == [1, 2, 3, 4, 5]


def test_completeness_seq_is_scoped_per_boundary(emitter):
    assert emitter._next_completeness("boundary-a") == (0, 1)
    assert emitter._next_completeness("boundary-b") == (0, 1)
    assert emitter._next_completeness("boundary-a") == (1, 2)


def test_completeness_seq_no_dupes_under_concurrency(emitter):
    # Concurrent issuance must never share or skip a number: the assigned set is
    # exactly 0..N-1, which is what makes a missing seq a provable gap.
    import threading

    out: list[int] = []
    lock = threading.Lock()
    n = 200

    def worker() -> None:
        seq, _ = emitter._next_completeness("vaara-mcp-proxy")
        with lock:
            out.append(seq)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(out) == list(range(n))


def test_emitted_authz_receipts_carry_contiguous_completeness(emitter):
    # Two allowed calls under the same boundary get seq 0 then 1 (runningCount
    # 1 then 2), and the completeness boundaryId matches the coverage boundary.
    args = {"path": "/tmp/x"}
    seqs = []
    for _ in range(2):
        att, counter = emitter.emit_attestation(
            tool_name="read_file", arguments=args,
            upstream_name="default", tenant_id="t1",
        )
        cred = emitter.emit_grant(
            attestation=att, counter=counter, tool_name="read_file",
            upstream_name="default", tenant_id="t1",
        )
        auth = emitter.emit_authorization_receipt(
            credential=cred, runtime_args=args, attestation=att, counter=counter,
            upstream_name="default", tenant_id="t1",
        )
        assert auth is not None
        block = auth.evidence["completeness"]
        assert block["boundaryId"] == auth.evidence["coverage"]["boundary"]
        seqs.append((block["seq"], block["runningCount"]))
    assert seqs == [(0, 1), (1, 2)]
