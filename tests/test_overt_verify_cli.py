"""Tests for ``vaara overt verify`` — OVERT 1.0 Base Envelope reference verifier."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("cbor2")
pytest.importorskip("cryptography")

import cbor2
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaara.attestation import emit_base_envelope
from vaara.attestation.iap import envelope_to_canonical_cbor
from vaara.attestation.overt import (
    encoder_binary_identity,
    make_request_commitment,
)
from vaara.cli import main


def _emit_test_envelope(tmp_path: Path) -> tuple[Path, Path]:
    sk = Ed25519PrivateKey.generate()
    pub_raw = sk.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    op_key = os.urandom(32)
    env = emit_base_envelope(
        signing_key=sk,
        request_commitment=make_request_commitment(
            b"request-bytes", operator_key=op_key,
        ),
        encoder_binary_identity=encoder_binary_identity(
            arbiter_version="vaara-test", policy_hash=b"p" * 32,
        ),
        non_content_metadata={
            "action_class": "tool_call", "decision": "allow",
        },
        monotonic_counter=1,
        arbiter_instance_identifier=os.urandom(16),
    )
    receipt = tmp_path / "env.cbor"
    pubkey = tmp_path / "pub.bin"
    receipt.write_bytes(envelope_to_canonical_cbor(env))
    pubkey.write_bytes(pub_raw)
    return receipt, pubkey


def test_overt_verify_accepts_valid_envelope(tmp_path, capsys):
    receipt, pubkey = _emit_test_envelope(tmp_path)
    rc = main([
        "overt", "verify", str(receipt), "--pubkey-file", str(pubkey),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"valid": true' in out
    assert '"key_identifier":' in out


def test_overt_verify_rejects_wrong_pubkey(tmp_path, capsys):
    receipt, _ = _emit_test_envelope(tmp_path)
    wrong = tmp_path / "wrong.bin"
    wrong.write_bytes(os.urandom(32))
    rc = main([
        "overt", "verify", str(receipt), "--pubkey-file", str(wrong),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "signature verification failed" in err


def test_overt_verify_pubkey_hex_form_works(tmp_path, capsys):
    receipt, pubkey = _emit_test_envelope(tmp_path)
    rc = main([
        "overt", "verify", str(receipt),
        "--pubkey-hex", pubkey.read_bytes().hex(),
    ])
    assert rc == 0


def test_overt_verify_rejects_bad_hex(tmp_path, capsys):
    receipt, _ = _emit_test_envelope(tmp_path)
    rc = main([
        "overt", "verify", str(receipt), "--pubkey-hex", "not-hex",
    ])
    assert rc == 2
    assert "not valid hex" in capsys.readouterr().err


def test_overt_verify_rejects_wrong_pubkey_length(tmp_path, capsys):
    receipt, _ = _emit_test_envelope(tmp_path)
    short = tmp_path / "short.bin"
    short.write_bytes(b"\x00" * 16)
    rc = main([
        "overt", "verify", str(receipt), "--pubkey-file", str(short),
    ])
    assert rc == 2
    assert "32 raw bytes" in capsys.readouterr().err


def test_overt_verify_rejects_missing_receipt(tmp_path, capsys):
    pubkey = tmp_path / "pub.bin"
    pubkey.write_bytes(os.urandom(32))
    rc = main([
        "overt", "verify", str(tmp_path / "nope.cbor"),
        "--pubkey-file", str(pubkey),
    ])
    assert rc == 2
    assert "not a file" in capsys.readouterr().err


def test_overt_verify_rejects_bad_cbor(tmp_path, capsys):
    """Either CBOR decode fails, or it decodes to a non-map. Both to exit 1."""
    bad = tmp_path / "bad.cbor"
    bad.write_bytes(b"not cbor")
    pubkey = tmp_path / "pub.bin"
    pubkey.write_bytes(os.urandom(32))
    rc = main([
        "overt", "verify", str(bad), "--pubkey-file", str(pubkey),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "CBOR" in err or "decode" in err or "must decode to a map" in err


def test_overt_verify_rejects_envelope_missing_required_field(
    tmp_path, capsys,
):
    receipt, pubkey = _emit_test_envelope(tmp_path)
    decoded = cbor2.loads(receipt.read_bytes())
    decoded.pop("signature")
    receipt.write_bytes(cbor2.dumps(decoded, canonical=True))
    rc = main([
        "overt", "verify", str(receipt), "--pubkey-file", str(pubkey),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "missing required fields" in err
    assert "signature" in err


def test_overt_verify_rejects_envelope_with_extra_field(tmp_path, capsys):
    """OVERT 1.0 schema is closed. Unknown fields must be rejected."""
    receipt, pubkey = _emit_test_envelope(tmp_path)
    decoded = cbor2.loads(receipt.read_bytes())
    decoded["smuggled_field"] = b"x"
    receipt.write_bytes(cbor2.dumps(decoded, canonical=True))
    rc = main([
        "overt", "verify", str(receipt), "--pubkey-file", str(pubkey),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown fields" in err
    assert "smuggled_field" in err


def test_overt_verify_rejects_tampered_signature(tmp_path, capsys):
    receipt, pubkey = _emit_test_envelope(tmp_path)
    decoded = cbor2.loads(receipt.read_bytes())
    sig = bytearray(decoded["signature"])
    sig[0] ^= 0xFF
    decoded["signature"] = bytes(sig)
    receipt.write_bytes(cbor2.dumps(decoded, canonical=True))
    rc = main([
        "overt", "verify", str(receipt), "--pubkey-file", str(pubkey),
    ])
    assert rc == 1
    assert "signature verification failed" in capsys.readouterr().err
