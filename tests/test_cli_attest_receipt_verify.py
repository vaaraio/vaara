"""Tests for ``vaara keygen --attest``, ``vaara attest verify`` and
``vaara receipt verify`` — the v0.44 reference-verifier CLI surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("rfc8785")

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaara.attestation.receipt import (
    OutcomeDerived,
    emit_receipt,
    make_back_link,
    make_result_digest,
)
from vaara.attestation.sep2787 import (
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)
from vaara.cli import main

_ISS = "vaara-mcp-proxy"
_SUB = "tenantA/files"
_ARGS = {"path": "/tmp/x"}


def _write_pubkey(tmp_path: Path, key) -> Path:
    pub_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    p = tmp_path / "pub.pem"
    p.write_bytes(pub_pem)
    return p


def _emit_attestation(key, *, alg: str = "ES256", exp_seconds: int = 300, iat=None):
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="tools/call/read_file"),
        payload_derived=PayloadDerived(
            tool_calls=(
                ToolCallBinding(
                    name="read_file",
                    server_fingerprint="cmd:sha256:abc",
                    args=make_args_digest(_ARGS),
                ),
            ),
        ),
        iss=_ISS,
        sub=_SUB,
        secret_version="deadbeef",
        alg=alg,
        signing_material=key,
        exp_seconds=exp_seconds,
        iat=iat,
    )


def _emit_receipt(key, attestation, *, alg="ES256", result_commitment=None):
    return emit_receipt(
        back_link=make_back_link(attestation),
        outcome_derived=OutcomeDerived(
            status="executed",
            completed_at="2026-05-30T00:00:00Z",
            result_commitment=result_commitment,
        ),
        iss=_ISS,
        sub=_SUB,
        secret_version="deadbeef",
        alg=alg,
        signing_material=key,
    )


def _es256_pair(tmp_path, **att_kwargs):
    key = ec.generate_private_key(ec.SECP256R1())
    pub = _write_pubkey(tmp_path, key)
    att = _emit_attestation(key, **att_kwargs)
    receipt = _emit_receipt(key, att)
    att_path = tmp_path / "attest.json"
    receipt_path = tmp_path / "receipt.json"
    att_path.write_text(json.dumps(att.to_dict(), indent=2))
    receipt_path.write_text(json.dumps(receipt.to_dict(), indent=2))
    return key, pub, att, att_path, receipt_path


# --- attest verify --------------------------------------------------------


def test_attest_verify_accepts_valid_es256(tmp_path, capsys):
    _, pub, _, att_path, _ = _es256_pair(tmp_path)
    rc = main(["attest", "verify", str(att_path), "--pubkey-file", str(pub)])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"valid": true' in out
    assert '"ttl_expired": false' in out
    assert '"intent": "tools/call/read_file"' in out


def test_attest_verify_rejects_wrong_pubkey(tmp_path, capsys):
    _, _, _, att_path, _ = _es256_pair(tmp_path)
    wrong = ec.generate_private_key(ec.SECP256R1())
    wrong_pub = tmp_path / "wrong.pem"
    wrong_pub.write_bytes(
        wrong.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    rc = main(["attest", "verify", str(att_path), "--pubkey-file", str(wrong_pub)])
    assert rc == 1
    assert "signature verification failed" in capsys.readouterr().err


def test_attest_verify_alg_material_mismatch(tmp_path, capsys):
    """An HS256 secret offered for an ES256 envelope is rejected up front."""
    _, _, _, att_path, _ = _es256_pair(tmp_path)
    secret = tmp_path / "secret.bin"
    secret.write_bytes(b"x" * 32)
    rc = main([
        "attest", "verify", str(att_path), "--hs256-secret-file", str(secret),
    ])
    assert rc == 2
    assert "alg is ES256" in capsys.readouterr().err


def test_attest_verify_hs256_secret(tmp_path, capsys):
    secret = b"k" * 32
    secret_path = tmp_path / "secret.bin"
    secret_path.write_bytes(secret)
    att = _emit_attestation(secret, alg="HS256")
    att_path = tmp_path / "attest.json"
    att_path.write_text(json.dumps(att.to_dict(), indent=2))
    rc = main([
        "attest", "verify", str(att_path),
        "--hs256-secret-file", str(secret_path),
    ])
    assert rc == 0
    assert '"valid": true' in capsys.readouterr().out


def test_attest_verify_ttl_reported_not_enforced_by_default(tmp_path, capsys):
    _, pub, _, att_path, _ = _es256_pair(
        tmp_path, exp_seconds=1, iat="2020-01-01T00:00:00Z",
    )
    rc = main(["attest", "verify", str(att_path), "--pubkey-file", str(pub)])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"valid": true' in out
    assert '"ttl_expired": true' in out


def test_attest_verify_enforce_ttl_fails_on_expired(tmp_path, capsys):
    _, pub, _, att_path, _ = _es256_pair(
        tmp_path, exp_seconds=1, iat="2020-01-01T00:00:00Z",
    )
    rc = main([
        "attest", "verify", str(att_path), "--pubkey-file", str(pub),
        "--enforce-ttl",
    ])
    assert rc == 1
    assert "TTL has expired" in capsys.readouterr().err


def test_attest_verify_missing_file(tmp_path, capsys):
    pub = tmp_path / "pub.pem"
    pub.write_bytes(b"x")
    rc = main([
        "attest", "verify", str(tmp_path / "nope.json"),
        "--pubkey-file", str(pub),
    ])
    assert rc == 2
    assert "not a file" in capsys.readouterr().err


def test_attest_verify_bad_json(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    pub = tmp_path / "pub.pem"
    pub.write_bytes(b"x")
    rc = main(["attest", "verify", str(bad), "--pubkey-file", str(pub)])
    assert rc == 1
    assert "cannot read JSON" in capsys.readouterr().err


def test_attest_verify_valid_json_but_not_attestation(tmp_path, capsys):
    notatt = tmp_path / "x.json"
    notatt.write_text(json.dumps({"hello": "world"}))
    pub = tmp_path / "pub.pem"
    pub.write_bytes(b"x")
    rc = main(["attest", "verify", str(notatt), "--pubkey-file", str(pub)])
    assert rc == 1
    assert "not a valid SEP-2787 attestation" in capsys.readouterr().err


# --- receipt verify -------------------------------------------------------


def test_receipt_verify_accepts_valid_pair(tmp_path, capsys):
    _, pub, _, att_path, receipt_path = _es256_pair(tmp_path)
    rc = main([
        "receipt", "verify", str(receipt_path),
        "--attestation", str(att_path), "--pubkey-file", str(pub),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"receipt_signature_valid": true' in out
    assert '"attestation_signature_valid": true' in out
    assert '"back_link_valid": true' in out
    assert '"result_commitment_valid": null' in out


def test_receipt_verify_detects_broken_backlink(tmp_path, capsys):
    key, pub, att, att_path, receipt_path = _es256_pair(tmp_path)
    # A receipt that signs cleanly but is checked against a different
    # attestation: the back-link must fail.
    other_att = _emit_attestation(key)
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps(other_att.to_dict(), indent=2))
    rc = main([
        "receipt", "verify", str(receipt_path),
        "--attestation", str(other_path), "--pubkey-file", str(pub),
    ])
    assert rc == 1
    assert '"back_link_valid": false' in capsys.readouterr().out


def test_receipt_verify_wrong_pubkey(tmp_path, capsys):
    _, _, _, att_path, receipt_path = _es256_pair(tmp_path)
    wrong = ec.generate_private_key(ec.SECP256R1())
    wrong_pub = tmp_path / "wrong.pem"
    wrong_pub.write_bytes(
        wrong.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    rc = main([
        "receipt", "verify", str(receipt_path),
        "--attestation", str(att_path), "--pubkey-file", str(wrong_pub),
    ])
    assert rc == 1
    assert '"receipt_signature_valid": false' in capsys.readouterr().out


def test_receipt_verify_result_commitment_matches(tmp_path, capsys):
    key = ec.generate_private_key(ec.SECP256R1())
    pub = _write_pubkey(tmp_path, key)
    att = _emit_attestation(key)
    result_obj = {"content": "file body", "bytes": 9}
    receipt = _emit_receipt(
        key, att, result_commitment=make_result_digest(result_obj),
    )
    att_path = tmp_path / "attest.json"
    receipt_path = tmp_path / "receipt.json"
    result_path = tmp_path / "result.json"
    att_path.write_text(json.dumps(att.to_dict(), indent=2))
    receipt_path.write_text(json.dumps(receipt.to_dict(), indent=2))
    result_path.write_text(json.dumps(result_obj))

    rc = main([
        "receipt", "verify", str(receipt_path),
        "--attestation", str(att_path), "--pubkey-file", str(pub),
        "--result", str(result_path),
    ])
    assert rc == 0
    assert '"result_commitment_valid": true' in capsys.readouterr().out


def test_receipt_verify_result_commitment_mismatch(tmp_path, capsys):
    key = ec.generate_private_key(ec.SECP256R1())
    pub = _write_pubkey(tmp_path, key)
    att = _emit_attestation(key)
    receipt = _emit_receipt(
        key, att, result_commitment=make_result_digest({"content": "real"}),
    )
    att_path = tmp_path / "attest.json"
    receipt_path = tmp_path / "receipt.json"
    result_path = tmp_path / "result.json"
    att_path.write_text(json.dumps(att.to_dict(), indent=2))
    receipt_path.write_text(json.dumps(receipt.to_dict(), indent=2))
    result_path.write_text(json.dumps({"content": "tampered"}))

    rc = main([
        "receipt", "verify", str(receipt_path),
        "--attestation", str(att_path), "--pubkey-file", str(pub),
        "--result", str(result_path),
    ])
    assert rc == 1
    assert '"result_commitment_valid": false' in capsys.readouterr().out


def test_receipt_verify_commitment_present_without_result_flag(tmp_path, capsys):
    key = ec.generate_private_key(ec.SECP256R1())
    pub = _write_pubkey(tmp_path, key)
    att = _emit_attestation(key)
    receipt = _emit_receipt(
        key, att, result_commitment=make_result_digest({"content": "real"}),
    )
    att_path = tmp_path / "attest.json"
    receipt_path = tmp_path / "receipt.json"
    att_path.write_text(json.dumps(att.to_dict(), indent=2))
    receipt_path.write_text(json.dumps(receipt.to_dict(), indent=2))

    rc = main([
        "receipt", "verify", str(receipt_path),
        "--attestation", str(att_path), "--pubkey-file", str(pub),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"result_commitment_valid": null' in out
    assert "pass --result" in out


# --- keygen --attest ------------------------------------------------------


def test_keygen_attest_emits_p256_without_dev(tmp_path, capsys):
    out = tmp_path / "attest_key.pem"
    rc = main(["keygen", "--attest", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert (tmp_path / "attest_key.pem.pub").exists()
    key = serialization.load_pem_private_key(out.read_bytes(), password=None)
    assert isinstance(key, ec.EllipticCurvePrivateKey)
    assert key.curve.name == "secp256r1"
    stdout = capsys.readouterr().out
    assert "secretVersion:" in stdout
    assert "ES256" in stdout


def test_keygen_attest_roundtrips_through_verify(tmp_path, capsys):
    """Key from keygen --attest signs an attestation the verify command accepts."""
    out = tmp_path / "k.pem"
    assert main(["keygen", "--attest", "--out", str(out)]) == 0
    capsys.readouterr()
    key = serialization.load_pem_private_key(out.read_bytes(), password=None)
    att = _emit_attestation(key)
    att_path = tmp_path / "attest.json"
    att_path.write_text(json.dumps(att.to_dict(), indent=2))
    rc = main([
        "attest", "verify", str(att_path),
        "--pubkey-file", str(tmp_path / "k.pem.pub"),
    ])
    assert rc == 0
    assert '"valid": true' in capsys.readouterr().out


def test_keygen_default_still_ed25519_and_requires_dev(tmp_path, capsys):
    out = tmp_path / "ed.pem"
    rc = main(["keygen", "--out", str(out)])
    assert rc == 2  # still gated on --dev
    assert not out.exists()
    rc = main(["keygen", "--dev", "--out", str(out)])
    assert rc == 0
    key = serialization.load_pem_private_key(out.read_bytes(), password=None)
    assert isinstance(key, Ed25519PrivateKey)
