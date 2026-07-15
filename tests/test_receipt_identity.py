"""Resolvable agent identity (did:web) for execution receipts.

Level-2 pinned-resolvable verification: a receipt whose ``iss`` is a
did:web identity, checked against a DID document the verifier already
holds. See ``docs/design/resolvable-agent-identity-spec.md``.
"""

from __future__ import annotations

import base64
import importlib.util

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives.asymmetric import ec, rsa  # noqa: E402

from vaara.attestation._attest_types import AttestationError  # noqa: E402
from vaara.attestation.receipt import (  # noqa: E402
    OutcomeDerived,
    did_web_to_url,
    emit_receipt,
    make_back_link,
    verify_receipt_identity,
)
from vaara.attestation.tool_call_attestation import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)

HS_SECRET = b"\x42" * 32
DID = "did:web:agents.example.com:billing"
KEYID = DID + "#key-2026"


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _ec_jwk(public_key: ec.EllipticCurvePublicKey) -> dict:
    nums = public_key.public_numbers()
    return {
        "kty": "EC",
        "crv": "P-256",
        "x": _b64u(nums.x.to_bytes(32, "big")),
        "y": _b64u(nums.y.to_bytes(32, "big")),
    }


def _rsa_jwk(public_key: rsa.RSAPublicKey) -> dict:
    nums = public_key.public_numbers()
    n_len = (nums.n.bit_length() + 7) // 8
    e_len = (nums.e.bit_length() + 7) // 8
    return {
        "kty": "RSA",
        "n": _b64u(nums.n.to_bytes(n_len, "big")),
        "e": _b64u(nums.e.to_bytes(e_len, "big")),
    }


def _did_document(jwk: dict, *, doc_id: str = DID, keyid: str = KEYID) -> dict:
    return {
        "id": doc_id,
        "verificationMethod": [
            {
                "id": keyid,
                "type": "JsonWebKey2020",
                "controller": doc_id,
                "publicKeyJwk": jwk,
            }
        ],
    }


def _attestation():
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="charge_card",
        server_fingerprint="sha256:" + "1" * 64,
        args=make_args_digest({"amount": 4200}),
    ),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="settle invoice"),
        payload_derived=payload,
        iss="issuer://test",
        sub="agent:billing",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )


def _emit(*, alg, signing_material, iss=DID):
    return emit_receipt(
        back_link=make_back_link(_attestation()),
        outcome_derived=OutcomeDerived(status="executed", completed_at="2026-05-29T10:00:00Z"),
        iss=iss,
        sub=iss,
        secret_version="v1",
        alg=alg,
        signing_material=signing_material,
    )


def test_es256_receipt_binds_to_matching_did_document():
    priv = ec.generate_private_key(ec.SECP256R1())
    receipt = _emit(alg="ES256", signing_material=priv)
    result = verify_receipt_identity(receipt, _did_document(_ec_jwk(priv.public_key())))
    assert (result.resolved, result.bound, result.keyid) == (True, True, KEYID)


def test_rsa_receipt_binds_to_matching_did_document():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    receipt = _emit(alg="RS256", signing_material=priv)
    result = verify_receipt_identity(receipt, _did_document(_rsa_jwk(priv.public_key())))
    assert result.resolved is True and result.bound is True


def test_wrong_key_in_document_does_not_bind():
    priv = ec.generate_private_key(ec.SECP256R1())
    other = ec.generate_private_key(ec.SECP256R1())
    receipt = _emit(alg="ES256", signing_material=priv)
    result = verify_receipt_identity(receipt, _did_document(_ec_jwk(other.public_key())))
    assert result.resolved is True and result.bound is False


def test_document_id_mismatch_is_unresolved():
    priv = ec.generate_private_key(ec.SECP256R1())
    receipt = _emit(alg="ES256", signing_material=priv)
    doc = _did_document(_ec_jwk(priv.public_key()), doc_id="did:web:evil.example.com")
    result = verify_receipt_identity(receipt, doc)
    assert result.resolved is False and result.bound is False


def test_hs256_receipt_is_never_resolvable():
    receipt = _emit(alg="HS256", signing_material=HS_SECRET)
    result = verify_receipt_identity(receipt, _did_document({"kty": "oct"}))
    assert result.resolved is False
    assert "no resolvable public key" in result.reason


def test_opaque_iss_is_not_failed_but_unresolved():
    receipt = _emit(alg="HS256", signing_material=HS_SECRET, iss="issuer://opaque")
    result = verify_receipt_identity(receipt, {"id": "issuer://opaque"})
    assert result.resolved is False
    assert "not a did:web" in result.reason


def test_expected_keyid_selects_the_named_method():
    priv = ec.generate_private_key(ec.SECP256R1())
    receipt = _emit(alg="ES256", signing_material=priv)
    doc = _did_document(_ec_jwk(priv.public_key()))
    assert verify_receipt_identity(receipt, doc, expected_keyid=KEYID).bound is True
    missing = verify_receipt_identity(receipt, doc, expected_keyid=DID + "#absent")
    assert missing.resolved is True and missing.bound is False
    assert "no verification method" in missing.reason


def test_signature_matches_second_key_in_document():
    priv = ec.generate_private_key(ec.SECP256R1())
    decoy = ec.generate_private_key(ec.SECP256R1())
    receipt = _emit(alg="ES256", signing_material=priv)
    doc = {
        "id": DID,
        "verificationMethod": [
            {"id": DID + "#old", "publicKeyJwk": _ec_jwk(decoy.public_key())},
            {"id": DID + "#new", "publicKeyJwk": _ec_jwk(priv.public_key())},
        ],
    }
    result = verify_receipt_identity(receipt, doc)
    assert result.bound is True and result.keyid == DID + "#new"


def test_tampered_signature_does_not_bind():
    priv = ec.generate_private_key(ec.SECP256R1())
    receipt = _emit(alg="ES256", signing_material=priv)
    tampered = receipt.__class__(
        version=receipt.version,
        alg=receipt.alg,
        back_link=receipt.back_link,
        receipt_asserted=receipt.receipt_asserted,
        outcome_derived=receipt.outcome_derived,
        signature="00" + receipt.signature[2:],
    )
    doc = _did_document(_ec_jwk(priv.public_key()))
    assert verify_receipt_identity(tampered, doc).bound is False


def test_did_web_to_url_mappings():
    assert did_web_to_url("did:web:example.com") == "https://example.com/.well-known/did.json"
    assert did_web_to_url("did:web:example.com:billing") == "https://example.com/billing/did.json"
    assert (
        did_web_to_url("did:web:example.com:agents:billing")
        == "https://example.com/agents/billing/did.json"
    )
    assert did_web_to_url("did:web:localhost%3A8080") == "https://localhost:8080/.well-known/did.json"
    with pytest.raises(AttestationError):
        did_web_to_url("did:key:z6Mk")
    with pytest.raises(AttestationError):
        did_web_to_url("did:web:")
