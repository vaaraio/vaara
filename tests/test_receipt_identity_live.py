"""Tests for level-3 live-resolvable agent identity (did:web).

The fetcher is injected throughout, so no test makes a network call. These
cover resolution, the auditable resolution record, the cache (hit, TTL
expiry), deactivation, and revocation-in-time (revoked before vs after
issuance), all composed over the unchanged level-2 check.
"""

from __future__ import annotations

import base64
import importlib.util
import json

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation._attest_types import AttestationError  # noqa: E402
from vaara.attestation.receipt import (  # noqa: E402
    DidDocumentCache,
    OutcomeDerived,
    emit_receipt,
    https_fetch,
    make_back_link,
    verify_receipt_identity_live,
)
from vaara.attestation.tool_call_attestation import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)

DID = "did:web:agents.example.com:billing"
KEYID = DID + "#key-2026"
ISSUED = "2026-05-29T10:00:00Z"
SCALAR_A = 0x1F2E3D4C5B6A79887766554433221100FFEEDDCCBBAA99887766554433221101
SCALAR_B = 0x0A1B2C3D4E5F60718293A4B5C6D7E8F9001122334455667788990AABBCCDDEEFF


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _ec_jwk(public_key: ec.EllipticCurvePublicKey) -> dict:
    nums = public_key.public_numbers()
    return {
        "kty": "EC", "crv": "P-256",
        "x": _b64u(nums.x.to_bytes(32, "big")),
        "y": _b64u(nums.y.to_bytes(32, "big")),
    }


def _did_document(public_key, *, revoked=None, deactivated=False) -> dict:
    method = {
        "id": KEYID, "type": "JsonWebKey2020", "controller": DID,
        "publicKeyJwk": _ec_jwk(public_key),
    }
    if revoked is not None:
        method["revoked"] = revoked
    doc = {"id": DID, "verificationMethod": [method]}
    if deactivated:
        doc["deactivated"] = True
    return doc


def _receipt(key):
    attestation = emit_attestation(
        planner_declared=PlannerDeclared(intent="settle invoice"),
        payload_derived=PayloadDerived(tool_calls=(ToolCallBinding(
            name="charge_card", server_fingerprint="sha256:" + "1" * 64,
            args=make_args_digest({"amount": 4200}),
        ),)),
        iss="issuer://test", sub="agent:billing", secret_version="v1",
        alg="HS256", signing_material=b"\x42" * 32,
        nonce="att-nonce-0001", iat="2026-05-29T09:59:59Z",
    )
    return emit_receipt(
        back_link=make_back_link(attestation),
        outcome_derived=OutcomeDerived(status="executed", completed_at=ISSUED),
        iss=DID, sub=DID, secret_version="v1", alg="ES256",
        signing_material=key, nonce="rcpt-nonce-0001", iat=ISSUED,
    )


def _fetcher_for(document: dict):
    raw = json.dumps(document).encode("utf-8")
    calls = {"n": 0}

    def fetch(url: str) -> bytes:
        calls["n"] += 1
        return raw

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


@pytest.fixture
def key_a():
    return ec.derive_private_key(SCALAR_A, ec.SECP256R1())


@pytest.fixture
def key_b():
    return ec.derive_private_key(SCALAR_B, ec.SECP256R1())


def test_bound_and_trusted(key_a):
    receipt = _receipt(key_a)
    fetch = _fetcher_for(_did_document(key_a.public_key()))
    result = verify_receipt_identity_live(receipt, fetcher=fetch, now="2026-06-03T00:00:00Z")
    assert result.resolved and result.bound and result.trusted
    assert result.keyid == KEYID
    assert not result.revoked and not result.deactivated
    assert result.resolution is not None
    assert result.resolution.url == "https://agents.example.com/billing/did.json"
    assert result.resolution.document_digest.startswith("sha256:")
    assert result.resolution.fetched_at == "2026-06-03T00:00:00Z"
    assert result.resolution.from_cache is False


def test_unbound_when_document_lists_a_different_key(key_a, key_b):
    receipt = _receipt(key_a)
    fetch = _fetcher_for(_did_document(key_b.public_key()))
    result = verify_receipt_identity_live(receipt, fetcher=fetch)
    assert result.resolved and not result.bound and not result.trusted


def test_revoked_before_issuance_is_not_trusted(key_a):
    receipt = _receipt(key_a)
    doc = _did_document(key_a.public_key(), revoked="2026-05-29T09:00:00Z")
    result = verify_receipt_identity_live(receipt, fetcher=_fetcher_for(doc))
    assert result.bound is True  # signature still matches the key
    assert result.revoked is True and result.trusted is False
    assert result.revoked_at == "2026-05-29T09:00:00Z"
    assert "revoked" in result.reason


def test_revoked_after_issuance_still_trusted(key_a):
    receipt = _receipt(key_a)
    doc = _did_document(key_a.public_key(), revoked="2026-06-01T00:00:00Z")
    result = verify_receipt_identity_live(receipt, fetcher=_fetcher_for(doc))
    assert result.bound and result.trusted
    assert result.revoked is False
    assert result.revoked_at == "2026-06-01T00:00:00Z"  # surfaced even on a pass


def test_revoked_exactly_at_issuance_is_not_trusted(key_a):
    receipt = _receipt(key_a)
    doc = _did_document(key_a.public_key(), revoked=ISSUED)
    result = verify_receipt_identity_live(receipt, fetcher=_fetcher_for(doc))
    assert result.revoked is True and result.trusted is False


def test_deactivated_document_is_not_trusted(key_a):
    receipt = _receipt(key_a)
    doc = _did_document(key_a.public_key(), deactivated=True)
    result = verify_receipt_identity_live(receipt, fetcher=_fetcher_for(doc))
    assert result.deactivated is True and result.trusted is False
    assert "deactivated" in result.reason


def test_plain_string_iss_is_not_resolved(key_a):
    receipt = _receipt(key_a)
    object.__setattr__(receipt.receipt_asserted, "iss", "billing-agent")
    fetch = _fetcher_for(_did_document(key_a.public_key()))
    result = verify_receipt_identity_live(receipt, fetcher=fetch)
    assert result.resolved is False and result.resolution is None
    assert fetch.calls["n"] == 0  # no fetch attempted


def test_fetch_failure_is_a_resolution_failure(key_a):
    receipt = _receipt(key_a)

    def boom(url: str) -> bytes:
        raise OSError("connection refused")

    result = verify_receipt_identity_live(receipt, fetcher=boom)
    assert result.resolved is False and not result.trusted
    assert "fetch failed" in result.reason


def test_cache_hit_avoids_second_fetch(key_a):
    receipt = _receipt(key_a)
    fetch = _fetcher_for(_did_document(key_a.public_key()))
    cache = DidDocumentCache(ttl_seconds=3600)
    first = verify_receipt_identity_live(receipt, fetcher=fetch, cache=cache, now_epoch=1000.0)
    second = verify_receipt_identity_live(receipt, fetcher=fetch, cache=cache, now_epoch=1500.0)
    assert fetch.calls["n"] == 1
    assert first.resolution.from_cache is False
    assert second.resolution.from_cache is True
    assert second.trusted


def test_cache_ttl_expiry_refetches(key_a):
    receipt = _receipt(key_a)
    fetch = _fetcher_for(_did_document(key_a.public_key()))
    cache = DidDocumentCache(ttl_seconds=100)
    verify_receipt_identity_live(receipt, fetcher=fetch, cache=cache, now_epoch=1000.0)
    verify_receipt_identity_live(receipt, fetcher=fetch, cache=cache, now_epoch=1201.0)
    assert fetch.calls["n"] == 2


def test_malformed_document_fails_closed(key_a):
    receipt = _receipt(key_a)
    result = verify_receipt_identity_live(receipt, fetcher=lambda url: b"not json")
    assert result.resolved is False and "JSON" in result.reason


def test_https_fetch_rejects_non_https():
    with pytest.raises(AttestationError, match="not HTTPS"):
        https_fetch("http://agents.example.com/.well-known/did.json")
