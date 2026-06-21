#!/usr/bin/env python3
"""Regenerate the ap2_v0 conformance vectors (AP2 <-> Vaara binding profile).

Imports Vaara to MINT the vaara.receipt/v1 side; the sibling
``_check_independent.py`` imports no Vaara and only RECOMPUTES. See README.md
for the full mapping. The AP2 checkout produces a Payment Evidence Frame (PEF,
AP2 PR #274), content-addressed as ``frame_id``; each post-checkout action is a
vaara.authorization/v0 receipt that names the checkout by that address
(``evidenceRef.ref`` = ``ap2:checkout/<frame_id>``), declares the AP2 task scope
as ``coverage.boundary``, and carries a signed ``completeness`` block. The
``dropped`` case withholds seq 1, which the held running count still proves
exists.

Run: tests/vectors/ap2_v0/_generate.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import rfc8785
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from vaara.attestation._sep2787_canonical import iso8601_to_epoch, make_args_digest
from vaara.credential import Capability, GrantBinding, GrantScope, emit_grant, verify_grant
from vaara.credential._authorization_receipt import mint_authorization_receipt

HERE = Path(__file__).resolve().parent

_SCALAR = 0x42C0FFEE_1337_0BADBEEF_CAFEBABE_0DDF00D_1234567890ABCDEF_42424242  # test key
DIGEST = "sha256:" + "ab" * 32
NONCE = "att-nonce-ap2"
IAT = "2026-06-21T09:00:00Z"
DECIDED_AT = "2026-06-21T09:00:05Z"
BOUNDARY = "ap2:task/checkout-7f3a"  # the AP2 task scope = coverage boundary
SERVER_FINGERPRINT = "manifest:sha256:" + "cd" * 32
STREAM_LEN = 3
DROPPED_SEQ = 1
CANON = "urn:x402:canonicalisation:jcs-rfc8785-v1"

CAPS = (
    Capability("region", "in", ("EU", "US")),
    Capability("items", "le", "10"),
    Capability("orderRef", "eq", "cart-7f3a"),
)
MINT_ARGS = {"region": "EU", "items": 3, "orderRef": "cart-7f3a"}
COMMIT = make_args_digest(MINT_ARGS).projection_digest
RUNTIME_ARGS = {"region": "EU", "items": 3, "orderRef": "cart-7f3a"}


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _sha256_hex(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_pef() -> dict:
    """AP2 Payment Evidence Frame (#274). Inner ``receipt`` is representative;
    the binding rests on the #274 content-addressing discipline, not field names."""
    receipt = {
        "schema": "ap2.checkout-receipt/v0",
        "cartId": "cart-7f3a",
        "merchant": "merchant:acme",
        "amount": "49.99",
        "currency": "USD",
        "paymentMethod": "card",
        "payerAgent": "agent:checkout-bot",
        "settledAtMs": 1779267600000,
    }
    core = {  # frame_id preimage excludes frame_id and signature (#274)
        "canon_version": CANON,
        "claim_type": "payment.checkout",
        "frame_provider_did": "did:web:acme.example",
        "frame_timestamp_ms": 1779267600500,
        "receipt": receipt,
        "receipt_hash": _sha256_hex(_jcs(receipt)),
    }
    return {**core, "frame_id": _sha256_hex(_jcs(core))}


def main() -> int:
    priv = ec.derive_private_key(_SCALAR, ec.SECP256R1())
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    (HERE / "keys").mkdir(parents=True, exist_ok=True)
    (HERE / "keys" / "es256_public.pem").write_bytes(pub_pem)

    pef = _build_pef()
    _write(HERE / "checkout" / "pef.json", pef)
    checkout_ref = f"ap2:checkout/{pef['frame_id']}"

    grant = emit_grant(
        scope=GrantScope(
            tool_name="fulfillment.dispatch", args_commitment=COMMIT, tenant_id="tenant-a"
        ),
        binding=GrantBinding(attestation_digest=DIGEST, attestation_nonce=NONCE),
        iss="vaara-mcp-proxy",
        sub="tenant-a/upstream",
        secret_version="key-v1",
        alg="ES256",
        signing_material=priv,
        exp_seconds=60,
        iat=IAT,
        nonce="grant-nonce-1",
        capabilities=CAPS,
    )
    verdict = verify_grant(
        grant,
        verifying_material=priv.public_key(),
        runtime_tool_name="fulfillment.dispatch",
        runtime_args=RUNTIME_ARGS,
        runtime_tenant_id="tenant-a",
        known_attestation_digests=frozenset({DIGEST}),
        now=iso8601_to_epoch(IAT) + 5,
    )
    _write(HERE / "grant.json", grant.to_dict())

    coverage = {
        "boundary": BOUNDARY,
        "serverFingerprint": SERVER_FINGERPRINT,
        "scope": "only post-checkout actions routed through the AP2 task are observed",
    }
    receipts = []  # mint once; both cases share the byte-identical receipts
    for seq in range(STREAM_LEN):
        auth = mint_authorization_receipt(
            credential=grant,
            runtime_args=RUNTIME_ARGS,
            verdict=verdict,
            iss="vaara-mcp-proxy",
            sub="tenant-a/upstream",
            secret_version="key-v1",
            alg="ES256",
            signing_material=priv,
            decided_at=DECIDED_AT,
            nonce=f"decision-nonce-{seq}",
            ref=checkout_ref,
            coverage=coverage,
            completeness={"boundaryId": BOUNDARY, "seq": seq, "runningCount": seq + 1},
        )
        receipts.append({"record": auth.record.to_dict(), "evidence": auth.evidence})

    slug = BOUNDARY.replace(":", "-").replace("/", "-")
    for seq, authz in enumerate(receipts):
        name = f"{slug}-{seq:04d}-authz.json"
        _write(HERE / "complete" / name, authz)
        if seq != DROPPED_SEQ:
            _write(HERE / "dropped" / name, authz)

    base = {"all_signatures_ok": True, "all_evidence_bindings_resolve": True,
            "all_receipts_name_checkout": True}
    expected = {
        "frame_id_recomputes": True,
        "receipt_hash_recomputes": True,
        "complete": {**base, "contiguity": {
            "ok": True, "present": STREAM_LEN, "expected": STREAM_LEN, "missingSeqs": []}},
        "dropped": {**base, "contiguity": {
            "ok": False, "present": STREAM_LEN - 1, "expected": STREAM_LEN,
            "missingSeqs": [DROPPED_SEQ]}},
    }
    _write(HERE / "expected.json", expected)
    print("wrote ap2_v0 vectors: checkout/pef.json, complete, dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
