#!/usr/bin/env python3
"""Generate the agent_identity_v0 conformance vectors.

Emits three cases with Vaara, using did:web resolvable agent identity:

- ``bound.json``: an ES256 receipt whose ``iss`` is a did:web identity,
  plus a DID document that lists the signing key. Expected: resolved,
  bound, trusted.
- ``unbound.json``: the same receipt against a DID document that lists a
  different key. Expected: resolved, not bound, not trusted.
- ``revoked.json``: the receipt against a document that lists the signing
  key but marks it ``revoked`` before the receipt's ``iat``. Expected:
  resolved and bound (the signature matches the key) but revoked and not
  trusted (the key was revoked at or before issuance). This is the level-3
  revocation-in-time property, reproducible offline from the captured
  document because the comparison is purely the receipt ``iat`` against the
  method ``revoked`` instant.

ECDSA signatures are randomized, so re-running this overwrites the cases
with fresh but equivalent vectors. The committed JSON is the vector;
``_check_independent.py`` verifies whatever is committed. Run from the
repo root: ``python tests/vectors/agent_identity_v0/_generate.py``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ec

from vaara.attestation.receipt import (
    OutcomeDerived,
    emit_receipt,
    make_back_link,
)
from vaara.attestation.sep2787 import (
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)

HERE = Path(__file__).resolve().parent
DID = "did:web:agents.example.com:billing"
KEYID = DID + "#key-2026"
HS_SECRET = b"\x42" * 32

# Fixed private scalars so regeneration uses stable keys (only the ECDSA
# nonce varies). Key A signs; key B is the decoy in the unbound document.
SCALAR_A = 0x1F2E3D4C5B6A79887766554433221100FFEEDDCCBBAA99887766554433221101
SCALAR_B = 0x0A1B2C3D4E5F60718293A4B5C6D7E8F9001122334455667788990AABBCCDDEEFF


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


RECEIPT_IAT = "2026-05-29T10:00:00Z"
REVOKED_BEFORE_IAT = "2026-05-29T09:30:00Z"


def _did_document(jwk: dict, *, revoked: str | None = None) -> dict:
    method = {
        "id": KEYID,
        "type": "JsonWebKey2020",
        "controller": DID,
        "publicKeyJwk": jwk,
    }
    if revoked is not None:
        method["revoked"] = revoked
    return {"id": DID, "verificationMethod": [method]}


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
        nonce="att-nonce-fixed-0001",
        iat="2026-05-29T09:59:59Z",
    )


def main() -> None:
    key_a = ec.derive_private_key(SCALAR_A, ec.SECP256R1())
    key_b = ec.derive_private_key(SCALAR_B, ec.SECP256R1())

    receipt = emit_receipt(
        back_link=make_back_link(_attestation()),
        outcome_derived=OutcomeDerived(status="executed", completed_at="2026-05-29T10:00:00Z"),
        iss=DID,
        sub=DID,
        secret_version="v1",
        alg="ES256",
        signing_material=key_a,
        nonce="rcpt-nonce-fixed-0001",
        iat="2026-05-29T10:00:00Z",
    )
    receipt_dict = receipt.to_dict()

    jwk_a = _ec_jwk(key_a.public_key())

    (HERE / "bound.json").write_text(json.dumps({
        "receipt": receipt_dict,
        "didDocument": _did_document(jwk_a),
    }, indent=2, sort_keys=True) + "\n")

    (HERE / "unbound.json").write_text(json.dumps({
        "receipt": receipt_dict,
        "didDocument": _did_document(_ec_jwk(key_b.public_key())),
    }, indent=2, sort_keys=True) + "\n")

    (HERE / "revoked.json").write_text(json.dumps({
        "receipt": receipt_dict,
        "didDocument": _did_document(jwk_a, revoked=REVOKED_BEFORE_IAT),
    }, indent=2, sort_keys=True) + "\n")

    (HERE / "expected.json").write_text(json.dumps({
        "bound": {
            "resolved": True, "bound": True, "keyid": KEYID,
            "revoked": False, "trusted": True,
        },
        "unbound": {
            "resolved": True, "bound": False, "keyid": None,
            "revoked": False, "trusted": False,
        },
        "revoked": {
            "resolved": True, "bound": True, "keyid": KEYID,
            "revoked": True, "trusted": False,
        },
    }, indent=2, sort_keys=True) + "\n")

    print("wrote bound.json, unbound.json, revoked.json, expected.json")


if __name__ == "__main__":
    main()
