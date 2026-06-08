#!/usr/bin/env python3
"""Generate the key_rotation_v0 conformance vectors.

Emits cases that exercise ``verify_receipt_retained``: a record signed by a
key that is later rotated out, audited against the DID document the regulator
archived. One ES256 receipt (signed by ``#key-2026``, the key that is retired
later) is checked against documents whose verification methods carry
``validFrom`` / ``validUntil`` (and, in one case, ``revoked``) markers, with
and without a time anchor.

ECDSA signatures are randomized, so re-running overwrites the cases with fresh
but equivalent vectors. ``expected.json`` is produced by Vaara itself; the
committed ``_check_independent.py`` reproduces every verdict without importing
Vaara. Run from the repo root:
``python tests/vectors/key_rotation_v0/_generate.py``.
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
    parse_receipt,
    verify_receipt_retained,
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
KEYID_A = DID + "#key-2026"   # signs; retired later
KEYID_B = DID + "#key-2028"   # the rotated-in key (decoy, never signs)
HS_SECRET = b"\x42" * 32
SCALAR_A = 0x1F2E3D4C5B6A79887766554433221100FFEEDDCCBBAA99887766554433221101
SCALAR_B = 0x0A1B2C3D4E5F60718293A4B5C6D7E8F9001122334455667788990AABBCCDDEEFF

IAT = "2026-05-29T10:00:00Z"
ACTIVATED = "2026-01-01T00:00:00Z"
RETIRED = "2028-01-01T00:00:00Z"
ANCHOR_BEFORE_RETIREMENT = "2026-05-29T10:05:00Z"
ANCHOR_AFTER_RETIREMENT = "2028-06-01T00:00:00Z"
REVOKED_BEFORE_IAT = "2026-05-29T09:30:00Z"

COMPARE = (
    "bound", "keyid", "within_window", "window_recorded", "not_before",
    "not_after", "revoked", "revoked_at", "time_basis",
    "anchored_before_retirement", "anchored_before_revocation",
    "verifiable", "corroborated",
)


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _ec_jwk(public_key: ec.EllipticCurvePublicKey) -> dict:
    nums = public_key.public_numbers()
    return {
        "kty": "EC", "crv": "P-256",
        "x": _b64u(nums.x.to_bytes(32, "big")),
        "y": _b64u(nums.y.to_bytes(32, "big")),
    }


def _method(keyid: str, jwk: dict, **markers: str) -> dict:
    method = {"id": keyid, "type": "JsonWebKey2020", "controller": DID,
              "publicKeyJwk": jwk}
    method.update(markers)
    return method


def _doc(*methods: dict) -> dict:
    return {"id": DID, "verificationMethod": list(methods)}


def _attestation():
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="charge_card", server_fingerprint="sha256:" + "1" * 64,
        args=make_args_digest({"amount": 4200}),
    ),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="settle invoice"),
        payload_derived=payload, iss="issuer://test", sub="agent:billing",
        secret_version="v1", alg="HS256", signing_material=HS_SECRET,
        nonce="att-nonce-fixed-0001", iat="2026-05-29T09:59:59Z",
    )


def main() -> None:
    key_a = ec.derive_private_key(SCALAR_A, ec.SECP256R1())
    key_b = ec.derive_private_key(SCALAR_B, ec.SECP256R1())
    jwk_a = _ec_jwk(key_a.public_key())
    jwk_b = _ec_jwk(key_b.public_key())

    receipt = emit_receipt(
        back_link=make_back_link(_attestation()),
        outcome_derived=OutcomeDerived(status="executed", completed_at=IAT),
        iss=DID, sub=DID, secret_version="v1", alg="ES256",
        signing_material=key_a, nonce="rcpt-nonce-fixed-0001", iat=IAT,
    )
    rec = receipt.to_dict()

    in_window = _method(KEYID_A, jwk_a, validFrom=ACTIVATED, validUntil=RETIRED)
    cases = [
        {"name": "retired_key_no_anchor", "didDocument": _doc(in_window)},
        {"name": "retired_key_anchored", "didDocument": _doc(in_window),
         "anchoredTime": ANCHOR_BEFORE_RETIREMENT},
        {"name": "signed_after_retirement",
         "didDocument": _doc(_method(KEYID_A, jwk_a, validFrom=ACTIVATED,
                                     validUntil="2026-05-01T00:00:00Z"))},
        {"name": "signed_before_activation",
         "didDocument": _doc(_method(KEYID_A, jwk_a,
                                     validFrom="2026-06-01T00:00:00Z"))},
        {"name": "revoked_before_issuance",
         "didDocument": _doc(_method(KEYID_A, jwk_a, validFrom=ACTIVATED,
                                     validUntil=RETIRED,
                                     revoked=REVOKED_BEFORE_IAT))},
        {"name": "anchor_after_retirement", "didDocument": _doc(in_window),
         "anchoredTime": ANCHOR_AFTER_RETIREMENT},
        {"name": "unbounded_key", "didDocument": _doc(_method(KEYID_A, jwk_a))},
        {"name": "wrong_key", "didDocument": _doc(_method(KEYID_B, jwk_b,
                                                          validFrom=ACTIVATED))},
    ]
    for case in cases:
        case["receipt"] = rec

    expected = {}
    for case in cases:
        result = verify_receipt_retained(
            parse_receipt(case["receipt"]), case["didDocument"],
            anchored_time=case.get("anchoredTime"),
        )
        d = result.to_dict()
        expected[case["name"]] = {k: d[k] for k in COMPARE}

    (HERE / "cases.json").write_text(
        json.dumps({"cases": cases, "did": DID, "keyid": KEYID_A},
                   indent=2, sort_keys=True) + "\n")
    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} cases and expected.json")


if __name__ == "__main__":
    main()
