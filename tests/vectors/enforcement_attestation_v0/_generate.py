#!/usr/bin/env python3
"""Generate the enforcement_attestation_v0 conformance vectors.

Emits SEV-SNP attestation reports bound to a SEP-2828 execution record and runs
them through ``verify_enforcement``. Each case carries the record, a base64
report blob, the VCEK PEM the report signature is checked against, an optional
pinned measurement, and the strict flag. The cases exercise every verdict
branch: the clean ``bound`` tier, a pinned measurement match and mismatch, a
report bound to a different record (substitution), the signature-malleability
defence (the binding is over the full record including its signature), a flipped
signature, a wrong VCEK, an unsupported signature algorithm, a truncated report,
and strict mode (unreachable in v0).

The record signature and the report's ECDSA signature are randomized, so
re-running overwrites the cases with fresh but equivalent vectors. ``expected.json``
is produced by Vaara; the committed ``_check_independent.py`` reproduces every
verdict without importing Vaara. Run from the repo root:
``python tests/vectors/enforcement_attestation_v0/_generate.py``.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from vaara.attestation.receipt import (
    OutcomeDerived,
    bind_record_to_report_data,
    emit_receipt,
    make_back_link,
    verify_enforcement,
)
from vaara.attestation.tool_call_attestation import (
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)
from vaara.attestation.tee import (
    MockSEVSNPAttester,
    SEV_SNP_BODY_SIZE,
)

HERE = Path(__file__).resolve().parent
DID = "did:web:vendor-a.example:billing"
IAT = "2026-05-29T10:00:00Z"
# Fixed P-384 VCEK scalars: the trusted endorsement key and an unrelated one.
VCEK_SCALAR = 0x1F2E3D4C5B6A79887766554433221100FFEEDDCCBBAA998877665544332211001122334455667788
OTHER_VCEK_SCALAR = 0x0A1B2C3D4E5F60718293A4B5C6D7E8F900112233445566778899AABBCCDDEEFF00FFEEDDCCBBAA9988
# Fixed P-256 record signing key.
RECORD_SCALAR = 0x51A8B6C4D2E0F1A3B5C7D9E0F2A4B6C8DAE0F2A4B6C8DAE0F2A4B6C8DAE0F201
MEASUREMENT = bytes(range(1, 49))          # 48 bytes, the vetted launch image
WRONG_MEASUREMENT = b"\xff" * 48           # a different image

COMPARE = (
    "tier", "parsed", "report_version", "signature_algo_ok", "signature_valid",
    "bound", "report_data_expected", "report_data_actual", "measurement",
    "measurement_basis", "vcek_chain_basis", "enforcement_logic_basis",
    "strict", "ok",
)


def _attestation():
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="charge_card", server_fingerprint="sha256:" + "1" * 64,
        args=make_args_digest({"amount": 4200}),
    ),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="settle invoice"),
        payload_derived=payload, iss="issuer://test", sub="agent:billing",
        secret_version="v1", alg="HS256", signing_material=b"\x42" * 32,
        nonce="att-nonce-fixed-0001", iat="2026-05-29T09:59:59Z",
    )


def _receipt(key: ec.EllipticCurvePrivateKey, nonce: str) -> dict:
    return emit_receipt(
        back_link=make_back_link(_attestation()),
        outcome_derived=OutcomeDerived(status="executed", completed_at=IAT),
        iss=DID, sub=DID, secret_version="v1", alg="ES256",
        signing_material=key, nonce=nonce, iat=IAT,
    ).to_dict()


def _b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _jwk_p384(pub: ec.EllipticCurvePublicKey) -> dict:
    """A P-384 public key as a JWK: public verification material, not a secret.

    The VCEK is stored this way (like the DID-document keys in the other vector
    suites) rather than as a serialised PEM, so the committed cases carry no
    key-shaped blob; the checker and the generator reconstruct the key from it.
    """
    n = pub.public_numbers()
    return {"kty": "EC", "crv": "P-384",
            "x": _b64u(n.x.to_bytes(48, "big")), "y": _b64u(n.y.to_bytes(48, "big"))}


def _jwk_to_pem(jwk: dict) -> bytes:
    """Reconstruct the SubjectPublicKeyInfo PEM the verifier consumes from a JWK."""
    def _i(v: str) -> int:
        return int.from_bytes(
            base64.urlsafe_b64decode(v + "=" * (-len(v) % 4)), "big")
    pub = ec.EllipticCurvePublicNumbers(
        _i(jwk["x"]), _i(jwk["y"]), ec.SECP384R1()).public_key()
    return pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def main() -> None:
    vcek = ec.derive_private_key(VCEK_SCALAR, ec.SECP384R1())
    other_vcek = ec.derive_private_key(OTHER_VCEK_SCALAR, ec.SECP384R1())
    record_key = ec.derive_private_key(RECORD_SCALAR, ec.SECP256R1())
    vcek_jwk = _jwk_p384(vcek.public_key())
    other_vcek_jwk = _jwk_p384(other_vcek.public_key())

    record = _receipt(record_key, "rcpt-nonce-fixed-0001")
    other_record = _receipt(record_key, "rcpt-nonce-fixed-0002")

    attester = MockSEVSNPAttester(vcek, measurement=MEASUREMENT)
    report = attester.emit(bind_record_to_report_data(record))

    # Sig-stripped variant of the genuine record: the report still binds the
    # genuine record, so this variant must NOT bind (full-record preimage).
    variant = copy.deepcopy(record)
    variant["signature"] = "00" * 64

    bad_sig = bytearray(report)
    bad_sig[SEV_SNP_BODY_SIZE + 4] ^= 0x01     # flip one byte of the signature

    wrong_algo = bytearray(report)
    wrong_algo[0x034:0x038] = b"\x00\x00\x00\x00"  # signature_algo -> 0 (invalid)

    def b64(blob: bytes) -> str:
        return base64.b64encode(bytes(blob)).decode("ascii")

    cases = [
        {"name": "clean_bound", "record": record, "report_b64": b64(report),
         "vcek_jwk": vcek_jwk, "expected_measurement": None, "strict": False},
        {"name": "pinned_measurement_match", "record": record,
         "report_b64": b64(report), "vcek_jwk": vcek_jwk,
         "expected_measurement": MEASUREMENT.hex(), "strict": False},
        {"name": "pin_mismatch", "record": record, "report_b64": b64(report),
         "vcek_jwk": vcek_jwk, "expected_measurement": WRONG_MEASUREMENT.hex(),
         "strict": False},
        {"name": "bound_to_different_record", "record": other_record,
         "report_b64": b64(report), "vcek_jwk": vcek_jwk,
         "expected_measurement": None, "strict": False},
        {"name": "signature_malleable_variant", "record": variant,
         "report_b64": b64(report), "vcek_jwk": vcek_jwk,
         "expected_measurement": None, "strict": False},
        {"name": "bad_signature", "record": record, "report_b64": b64(bad_sig),
         "vcek_jwk": vcek_jwk, "expected_measurement": None, "strict": False},
        {"name": "vcek_mismatch", "record": record, "report_b64": b64(report),
         "vcek_jwk": other_vcek_jwk, "expected_measurement": None,
         "strict": False},
        {"name": "wrong_signature_algo", "record": record,
         "report_b64": b64(wrong_algo), "vcek_jwk": vcek_jwk,
         "expected_measurement": None, "strict": False},
        {"name": "truncated_report", "record": record,
         "report_b64": b64(report[:100]), "vcek_jwk": vcek_jwk,
         "expected_measurement": None, "strict": False},
        {"name": "strict_unmet_no_kds", "record": record,
         "report_b64": b64(report), "vcek_jwk": vcek_jwk,
         "expected_measurement": MEASUREMENT.hex(), "strict": True},
    ]

    expected = {}
    for case in cases:
        verdict = verify_enforcement(
            case["record"],
            base64.b64decode(case["report_b64"]),
            _jwk_to_pem(case["vcek_jwk"]),
            expected_measurement=case["expected_measurement"],
            strict=case["strict"],
        )
        d = verdict.to_dict()
        expected[case["name"]] = {k: d[k] for k in COMPARE}

    (HERE / "cases.json").write_text(
        json.dumps({"cases": cases, "did": DID}, indent=2, sort_keys=True) + "\n")
    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} cases and expected.json")


if __name__ == "__main__":
    main()
