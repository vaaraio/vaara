#!/usr/bin/env python3
"""Generate the v0 execution-receipt conformance vectors.

Writes pinned keys and signed fixtures under
``tests/vectors/execution_receipt_v0/``. Run once and commit the output;
re-running regenerates keys and signatures. A second implementation
reads the committed fixtures with ``_check_independent.py`` and must
reproduce the same canonical bytes and verification verdicts.

Usage: python scripts/generate_receipt_vectors.py
"""

from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from vaara.attestation.receipt import (
    OutcomeDerived,
    emit_receipt,
    make_back_link,
    make_result_digest,
    make_result_projection,
)
from vaara.attestation.sep2787 import (
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "vectors" / "execution_receipt_v0"
HS_SECRET = bytes.fromhex("42" * 32)
IAT = "2026-05-29T10:00:00Z"
RESULT = {"deleted": True, "path": "/archive/2024-Q3.md"}


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _attestation(secret: bytes, nonce: str = "fixed-attestation-nonce-000"):
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="delete_file",
        server_fingerprint="sha256:" + "1" * 64,
        args=make_args_digest({"path": "/archive/2024-Q3.md"}),
    ),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="archive obsolete report"),
        payload_derived=payload,
        iss="issuer://test",
        sub="agent:archiver",
        secret_version="v1",
        alg="HS256",
        signing_material=secret,
        nonce=nonce,
        iat=IAT,
    )


def _emit_keys() -> dict:
    es = ec.generate_private_key(ec.SECP256R1())
    rs = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    keys = OUT / "keys"
    keys.mkdir(parents=True, exist_ok=True)
    (keys / "hs256_secret.bin").write_bytes(HS_SECRET)
    (keys / "es256_private.pem").write_bytes(es.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    (keys / "es256_public.pem").write_bytes(es.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    (keys / "rs256_private.pem").write_bytes(rs.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    (keys / "rs256_public.pem").write_bytes(rs.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    return {"ES256": es, "RS256": rs}


def _case(name: str, *, attestation, receipt, expected: dict,
          runtime_result=None) -> None:
    d = OUT / "normative" / name
    _write(d / "attestation.json", attestation.to_dict())
    _write(d / "receipt.json", receipt.to_dict())
    _write(d / "expected.json", expected)
    if runtime_result is not None:
        _write(d / "runtime_result.json", runtime_result)


def main() -> None:
    asym = _emit_keys()
    att = _attestation(HS_SECRET)
    common = dict(iss="issuer://test", sub="agent:archiver",
                  secret_version="v1", iat=IAT,
                  nonce="fixed-receipt-nonce-0001")

    # Positive: HS256, executed, hash-only result digest.
    r = emit_receipt(
        back_link=make_back_link(att),
        outcome_derived=OutcomeDerived(
            status="executed", completed_at=IAT,
            result_commitment=make_result_digest(RESULT)),
        alg="HS256", signing_material=HS_SECRET, **common)
    _case("hs256_executed_digest", attestation=att, receipt=r,
          runtime_result=RESULT,
          expected={"signature_ok": True, "back_link_ok": True,
                    "result_commitment_ok": True})

    # Positive: ES256, executed, identity projection of the result.
    r = emit_receipt(
        back_link=make_back_link(att),
        outcome_derived=OutcomeDerived(
            status="executed", completed_at=IAT,
            result_commitment=make_result_projection(RESULT)),
        alg="ES256", signing_material=asym["ES256"], **common)
    _case("es256_executed_projection", attestation=att, receipt=r,
          runtime_result=RESULT,
          expected={"signature_ok": True, "back_link_ok": True,
                    "result_commitment_ok": True})

    # Positive: RS256, refused, no result commitment.
    r = emit_receipt(
        back_link=make_back_link(att),
        outcome_derived=OutcomeDerived(status="refused", completed_at=IAT),
        alg="RS256", signing_material=asym["RS256"], **common)
    _case("rs256_refused_no_commitment", attestation=att, receipt=r,
          expected={"signature_ok": True, "back_link_ok": True,
                    "result_commitment_ok": None})

    # Negative: result commitment does not bind the given runtime result.
    r = emit_receipt(
        back_link=make_back_link(att),
        outcome_derived=OutcomeDerived(
            status="executed", completed_at=IAT,
            result_commitment=make_result_digest(RESULT)),
        alg="HS256", signing_material=HS_SECRET, **common)
    _case("neg_result_mismatch", attestation=att, receipt=r,
          runtime_result={"deleted": False},
          expected={"signature_ok": True, "back_link_ok": True,
                    "result_commitment_ok": False})

    # Negative: receipt back-links to a different attestation. The stored
    # attestation.json is `att`; the receipt pins `other` (distinct nonce),
    # so back-link verification against the stored attestation must fail.
    other = _attestation(HS_SECRET, nonce="other-attestation-nonce-999")
    r = emit_receipt(
        back_link=make_back_link(other),
        outcome_derived=OutcomeDerived(status="executed", completed_at=IAT,
                                       result_commitment=make_result_digest(RESULT)),
        alg="HS256", signing_material=HS_SECRET, **common)
    _case("neg_broken_backlink", attestation=att, receipt=r,
          runtime_result=RESULT,
          expected={"signature_ok": True, "back_link_ok": False,
                    "result_commitment_ok": True})

    # Negative: a valid executed receipt is replayed with one signed field
    # substituted (outcome status executed -> refused) while the original
    # signature is kept. The back-link and result commitment still verify, so
    # only the signature catches the forgery. This is the replay-with-field-
    # substitution case (distinct from a stale verifier clock): the signed
    # envelope, not any single sub-check, is what binds the outcome claim.
    valid = emit_receipt(
        back_link=make_back_link(att),
        outcome_derived=OutcomeDerived(
            status="executed", completed_at=IAT,
            result_commitment=make_result_digest(RESULT)),
        alg="HS256", signing_material=HS_SECRET, **common)
    tampered = valid.to_dict()
    tampered["outcomeDerived"]["status"] = "refused"
    d = OUT / "normative" / "neg_replay_substituted_field"
    _write(d / "attestation.json", att.to_dict())
    _write(d / "receipt.json", tampered)
    _write(d / "runtime_result.json", RESULT)
    _write(d / "expected.json", {"signature_ok": False, "back_link_ok": True,
                                 "result_commitment_ok": True})

    print(f"wrote vectors under {OUT}")


if __name__ == "__main__":
    main()
