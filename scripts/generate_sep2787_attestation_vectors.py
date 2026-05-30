#!/usr/bin/env python3
"""Generate the v0 SEP-2787 attestation conformance vectors.

Writes pinned keys and signed fixtures under
``tests/vectors/sep2787_attestation_v0/``. Run once and commit the
output; re-running regenerates the asymmetric keys and signatures. A
second implementation reads the committed fixtures with
``_check_independent.py`` and must reproduce the same canonical bytes
and verification verdicts.

The receipt vectors (``tests/vectors/execution_receipt_v0/``) cover the
post-execution sibling. These cover the attestation itself: the three
verification dimensions ``verify_attestation`` owns (signature, TTL) plus
the step-5 argument commitment exposed as ``verify_args_commitment``.

TTL is evaluated at a fixed instant, ``EVAL_NOW_ISO`` below, which the
independent checker pins too. The expired case carries an older ``iat``
so its deadline falls before that instant while every other case stays
inside its window.

Usage: python scripts/generate_sep2787_attestation_vectors.py
"""

from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from vaara.attestation.sep2787 import (
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
    make_args_projection,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "vectors" / "sep2787_attestation_v0"
HS_SECRET = bytes.fromhex("42" * 32)
IAT = "2026-05-29T10:00:00Z"
OLD_IAT = "2026-05-29T08:00:00Z"
# The independent checker evaluates TTL at this instant. 30s past IAT,
# well inside the 300s default window; the expired case (OLD_IAT, two
# hours earlier) is already past its deadline here.
EVAL_NOW_ISO = "2026-05-29T10:00:30Z"

ARGS = {"path": "/archive/2024-Q3.md", "recursive": False}
OTHER_ARGS = {"path": "/keep/forever.md", "recursive": False}
COMMON = dict(iss="issuer://test", sub="agent:archiver", secret_version="v1")


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _attest(*, alg, signing_material, args, nonce, iat=IAT):
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="delete_file",
        server_fingerprint="sha256:" + "1" * 64,
        args=args,
    ),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="archive obsolete report"),
        payload_derived=payload,
        alg=alg, signing_material=signing_material,
        nonce=nonce, iat=iat, **COMMON,
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


def _case(name: str, *, attestation_dict: dict, expected: dict,
          runtime_args=None) -> None:
    d = OUT / "normative" / name
    _write(d / "attestation.json", attestation_dict)
    _write(d / "expected.json", expected)
    if runtime_args is not None:
        _write(d / "runtime_args.json", runtime_args)


def main() -> None:
    asym = _emit_keys()

    # Positive: HS256, hash-only-identity commitment, runtime args in hand.
    att = _attest(alg="HS256", signing_material=HS_SECRET,
                  args=make_args_digest(ARGS), nonce="att-nonce-hs-0001")
    _case("hs256_digest_identity", attestation_dict=att.to_dict(),
          runtime_args=ARGS,
          expected={"signature_ok": True, "ttl_ok": True,
                    "args_commitment_ok": True, "projection_match": True})

    # Positive: ES256, identity projection of the full args.
    att = _attest(alg="ES256", signing_material=asym["ES256"],
                  args=make_args_projection(ARGS), nonce="att-nonce-es-0002")
    _case("es256_projection_identity", attestation_dict=att.to_dict(),
          runtime_args=ARGS,
          expected={"signature_ok": True, "ttl_ok": True,
                    "args_commitment_ok": True, "projection_match": True})

    # Positive: RS256, signature + TTL only. No runtime args supplied, so
    # the argument-commitment step is not composed (verify_attestation
    # covers steps 1 and 3; step 5 is the caller's, run once the runtime
    # arguments are in hand). args_commitment_ok is null.
    att = _attest(alg="RS256", signing_material=asym["RS256"],
                  args=make_args_projection(ARGS), nonce="att-nonce-rs-0003")
    _case("rs256_signature_ttl_only", attestation_dict=att.to_dict(),
          expected={"signature_ok": True, "ttl_ok": True,
                    "args_commitment_ok": None, "projection_match": None})

    # Negative: signature tampered. The signed body is unchanged, so TTL
    # and the argument commitment still hold; only the signature fails.
    att = _attest(alg="HS256", signing_material=HS_SECRET,
                  args=make_args_digest(ARGS), nonce="att-nonce-hs-0004")
    bad = att.to_dict()
    last = bad["signature"][-1]
    bad["signature"] = bad["signature"][:-1] + ("0" if last != "0" else "1")
    _case("neg_bad_signature", attestation_dict=bad, runtime_args=ARGS,
          expected={"signature_ok": False, "ttl_ok": True,
                    "args_commitment_ok": True, "projection_match": True})

    # Negative: signature valid but the envelope is past its TTL deadline
    # at EVAL_NOW (older iat, default 300s window).
    att = _attest(alg="ES256", signing_material=asym["ES256"],
                  args=make_args_projection(ARGS), nonce="att-nonce-es-0005",
                  iat=OLD_IAT)
    _case("neg_expired", attestation_dict=att.to_dict(), runtime_args=ARGS,
          expected={"signature_ok": True, "ttl_ok": False,
                    "args_commitment_ok": True, "projection_match": True})

    # Negative: signature and TTL valid, but the hash-only-identity
    # commitment binds ARGS while the runtime arguments are OTHER_ARGS.
    att = _attest(alg="HS256", signing_material=HS_SECRET,
                  args=make_args_digest(ARGS), nonce="att-nonce-hs-0006")
    _case("neg_args_mismatch", attestation_dict=att.to_dict(),
          runtime_args=OTHER_ARGS,
          expected={"signature_ok": True, "ttl_ok": True,
                    "args_commitment_ok": False, "projection_match": None})

    print(f"wrote vectors under {OUT}")
    print(f"TTL evaluated at {EVAL_NOW_ISO}")


if __name__ == "__main__":
    main()
