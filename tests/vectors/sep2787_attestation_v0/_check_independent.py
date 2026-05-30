#!/usr/bin/env python3
"""Independent conformance checker for the v0 SEP-2787 attestation vectors.

Imports only the standard library plus ``cryptography`` and ``rfc8785``.
It does not import Vaara. It reads the committed fixtures from disk and
reproduces, for each case, the three verification dimensions: signature
over the JCS-canonical envelope body, TTL against a pinned instant, and
(when runtime arguments are supplied) the step-5 argument commitment.
The results are compared against ``expected.json``.

A second implementation that can run this file (or reproduce its logic)
demonstrates that the attestation format is consumable without depending
on Vaara. Run:
``python tests/vectors/sep2787_attestation_v0/_check_independent.py``.
Exit code 0 means every case matched its expected verdict.

HS256 and RS256 signatures are deterministic, so a second implementation
re-signing reproduces the stored signature exactly. ES256 is randomised,
so the ES256 fixtures are verified against the stored public key rather
than reproduced bit for bit.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

HERE = Path(__file__).resolve().parent
KEYS = HERE / "keys"

# Must match EVAL_NOW_ISO in scripts/generate_sep2787_attestation_vectors.py
# and the default clock_skew_seconds in vaara.attestation verify_attestation.
EVAL_NOW = datetime(2026, 5, 29, 10, 0, 30, tzinfo=timezone.utc).timestamp()
CLOCK_SKEW_SECONDS = 30

_SIGNED_KEYS = ("version", "alg", "plannerDeclared", "issuerAsserted",
                "payloadDerived")


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _sha256_hex(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _signing_payload(att: dict) -> bytes:
    return _jcs({k: att[k] for k in _SIGNED_KEYS})


def verify_signature(att: dict) -> bool:
    payload = _signing_payload(att)
    alg = att["alg"]
    sig = att["signature"]
    if alg == "HS256":
        secret = (KEYS / "hs256_secret.bin").read_bytes()
        expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig)
    if alg == "ES256":
        pub = serialization.load_pem_public_key(
            (KEYS / "es256_public.pem").read_bytes())
        if len(sig) != 128:
            return False
        try:
            raw = bytes.fromhex(sig)
        except ValueError:
            return False
        der = encode_dss_signature(
            int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
        try:
            pub.verify(der, payload, ec.ECDSA(hashes.SHA256()))
            return True
        except InvalidSignature:
            return False
    if alg == "RS256":
        pub = serialization.load_pem_public_key(
            (KEYS / "rs256_public.pem").read_bytes())
        try:
            pub.verify(bytes.fromhex(sig), payload,
                       padding.PKCS1v15(), hashes.SHA256())
            return True
        except (InvalidSignature, ValueError):
            return False
    return False


def _iso_epoch(iso: str) -> float:
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    return datetime.fromisoformat(iso).timestamp()


def verify_ttl(att: dict) -> bool:
    issuer = att["issuerAsserted"]
    deadline = _iso_epoch(issuer["iat"]) + issuer["expSeconds"] + CLOCK_SKEW_SECONDS
    return EVAL_NOW <= deadline


def _parse_hash_only_identity(projection: str):
    try:
        obj = json.loads(projection)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"digest"}:
        return None
    digest = obj["digest"]
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        return None
    return digest


def verify_args_commitment(att: dict, runtime_args):
    """Return (ok, projection_match) for the single tool-call commitment.

    Mirrors vaara.attestation verify_args_commitment for the
    ArgsProjection shapes the vectors use (identity, hash-only-identity,
    and redacted). ArgsRef is not exercised by these fixtures.
    """
    args = att["payloadDerived"]["toolCalls"][0]["args"]
    if "projection" not in args:
        raise ValueError("vector uses a non-projection commitment")
    projection = args["projection"]
    pbytes = projection.encode("utf-8")
    if _sha256_hex(pbytes) != args["projectionDigest"]:
        return False, None
    runtime_canonical = _jcs(runtime_args)
    hash_only = _parse_hash_only_identity(projection)
    if hash_only is not None:
        if hash_only != _sha256_hex(runtime_canonical):
            return False, None
        return True, True
    return True, pbytes == runtime_canonical


def main() -> int:
    failures = 0
    cases = sorted((HERE / "normative").iterdir())
    for case in cases:
        if not case.is_dir():
            continue
        att = json.loads((case / "attestation.json").read_text())
        expected = json.loads((case / "expected.json").read_text())
        got = {
            "signature_ok": verify_signature(att),
            "ttl_ok": verify_ttl(att),
        }
        runtime_args_path = case / "runtime_args.json"
        if runtime_args_path.exists():
            ok, pm = verify_args_commitment(
                att, json.loads(runtime_args_path.read_text()))
            got["args_commitment_ok"] = ok
            got["projection_match"] = pm
        else:
            got["args_commitment_ok"] = None
            got["projection_match"] = None
        ok = got == expected
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {case.name}: {got}")
    print(f"\n{len(cases) - failures}/{len(cases)} cases matched expected.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
