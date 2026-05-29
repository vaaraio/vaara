#!/usr/bin/env python3
"""Independent conformance checker for the v0 execution-receipt vectors.

Imports only the standard library plus ``cryptography`` and ``rfc8785``.
It does not import Vaara. It reads the committed fixtures from disk and
reproduces the canonical bytes, signature verification, back-link
verification, and result-commitment verification for each case, then
compares against ``expected.json``.

A second implementation that can run this file (or reproduce its logic)
demonstrates that the receipt format is consumable without depending on
Vaara. Run: ``python tests/vectors/execution_receipt_v0/_check_independent.py``.
Exit code 0 means every case matched its expected verdict.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sys
from pathlib import Path

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

HERE = Path(__file__).resolve().parent
KEYS = HERE / "keys"


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _sha256_hex(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _signing_payload(receipt: dict) -> bytes:
    body = {k: receipt[k] for k in (
        "version", "alg", "backLink", "outcomeDerived", "receiptAsserted")}
    return _jcs(body)


def verify_signature(receipt: dict) -> bool:
    payload = _signing_payload(receipt)
    alg = receipt["alg"]
    sig = receipt["signature"]
    if alg == "HS256":
        secret = (KEYS / "hs256_secret.bin").read_bytes()
        expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig)
    if alg == "ES256":
        pub = serialization.load_pem_public_key(
            (KEYS / "es256_public.pem").read_bytes())
        if len(sig) != 128:
            return False
        raw = bytes.fromhex(sig)
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
        except InvalidSignature:
            return False
    return False


def verify_back_link(receipt: dict, attestation: dict) -> bool:
    expected = _sha256_hex(_jcs(attestation))
    bl = receipt["backLink"]
    if not hmac.compare_digest(bl["attestationDigest"], expected):
        return False
    return bl["attestationNonce"] == attestation["issuerAsserted"]["nonce"]


def verify_result_commitment(receipt: dict, runtime_result) -> bool:
    commitment = receipt["outcomeDerived"].get("resultCommitment")
    if commitment is None:
        raise ValueError("case has no resultCommitment")
    projection = commitment["projection"]
    pbytes = projection.encode("utf-8")
    if _sha256_hex(pbytes) != commitment["projectionDigest"]:
        return False
    runtime_canonical = _jcs(runtime_result)
    obj = json.loads(projection)
    if isinstance(obj, dict) and set(obj) == {"digest"}:
        return obj["digest"] == _sha256_hex(runtime_canonical)
    return pbytes == runtime_canonical


def main() -> int:
    failures = 0
    cases = sorted((HERE / "normative").iterdir())
    for case in cases:
        if not case.is_dir():
            continue
        receipt = json.loads((case / "receipt.json").read_text())
        attestation = json.loads((case / "attestation.json").read_text())
        expected = json.loads((case / "expected.json").read_text())
        got = {
            "signature_ok": verify_signature(receipt),
            "back_link_ok": verify_back_link(receipt, attestation),
        }
        rr = case / "runtime_result.json"
        if expected["result_commitment_ok"] is None:
            got["result_commitment_ok"] = None
        else:
            runtime_result = json.loads(rr.read_text())
            got["result_commitment_ok"] = verify_result_commitment(
                receipt, runtime_result)
        ok = got == expected
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {case.name}: {got}")
    print(f"\n{len(cases) - failures}/{len(cases)} cases matched expected.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
