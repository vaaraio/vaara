#!/usr/bin/env python3
"""Independent conformance checker for the agent_identity_v0 vectors.

Imports only the standard library plus ``cryptography`` and ``rfc8785``.
It does not import Vaara. For each committed case it reproduces level-2
pinned-resolvable identity verification: confirm the receipt's did:web
``iss`` matches the DID document id, then verify the ES256 receipt
signature against a verification key the document lists. Verdicts are
compared against ``expected.json``.

A second implementation that can run this file (or reproduce its logic)
demonstrates resolvable agent identity is consumable without depending on
Vaara. Run: ``python tests/vectors/agent_identity_v0/_check_independent.py``.
Exit code 0 means every case matched its expected verdict.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

HERE = Path(__file__).resolve().parent
_RECEIPT_BLOCKS = ("version", "alg", "backLink", "outcomeDerived", "receiptAsserted")


def _b64u_to_int(value: str) -> int:
    pad = "=" * (-len(value) % 4)
    return int.from_bytes(base64.urlsafe_b64decode(value + pad), "big")


def _jwk_to_ec_key(jwk: dict):
    if jwk.get("kty") != "EC" or jwk.get("crv") != "P-256":
        return None
    numbers = ec.EllipticCurvePublicNumbers(
        _b64u_to_int(jwk["x"]), _b64u_to_int(jwk["y"]), ec.SECP256R1()
    )
    return numbers.public_key()


def _es256_verifies(payload: bytes, signature_hex: str, public_key) -> bool:
    if len(signature_hex) != 128:
        return False
    try:
        raw = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    der = encode_dss_signature(
        int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
    )
    try:
        public_key.verify(der, payload, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


def _evaluate(case: dict) -> dict:
    receipt = case["receipt"]
    doc = case["didDocument"]
    iss = receipt["receiptAsserted"]["iss"]
    if not iss.startswith("did:web:") or doc.get("id") != iss:
        return {"resolved": False, "bound": False, "keyid": None}
    if receipt["alg"] != "ES256":
        return {"resolved": False, "bound": False, "keyid": None}

    payload = rfc8785.dumps({k: receipt[k] for k in _RECEIPT_BLOCKS})
    for method in doc.get("verificationMethod", []):
        key = _jwk_to_ec_key(method.get("publicKeyJwk", {}))
        if key is None:
            continue
        if _es256_verifies(payload, receipt["signature"], key):
            return {"resolved": True, "bound": True, "keyid": method.get("id")}
    return {"resolved": True, "bound": False, "keyid": None}


def main() -> int:
    expected = json.loads((HERE / "expected.json").read_text())
    failures = []
    for name in ("bound", "unbound"):
        case = json.loads((HERE / f"{name}.json").read_text())
        got = _evaluate(case)
        want = expected[name]
        if got != want:
            failures.append(f"{name}: expected {want}, got {got}")
        else:
            print(f"{name}: OK {got}")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print("\nall agent_identity_v0 cases matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
