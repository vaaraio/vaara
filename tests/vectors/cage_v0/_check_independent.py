#!/usr/bin/env python3
"""Independent checker for the cage block of vaara.trail-decision/v0.

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
NOT import Vaara. The valid receipts were written by the engine's own sink over
a real SQLite trail.

Per file:

  signature    the ES256 signature verifies over the canonical (version, alg,
               backLink, decisionDerived, issuerAsserted) blocks against
               issuer-es256.pub.pem.
  evidence     sha256 over JCS(evidence) equals the receipt's
               decisionDerived.evidenceRef.digest.
  cage         the evidence's cage block keeps the rules of the block:
               driver is a non-empty string and confirmed a boolean;
               driver "none" carries those two members only, with confirmed
               false; any other driver carries a basis, and confirmed true
               needs a basis other than "none" or "declared"; configDigest,
               when present, is "sha256:" and 64 lowercase hex. Members this
               checker does not know are ignored. A record without a cage
               block passes: it makes no claim.

Run: tests/vectors/cage_v0/_check_independent.py
Exit 0 means every verdict matched expected.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.primitives.serialization import load_pem_public_key

HERE = Path(__file__).resolve().parent
_SIGNED_KEYS = ("version", "alg", "backLink", "decisionDerived", "issuerAsserted")
_JCS_ALIASES = ("jcs-rfc8785", "JCS", "jcs-json-v1")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_STRINGS = ("upstream", "configDigest", "basis", "name")


def _sha256(obj) -> str:
    return "sha256:" + hashlib.sha256(rfc8785.dumps(obj)).hexdigest()


def _signature_ok(receipt, pub) -> bool:
    if receipt.get("alg") != "ES256":
        return False
    sig = receipt.get("signature", "")
    if len(sig) != 128:
        return False
    raw = bytes.fromhex(sig)
    der = encode_dss_signature(
        int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
    )
    try:
        pub.verify(der, rfc8785.dumps({k: receipt[k] for k in _SIGNED_KEYS}),
                   ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, KeyError):
        return False


def _evidence_ok(evidence, receipt) -> bool:
    ref = receipt.get("decisionDerived", {}).get("evidenceRef", {})
    if ref.get("canonicalization") not in _JCS_ALIASES:
        return False
    return _sha256(evidence) == ref.get("digest")


def _cage_ok(evidence) -> bool:
    if "cage" not in evidence:
        return True
    cage = evidence["cage"]
    if not isinstance(cage, dict):
        return False
    driver, confirmed = cage.get("driver"), cage.get("confirmed")
    if not isinstance(driver, str) or not driver or not isinstance(confirmed, bool):
        return False
    if driver == "none":
        return confirmed is False and not any(k in cage for k in _STRINGS)
    if any(k in cage and not isinstance(cage[k], str) for k in _STRINGS):
        return False
    basis = cage.get("basis")
    if not basis:
        return False
    if confirmed and basis in ("none", "declared"):
        return False
    if "configDigest" in cage and not _DIGEST.match(cage["configDigest"]):
        return False
    return True


def main() -> int:
    pub = load_pem_public_key((HERE / "issuer-es256.pub.pem").read_bytes())
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))

    got = {}
    for name in sorted(expected):
        body = json.loads((HERE / name).read_text(encoding="utf-8"))
        receipt, evidence = body["receipt"], body["evidence"]
        got[name] = {
            "cage": _cage_ok(evidence),
            "evidence": _evidence_ok(evidence, receipt),
            "signature": _signature_ok(receipt, pub),
        }

    ok = got == expected
    for name, verdicts in got.items():
        for k, v in verdicts.items():
            print(f"[{'OK' if v == expected[name][k] else 'MISMATCH'}] {name}.{k}: {v}")
    print(f"\n{'all verdicts matched expected' if ok else 'MISMATCH vs expected'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
