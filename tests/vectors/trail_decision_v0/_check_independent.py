#!/usr/bin/env python3
"""Independent checker for the vaara.trail-decision/v0 profile.

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
NOT import Vaara. The receipts here were written by the engine's own sink over
a real SQLite trail; ``trail-hashes.json`` holds that trail's record hashes, so
the checker confirms each receipt against the trail as well as against itself.

Per file:

  signature    the ES256 signature verifies over the canonical (version, alg,
               backLink, decisionDerived, issuerAsserted) blocks against
               issuer-es256.pub.pem.
  evidence     sha256 over JCS(evidence) equals the receipt's
               decisionDerived.evidenceRef.digest.
  trail        the trail holds a record with the evidence's recordId, and its
               hash, written as sha256:<hex>, equals the evidence's recordHash.

Run: tests/vectors/trail_decision_v0/_check_independent.py
Exit 0 means every verdict matched expected.json.
"""

from __future__ import annotations

import hashlib
import json
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


def _trail_ok(evidence, trail) -> bool:
    stored = trail.get(evidence.get("recordId", ""))
    return stored is not None and "sha256:" + stored == evidence.get("recordHash")


def main() -> int:
    pub = load_pem_public_key((HERE / "issuer-es256.pub.pem").read_bytes())
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    trail = json.loads((HERE / "trail-hashes.json").read_text(encoding="utf-8"))

    got = {}
    for name in sorted(expected):
        body = json.loads((HERE / name).read_text(encoding="utf-8"))
        receipt, evidence = body["receipt"], body["evidence"]
        got[name] = {
            "evidence": _evidence_ok(evidence, receipt),
            "signature": _signature_ok(receipt, pub),
            "trail": _trail_ok(evidence, trail),
        }

    ok = got == expected
    for name, verdicts in got.items():
        for k, v in verdicts.items():
            print(f"[{'OK' if v == expected[name][k] else 'MISMATCH'}] {name}.{k}: {v}")
    print(f"\n{'all verdicts matched expected' if ok else 'MISMATCH vs expected'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
