#!/usr/bin/env python3
"""Independent checker for the vaara.contiguity/v0 vectors.

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
NOT import Vaara. For each case it reproduces, from the held receipts alone, the
verdict a third party reaches with nothing but the issuer's public key:

  all_signatures_ok            every held receipt's ES256 signature verifies over
                               the canonical signed blocks.
  all_evidence_bindings_resolve   sha256 over JCS(evidence) equals each receipt's
                               decisionDerived.evidenceRef.digest, so the signed
                               receipt commits to the completeness block.
  contiguity                   re-running the gap check here, in plain Python,
                               reproduces ok / present / expected / missingSeqs.

The headline case is ``dropped``: with the seq-2 receipt withheld, the four held
receipts still carry the signed running count that says five exist, so seq 2 is a
provable gap. No issuer access, no external witness, no Vaara code.

Run: tests/vectors/contiguity_v0/_check_independent.py
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
_CASES = ("complete", "dropped")
_SIGNED_KEYS = ("version", "alg", "backLink", "decisionDerived", "issuerAsserted")


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _sha256_hex(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _load_case(case: str) -> list[dict]:
    """Held authz records for a case, sorted by filename (issuance order)."""
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((HERE / case).glob("*-authz.json"))
    ]


# --- per-receipt verdicts, reproduced from scratch (no Vaara import) ----------


def _evidence_binding_resolves(authz) -> bool:
    receipt = authz.get("record", {})
    evidence = authz.get("evidence", {})
    ref = receipt.get("decisionDerived", {}).get("evidenceRef")
    if not isinstance(ref, dict) or ref.get("canonicalization") != "JCS":
        return False
    return _sha256_hex(_jcs(evidence)) == ref.get("digest")


def _receipt_signature_ok(authz, pub) -> bool:
    receipt = authz.get("record", {})
    if receipt.get("alg") != "ES256":
        return False
    sig = receipt.get("signature", "")
    if len(sig) != 128:
        return False
    payload = _jcs({k: receipt[k] for k in _SIGNED_KEYS})
    raw = bytes.fromhex(sig)
    der = encode_dss_signature(
        int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
    )
    try:
        pub.verify(der, payload, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


# --- contiguity, reproduced from scratch --------------------------------------


def _contiguity(records) -> dict:
    """Reproduce verify_contiguity over the held evidence completeness blocks."""
    blocks = [
        ev["completeness"]
        for authz in records
        if isinstance((ev := authz.get("evidence")), dict)
        and isinstance(ev.get("completeness"), dict)
    ]
    if not blocks:
        return {"ok": True, "present": 0, "expected": 0, "missingSeqs": []}

    seqs = [int(b["seq"]) for b in blocks]
    counts = [int(b["runningCount"]) for b in blocks]
    expected = max(max(seqs) + 1, max(counts))
    missing = sorted(set(range(expected)) - set(seqs))
    duplicates = len(seqs) != len(set(seqs))
    mismatch = any(int(b["runningCount"]) != int(b["seq"]) + 1 for b in blocks)
    present = len(blocks)
    ok = not missing and not duplicates and not mismatch and present == expected
    return {"ok": ok, "present": present, "expected": expected, "missingSeqs": missing}


def main() -> int:
    pub = load_pem_public_key((HERE / "keys" / "es256_public.pem").read_bytes())
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))

    got = {}
    for case in _CASES:
        records = _load_case(case)
        got[case] = {
            "all_signatures_ok": all(_receipt_signature_ok(r, pub) for r in records),
            "all_evidence_bindings_resolve": all(
                _evidence_binding_resolves(r) for r in records
            ),
            "contiguity": _contiguity(records),
        }

    ok = got == expected
    for case in _CASES:
        for k, v in got[case].items():
            mark = "OK" if (v is True or (k == "contiguity" and v == expected[case][k])) else "FAIL"
            print(f"[{mark}] {case}.{k}: {v}")
    print(f"\n{'all verdicts matched expected' if ok else 'MISMATCH vs expected'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
