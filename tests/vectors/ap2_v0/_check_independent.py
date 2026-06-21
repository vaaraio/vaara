#!/usr/bin/env python3
"""Independent checker for the AP2 <-> Vaara binding profile (ap2_v0).

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
NOT import Vaara. It reproduces, from the committed AP2 Payment Evidence Frame
and the held vaara.receipt/v1 authorization receipts alone, every verdict a
third party reaches with nothing but the issuer's public key:

  frame_id_recomputes        sha256 over JCS(frame), with frame_id and signature
                             excluded from the preimage, equals the PEF
                             frame_id (AP2 PR #274). The checkout is
                             content-addressed.
  receipt_hash_recomputes    sha256 over JCS(inner receipt) equals the PEF
                             receipt_hash: the AP2 Checkout Receipt is itself
                             content-addressed inside the frame.
  all_receipts_name_checkout every held receipt's evidenceRef.ref equals
                             ap2:checkout/<frame_id>, under signature: the
                             post-checkout action names the checkout it followed.
  all_evidence_bindings_resolve  sha256 over JCS(evidence) equals each receipt's
                             decisionDerived.evidenceRef.digest.
  all_signatures_ok          every receipt's ES256 signature verifies over the
                             canonical signed blocks.
  contiguity                 re-running the gap check over the held completeness
                             blocks reproduces ok / present / expected /
                             missingSeqs.

The headline case is ``dropped``: with the seq-1 receipt withheld, the held
receipts still carry the signed running count that says three exist, so seq 1 is
a provable gap inside the AP2 task boundary. No AP2 access, no external witness.

Run: tests/vectors/ap2_v0/_check_independent.py
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
_PREIMAGE_EXCLUDED = ("frame_id", "signature")


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


# --- AP2 frame, reproduced from scratch (no Vaara import) ---------------------


def _frame_id_recomputes(pef: dict) -> bool:
    core = {k: v for k, v in pef.items() if k not in _PREIMAGE_EXCLUDED}
    return _sha256_hex(_jcs(core)) == pef.get("frame_id")


def _receipt_hash_recomputes(pef: dict) -> bool:
    return _sha256_hex(_jcs(pef.get("receipt"))) == pef.get("receipt_hash")


# --- per-receipt verdicts -----------------------------------------------------


def _names_checkout(authz, checkout_ref) -> bool:
    ref = authz.get("record", {}).get("decisionDerived", {}).get("evidenceRef", {})
    return isinstance(ref, dict) and ref.get("ref") == checkout_ref


def _evidence_binding_resolves(authz) -> bool:
    receipt = authz.get("record", {})
    ref = receipt.get("decisionDerived", {}).get("evidenceRef")
    if not isinstance(ref, dict) or ref.get("canonicalization") != "JCS":
        return False
    return _sha256_hex(_jcs(authz.get("evidence", {}))) == ref.get("digest")


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
    pef = json.loads((HERE / "checkout" / "pef.json").read_text(encoding="utf-8"))
    checkout_ref = f"ap2:checkout/{pef.get('frame_id')}"

    got = {
        "frame_id_recomputes": _frame_id_recomputes(pef),
        "receipt_hash_recomputes": _receipt_hash_recomputes(pef),
    }
    for case in _CASES:
        records = _load_case(case)
        got[case] = {
            "all_signatures_ok": all(_receipt_signature_ok(r, pub) for r in records),
            "all_evidence_bindings_resolve": all(
                _evidence_binding_resolves(r) for r in records
            ),
            "all_receipts_name_checkout": all(
                _names_checkout(r, checkout_ref) for r in records
            ),
            "contiguity": _contiguity(records),
        }

    ok = got == expected
    for key in ("frame_id_recomputes", "receipt_hash_recomputes"):
        print(f"[{'OK' if got[key] else 'FAIL'}] {key}: {got[key]}")
    for case in _CASES:
        for k, v in got[case].items():
            mark = "OK" if v == expected.get(case, {}).get(k) else "FAIL"
            print(f"[{mark}] {case}.{k}: {v}")
    print(f"\n{'all verdicts matched expected' if ok else 'MISMATCH vs expected'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
