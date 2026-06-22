#!/usr/bin/env python3
"""Independent checker for the sealed-class enforcement gate (class_gate_v0).

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
NOT import Vaara. From the held vaara.receipt/v1 receipts and the signed terminal
seal alone, with nothing but the issuer's public key, it reproduces every verdict:

  all_signatures_ok   every held receipt's ES256 signature verifies, the seal
                      included, so the sealed ``maxClass`` bound is under signature.
  contiguity          the seal-aware gap check over the held completeness blocks
                      reproduces ok / present / expected / missingSeqs.
  permit / reason / worstCaseClass
                      the gate decision: the consumer holds ``permittedClasses``
                      and permits iff the sealed worst-case class is a member,
                      failing closed (``unbounded_no_sealed_class``) when no class
                      is sealed. Membership, not an ordering over class labels.

The point is ``permit_gap_bounded``: an interior receipt is withheld, so the
boundary has a provable gap, yet the gate permits, because the seal bounds the
missing record's worst case at the permitted class. A third party reaches that
decision here from the committed bytes alone, no Vaara and no log.

Run: tests/vectors/class_gate_v0/_check_independent.py
Exit 0 means every verdict matched expected.json.
"""

from __future__ import annotations

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


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _load_case(case: str) -> list[dict]:
    """Held items for a case, sorted by filename (issuance order, seal last)."""
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((HERE / case).glob("*-authz.json"))
    ]


def _receipt_signature_ok(item: dict, pub) -> bool:
    receipt = item.get("record", {})
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


def _completeness_blocks(items) -> list[dict]:
    return [
        ev["completeness"]
        for item in items
        if isinstance((ev := item.get("evidence")), dict)
        and isinstance(ev.get("completeness"), dict)
    ]


def _contiguity_and_seal(items):
    """Seal-aware gap check; returns (contiguity_dict, sealed_max_class)."""
    blocks = _completeness_blocks(items)
    seq_blocks = [b for b in blocks if not b.get("sealed")]
    sealed_total = max((int(b["total"]) for b in blocks if b.get("sealed")), default=0)
    max_class = next(
        (str(b["maxClass"]) for b in blocks if b.get("sealed") and b.get("maxClass") is not None),
        None,
    )
    if not seq_blocks:
        ok = sealed_total == 0
        return (
            {"ok": ok, "present": 0, "expected": sealed_total,
             "missingSeqs": list(range(sealed_total))},
            max_class,
        )
    seqs = [int(b["seq"]) for b in seq_blocks]
    counts = [int(b["runningCount"]) for b in seq_blocks]
    expected = max(max(seqs) + 1, max(counts), sealed_total)
    missing = sorted(set(range(expected)) - set(seqs))
    duplicates = len(seqs) != len(set(seqs))
    mismatch = any(int(b["runningCount"]) != int(b["seq"]) + 1 for b in seq_blocks)
    present = len(seq_blocks)
    ok = not missing and not duplicates and not mismatch and present == expected
    return (
        {"ok": ok, "present": present, "expected": expected, "missingSeqs": missing},
        max_class,
    )


def _gate(max_class, permitted) -> dict:
    """Membership gate, fail-closed when no class is sealed."""
    if max_class is None:
        return {"permit": False, "reason": "unbounded_no_sealed_class", "worstCaseClass": None}
    permit = max_class in permitted
    return {
        "permit": permit,
        "reason": "permitted" if permit else "class_not_permitted",
        "worstCaseClass": max_class,
    }


def main() -> int:
    pub = load_pem_public_key((HERE / "keys" / "es256_public.pem").read_bytes())
    spec = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    permitted = spec["permittedClasses"]
    expected = spec["cases"]

    got = {}
    for case in expected:
        items = _load_case(case)
        contiguity, max_class = _contiguity_and_seal(items)
        decision = _gate(max_class, permitted)
        got[case] = {
            "all_signatures_ok": all(_receipt_signature_ok(i, pub) for i in items),
            "permit": decision["permit"],
            "reason": decision["reason"],
            "worstCaseClass": decision["worstCaseClass"],
            "contiguity": contiguity,
        }

    ok = got == expected
    for case in expected:
        for k, v in got[case].items():
            mark = "OK" if v == expected.get(case, {}).get(k) else "FAIL"
            print(f"[{mark}] {case}.{k}: {v}")
    print(f"\n{'all verdicts matched expected' if ok else 'MISMATCH vs expected'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
