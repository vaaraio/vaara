#!/usr/bin/env python3
"""Independent checker for the generic external-execution-evidence binding (external_evidence_v0).

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
NOT import Vaara. It reproduces, from the held ``external_execution_evidence``
slots and the vaara.receipt/v1 authorization receipts alone, every verdict a
third party reaches with nothing but the issuer's public key:

  all_slots_resolve          for each held item, the external slot points at the
                             recomputable receipt: slot.evidenceHash equals the
                             receipt's decisionDerived.evidenceRef.digest and
                             equals sha256(JCS(evidence)); the receipt names the
                             linked call (evidenceRef.ref == mcp:call/<linkedCallId>);
                             and slot.evidenceType equals the receipt's evidence
                             schema. The slot is a view onto a producer the holder
                             recomputes, not an opaque hash a service vouches for.
  all_evidence_bindings_resolve  sha256 over JCS(evidence) equals each receipt's
                             decisionDerived.evidenceRef.digest.
  all_signatures_ok          every receipt's ES256 signature verifies over the
                             canonical signed blocks.
  contiguity                 re-running the gap check over the held completeness
                             blocks reproduces ok / present / expected / missingSeqs.

The headline case is ``dropped``: the seq-1 item is withheld, slot and receipt
both. A verifier holding only the external slots sees {0, 2} with no inherent
ordering or count and cannot tell call 1 ever existed. The held receipts carry the
signed running count that says three exist, so seq 1 is a provable gap inside the
trace boundary, from the held set alone. That is the half the slot does not give.

Run: tests/vectors/external_evidence_v0/_check_independent.py
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
_AUTHORIZATION_SCHEMA = "vaara.authorization/v0"


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _sha256_hex(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _load_case(case: str) -> list[dict]:
    """Held items for a case, sorted by filename (issuance order)."""
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((HERE / case).glob("*-authz.json"))
    ]


def _evidence_ref(item: dict) -> dict:
    ref = item.get("record", {}).get("decisionDerived", {}).get("evidenceRef")
    return ref if isinstance(ref, dict) else {}


def _slot_resolves(item: dict) -> bool:
    slot = item.get("slot")
    if not isinstance(slot, dict):
        return False
    ref = _evidence_ref(item)
    evidence_digest = _sha256_hex(_jcs(item.get("evidence", {})))
    return (
        slot.get("evidenceHash") == ref.get("digest")
        and slot.get("evidenceHash") == evidence_digest
        and ref.get("ref") == "mcp:call/" + str(slot.get("linkedCallId"))
        and slot.get("evidenceType") == ref.get("schema") == _AUTHORIZATION_SCHEMA
    )


def _evidence_binding_resolves(item: dict) -> bool:
    ref = _evidence_ref(item)
    if ref.get("canonicalization") != "JCS":
        return False
    return _sha256_hex(_jcs(item.get("evidence", {}))) == ref.get("digest")


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


def _contiguity(items) -> dict:
    blocks = [
        ev["completeness"]
        for item in items
        if isinstance((ev := item.get("evidence")), dict)
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
        items = _load_case(case)
        got[case] = {
            "all_signatures_ok": all(_receipt_signature_ok(i, pub) for i in items),
            "all_evidence_bindings_resolve": all(
                _evidence_binding_resolves(i) for i in items
            ),
            "all_slots_resolve": all(_slot_resolves(i) for i in items),
            "contiguity": _contiguity(items),
        }

    ok = got == expected
    for case in _CASES:
        for k, v in got[case].items():
            mark = "OK" if v == expected.get(case, {}).get(k) else "FAIL"
            print(f"[{mark}] {case}.{k}: {v}")
    print(f"\n{'all verdicts matched expected' if ok else 'MISMATCH vs expected'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
