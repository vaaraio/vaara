#!/usr/bin/env python3
"""Independent checker for the sealed-class enforcement gate (class_gate_v0).

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
NOT import Vaara. From the held vaara.receipt/v1 receipts and the signed terminal
seal alone, with nothing but the issuer's public key, it reproduces every verdict:

  all_signatures_ok   every held receipt's ES256 signature verifies. The signature
                      covers the ``record`` half only (``_SIGNED_KEYS``); it does
                      not, on its own, reach the ``evidence`` block where
                      ``maxClass`` lives.
  all_evidence_bound  every held receipt's ``evidence`` block recomputes to the
                      ``decisionDerived.evidenceRef.digest`` carried in its signed
                      record. THIS is what puts the sealed ``maxClass`` under
                      signature: the digest is signed, and the binding proves the
                      evidence (hence the class) is the evidence that was signed.
                      Tamper ``maxClass`` and this fails even though the record
                      signature still verifies.
  contiguity          the seal-aware gap check over the held completeness blocks
                      reproduces ok / present / expected / missingSeqs.
  permit / reason / worstCaseClass
                      the gate decision: the consumer holds ``permittedClasses``
                      and permits iff the sealed worst-case class is a member,
                      failing closed (``unbounded_no_sealed_class``) when no class
                      is sealed. The sealed ``maxClass`` is consumed ONLY from a
                      seal whose evidence binds; an unbound (relabeled) seal
                      contributes no class, so the gate fails closed. Membership,
                      not an ordering over class labels.

The point is ``permit_gap_bounded``: an interior receipt is withheld, so the
boundary has a provable gap, yet the gate permits, because the seal bounds the
missing record's worst case at the permitted class. ``deny_relabeled`` is the
adversary: a seal whose ``maxClass`` was relabeled to a permitted class after
signing; the record signature still verifies but the evidence no longer binds, so
the gate refuses to consume the class and fails closed. A third party reaches
both decisions here from the committed bytes alone, no Vaara and no log.

Run: tests/vectors/class_gate_v0/_check_independent.py
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


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _evidence_bound_ok(item: dict) -> bool:
    """The evidence block recomputes to the digest its signed record commits to.

    ``record.decisionDerived.evidenceRef.digest`` is inside the ES256-signed
    payload (``decisionDerived`` is a ``_SIGNED_KEYS`` member). It pins
    ``sha256:`` + sha256(JCS(evidence)). Recomputing it and comparing is what
    carries the unsigned ``evidence`` block (where ``maxClass`` lives) under
    signature. A relabeled ``maxClass`` breaks this while the record signature,
    which never covered the evidence, still verifies.
    """
    evidence = item.get("evidence")
    if not isinstance(evidence, dict):
        return False
    ref = item.get("record", {}).get("decisionDerived", {}).get("evidenceRef", {})
    signed = ref.get("digest")
    if ref.get("canonicalization") != "JCS" or not isinstance(signed, str):
        return False
    computed = "sha256:" + hashlib.sha256(_jcs(evidence)).hexdigest()
    return computed == signed


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


def _seq_blocks(items) -> list[dict]:
    """Per-call (non-seal) completeness blocks, the input to the gap check."""
    return [
        ev["completeness"]
        for item in items
        if isinstance((ev := item.get("evidence")), dict)
        and isinstance(ev.get("completeness"), dict)
        and not ev["completeness"].get("sealed")
    ]


def _sealed_blocks(items) -> list[dict]:
    """Sealed completeness blocks from seals whose evidence binds to signature.

    An unbound seal is not trusted: it contributes neither its ``total`` nor its
    ``maxClass``. A relabeled seal therefore reverts to "no seal" — a bounded gap
    becomes an unbounded one and the gate fails closed.
    """
    out = []
    for item in items:
        if not _evidence_bound_ok(item):
            continue
        comp = item["evidence"].get("completeness")
        if isinstance(comp, dict) and comp.get("sealed"):
            out.append(comp)
    return out


def _contiguity_and_seal(items):
    """Seal-aware gap check; returns (contiguity_dict, sealed_max_class)."""
    seq_blocks = _seq_blocks(items)
    sealed = _sealed_blocks(items)
    sealed_total = max((int(b["total"]) for b in sealed), default=0)
    max_class = next(
        (str(b["maxClass"]) for b in sealed if b.get("maxClass") is not None),
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
            "all_evidence_bound": all(_evidence_bound_ok(i) for i in items),
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
