#!/usr/bin/env python3
"""Independent checker for the vaara.authorization/v0 profile.

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
NOT import Vaara. For each committed case it reproduces five verdicts a third
party can confirm with only the grant, the runtime arguments, the evidence
record, and the receipt in hand, trusting nothing but the issuer's public key.

The headline case is ``deny``: the checker re-runs the capability evaluation in
plain Python and confirms the refusal was correct, then verifies the signature
over the receipt. That is a portable proof of a non-action, the artifact a
kernel egress filter structurally cannot produce.

Per case:

  grant_fingerprint_recomputes   sha256 over JCS(grant) equals the evidence's
                                 grantFingerprint: the exact signed grant.
  args_commitment_recomputes     sha256 over JCS(args) equals the evidence's
                                 argsCommitment: the exact arguments, while the
                                 raw arguments never enter the receipt.
  verdict_recomputes             re-running evaluate(capabilities, args) here,
                                 with no Vaara code, reproduces the evidence
                                 verdict/reason and the receipt decision.
  evidence_binding_resolves      sha256 over JCS(evidence) equals the receipt's
                                 decisionDerived.evidenceRef.digest.
  receipt_signature_ok           the ES256 signature verifies over the canonical
                                 (version, alg, backLink, decisionDerived,
                                 issuerAsserted) blocks against the public key.

Run: tests/vectors/authorization_v0/_check_independent.py
Exit 0 means every verdict matched expected.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.primitives import hashes

HERE = Path(__file__).resolve().parent
_CASES = ("allow", "deny")
_SIGNED_KEYS = ("version", "alg", "backLink", "decisionDerived", "issuerAsserted")


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _sha256_hex(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _load(case: str, name: str):
    return json.loads((HERE / case / name).read_text(encoding="utf-8"))


# --- capability evaluation, reproduced from scratch (no Vaara import) ---------


def _as_decimal(v):
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        return None
    try:
        return Decimal(str(v))
    except InvalidOperation:
        return None


def _check(cap, actual) -> bool:
    op, value = cap["op"], cap["value"]
    if op == "eq":
        return not isinstance(actual, bool) and str(actual) == value
    if op == "in":
        return not isinstance(actual, bool) and str(actual) in value
    a, bound = _as_decimal(actual), _as_decimal(value)
    if a is None or bound is None:
        return False
    return a <= bound if op == "le" else a >= bound


def _evaluate(capabilities, args):
    if not isinstance(args, dict):
        return (False, "capability_exceeded")
    named = {c["arg"] for c in capabilities}
    for key in args:
        if key not in named:
            return (False, "capability_uncovered")
    for cap in capabilities:
        if cap["arg"] not in args or not _check(cap, args[cap["arg"]]):
            return (False, "capability_exceeded")
    return (True, "ok")


# --- the five recompute verdicts ----------------------------------------------


def _grant_fingerprint_recomputes(grant, evidence) -> bool:
    return _sha256_hex(_jcs(grant)) == evidence.get("grantFingerprint")


def _args_commitment_recomputes(args, evidence) -> bool:
    return _sha256_hex(_jcs(args)) == evidence.get("argsCommitment")


def _verdict_recomputes(grant, args, evidence, receipt) -> bool:
    ok, reason = _evaluate(grant.get("capabilities", []), args)
    verdict = "allow" if ok else "deny"
    if (verdict, reason) != (evidence.get("verdict"), evidence.get("reason")):
        return False
    dd = receipt.get("decisionDerived", {})
    decision = "allow" if ok else "block"
    return dd.get("decision") == decision and dd.get("reason") == reason


def _evidence_binding_resolves(evidence, receipt) -> bool:
    ref = receipt.get("decisionDerived", {}).get("evidenceRef")
    if not isinstance(ref, dict) or ref.get("canonicalization") != "JCS":
        return False
    return _sha256_hex(_jcs(evidence)) == ref.get("digest")


def _receipt_signature_ok(receipt, pub) -> bool:
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


def main() -> int:
    pub = load_pem_public_key((HERE / "keys" / "es256_public.pem").read_bytes())
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))

    got = {}
    for case in _CASES:
        grant = _load(case, "grant.json")
        args = _load(case, "args.json")
        evidence = _load(case, "evidence.json")
        receipt = _load(case, "receipt.json")
        got[case] = {
            "grant_fingerprint_recomputes": _grant_fingerprint_recomputes(grant, evidence),
            "args_commitment_recomputes": _args_commitment_recomputes(args, evidence),
            "verdict_recomputes": _verdict_recomputes(grant, args, evidence, receipt),
            "evidence_binding_resolves": _evidence_binding_resolves(evidence, receipt),
            "receipt_signature_ok": _receipt_signature_ok(receipt, pub),
        }

    ok = got == expected
    for case in _CASES:
        for k, v in got[case].items():
            print(f"[{'OK' if v else 'FAIL'}] {case}.{k}: {v}")
    print(f"\n{'all verdicts matched expected' if ok else 'MISMATCH vs expected'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
