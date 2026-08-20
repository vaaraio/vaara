#!/usr/bin/env python3
"""Independent checker for release_condition_v0: does a receipt release money?

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
not import Vaara. Every verdict below is recomputed from the bytes of the case
file and the committed public keys, so a passing run is a property of those
bytes rather than of the reference implementation.

The direction is the inversion this suite exists to pin. Elsewhere a payment
gates access and the settlement evidence lands inside a receipt. Here money is
held against a signed release condition, and a Vaara receipt proving the
authorised action happened is what releases it.

Four states, and the reason space is partitioned so they cannot collapse:

  released  the authorised action is proved; the held value moves
  held      the evidence is sound and does not satisfy the condition, or none
            has been presented yet
  expired   the window named by the condition closed
  refused   the presented artifact fails as evidence: a broken condition
            signature, a receipt under a key the condition does not pin, a
            broken receipt signature, or evidence that does not resolve to the
            digest the receipt signed

That partition is the whole point. A verifier that proved nothing must never
read as green, and must never read as the same false as a genuine failure:
``held`` because nothing arrived and ``refused`` because a receipt was tampered
with are different facts, and answering both with one boolean discards the
difference between "not yet" and "no".

The order is soundness, then the clock, then sufficiency. Soundness comes first
so a closed window cannot swallow a tampering finding; the clock comes before
sufficiency so a closed window is reported as the reason the money is not moving.

Both signatures cover the JCS encoding of their own document with the
``signature`` field removed: Ed25519 for the condition, ES256 (raw r||s hex) for
the receipt. A reader needs no Vaara-specific canonicalization to reproduce them.

Run: python3 tests/vectors/release_condition_v0/_check_independent.py
Exit 0 means every case produced the state and reason recorded in expected.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

HERE = Path(__file__).resolve().parent

SCHEMA = "vaara.release-condition/v0"
RECEIPT_ALG = "ES256"

# SPEC.md section 1: "jcs-rfc8785", "JCS" and "jcs-json-v1" are accepted aliases
# for the same algorithm; consumers MUST accept all three.
_JCS_ALIASES = ("jcs-rfc8785", "JCS", "jcs-json-v1")

_REQUIRED_CONDITION_KEYS = (
    "alg", "conditionId", "holds", "issuer", "notAfter", "requires", "schema",
    "version", "signature",
)
_REQUIRED_HELD_KEYS = ("amount", "asset", "network", "payee")
_REQUIRED_REQUIRES_KEYS = (
    "actionDigest", "decision", "evidenceSchema", "grantFingerprint",
    "receiptIssuer", "receiptKeyFingerprint",
)
_DIGEST_REQUIRES_KEYS = ("actionDigest", "grantFingerprint", "receiptKeyFingerprint")
_RECEIPT_BLOCKS = ("version", "alg", "backLink", "decisionDerived", "issuerAsserted")

# reason -> state. The partition, as data. Nothing below picks a state directly.
REASON_STATE = {
    "receipt_matches": "released",
    "receipt_absent": "held",
    "evidence_schema_mismatch": "held",
    "issuer_not_accepted": "held",
    "decision_not_accepted": "held",
    "authorization_mismatch": "held",
    "action_digest_mismatch": "held",
    "condition_expired": "expired",
    "condition_malformed": "refused",
    "condition_key_absent": "refused",
    "condition_signature_invalid": "refused",
    "receipt_malformed": "refused",
    "receipt_key_absent": "refused",
    "receipt_key_untrusted": "refused",
    "receipt_signature_invalid": "refused",
    "evidence_digest_mismatch": "refused",
}


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _digest(obj) -> str:
    return "sha256:" + hashlib.sha256(_jcs(obj)).hexdigest()


def _without_signature(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "signature"}


def _epoch(iso: str):
    try:
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        return datetime.fromisoformat(iso).timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


def _load_public_key(rel_path: str):
    return serialization.load_pem_public_key((HERE / rel_path).read_bytes())


def key_fingerprint(public_key) -> str:
    """sha256 over the SubjectPublicKeyInfo DER, so PEM formatting cannot move it."""
    der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return "sha256:" + hashlib.sha256(der).hexdigest()


def condition_well_formed(condition) -> bool:
    if not isinstance(condition, dict) or condition.get("schema") != SCHEMA:
        return False
    if any(k not in condition for k in _REQUIRED_CONDITION_KEYS):
        return False
    if not isinstance(condition["version"], int) or isinstance(condition["version"], bool):
        return False
    for key in ("alg", "conditionId", "issuer", "signature"):
        if not isinstance(condition[key], str) or not condition[key]:
            return False
    held, requires = condition["holds"], condition["requires"]
    if not isinstance(held, dict) or not isinstance(requires, dict):
        return False
    if any(not isinstance(held.get(k), str) or not held.get(k) for k in _REQUIRED_HELD_KEYS):
        return False
    if any(
        not isinstance(requires.get(k), str) or not requires.get(k)
        for k in _REQUIRED_REQUIRES_KEYS
    ):
        return False
    if any(not requires[k].startswith("sha256:") for k in _DIGEST_REQUIRES_KEYS):
        return False
    return _epoch(condition["notAfter"]) is not None


def condition_signature_ok(condition: dict, public_key) -> bool:
    try:
        signature = bytes.fromhex(condition["signature"])
    except (ValueError, TypeError):
        return False
    try:
        public_key.verify(signature, _jcs(_without_signature(condition)))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def receipt_well_formed(receipt) -> bool:
    if not isinstance(receipt, dict):
        return False
    if any(k not in receipt for k in (*_RECEIPT_BLOCKS, "signature")):
        return False
    derived = receipt["decisionDerived"]
    asserted = receipt["issuerAsserted"]
    if not isinstance(derived, dict) or not isinstance(asserted, dict):
        return False
    if "decision" not in derived or "decidedAt" not in derived:
        return False
    return isinstance(asserted.get("iss"), str) and bool(asserted["iss"])


def receipt_signature_ok(receipt: dict, public_key) -> bool:
    """ES256 over JCS({version, alg, backLink, decisionDerived, issuerAsserted}).

    The signed blocks are exactly the wire record minus its own signature, so
    this is the same "canonicalize the document without its signature" rule the
    condition uses one function above.
    """
    signature = receipt.get("signature", "")
    if not isinstance(signature, str) or len(signature) != 128:
        return False
    try:
        raw = bytes.fromhex(signature)
    except ValueError:
        return False
    payload = _jcs({k: receipt[k] for k in _RECEIPT_BLOCKS})
    der = encode_dss_signature(
        int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
    )
    try:
        public_key.verify(der, payload, ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def evaluate(case: dict) -> str:
    """Recompute the release reason for one case. Returns a REASON_STATE key."""
    condition = case.get("condition")
    now = _epoch(case["now"])
    if now is None:
        raise ValueError("case 'now' is not an ISO 8601 instant")

    # 1. the condition must be sound
    if not condition_well_formed(condition):
        return "condition_malformed"
    condition_key_path = case.get("condition_key")
    if not condition_key_path:
        return "condition_key_absent"
    if not condition_signature_ok(condition, _load_public_key(condition_key_path)):
        return "condition_signature_invalid"

    requires = condition["requires"]
    receipt = case.get("receipt")
    evidence = case.get("evidence")

    # 2. a presented receipt must be sound, BEFORE the clock is consulted, so an
    #    expired window cannot swallow a forgery.
    if receipt is not None:
        if not receipt_well_formed(receipt):
            return "receipt_malformed"
        receipt_key_path = case.get("receipt_key")
        if not receipt_key_path:
            return "receipt_key_absent"
        if receipt["alg"] != RECEIPT_ALG:
            return "receipt_key_untrusted"
        public_key = _load_public_key(receipt_key_path)
        if key_fingerprint(public_key) != requires["receiptKeyFingerprint"]:
            return "receipt_key_untrusted"
        if not receipt_signature_ok(receipt, public_key):
            return "receipt_signature_invalid"
        ref = receipt["decisionDerived"].get("evidenceRef")
        if not isinstance(ref, dict) or ref.get("canonicalization") not in _JCS_ALIASES:
            return "evidence_digest_mismatch"
        if not isinstance(evidence, dict) or _digest(evidence) != ref.get("digest"):
            return "evidence_digest_mismatch"

    # 3. the clock
    if now > _epoch(condition["notAfter"]):
        return "condition_expired"

    # 4. sound evidence, but is it sufficient?
    if receipt is None:
        return "receipt_absent"
    ref = receipt["decisionDerived"]["evidenceRef"]
    required_schema = requires["evidenceSchema"]
    if ref.get("schema") != required_schema or evidence.get("schema") != required_schema:
        return "evidence_schema_mismatch"
    if receipt["issuerAsserted"]["iss"] != requires["receiptIssuer"]:
        return "issuer_not_accepted"
    if receipt["decisionDerived"]["decision"] != requires["decision"]:
        return "decision_not_accepted"
    if evidence.get("grantFingerprint") != requires["grantFingerprint"]:
        return "authorization_mismatch"
    if evidence.get("argsCommitment") != requires["actionDigest"]:
        return "action_digest_mismatch"

    return "receipt_matches"


def _partition_is_total() -> bool:
    """Required check: every reason has exactly one state, and all four occur.

    A corpus that graded only the cases would still pass if two negatives were
    merged into one state, which is the failure this suite exists to prevent.
    """
    states = set(REASON_STATE.values())
    return states == {"released", "held", "expired", "refused"} and len(
        REASON_STATE
    ) == len(set(REASON_STATE))


def main() -> int:
    expected_path = HERE / "expected.json"
    if not expected_path.exists():
        print("expected.json not found — run _generate.py first", file=sys.stderr)
        return 1

    expected = json.loads(expected_path.read_text(encoding="utf-8"))["cases"]
    cases_dir = HERE / "cases"
    failures = []

    if not _partition_is_total():
        print("  FAIL  reason space is not a partition over the four states")
        failures.append("reason_partition")
    else:
        print("  PASS  reason space partitions cleanly over the four states")

    for path in sorted(cases_dir.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        reason = evaluate(case)
        state = REASON_STATE[reason]
        want_reason = case["expected_reason"]
        want_state = case["expected_state"]
        ok = reason == want_reason and state == want_state
        print(
            f"  {'PASS' if ok else 'FAIL'}  {path.stem:<28}"
            f"  computed={state}/{reason}  expected={want_state}/{want_reason}"
        )
        if not ok:
            failures.append(path.stem)

    # Cross-check against expected.json itself, so a deleted case file is a
    # failure rather than a smaller, still-green run.
    for name, meta in expected.items():
        path = cases_dir / f"{name}.json"
        if not path.exists():
            print(f"  MISSING  {name}", file=sys.stderr)
            failures.append(name)
            continue
        case = json.loads(path.read_text(encoding="utf-8"))
        reason = evaluate(case)
        if reason != meta["expected_reason"] or REASON_STATE[reason] != meta["expected_state"]:
            failures.append(f"{name}(expected.json cross-check)")

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1

    print(f"\nall {len(list(cases_dir.glob('*.json')))} cases pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
