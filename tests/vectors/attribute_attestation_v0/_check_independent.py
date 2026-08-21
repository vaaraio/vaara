#!/usr/bin/env python3
"""Independent checker for attribute_attestation_v0: is this value worth anything?

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
not import Vaara. Every verdict is recomputed from the bytes of the case file and
the committed public key.

An attribute attestation binds a subject to values and states, per value, where
that value came from. The standing is a closed, ordered set:

    undeclared        < operator_declared < measured < protocol_defined

A relying party names the floor it will accept. ``protocol_defined`` outranks
``measured`` because a value fixed by a specification cannot be wrong while a
measurement can come from a broken sensor.

Four states, and the reason space is partitioned so they cannot collapse:

  accepted  the attribute is present, in window, and at or above the floor
  withheld  the attestation is sound and does not answer what was asked: the
            attribute is absent, the subject or issuer is not the one asked
            about, or the value's source is below the floor
  expired   now is outside notBefore..notAfter
  refused   the artifact fails as evidence: malformed, unsigned by the pinned
            key, or carrying a standing outside the closed set

The distinction between withheld and refused is the whole point. A value the
supplier typed in is sound evidence of a claim and no evidence of a fact, and
reporting that the same way as a forgery throws away the difference between
"weaker than you asked for" and "someone edited this".

Order is soundness, then the clock, then sufficiency. Soundness runs first so an
expired window cannot swallow a broken signature.

The signature is Ed25519 over the JCS encoding of the attestation with its own
``signature`` member removed, the same rule the release condition and the
data-locality record use.

Run: python3 tests/vectors/attribute_attestation_v0/_check_independent.py
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
from cryptography.hazmat.primitives import serialization

HERE = Path(__file__).resolve().parent
SCHEMA = "vaara.attribute-attestation/v0"

RANK = {
    "undeclared": 0,
    "operator_declared": 1,
    "measured": 2,
    "protocol_defined": 3,
}

REASON_STATE = {
    "attribute_attested": "accepted",
    "attribute_absent": "withheld",
    "subject_mismatch": "withheld",
    "issuer_not_accepted": "withheld",
    "source_below_floor": "withheld",
    "outside_validity_window": "expired",
    "attestation_malformed": "refused",
    "key_absent": "refused",
    "signature_invalid": "refused",
}

_REQUIRED = ("alg", "attestationId", "attributes", "issuer", "notAfter",
             "notBefore", "schema", "signature", "subject", "version")


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _digest(obj) -> str:
    return "sha256:" + hashlib.sha256(_jcs(obj)).hexdigest()


def _epoch(iso):
    try:
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        return datetime.fromisoformat(iso).timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


def well_formed(a) -> bool:
    if not isinstance(a, dict) or a.get("schema") != SCHEMA:
        return False
    if any(k not in a for k in _REQUIRED):
        return False
    if not isinstance(a["version"], int) or isinstance(a["version"], bool):
        return False
    for k in ("alg", "attestationId", "issuer", "signature"):
        if not isinstance(a[k], str) or not a[k]:
            return False
    subj = a["subject"]
    if not isinstance(subj, dict):
        return False
    if any(not isinstance(subj.get(k), str) or not subj.get(k) for k in ("id", "kind")):
        return False
    attrs = a["attributes"]
    if not isinstance(attrs, list) or not attrs:
        return False
    seen = set()
    for at in attrs:
        if not isinstance(at, dict):
            return False
        if any(not isinstance(at.get(k), str) or not at.get(k)
               for k in ("name", "source", "value")):
            return False
        # An unrecognised standing is malformed, never silently floored: a
        # verifier that downgrades what it does not understand lets a forger
        # introduce a standing of their own.
        if at["source"] not in RANK:
            return False
        if at["name"] in seen:
            return False
        seen.add(at["name"])
    return all(_epoch(a[f]) is not None for f in ("notBefore", "notAfter"))


def signature_ok(a: dict, public_key) -> bool:
    body = _jcs({k: v for k, v in a.items() if k != "signature"})
    try:
        sig = bytes.fromhex(a["signature"])
    except (ValueError, TypeError):
        return False
    try:
        public_key.verify(sig, body)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def evaluate(case: dict) -> str:
    """Recompute the reason for one case. Returns a REASON_STATE key."""
    a = case.get("attestation")
    q = case["query"]
    now = _epoch(case["now"])
    if now is None:
        raise ValueError("case 'now' is not an ISO 8601 instant")

    # 1. sound as an artifact
    if not well_formed(a):
        return "attestation_malformed"
    key_path = case.get("issuer_key")
    if not key_path:
        return "key_absent"
    pub = serialization.load_pem_public_key((HERE / key_path).read_bytes())
    if not signature_ok(a, pub):
        return "signature_invalid"

    # 2. the clock
    if not (_epoch(a["notBefore"]) <= now <= _epoch(a["notAfter"])):
        return "outside_validity_window"

    # 3. does it answer what was asked
    if q.get("subject_id") is not None and a["subject"]["id"] != q["subject_id"]:
        return "subject_mismatch"
    accepted = q.get("accepted_issuers")
    if accepted is not None and a["issuer"] not in accepted:
        return "issuer_not_accepted"
    match = next((x for x in a["attributes"] if x["name"] == q["name"]), None)
    if match is None:
        return "attribute_absent"
    if RANK[match["source"]] < RANK[q["minimum_source"]]:
        return "source_below_floor"
    return "attribute_attested"


def _partition_is_total() -> bool:
    """Required check: the four states all occur and no reason is duplicated."""
    return (set(REASON_STATE.values()) == {"accepted", "withheld", "expired", "refused"}
            and len(REASON_STATE) == len(set(REASON_STATE)))


def _ladder_is_total_order() -> bool:
    """Required check: the standing ladder has no ties and floors at undeclared."""
    vals = list(RANK.values())
    return len(set(vals)) == len(vals) and RANK["undeclared"] == min(vals)


def main() -> int:
    expected_path = HERE / "expected.json"
    if not expected_path.exists():
        print("expected.json not found, run _generate.py first", file=sys.stderr)
        return 1
    expected = json.loads(expected_path.read_text(encoding="utf-8"))["cases"]
    cases_dir = HERE / "cases"
    failures = []

    for label, ok in (("reason space partitions over the four states", _partition_is_total()),
                      ("standing ladder is a total order", _ladder_is_total_order())):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            failures.append(label)

    for path in sorted(cases_dir.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        reason = evaluate(case)
        state = REASON_STATE[reason]
        ok = reason == case["expected_reason"] and state == case["expected_state"]
        print(f"  {'PASS' if ok else 'FAIL'}  {path.stem:<30}"
              f"  computed={state}/{reason}"
              f"  expected={case['expected_state']}/{case['expected_reason']}")
        if not ok:
            failures.append(path.stem)

    for name, meta in expected.items():
        path = cases_dir / f"{name}.json"
        if not path.exists():
            print(f"  MISSING  {name}", file=sys.stderr)
            failures.append(name)
            continue
        case = json.loads(path.read_text(encoding="utf-8"))
        reason = evaluate(case)
        if (reason != meta["expected_reason"]
                or REASON_STATE[reason] != meta["expected_state"]):
            failures.append(f"{name}(expected.json cross-check)")

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    print(f"\nall {len(list(cases_dir.glob('*.json')))} cases pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
