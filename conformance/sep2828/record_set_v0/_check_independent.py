#!/usr/bin/env python3
"""Independent set-conformance checker for the v0 record-set vectors.

A second implementation of the SEP-2828 set-level rules, written from the
schema alone with only the standard library. It does not import Vaara. For
each committed set it classifies every record (decision or outcome),
reproduces the aggregate verdict, the conform count, the status and verdict
tallies, and the cross-record findings, then compares against
``expected.json``.

The receiving side of the evidence has to be checkable by a neutral party
with no shared code: not just "is each record well-formed" but "is the set
faithful" (no call recorded twice), "is it complete" (every authorised act
left an outcome, every recorded act traces to a decision), and "is every
executed action committed to its result". This file shows all of that falls
out of the records alone, with nothing but a hash function and the schema.

Run: ``python tests/vectors/record_set_v0/_check_independent.py``.
Exit 0 means every set matched its expected verdict.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALID_ALGS = {"HS256", "ES256", "RS256"}
VALID_STATUSES = {"executed", "refused", "errored"}
VALID_VERDICTS = {"allow", "block", "escalate"}
ACTING_VERDICTS = {"allow", "escalate"}
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX_RE = re.compile(r"^[0-9a-f]+$")


def classify(doc) -> str:
    """decision / outcome / unknown, by which derived block the record carries."""
    if not isinstance(doc, dict):
        return "unknown"
    has_d = isinstance(doc.get("decisionDerived"), dict)
    has_o = isinstance(doc.get("outcomeDerived"), dict)
    if has_d and not has_o:
        return "decision"
    if has_o and not has_d:
        return "outcome"
    return "unknown"


def _envelope_ok(doc, asserted_key) -> bool:
    """Shared wire-schema checks for both record types."""
    if doc.get("version") != 1:
        return False
    alg = doc.get("alg")
    if alg not in VALID_ALGS:
        return False
    sig = doc.get("signature")
    if not (isinstance(sig, str) and HEX_RE.match(sig) and len(sig) % 2 == 0):
        return False
    bl = doc.get("backLink")
    if not isinstance(bl, dict):
        return False
    if not (isinstance(bl.get("attestationDigest"), str)
            and DIGEST_RE.match(bl["attestationDigest"])):
        return False
    if not isinstance(bl.get("attestationNonce"), str):
        return False
    a = doc.get(asserted_key)
    if not isinstance(a, dict):
        return False
    if any(f not in a for f in ("alg", "iat", "iss", "nonce", "secretVersion", "sub")):
        return False
    return a.get("alg") == alg


def conforms_outcome(doc) -> bool:
    if not _envelope_ok(doc, "receiptAsserted"):
        return False
    od = doc["outcomeDerived"]
    if od.get("status") not in VALID_STATUSES:
        return False
    if not isinstance(od.get("completedAt"), str):
        return False
    rc = od.get("resultCommitment")
    if isinstance(rc, dict) and "projection" in rc:
        proj, pdg = rc.get("projection"), rc.get("projectionDigest")
        if not (isinstance(proj, str) and isinstance(pdg, str)):
            return False
        if "sha256:" + hashlib.sha256(proj.encode("utf-8")).hexdigest() != pdg:
            return False
    return True


def conforms_decision(doc) -> bool:
    if not _envelope_ok(doc, "issuerAsserted"):
        return False
    dd = doc["decisionDerived"]
    if dd.get("decision") not in VALID_VERDICTS:
        return False
    return isinstance(dd.get("decidedAt"), str)


def _call_key(doc):
    bl = doc["backLink"]
    return bl["attestationDigest"], bl["attestationNonce"]


def _by_call(records):
    out = {}
    for name, doc in records:
        out.setdefault(_call_key(doc), []).append(name)
    return out


def check_set(records):
    """Reproduce the set verdict from (name, doc) pairs."""
    decisions, outcomes = [], []
    for name, doc in records:
        kind = classify(doc)
        if kind == "decision" and conforms_decision(doc):
            decisions.append((name, doc))
        elif kind == "outcome" and conforms_outcome(doc):
            outcomes.append((name, doc))

    findings = []

    for names in _by_call(outcomes).values():
        if len(names) > 1:
            findings.append({"id": "duplicate_call", "severity": "required",
                             "records": sorted(names)})

    if decisions and outcomes:
        decision_calls = _by_call(decisions)
        outcome_calls = _by_call(outcomes)
        acting = {k: v for k, v in decision_calls.items()
                  if any(d["decisionDerived"].get("decision") in ACTING_VERDICTS
                         for n, d in decisions if _call_key(d) == k)}
        no_outcome = sorted(n for k, v in acting.items()
                            if k not in outcome_calls for n in v)
        no_decision = sorted(n for k, v in outcome_calls.items()
                             if k not in decision_calls for n in v)
        if no_outcome:
            findings.append({"id": "decision_without_outcome", "severity": "advisory",
                             "records": no_outcome})
        if no_decision:
            findings.append({"id": "outcome_without_decision", "severity": "advisory",
                             "records": no_decision})

    gap = sorted(n for n, d in outcomes
                 if d["outcomeDerived"].get("status") == "executed"
                 and d["outcomeDerived"].get("resultCommitment") is None)
    if gap:
        findings.append({"id": "executed_without_result_commitment",
                         "severity": "advisory", "records": gap})

    findings.sort(key=lambda f: (f["id"], f["records"]))

    status_counts = {}
    for _n, d in outcomes:
        s = d["outcomeDerived"]["status"]
        status_counts[s] = status_counts.get(s, 0) + 1
    verdict_counts = {}
    for _n, d in decisions:
        v = d["decisionDerived"]["decision"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    conforming = len(decisions) + len(outcomes)
    all_conform = conforming == len(records)
    no_required = not any(f["severity"] == "required" for f in findings)
    return {
        "conforms": all_conform and no_required,
        "total": len(records),
        "conforming": conforming,
        "statusCounts": dict(sorted(status_counts.items())),
        "verdictCounts": dict(sorted(verdict_counts.items())),
        "findings": findings,
    }


def main() -> int:
    expected = json.loads((HERE / "expected.json").read_text())
    failures = 0
    for name in sorted(expected):
        files = sorted((HERE / "sets" / name).glob("*.json"))
        records = [(p.name, json.loads(p.read_text())) for p in files]
        got = check_set(records)
        ok = got == expected[name]
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {got['conforming']}/{got['total']} conform")
        if not ok:
            print("  want:", expected[name])
            print("  got :", got)
    print(f"\n{len(expected) - failures}/{len(expected)} sets matched expected.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
