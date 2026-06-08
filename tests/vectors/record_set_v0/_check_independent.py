#!/usr/bin/env python3
"""Independent set-conformance checker for the v0 record-set vectors.

A second implementation of the SEP-2828 set-level rules, written from the
schema alone with only the standard library. It does not import Vaara. For
each committed set it reproduces the aggregate verdict, the conform count,
the status tally, and the cross-record findings, then compares against
``expected.json``.

The receiving side of the evidence has to be checkable by a neutral party
with no shared code: not just "is each record well-formed" but "is the set
faithful" (no call recorded twice) and "is it complete" (no executed action
left without a committed result). This file demonstrates that both fall out
of the records alone, with nothing but a hash function and the schema.

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
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX_RE = re.compile(r"^[0-9a-f]+$")


def conforms(doc) -> bool:
    """Minimal per-record conformance: enough to gate set-level reasoning."""
    if not isinstance(doc, dict):
        return False
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
    ra = doc.get("receiptAsserted")
    if not isinstance(ra, dict):
        return False
    if any(f not in ra for f in ("alg", "iat", "iss", "nonce", "secretVersion", "sub")):
        return False
    if ra.get("alg") != alg:
        return False
    od = doc.get("outcomeDerived")
    if not isinstance(od, dict):
        return False
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


def check_set(records):
    """Reproduce the set verdict from (name, doc) pairs."""
    conforming = [(n, d) for n, d in records if conforms(d)]
    findings = []

    by_call = {}
    for name, doc in conforming:
        bl = doc["backLink"]
        by_call.setdefault((bl["attestationDigest"], bl["attestationNonce"]), []).append(name)
    for names in by_call.values():
        if len(names) > 1:
            findings.append({"id": "duplicate_call", "severity": "required",
                             "records": sorted(names)})

    gap = sorted(n for n, d in conforming
                 if d["outcomeDerived"].get("status") == "executed"
                 and d["outcomeDerived"].get("resultCommitment") is None)
    if gap:
        findings.append({"id": "executed_without_result_commitment",
                         "severity": "advisory", "records": gap})

    findings.sort(key=lambda f: (f["id"], f["records"]))

    counts = {}
    for _n, d in conforming:
        counts[d["outcomeDerived"]["status"]] = counts.get(d["outcomeDerived"]["status"], 0) + 1

    all_conform = len(conforming) == len(records)
    no_required = not any(f["severity"] == "required" for f in findings)
    return {
        "conforms": all_conform and no_required,
        "total": len(records),
        "conforming": len(conforming),
        "statusCounts": dict(sorted(counts.items())),
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
