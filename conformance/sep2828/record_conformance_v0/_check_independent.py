#!/usr/bin/env python3
"""Independent conformance checker for the v0 record-conformance vectors.

A second implementation of the SEP-2828 record conformance rules,
written from the schema alone with only the standard library
(``hashlib`` + ``re`` + ``json``). It does not import Vaara. For each
committed record it reproduces the conformance verdict, the set of
failed required checks, and the set of advisory warnings, then compares
against ``expected.json``.

A record format that a neutral party is meant to check without trusting
the producer must be checkable from a second implementation with no
shared code. That is exactly what this file demonstrates: the one
binding the record proves about itself (``projectionDigest`` over the
projection bytes) recomputes here with nothing but a hash function.

Run: ``python tests/vectors/record_conformance_v0/_check_independent.py``.
Exit 0 means every case matched its expected verdict.
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
SIG_HEX_LEN = {"HS256": 64, "ES256": 128}


def conformance(doc):
    """Return (conforms, required_failed sorted, advisories sorted)."""
    req_fail, adv = [], []

    def req(check_id, ok):
        if not ok:
            req_fail.append(check_id)

    def advisory(check_id, ok):
        if not ok:
            adv.append(check_id)

    if not isinstance(doc, dict):
        return False, ["top_level_object"], []

    req("version", doc.get("version") == 1)
    alg = doc.get("alg")
    req("alg_supported", alg in VALID_ALGS)

    sig = doc.get("signature")
    sig_hex = isinstance(sig, str) and bool(HEX_RE.match(sig)) and len(sig) % 2 == 0
    req("signature_hex", sig_hex)
    if sig_hex and isinstance(alg, str) and alg in SIG_HEX_LEN:
        advisory("signature_length", len(sig) == SIG_HEX_LEN[alg])

    bl = doc.get("backLink")
    if not isinstance(bl, dict):
        req("back_link_present", False)
    else:
        req("back_link_present", "attestationDigest" in bl and "attestationNonce" in bl)
        ad = bl.get("attestationDigest")
        req("back_link_digest_format", isinstance(ad, str) and bool(DIGEST_RE.match(ad)))
        req("back_link_nonce_type", isinstance(bl.get("attestationNonce"), str))

    ra = doc.get("receiptAsserted")
    if not isinstance(ra, dict):
        req("receipt_asserted_present", False)
    else:
        fields = ("alg", "iat", "iss", "nonce", "secretVersion", "sub")
        req("receipt_asserted_present", all(f in ra for f in fields))
        req("receipt_asserted_alg_matches", ra.get("alg") == alg)

    od = doc.get("outcomeDerived")
    if not isinstance(od, dict):
        req("outcome_present", False)
    else:
        status = od.get("status")
        req("outcome_present", "status" in od and "completedAt" in od)
        req("status_valid", status in VALID_STATUSES)
        req("completed_at_type", isinstance(od.get("completedAt"), str))
        rc = od.get("resultCommitment")
        if rc is not None:
            _result_commitment(rc, status, req, advisory)
        dd = od.get("decisionDigest")
        if dd is not None:
            req("decision_digest_format", isinstance(dd, str) and bool(DIGEST_RE.match(dd)))

    return not req_fail, sorted(req_fail), sorted(adv)


def _result_commitment(rc, status, req, advisory):
    if not isinstance(rc, dict):
        req("result_commitment_shape", False)
        return
    if "projection" in rc:
        proj, pdg = rc.get("projection"), rc.get("projectionDigest")
        shape_ok = isinstance(proj, str) and isinstance(pdg, str)
        req("result_commitment_shape", shape_ok)
        if shape_ok:
            req("result_commitment_digest_format", bool(DIGEST_RE.match(pdg)))
            recomputed = "sha256:" + hashlib.sha256(proj.encode("utf-8")).hexdigest()
            req("result_commitment_self_consistent", recomputed == pdg)
    elif "ref" in rc:
        dg = rc.get("digest")
        req("result_commitment_shape", isinstance(rc.get("ref"), str) and isinstance(dg, str))
        req("result_commitment_digest_format", isinstance(dg, str) and bool(DIGEST_RE.match(dg)))
    else:
        req("result_commitment_shape", False)
    if status == "refused":
        advisory("refused_has_no_result", False)


def main() -> int:
    expected = json.loads((HERE / "expected.json").read_text())
    failures = 0
    for name in sorted(expected):
        want = expected[name]
        doc = json.loads((HERE / "records" / f"{name}.json").read_text())
        conforms, req_fail, adv = conformance(doc)
        got = {"conforms": conforms, "requiredFailed": req_fail, "advisories": adv}
        ok = got == want
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {got}")
    print(f"\n{len(expected) - failures}/{len(expected)} cases matched expected.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
