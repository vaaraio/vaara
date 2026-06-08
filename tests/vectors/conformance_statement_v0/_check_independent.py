#!/usr/bin/env python3
"""Independent check that the conformance statement tells the truth.

The statement claims three things: the corpus bytes match their manifest, this
implementation reproduced every recorded verdict, and the emitter's own records
conform. A renderer could print CONFORMS over any of those without it being so.
This checker closes that gap without importing Vaara.

It re-derives each claim from the real corpus and the committed emitter records:

* corpus integrity, by recomputing every file digest and the corpusDigest;
* self-test, by running the corpus's own Vaara-free runner (``run.py``) and
  confirming the neutral suite reproduces every case the statement counts;
* the emitter records verdict, with a second implementation of the SEP-2828
  set check (per-record conformance plus the required unique-call property that
  gates it) written from the schema alone.

Then it parses each committed golden page and the structured ``expected.json``
and asserts both state exactly what this independent derivation found. Standard
library only (hashlib, json, re, subprocess). Run:
``python tests/vectors/conformance_statement_v0/_check_independent.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CORPUS = REPO / "conformance" / "sep2828"
EMITTER = HERE / "emitter_records"
PAGES = HERE / "pages"

VALID_ALGS = {"HS256", "ES256", "RS256"}
VALID_STATUSES = {"executed", "refused", "errored"}
VALID_VERDICTS = {"allow", "block", "escalate"}
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
HEX_RE = re.compile(r"^[0-9a-f]+$")

SCENARIO_RECORDS = {
    "selftest_only": None, "clean": "clean", "flawed": "flawed", "duplicate": "duplicate",
}


# ── Independent corpus + self-test derivation ─────────────────────────────────


def _file_digests(suites):
    out = {}
    for suite in suites:
        for path in sorted((CORPUS / suite).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            rel = path.relative_to(CORPUS).as_posix()
            out[rel] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _corpus_digest(files):
    lines = [f"{files[k].split(':', 1)[1]}  {k}" for k in sorted(files)]
    return "sha256:" + hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def corpus_facts():
    """Recompute the corpus identity and integrity from the bytes on disk."""
    manifest = json.loads((CORPUS / "MANIFEST.json").read_text())
    files = _file_digests(manifest["suites"])
    verified = files == manifest["files"] and _corpus_digest(files) == manifest["corpusDigest"]
    return {
        "name": manifest["corpus"],
        "version": manifest["version"],
        "corpusDigest": manifest["corpusDigest"],
        "fileCount": len(files),
        "verified": verified,
        "suites": list(manifest["suites"]),
    }


def self_test_facts(suites):
    """Per-suite case counts, corroborated by the corpus's own neutral runner."""
    runner = subprocess.run([sys.executable, "run.py"], cwd=CORPUS,
                            capture_output=True, text=True)
    reproduces_all = runner.returncode == 0 and "PASS: all" in runner.stdout
    per_suite = {}
    for suite in suites:
        cases = len(json.loads((CORPUS / suite / "expected.json").read_text()))
        per_suite[suite] = (cases, cases if reproduces_all else 0)
    return reproduces_all, per_suite


# ── Independent SEP-2828 set check (second implementation) ────────────────────


def _classify(doc):
    if not isinstance(doc, dict):
        return "unknown"
    has_d, has_o = isinstance(doc.get("decisionDerived"), dict), isinstance(
        doc.get("outcomeDerived"), dict)
    return "decision" if has_d and not has_o else "outcome" if has_o and not has_d else "unknown"


def _envelope_ok(doc, asserted_key):
    if doc.get("version") != 1 or doc.get("alg") not in VALID_ALGS:
        return False
    sig = doc.get("signature")
    if not (isinstance(sig, str) and HEX_RE.match(sig) and len(sig) % 2 == 0):
        return False
    bl = doc.get("backLink")
    if not (isinstance(bl, dict) and isinstance(bl.get("attestationDigest"), str)
            and DIGEST_RE.match(bl["attestationDigest"])
            and isinstance(bl.get("attestationNonce"), str)):
        return False
    a = doc.get(asserted_key)
    if not isinstance(a, dict) or any(
            f not in a for f in ("alg", "iat", "iss", "nonce", "secretVersion", "sub")):
        return False
    return a.get("alg") == doc.get("alg")


def _conforms(doc, kind):
    if kind == "decision":
        if not _envelope_ok(doc, "issuerAsserted"):
            return False
        dd = doc["decisionDerived"]
        return dd.get("decision") in VALID_VERDICTS and isinstance(dd.get("decidedAt"), str)
    if not _envelope_ok(doc, "receiptAsserted"):
        return False
    od = doc["outcomeDerived"]
    if od.get("status") not in VALID_STATUSES or not isinstance(od.get("completedAt"), str):
        return False
    rc = od.get("resultCommitment")
    if isinstance(rc, dict) and "projection" in rc:
        proj, pdg = rc.get("projection"), rc.get("projectionDigest")
        if not (isinstance(proj, str) and isinstance(pdg, str)):
            return False
        if "sha256:" + hashlib.sha256(proj.encode("utf-8")).hexdigest() != pdg:
            return False
    return True


def records_facts(scenario):
    """The set verdict for one emitter-records scenario, or None when there are none.

    Mirrors the conformance gate: every record conforms to its type's schema and
    no required cross-record property fails. The one required set property is
    unique calls (two outcome records pinning one call is a duplicate); pairing
    and coverage gaps are advisory and do not gate the verdict.
    """
    sub = SCENARIO_RECORDS[scenario]
    if sub is None:
        return None
    docs = [(p.name, json.loads(p.read_text())) for p in sorted((EMITTER / sub).glob("*.json"))]
    conforming = 0
    outcomes = []
    for _name, d in docs:
        kind = _classify(d)
        if kind in ("decision", "outcome") and _conforms(d, kind):
            conforming += 1
            if kind == "outcome":
                outcomes.append(d)
    by_call = {}
    for d in outcomes:
        bl = d["backLink"]
        key = (bl["attestationDigest"], bl["attestationNonce"])
        by_call[key] = by_call.get(key, 0) + 1
    duplicate = any(count > 1 for count in by_call.values())
    return {
        "total": len(docs),
        "conforming": conforming,
        "conforms": conforming == len(docs) and not duplicate,
    }


# ── Page parsing ──────────────────────────────────────────────────────────────

_VERDICT = re.compile(r"^\*\*Statement: (CONFORMS|NON-CONFORMING)\*\*$", re.M)
_CORPUS = re.compile(
    r"^Checked against corpus `(.+?)` version (\S+) \(corpusDigest `(sha256:[0-9a-f]{64})`\)\.$",
    re.M)
_VERIFIED = re.compile(r"^Verified: all (\d+) fixture files match", re.M)
_SUITE = re.compile(r"^- `(\S+?)`: (\d+)/(\d+) reproduced", re.M)
_RECORDS = re.compile(
    r"^(\d+) records? checked, (\d+) conforms?; your records (CONFORM|do NOT conform)\.$", re.M)


def parse_page(text):
    verdict = _VERDICT.search(text)
    corpus = _CORPUS.search(text)
    verified = _VERIFIED.search(text)
    suites = {m.group(1): (int(m.group(3)), int(m.group(2))) for m in _SUITE.finditer(text)}
    rec = _RECORDS.search(text)
    return {
        "verdict": verdict.group(1) if verdict else None,
        "corpus": (corpus.group(1), corpus.group(2), corpus.group(3)) if corpus else None,
        "verifiedCount": int(verified.group(1)) if verified else None,
        "suites": suites,
        "has_records": "## Your records" in text,
        "records": (int(rec.group(1)), int(rec.group(2)), rec.group(3) == "CONFORM")
        if rec else None,
    }


# ── Cross-check ───────────────────────────────────────────────────────────────


def main() -> int:
    corpus = corpus_facts()
    reproduces_all, per_suite = self_test_facts(corpus["suites"])
    expected = json.loads((HERE / "expected.json").read_text())
    failures = 0

    for scenario in sorted(SCENARIO_RECORDS):
        rec = records_facts(scenario)
        want_verdict = "CONFORMS" if (
            corpus["verified"] and reproduces_all and (rec is None or rec["conforms"])
        ) else "NON-CONFORMING"

        page = parse_page((PAGES / f"{scenario}.md").read_text())
        problems = []
        if page["verdict"] != want_verdict:
            problems.append(f"verdict {page['verdict']} != {want_verdict}")
        if page["corpus"] != (corpus["name"], corpus["version"], corpus["corpusDigest"]):
            problems.append(f"corpus identity {page['corpus']}")
        want_count = corpus["fileCount"] if corpus["verified"] else None
        if page["verifiedCount"] != want_count:
            problems.append(f"verified count {page['verifiedCount']} != {want_count}")
        if page["suites"] != per_suite:
            problems.append(f"suite counts {page['suites']} != {per_suite}")
        if (rec is None) == page["has_records"]:
            problems.append(f"records section presence wrong (rec={rec})")
        if rec is not None and page["records"] != (
                rec["total"], rec["conforming"], rec["conforms"]):
            problems.append(f"records line {page['records']} != {rec}")

        # The structured expected.json must agree with the same derivation.
        exp = expected[scenario]
        if exp["conforms"] != (want_verdict == "CONFORMS"):
            problems.append("expected.json conforms disagrees")
        if exp["corpus"]["corpusDigest"] != corpus["corpusDigest"]:
            problems.append("expected.json corpusDigest disagrees")
        if exp["selfTest"]["reproduced"] != sum(r for _c, r in per_suite.values()):
            problems.append("expected.json selfTest reproduced disagrees")
        if (exp["records"] is None) != (rec is None):
            problems.append("expected.json records presence disagrees")
        if rec is not None and exp["records"]["conforms"] != rec["conforms"]:
            problems.append("expected.json records conforms disagrees")

        ok = not problems
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {scenario}: page says {page['verdict']}")
        for p in problems:
            print(f"    {p}")

    total = len(SCENARIO_RECORDS)
    print(f"\n{total - failures}/{total} statements match the independent derivation.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
