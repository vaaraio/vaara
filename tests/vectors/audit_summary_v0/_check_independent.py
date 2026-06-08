#!/usr/bin/env python3
"""Independent check that the regulator page tells the truth.

The audit summary is a rendering, not a second judgement: every fact on the
page comes from ``check_record_set``, whose verdict is already reproduced from
the SEP-2828 schema with no Vaara import by ``record_set_v0/_check_independent.py``.
What that leaves unproven is whether the *prose* faithfully states that verdict,
or whether a renderer could quietly print CONFORMS over a non-conforming set.

This checker closes that gap without importing Vaara. It reads each committed
golden page as plain text, pulls out the claims it makes (the verdict word, the
"N records checked, M conform" counts, the finding ids and their records under
the Required and Advisory headings, and the count of non-conforming records),
and asserts they equal what ``record_set_v0/expected.json`` independently says
for the same case. The page is confirmed to match the machine verdict, by a
party that re-derived that verdict from the records alone.

Standard library only (re, json, pathlib). Run:
``python tests/vectors/audit_summary_v0/_check_independent.py``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAGES = HERE / "pages"
EXPECTED = HERE.parent / "record_set_v0" / "expected.json"

_VERDICT = re.compile(r"^\*\*Verdict: (CONFORMS|NON-CONFORMING)\*\*$", re.M)
_COUNTS = re.compile(r"^(\d+) records? checked, (\d+) conforms?\.$", re.M)
_FINDING_ID = re.compile(r"\*\*(.+?)\*\*")
_TRAILING_PARENS = re.compile(r"\(([^()]+)\)\s*$")
_NONCONFORMING = re.compile(r"^- `([^`]+)` \(")


def _section(lines, header):
    """The bullet lines under a `header`, skipping the blank line after it."""
    out = []
    if header not in lines:
        return out
    i = lines.index(header) + 1
    while i < len(lines) and not lines[i].strip():  # skip blank(s) after header
        i += 1
    while i < len(lines) and lines[i].startswith("- "):
        out.append(lines[i])
        i += 1
    return out


def parse_page(text):
    """Pull the claims a regulator would read off the page."""
    lines = text.splitlines()
    verdict_m = _VERDICT.search(text)
    counts_m = _COUNTS.search(text)

    findings = []
    for severity, header in (
        ("required", "Required (these gate conformance):"),
        ("advisory", "Advisory (gaps that do not gate conformance):"),
    ):
        for line in _section(lines, header):
            fid = _FINDING_ID.search(line)
            recs = _TRAILING_PARENS.search(line)
            if fid and recs:
                names = sorted(r.strip() for r in recs.group(1).split(","))
                findings.append({"id": fid.group(1), "severity": severity,
                                 "records": names})

    nonconforming = [m.group(1) for m in (_NONCONFORMING.match(ln) for ln in lines) if m]

    return {
        "verdict": verdict_m.group(1) if verdict_m else None,
        "total": int(counts_m.group(1)) if counts_m else None,
        "conforming": int(counts_m.group(2)) if counts_m else None,
        "findings": sorted(findings, key=lambda f: (f["id"], f["records"])),
        "nonconforming_count": len(nonconforming),
    }


def expected_claims(case):
    """The same claims, derived from the independent record-set verdict."""
    want_findings = sorted(
        ({"id": f["id"], "severity": f["severity"], "records": sorted(f["records"])}
         for f in case["findings"]),
        key=lambda f: (f["id"], f["records"]),
    )
    return {
        "verdict": "CONFORMS" if case["conforms"] else "NON-CONFORMING",
        "total": case["total"],
        "conforming": case["conforming"],
        "findings": want_findings,
        "nonconforming_count": case["total"] - case["conforming"],
    }


def main() -> int:
    expected = json.loads(EXPECTED.read_text())
    failures = 0
    for name in sorted(p.stem for p in PAGES.glob("*.md")):
        page = parse_page((PAGES / f"{name}.md").read_text())
        want = expected_claims(expected[name])
        ok = page == want
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {name}: page says {page['verdict']}")
        if not ok:
            print("  want:", want)
            print("  got :", page)
    total = len(list(PAGES.glob("*.md")))
    print(f"\n{total - failures}/{total} pages match the independent verdict.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
