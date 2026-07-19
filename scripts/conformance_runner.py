#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run every Vaara conformance suite and report a single pass/fail verdict.

Each suite under ``tests/vectors/<name>/`` ships an independent checker
(``_check_independent.py``) that imports no Vaara code and recomputes its
verdicts from the bytes of its own case files. This runner discovers every such
checker, invokes each in a subprocess, and aggregates the result into one exit
code and one optional machine-readable report.

The point is neutrality: the checkers decide, this runner only collects. It
imports no Vaara code and uses only the standard library, so the same
invocation grades the reference implementation and any outside one. Point
``--vectors-dir`` at a directory laid out the same way and it grades those bytes
instead. A format is conformant when its vectors pass here, not when a document
says they should.

    python scripts/conformance_runner.py                 # run all, exit 0 iff all pass
    python scripts/conformance_runner.py --list          # list discovered suites
    python scripts/conformance_runner.py --corpus tap_v0 # run one suite
    python scripts/conformance_runner.py --json report.json
    python scripts/conformance_runner.py --vectors-dir ./their_vectors

A suite that cannot run bare (it needs an external artifact passed as an
argument, or an optional dependency that is not installed) is reported SKIP,
never silently dropped, and does not on its own fail the run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
DEFAULT_VECTORS = REPO / "tests" / "vectors"
CHECKER = "_check_independent.py"

# A checker returns this (the conventional automake skip code) when an optional
# dependency it needs is not installed. Reported SKIP with a reason, never FAIL,
# so a base environment grades clean; the suite still runs where the extra is
# present.
SKIP_EXIT_CODE = 77

# Suites whose checker validates an artifact handed to it on the command line
# rather than a bare directory of case files. Reported SKIP (with reason) in the
# aggregate run; an explicit list so the gap reads as a gap, not as coverage.
NEEDS_ARGUMENT = {
    "article12_fold_v0": "checker validates a passed-in bundle zip, not a bare case directory",
}


def discover(vectors_dir: Path) -> list[str]:
    """Suite names (sorted) that carry an independent checker."""
    if not vectors_dir.is_dir():
        return []
    return sorted(p.parent.name for p in vectors_dir.glob(f"*/{CHECKER}") if p.is_file())


def _case_count(suite_dir: Path) -> Optional[int]:
    """Declared case count for the report. Prefer expected.json, else cases/*.json."""
    expected = suite_dir / "expected.json"
    if expected.is_file():
        try:
            cases = json.loads(expected.read_text(encoding="utf-8")).get("cases")
            if isinstance(cases, (dict, list)):
                return len(cases)
        except (OSError, ValueError):
            pass
    cases_dir = suite_dir / "cases"
    if cases_dir.is_dir():
        return len(list(cases_dir.glob("*.json")))
    return None


def run_suite(vectors_dir: Path, suite: str) -> dict[str, Any]:
    """Run one suite's checker and return a structured result row."""
    suite_dir = vectors_dir / suite
    if suite in NEEDS_ARGUMENT:
        return {
            "suite": suite, "status": "SKIP", "reason": NEEDS_ARGUMENT[suite],
            "cases": _case_count(suite_dir), "returncode": None, "duration_s": 0.0,
        }

    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(suite_dir / CHECKER)],
        capture_output=True, text=True, cwd=str(suite_dir),
    )
    duration = round(time.perf_counter() - start, 3)
    if proc.returncode == SKIP_EXIT_CODE:
        # The checker declared it cannot run here (an optional dependency is
        # absent). Take its own last stderr line as the reason.
        lines = proc.stderr.strip().splitlines()
        reason = lines[-1].removeprefix("SKIP: ") if lines else \
            "optional dependency not installed"
        return {
            "suite": suite, "status": "SKIP", "reason": reason,
            "cases": _case_count(suite_dir), "returncode": proc.returncode,
            "duration_s": duration,
        }
    status = "PASS" if proc.returncode == 0 else "FAIL"
    row: dict[str, Any] = {
        "suite": suite, "status": status, "cases": _case_count(suite_dir),
        "returncode": proc.returncode, "duration_s": duration,
    }
    if status == "FAIL":
        # Tail of the checker's own output, so a failure is actionable in place.
        row["output_tail"] = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
    return row


def _print_table(rows: list[dict[str, Any]]) -> None:
    width = max((len(r["suite"]) for r in rows), default=10)
    for r in rows:
        cases = "" if r["cases"] is None else f"{r['cases']:>3} cases"
        extra = f"  ({r['reason']})" if r["status"] == "SKIP" else ""
        print(f"  {r['status']:<4}  {r['suite']:<{width}}  {cases:<10}{extra}")
        for line in r.get("output_tail", []):
            print(f"            | {line}")


def build_report(rows: list[dict[str, Any]], vectors_dir: Path, stamp: str) -> dict[str, Any]:
    passed = [r for r in rows if r["status"] == "PASS"]
    failed = [r for r in rows if r["status"] == "FAIL"]
    skipped = [r for r in rows if r["status"] == "SKIP"]
    return {
        "tool": "vaara-conformance-runner",
        "generated_at": stamp,
        "vectors_dir": str(vectors_dir),
        "python": sys.version.split()[0],
        "totals": {
            "suites": len(rows), "passed": len(passed), "failed": len(failed),
            "skipped": len(skipped), "cases_passed": sum(r["cases"] or 0 for r in passed),
        },
        "all_passed": not failed,
        "suites": rows,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate runner over the Vaara conformance vector corpus."
    )
    parser.add_argument("--vectors-dir", type=Path, default=DEFAULT_VECTORS,
                        help="directory of suites to grade (default: repo tests/vectors)")
    parser.add_argument("--corpus", action="append", metavar="NAME",
                        help="run only this suite (repeatable)")
    parser.add_argument("--list", action="store_true", help="list discovered suites and exit")
    parser.add_argument("--json", type=Path, metavar="PATH",
                        help="write a machine-readable conformance report to this path")
    args = parser.parse_args(argv)

    vectors_dir = args.vectors_dir.resolve()
    suites = discover(vectors_dir)
    if not suites:
        print(f"no conformance suites found under {vectors_dir}", file=sys.stderr)
        return 2

    if args.list:
        for name in suites:
            tag = " (needs argument: skipped in aggregate run)" if name in NEEDS_ARGUMENT else ""
            print(f"{name}{tag}")
        return 0

    if args.corpus:
        unknown = [c for c in args.corpus if c not in suites]
        if unknown:
            print(f"unknown suite(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        suites = [s for s in suites if s in args.corpus]

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [run_suite(vectors_dir, s) for s in suites]
    _print_table(rows)

    report = build_report(rows, vectors_dir, stamp)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport written to {args.json}")

    t = report["totals"]
    print(f"\n{t['passed']} passed, {t['failed']} failed, {t['skipped']} skipped "
          f"({t['cases_passed']} cases) across {t['suites']} suites.")
    if t["failed"]:
        print(f"FAIL: {', '.join(r['suite'] for r in rows if r['status'] == 'FAIL')}")
        return 1
    print("PASS: every suite that ran matched its expected verdicts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
