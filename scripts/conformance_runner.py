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
import tempfile
import time
import urllib.parse
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
#
# The skip is a consequence of this runner's promise rather than a hole in the
# corpus. article12_fold_v0 validates a produced EU AI Act Article 12 regulator
# package, so something has to build the package first, and building it needs
# Vaara installed. The runner's whole point is that a stranger can grade the
# corpus with `pip install cryptography rfc8785` and no Vaara. Those two cannot
# both hold, so it skips, and `--with-vaara` is for the environment where the
# install already exists: our own CI, and anyone who happens to have it.
NEEDS_ARGUMENT = {
    "article12_fold_v0": "checker validates a passed-in bundle zip, not a bare case directory",
}

#: How to build the artifact a NEEDS_ARGUMENT suite grades, when Vaara is
#: importable. Each entry names the generator callable and the scenario it
#: builds; the runner hands the result to the suite's own checker unchanged.
BUILDABLE = {
    "article12_fold_v0": "full",
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


def _build_artifact(suite_dir: Path, suite: str, work: Path) -> Path | None:
    """Build the artifact a NEEDS_ARGUMENT suite grades. None if not possible.

    Imports the suite's own generator, which imports Vaara, so this only works
    where Vaara is installed. Any failure returns None and the suite falls back
    to being skipped with its usual reason: a runner that turned a missing
    optional install into a red suite would punish exactly the stranger this
    corpus is built for.
    """
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location(
            f"_gen_{suite}", suite_dir / "_generate.py"
        )
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        hcases = {c["name"]: c for c in json.loads(
            (gen.HANDOFF / "cases.json").read_text())["cases"]}
        ecases = {c["name"]: c for c in json.loads(
            (gen.ENFORCEMENT / "cases.json").read_text())["cases"]}
        return gen._build(gen.SCENARIOS[BUILDABLE[suite]], hcases, ecases, work)
    except Exception:  # noqa: BLE001
        return None


def run_suite(vectors_dir: Path, suite: str, with_vaara: bool = False) -> dict[str, Any]:
    """Run one suite's checker and return a structured result row."""
    suite_dir = vectors_dir / suite
    if suite in NEEDS_ARGUMENT:
        artifact = None
        if with_vaara and suite in BUILDABLE:
            work = Path(tempfile.mkdtemp(prefix=f"{suite}-"))
            artifact = _build_artifact(suite_dir, suite, work)
        if artifact is None:
            return {
                "suite": suite, "status": "SKIP", "reason": NEEDS_ARGUMENT[suite],
                "cases": _case_count(suite_dir), "returncode": None, "duration_s": 0.0,
            }
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(suite_dir / CHECKER), str(artifact)],
            capture_output=True, text=True, cwd=str(suite_dir),
        )
        duration = round(time.perf_counter() - start, 3)
        return {
            "suite": suite,
            "status": "PASS" if proc.returncode == 0 else "FAIL",
            "reason": "" if proc.returncode == 0 else proc.stderr.strip()[-400:],
            "cases": _case_count(suite_dir), "returncode": proc.returncode,
            "duration_s": duration,
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


#: Where a reproduction gets listed. The issue form is a self-service desk: a
#: workflow reads the issue, validates it, publishes the row and comments back
#: with a badge. Nothing is merged and no maintainer approves anything.
VCR_FORM = "https://github.com/vaaraio/vaara/issues/new"
VCR_TEMPLATE = "conformance-row.yml"


def _head_commit(repo_root: Path) -> str:
    """The commit these bytes were graded at, or "" outside a git checkout."""
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def submit_link(report: dict, repo_root: Path, selected=None) -> str:
    """A one-click link that opens the listing form already filled in.

    Added 2026-08-17. The runner used to print a table and stop, so anyone who
    wanted their result listed had to read numbers off a terminal and retype
    them into a web form. Everything the form needs except the person's own
    name and their public write-up is already known here, so it is filled in
    for them. GitHub issue forms accept prefill through query parameters keyed
    by field id.
    """
    totals = report["totals"]
    result = (f"{totals['passed']} passed, {totals['failed']} failed, "
              f"{totals['skipped']} skipped, {totals['cases_passed']} cases")
    suites = report.get("suites") or []
    # A skipped suite is still part of a full run, so a whole-corpus run says
    # so rather than listing forty-two names because one suite skipped.
    if selected is None:
        which = f"all {len(suites)} suites"
    else:
        names = [r["suite"] for r in suites]
        which = ", ".join(names[:6]) + (f" and {len(names) - 6} more"
                                        if len(names) > 6 else "")
    fields = {
        "template": VCR_TEMPLATE,
        "title": "VCR: reproduction by ",
        "suites": which,
        "result": result,
    }
    commit = _head_commit(repo_root)
    if commit:
        fields["at-commit"] = commit
    return VCR_FORM + "?" + urllib.parse.urlencode(fields)


def print_submit_block(report: dict, repo_root: Path, selected=None) -> None:
    """Two lines and a link. Shown for a failing run too.

    A run that does not pass is still a legitimate row, and an honest one. On
    2026-08-16 Pablo Play reported 0 of 9 on this corpus with a stated reason,
    and that reason was more informative than the count. Hiding the link
    behind a green run would collect only the flattering half.
    """
    link = submit_link(report, repo_root, selected)
    print()
    print("Want this result listed at vaara.io/conformance.html?")
    print("Open this link. Your commit, suites and totals are already filled in;")
    print("add your name and where you reported it, then submit. No install, no PR,")
    print("no approval step. A row that failed is as welcome as one that passed.")
    print()
    print(f"  {link}")


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
    parser.add_argument("--no-submit-link", action="store_true",
                        help="omit the listing link (for CI and scripted runs)")
    parser.add_argument("--with-vaara", action="store_true",
                        help="also grade suites whose checker needs a built artifact "
                             "(requires Vaara installed; skipped silently if not)")
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
    rows = [run_suite(vectors_dir, s, with_vaara=args.with_vaara) for s in suites]
    _print_table(rows)

    report = build_report(rows, vectors_dir, stamp)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport written to {args.json}")

    t = report["totals"]
    print(f"\n{t['passed']} passed, {t['failed']} failed, {t['skipped']} skipped "
          f"({t['cases_passed']} cases) across {t['suites']} suites.")
    failed = bool(t["failed"])
    if failed:
        print(f"FAIL: {', '.join(r['suite'] for r in rows if r['status'] == 'FAIL')}")
    else:
        print("PASS: every suite that ran matched its expected verdicts.")

    # Shown last so it is the thing still on screen when the run ends, and
    # shown whether or not the run passed. Deliberately not gated on isatty:
    # a piped or tee'd run is still a person reading the output, and the whole
    # value here is that nobody has to go looking for the form.
    if not args.no_submit_link:
        print_submit_block(report, Path(__file__).resolve().parent.parent, args.corpus)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
