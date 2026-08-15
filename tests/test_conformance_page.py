# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vaara Conformance Results has to say what the checkers actually did.

The page exists so a stranger can see every suite, its verdict, and every
outside party who reproduced the vectors. That is worth nothing if the numbers
on it are from a run three releases ago, and a stale conformance claim is worse
than no page: it invites someone to check and find the table wrong.

So the page is generated from the runner's own report and never edited by hand,
and this test regenerates it and fails if what is committed differs. It also
holds the two properties the page would quietly lose first: the honest limits
paragraph, and no off-origin resource loads on a site whose whole claim is that
it works with the network off.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "webpage" / "conformance.html"
RENDERER = ROOT / "scripts" / "render_conformance_page.py"
RUNNER = ROOT / "scripts" / "conformance_runner.py"
REPRODUCTIONS = ROOT / "conformance" / "reproductions.json"


def fresh_report(tmp_path) -> dict:
    """Run the corpus and hand back the report, refusing to guess on failure."""
    report = tmp_path / "report.json"
    run = subprocess.run(
        [sys.executable, str(RUNNER), "--json", str(report)],
        capture_output=True, text=True, timeout=900, cwd=ROOT,
    )
    assert report.exists(), f"runner produced no report:\n{run.stdout}\n{run.stderr}"
    return json.loads(report.read_text(encoding="utf-8"))


def page_count(page: str, key: str) -> int:
    found = re.search(rf"<b>(\d+)</b><span>{key}</span>", page)
    assert found, f"the page carries no {key} count at all"
    return int(found.group(1))


def require_the_same_corpus(report: dict) -> None:
    """Skip where the environment cannot reach every suite the page was built from.

    The committed page is generated with the optional extras installed, so a
    checkout without them skips suites the page counts as passing. Comparing
    the two reports a missing wheel as a stale page, which sends whoever reads
    the failure looking for a drift that is not there. A real edit to the page
    leaves the skip count alone, so this stays sharp against the thing it
    guards.
    """
    committed_skips = page_count(PAGE.read_text(encoding="utf-8"), "skipped")
    if report["totals"]["skipped"] > committed_skips:
        missing = [
            s["suite"] for s in report["suites"]
            if s["status"] == "SKIP"
        ]
        pytest.skip(
            f"environment skips {report['totals']['skipped']} suites against "
            f"{committed_skips} on the page, so it cannot reproduce it. "
            f"Install the extras to run this. Skipped here: {', '.join(missing)}"
        )


def test_page_exists():
    assert PAGE.is_file(), "webpage/conformance.html is missing"


def test_page_matches_a_fresh_run(tmp_path):
    """Regenerate from a live run and compare. Catches a stale committed page."""
    require_the_same_corpus(fresh_report(tmp_path))
    check = subprocess.run(
        [sys.executable, str(RENDERER), str(tmp_path / "report.json"), str(PAGE), "--check"],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert check.returncode == 0, check.stderr or check.stdout


def test_totals_on_the_page_are_the_real_totals(tmp_path):
    report = fresh_report(tmp_path)
    require_the_same_corpus(report)
    page = PAGE.read_text(encoding="utf-8")
    for key in ("suites", "passed", "failed", "skipped"):
        assert f"<b>{report['totals'][key]}</b><span>{key}</span>" in page, (
            f"the page does not carry the real {key} count ({report['totals'][key]})"
        )


def test_every_suite_is_listed(tmp_path):
    """No suite is quietly dropped, including the skips.

    This one holds in a reduced environment too: a suite that skips for a
    missing dependency still has to appear on the page, so the comparison is
    over names rather than verdicts.
    """
    report = fresh_report(tmp_path)
    page = PAGE.read_text(encoding="utf-8")
    missing = [s["suite"] for s in report["suites"] if s["suite"] not in page]
    assert not missing, f"suites run but not shown on the page: {missing}"


def test_the_honest_limits_paragraph_survives():
    """The first thing a marketing edit would delete."""
    # Whitespace-normalised: the sentences are line-wrapped in the source, and
    # a test that breaks on rewrapping gets deleted rather than fixed.
    page = re.sub(r"\s+", " ", PAGE.read_text(encoding="utf-8"))
    assert "there is no ratification process behind them" in page
    assert "Recompute is checkable by strangers. Authorship is not." in page


def test_no_off_origin_resource_loads():
    """vaara.io claims to work with the network off. A link is fine, a fetch is not."""
    page = PAGE.read_text(encoding="utf-8")
    loads = re.findall(r'(?:src|srcset)\s*=\s*"([^"]+)"', page)
    offsite = [u for u in loads if u.startswith(("http://", "https://", "//"))]
    assert not offsite, f"off-origin resource loads on the conformance page: {offsite}"


def test_reproductions_file_is_valid_and_every_record_is_public():
    """A reproduction row without a public record is an assertion, not evidence."""
    data = json.loads(REPRODUCTIONS.read_text(encoding="utf-8"))
    for row in data["reproductions"]:
        for field in ("date", "party", "result", "record", "record_held_by", "their_scoping"):
            assert row.get(field), f"reproduction row missing {field}: {row}"
        assert row["record"].startswith("https://"), (
            f"the record for {row['party']} is not a public link"
        )
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["date"]), row["date"]


def test_reproductions_are_not_the_maintainer():
    """The page is for outside reproductions. Vaara grading itself proves nothing."""
    data = json.loads(REPRODUCTIONS.read_text(encoding="utf-8"))
    for row in data["reproductions"]:
        assert "sirkkavaara" not in row["party"].lower(), (
            "the maintainer is not an independent reproduction"
        )


def test_page_is_linked_from_the_site():
    index = (ROOT / "webpage" / "index.html").read_text(encoding="utf-8")
    sitemap = (ROOT / "webpage" / "sitemap.xml").read_text(encoding="utf-8")
    assert "/conformance.html" in index, "nothing on the front page links to it"
    assert "conformance.html" in sitemap, "not in the sitemap, so it will not be indexed"
