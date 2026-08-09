# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""docs/PRIOR_ART.md is a priority-date claim, so its dates have to be right.

The document tells a reader comparing Vaara against newer work to check the
timeline instead of trusting marketing. That only holds if every row checks
out: 13 of 87 rows had a date that disagreed with the PyPI upload date, and
5 cited a version with no PyPI release and, for four of them, no git tag
either.

The network check is skipped when PyPI is unreachable, so an offline run
does not fail. The offline half — that every dated row is either verifiable
or explicitly daggered — always runs.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

PRIOR_ART = Path(__file__).resolve().parents[1] / "docs" / "PRIOR_ART.md"
CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"

#: Versions that never reached PyPI. Their dates come from CHANGELOG.md and
#: they carry a dagger in the table so a reader knows which record to check.
UNPUBLISHED = {"v0.1.0", "v0.3.0", "v0.4.1", "v0.71.0", "v1.56.0"}

ROW = re.compile(r"\| (v\d+\.\d+\.\d+)(†?), (\d{4}-\d{2}-\d{2}) \|")


def _rows() -> list[tuple[str, bool, str]]:
    return [
        (tag, bool(dagger), date)
        for tag, dagger, date in ROW.findall(PRIOR_ART.read_text())
    ]


def _pypi_dates() -> dict[str, str]:
    try:
        with urllib.request.urlopen(
            "https://pypi.org/pypi/vaara/json", timeout=20,
        ) as response:
            releases = json.load(response)["releases"]
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip(f"PyPI unreachable: {exc}")
    return {
        f"v{version}": files[0]["upload_time"][:10]
        for version, files in releases.items()
        if files
    }


def test_the_table_has_rows_to_check():
    assert len(_rows()) > 50


def test_every_dagger_marks_a_version_with_no_pypi_release():
    for tag, daggered, _ in _rows():
        if daggered:
            assert tag in UNPUBLISHED, f"{tag} is daggered but was published"


def test_every_unpublished_version_carries_its_dagger():
    """Without the mark a reader would check PyPI and find nothing."""
    for tag, daggered, _ in _rows():
        if tag in UNPUBLISHED:
            assert daggered, f"{tag} has no PyPI release but is not daggered"


def test_daggered_dates_match_the_changelog():
    """The changelog is the record those rows actually rest on."""
    changelog = CHANGELOG.read_text()
    for tag, daggered, date in _rows():
        if not daggered:
            continue
        heading = f"## [{tag[1:]}] - {date}"
        assert heading in changelog, f"{tag}: CHANGELOG has no {heading!r}"


@pytest.mark.network
def test_every_published_row_matches_the_pypi_upload_date():
    pypi = _pypi_dates()
    wrong = [
        (tag, date, pypi[tag])
        for tag, daggered, date in _rows()
        if not daggered and tag in pypi and pypi[tag] != date
    ]
    assert not wrong, "PRIOR_ART dates disagree with PyPI: " + "; ".join(
        f"{tag} says {claimed}, PyPI says {actual}" for tag, claimed, actual in wrong
    )


@pytest.mark.network
def test_no_undaggered_row_cites_a_version_pypi_does_not_have():
    pypi = _pypi_dates()
    missing = [tag for tag, daggered, _ in _rows() if not daggered and tag not in pypi]
    assert not missing, (
        f"rows cite versions with no PyPI release and no dagger: {sorted(set(missing))}"
    )


#: Directories a citation may point into. Anchoring on these keeps the check
#: on real repo paths and off the other slash-bearing things the table
#: quotes: media types (`vaara.receipt/v1`), MCP methods (`tools/call`),
#: npm scopes (`@vaara/client`) and home paths (`~/.vaara/...`).
_CITED_ROOTS = (
    "src/", "tests/", "docs/", "clients/", "conformance/", "bench/",
    "fuzz/", "scripts/", ".github/", "examples/",
)


def test_evidence_paths_in_the_table_exist():
    """A citation a reader cannot open is not evidence."""
    root = PRIOR_ART.resolve().parents[1]
    missing = set()
    for line in PRIOR_ART.read_text().splitlines():
        if not line.startswith("| "):
            continue
        for cited in re.findall(r"`([^`]+)`", line):
            candidate = cited.split(":")[0].strip()
            if not candidate.startswith(_CITED_ROOTS):
                continue
            if any(ch in candidate for ch in "*<> "):
                continue  # a glob or prose, not a single path
            if not (root / candidate).exists():
                missing.add(candidate)
    assert missing == set(), (
        f"PRIOR_ART cites paths that do not exist: {sorted(missing)}"
    )
