# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""docs/PRIOR_ART.md is a priority-date claim, so its dates have to be right.

The document tells a reader comparing Vaara against newer work to check the
timeline instead of trusting marketing. That only holds if every row checks
out, and 12 of 87 rows carried a date no public record supported.

A row is satisfied by the GitHub release date or the PyPI upload date; both
are public and, where both exist, they agree. PyPI alone is not the standard,
because yanking a release there used to delete its files: v0.1.0 and v0.71.0
shipped and were later yanked, so they are gone from PyPI while remaining
real releases. Those two carry a dagger in the table and are checked against
CHANGELOG.md instead.

Network checks skip when the service is unreachable, so an offline run does
not fail. The offline half always runs.
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

#: Releases that shipped and were later yanked, back when yanking on PyPI
#: deleted the files. They are real releases with real ship dates; the only
#: record that still carries them is CHANGELOG.md, so they carry a dagger.
YANKED = {"v0.1.0", "v0.71.0"}

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
        return {}  # unreachable: pytest.skip raises. Static analysers cannot
                   # see that, and read `releases` below as possibly unbound.
    return {
        f"v{version}": files[0]["upload_time"][:10]
        for version, files in releases.items()
        if files
    }


def _github_dates() -> dict[str, str]:
    """Release dates from the GitHub releases API.

    Unauthenticated and paginated; a rate-limited or offline run skips
    rather than failing, same as the PyPI check.
    """
    dates: dict[str, str] = {}
    for page in range(1, 4):
        url = (
            "https://api.github.com/repos/vaaraio/vaara/releases"
            f"?per_page=100&page={page}"
        )
        try:
            request = urllib.request.Request(
                url, headers={"Accept": "application/vnd.github+json"},
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                batch = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            pytest.skip(f"GitHub releases unreachable: {exc}")
            return {}  # unreachable, same reason as _pypi_dates
        if not batch:
            break
        for release in batch:
            stamp = release.get("published_at") or release.get("created_at")
            if release.get("tag_name") and stamp:
                dates[release["tag_name"]] = stamp[:10]
    return dates


def test_the_table_has_rows_to_check():
    assert len(_rows()) > 50


def test_only_the_yanked_releases_carry_a_dagger():
    for tag, daggered, _ in _rows():
        assert daggered == (tag in YANKED), (
            f"{tag}: dagger={daggered} but yanked={tag in YANKED}"
        )


def test_daggered_dates_match_the_changelog():
    """The changelog is the record those rows actually rest on."""
    changelog = CHANGELOG.read_text()
    for tag, daggered, date in _rows():
        if not daggered:
            continue
        heading = f"## [{tag[1:]}] - {date}"
        assert heading in changelog, f"{tag}: CHANGELOG has no {heading!r}"


@pytest.mark.network
def test_every_undaggered_date_matches_a_public_release_record():
    """GitHub or PyPI must back the date. Either one is enough.

    Not both: v1.56.0 has a GitHub release and never went to PyPI, and the
    GitHub release sometimes lands a day before the PyPI upload.
    """
    pypi, github = _pypi_dates(), _github_dates()
    wrong = []
    for tag, daggered, date in _rows():
        if daggered:
            continue
        known = {d for d in (pypi.get(tag), github.get(tag)) if d}
        if not known or date not in known:
            wrong.append((tag, date, sorted(known) or ["no public record"]))
    assert not wrong, "PRIOR_ART dates no public record supports: " + "; ".join(
        f"{tag} says {claimed}, records say {found}" for tag, claimed, found in wrong
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
