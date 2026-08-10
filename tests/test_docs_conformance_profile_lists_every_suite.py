# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The published profile has to name every suite the runner discovers.

conformance-profile.md is the document that defines what conformance
covers: "An implementation conforms to Profile v1 when, for every suite,
its own vectors reproduce the published verdicts under the same checker."
It said 37 suites and listed 37. The repository ships 43, and the runner
discovers all 43.

The six it never mentioned were acp_checkout_v0, agent_decision_v0,
credential_grant_v0, crewai_enforcement_v0, fallback_projection_v0 and
qualified_time_v0. A third party implementing to the published profile
would not have known they exist, which is the wrong direction for a
document whose whole job is to fix a target.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "conformance-profile.md"
VECTORS = ROOT / "tests" / "vectors"


def _shipped_suites() -> set[str]:
    """Exactly what `conformance_runner.py --list` discovers."""
    return {
        path.name
        for path in VECTORS.iterdir()
        if (path / "_check_independent.py").exists()
    }


def _documented_suites() -> set[str]:
    block = re.search(
        r"## Suites in Profile v1\s*\n+```\n(.*?)```", DOC.read_text(), re.S
    )
    assert block, "conformance-profile.md no longer lists its suites"
    return set(block.group(1).split())


def test_every_shipped_suite_is_in_the_published_profile():
    missing = _shipped_suites() - _documented_suites()
    assert not missing, (
        f"docs/conformance-profile.md does not list {sorted(missing)}, which "
        f"ship with checkers. The profile understates its own coverage."
    )


def test_the_profile_lists_no_suite_that_does_not_ship():
    invented = _documented_suites() - _shipped_suites()
    assert not invented, (
        f"docs/conformance-profile.md lists {sorted(invented)}, which is not "
        f"in tests/vectors/ with a checker"
    )


def test_the_stated_count_is_the_real_count():
    stated = re.search(r"Profile v1 covers the (\d+) suites", DOC.read_text())
    assert stated, "conformance-profile.md no longer states a suite count"
    assert int(stated.group(1)) == len(_shipped_suites())


def test_the_always_skipping_suite_is_named():
    """article12_fold_v0 takes a path argument, so it never runs bare."""
    text = DOC.read_text()
    assert "article12_fold_v0" in text
    for suite in ("pq_hybrid_v0", "qualified_time_v0"):
        assert suite in text, (
            f"{suite} skips without an optional dependency; the profile has to "
            f"say so or a reader reads the skip as a failure"
        )
