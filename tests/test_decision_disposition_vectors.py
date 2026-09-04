# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bind the shipped disposition check to decision_disposition_v0.

The vectors ship with `_check_independent.py`, which reimplements the rules
from the text and imports nothing from Vaara. This file runs the SHIPPED
implementation over the same cases and requires the same verdicts, so the two
cannot drift apart without a test going red.

That is the same arrangement the other suites use, and it is the reason a
published vector is worth more than a published assertion.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vaara._disposition import DispositionError, check

VECTORS = Path(__file__).parent / "vectors" / "decision_disposition_v0"


def _load(name):
    return json.loads((VECTORS / f"{name}.json").read_text())


CASES = _load("cases")
EXPECTED = _load("expected")


@pytest.mark.parametrize("name", sorted(CASES))
def test_shipped_check_agrees_with_the_vector(name):
    case, want = CASES[name], EXPECTED[name]
    approver, flag = case.get("approver"), case.get("human_disposed")

    if approver is None and flag is None:
        # Absent disposition never reaches check(); the trail omits the keys.
        assert want["conforming"] is True
        assert want["keys_present"] is False
        return

    if approver is None:
        # A bare flag is rejected by the trail before check() is reached, so
        # the vector's verdict is asserted against that rule rather than here.
        assert want["conforming"] is False
        assert want["reason_class"] == "human_claim_by_policy"
        return

    if want["conforming"]:
        got_approver, got_flag = check(approver, flag)
        assert got_approver == want["approver"]
        assert got_flag == want["human_disposed"]
    else:
        with pytest.raises(DispositionError):
            check(approver, flag)


def test_the_independent_checker_passes_on_its_own_bytes():
    """The checker a third party would run, run here on every commit."""
    result = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 disagreement(s)" in result.stdout


def test_the_suite_covers_the_three_allow_shaped_cases():
    """A regression guard on the vectors themselves. If someone deletes the
    replay case, the suite stops testing the thing it exists for."""
    allow_cases = {
        n for n, c in CASES.items()
        if c["decision"] == "allow" and EXPECTED[n]["conforming"]
    }
    assert {
        "policy_allow",
        "human_approved_at_escalation",
        "replayed_prior_approval",
    } <= allow_cases

    live = EXPECTED["human_approved_at_escalation"]
    replayed = EXPECTED["replayed_prior_approval"]
    assert live["human_disposed"] != replayed["human_disposed"]
