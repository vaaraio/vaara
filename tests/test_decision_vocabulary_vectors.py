# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gate the decision_vocabulary_v0 vectors through the independent checker.

The checker imports no Vaara and needs no optional dependency: a passing run
means the committed records prove, from their own bytes, that every verdict
stayed inside the closed three, that every refinement agreed with the verdict
beside it, and that a retry after a modify carried exactly the arguments the
gate proposed. The five adversarial cases prove the checker rejects the records
a gate without the projection would write.

The generator is gated too. Vectors that can drift from the code that produced
them are evidence of nothing, so `--check` regenerates and fails on any diff.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VECTORS = Path(__file__).resolve().parent / "vectors" / "decision_vocabulary_v0"


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_committed_vectors_are_what_vaara_produces_today():
    proc = subprocess.run(
        [sys.executable,
         str(ROOT / "scripts" / "build_decision_vocabulary_vectors.py"),
         "--check"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_adversarial_cases_really_are_non_conforming():
    """A corpus where nothing ever fails proves nothing about the checker."""
    import json

    expected = json.loads((VECTORS / "expected.json").read_text())
    rejected = {name for name, case in expected.items() if not case["conforms"]}
    assert rejected == {
        "verdict_outside_the_enum",
        "refinement_contradicts_verdict",
        "retry_bound_to_other_arguments",
        "chain_break",
    }


def test_no_record_says_allow_against_arguments_it_did_not_decide():
    """The claim the suite exists to make, asserted on the corpus itself.

    Only over the cases Vaara produced. The adversarial ones are committed
    precisely because they break this, which is what gives the checker teeth.
    """
    import json

    expected = json.loads((VECTORS / "expected.json").read_text())
    for case in sorted((VECTORS / "cases").glob("*.json")):
        if not expected[case.stem]["conforms"]:
            continue
        records = json.loads(case.read_text())
        for record in records:
            data = record.get("data") or {}
            if data.get("decision_detail") == "modify":
                assert data["decision"] == "deny", case.name
