"""Gate the governance_decision_v0 vectors through the independent checker.

The checker imports no Vaara: a passing run means the committed CrewAI
``GovernanceDecision`` / ``GovernanceOutcome`` corpus proves its own derivations,
its sealed completeness (mid-gap, suffix drop, and the unsealed residual), the
four fail-closed verdicts, and the non-ASCII canonicalization, from the held
bytes and the public key alone. This is the conformance bar, run in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

VECTORS = Path(__file__).resolve().parent / "vectors" / "governance_decision_v0"


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
