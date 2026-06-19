"""Gate the vaara.authorization/v0 vectors through the independent checker.

The checker imports no Vaara: a passing run means the committed deny-receipt
(and the allow-receipt) recompute from the grant, the arguments, and the public
key alone. This is the conformance bar the profile claims, run in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

VECTORS = Path(__file__).resolve().parents[1] / "vectors" / "authorization_v0"


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
