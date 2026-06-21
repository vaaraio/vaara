"""Gate the vaara.contiguity/v0 vectors through the independent checker.

The checker imports no Vaara: a passing run means the committed stream proves its
own completeness, and the dropped-receipt case proves the seq-2 gap, from the
held receipts and the public key alone. This is the conformance bar the
completeness property claims, run in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

VECTORS = Path(__file__).resolve().parents[1] / "vectors" / "contiguity_v0"


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
