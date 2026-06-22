"""Gate the tap_v0 binding vectors through the independent checker.

The checker imports no Vaara: a passing run means the committed Visa TAP request
and the held vaara.receipt/v1 receipts prove the binding from the public key
alone, across the in-progress / terminal lifecycle, with no TAP service and no
live endpoint. This is the conformance bar the TAP binding profile claims, run
in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

VECTORS = Path(__file__).resolve().parent / "vectors" / "tap_v0"


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
