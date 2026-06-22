"""Gate the class_gate_v0 vectors through the independent checker and the CLI.

The checker imports no Vaara: a passing run means the held receipts and the signed
seal prove the enforcement-time gate from the public key alone, the
``permit_gap_bounded`` case permitting over a real gap because the seal bounds it,
``deny_unbounded`` failing closed when no class is sealed. The same held streams
gate through the shipped ``vaara enforce-by-class`` CLI (exit 0 permit / 1 deny).
This is the conformance bar the sealed-class enforcement profile claims, run in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

VECTORS = Path(__file__).resolve().parent / "vectors" / "class_gate_v0"
_PERMIT = ["--permit", "data.read", "--permit", "data.write"]


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize(
    "case,code",
    [
        ("permit", 0),
        ("permit_gap_bounded", 0),
        ("deny_class", 1),
        ("deny_unbounded", 1),
    ],
)
def test_cli_enforce_by_class(case, code):
    proc = subprocess.run(
        [sys.executable, "-m", "vaara.cli", "enforce-by-class", str(VECTORS / case), *_PERMIT],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == code, proc.stdout + proc.stderr
