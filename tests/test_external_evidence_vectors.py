"""Gate the external_evidence_v0 binding vectors through the independent checker.

The checker imports no Vaara: a passing run means the held external-evidence slots
and the vaara.receipt/v1 receipts prove the binding from the public key alone, the
dropped case proving the seq-1 gap inside the trace boundary while every held slot
still resolves. The held streams also gate through the shipped
``vaara verify-contiguity`` CLI, since they carry standard completeness blocks.
This is the conformance bar the generic external-evidence binding profile claims,
run in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

VECTORS = Path(__file__).resolve().parent / "vectors" / "external_evidence_v0"


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("case,code", [("complete", 0), ("dropped", 1)])
def test_cli_verify_contiguity(case, code):
    proc = subprocess.run(
        [sys.executable, "-m", "vaara.cli", "verify-contiguity", str(VECTORS / case)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == code, proc.stdout + proc.stderr
