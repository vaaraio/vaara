"""Gate the vaara.contiguity/v0 vectors through the independent checker.

The checker imports no Vaara: a passing run means the committed stream proves its
own completeness, and the dropped-receipt case proves the seq-2 gap, from the
held receipts and the public key alone. This is the conformance bar the
completeness property claims, run in CI.
"""

from __future__ import annotations

import json
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


@pytest.mark.parametrize(
    "case,code", [("complete", 0), ("dropped", 1), ("fork", 1), ("replay", 1)]
)
def test_cli_verify_contiguity_exit_codes(case, code):
    key = VECTORS / "keys" / "es256_public.pem"
    proc = subprocess.run(
        [sys.executable, "-m", "vaara.cli", "verify-contiguity",
         str(VECTORS / case), "--key", str(key)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == code, proc.stdout + proc.stderr


@pytest.mark.parametrize(
    "case,kind,line",
    [
        ("fork", "fork", "duplicate seq: 2 (different records)"),
        ("replay", "replay", "duplicate seq: 2 (same record)"),
    ],
)
def test_cli_verify_contiguity_names_fork_or_replay(case, kind, line):
    key = VECTORS / "keys" / "es256_public.pem"
    base = [sys.executable, "-m", "vaara.cli", "verify-contiguity", str(VECTORS / case)]
    proc = subprocess.run(base + ["--key", str(key)], capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert line in proc.stdout
    proc = subprocess.run(base + ["--json"], capture_output=True, text=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["duplicates"] == [{"seq": 2, "kind": kind}]
    assert report["duplicateSeqs"] == [2]
