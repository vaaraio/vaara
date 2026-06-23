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
        ("deny_relabeled", 1),
    ],
)
def test_cli_enforce_by_class(case, code):
    proc = subprocess.run(
        [sys.executable, "-m", "vaara.cli", "enforce-by-class", str(VECTORS / case), *_PERMIT],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == code, proc.stdout + proc.stderr


def test_cli_relabeled_seal_fails_closed_keyless():
    # The relabel attack: maxClass changed to a permitted class after signing. The
    # keyless binding check must drop the unbound seal and deny.
    proc = subprocess.run(
        [sys.executable, "-m", "vaara.cli", "enforce-by-class",
         str(VECTORS / "deny_relabeled"), *_PERMIT],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "does not bind" in proc.stderr


@pytest.mark.parametrize("case,code", [("permit", 0), ("deny_relabeled", 1)])
def test_cli_enforce_by_class_with_key(case, code):
    # With --key the issuer signature is verified too; the honest permit still
    # permits and the relabeled seal still denies.
    key = VECTORS / "keys" / "es256_public.pem"
    proc = subprocess.run(
        [sys.executable, "-m", "vaara.cli", "enforce-by-class", str(VECTORS / case),
         *_PERMIT, "--key", str(key)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == code, proc.stdout + proc.stderr
