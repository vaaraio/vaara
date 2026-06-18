"""The credential_grant_v0 vectors: Vaara and the independent checker agree."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "conformance" / "sep2828" / "credential_grant_v0"


def _cases():
    return sorted(json.loads((VECTORS / "expected.json").read_text()))


def test_at_least_four_cases_present():
    assert len(_cases()) >= 4


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("name", _cases())
def test_in_tree_verifier_agrees_with_vectors(name):
    from vaara.attestation.receipt import attestation_digest
    from vaara.attestation.sep2787 import parse_attestation
    from vaara.credential import grant_from_dict, verify_grant

    d = VECTORS / "sets" / name
    expected = json.loads((VECTORS / "expected.json").read_text())[name]
    grant = grant_from_dict(json.loads((d / "grant.json").read_text()))
    att = parse_attestation(json.loads((d / "attestation.json").read_text()))
    inputs = json.loads((d / "inputs.json").read_text())

    verdict = verify_grant(
        grant,
        verifying_material=bytes.fromhex(inputs["keyHex"]),
        runtime_tool_name=inputs["toolName"],
        runtime_args=inputs["args"],
        runtime_tenant_id=inputs["tenantId"],
        known_attestation_digests=frozenset({attestation_digest(att)}),
        now=inputs["now"],
    )
    assert verdict.ok == expected["ok"]
    assert verdict.reason == expected["reason"]
