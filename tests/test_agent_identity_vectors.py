"""Conformance: the committed agent_identity_v0 vectors verify two ways.

1. Vaara's ``verify_receipt_identity`` reproduces each committed verdict.
2. The standalone ``_check_independent.py`` (no Vaara import) reproduces
   them too, proving the format is consumable without depending on Vaara.

See ``docs/design/resolvable-agent-identity-spec.md``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from vaara.attestation.receipt import (  # noqa: E402
    parse_receipt,
    verify_receipt_identity,
    verify_receipt_identity_live,
)

VECTORS = Path(__file__).resolve().parent / "vectors" / "agent_identity_v0"


def _expected():
    return json.loads((VECTORS / "expected.json").read_text())


@pytest.mark.parametrize("name", ["bound", "unbound", "revoked"])
def test_vaara_reproduces_vector_verdict(name):
    case = json.loads((VECTORS / f"{name}.json").read_text())
    receipt = parse_receipt(case["receipt"])
    result = verify_receipt_identity(receipt, case["didDocument"])
    want = _expected()[name]
    assert result.resolved is want["resolved"]
    assert result.bound is want["bound"]
    assert result.keyid == want["keyid"]


@pytest.mark.parametrize("name", ["bound", "unbound", "revoked"])
def test_vaara_live_reproduces_vector_verdict(name):
    # Level 3 against the captured document via an offline fetcher: the
    # full verdict, including revoked/trusted, reproduces with no network.
    case = json.loads((VECTORS / f"{name}.json").read_text())
    receipt = parse_receipt(case["receipt"])
    raw = json.dumps(case["didDocument"]).encode("utf-8")
    result = verify_receipt_identity_live(receipt, fetcher=lambda url: raw)
    want = _expected()[name]
    assert result.resolved is want["resolved"]
    assert result.bound is want["bound"]
    assert result.keyid == want["keyid"]
    assert result.revoked is want["revoked"]
    assert result.trusted is want["trusted"]


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
