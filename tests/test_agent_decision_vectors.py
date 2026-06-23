"""Gate the agent_decision_v0 conformance vector through the independent checker.

The checker imports no Vaara: a passing run means the DSSE-enveloped
``agent-decision/v0.1`` statement proves out from the bytes and the published
Ed25519 public key alone (the PAE digest and the signature), and the SEP-2828
mapping reproduces from the shipped declarative profile spec. This is the
recomputable ``{statement, expected-verdict}`` vector offered on
in-toto/attestation#554, run in CI.

A second test pins that the product's own ``normalize`` produces the same
mapping the vector commits, so the declarative profile and the vector cannot
drift apart silently.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

VECTORS = Path(__file__).resolve().parent / "vectors" / "agent_decision_v0"


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_module_reproduces_committed_mapping():
    from vaara.attestation.receipt import normalize

    statement = json.loads((VECTORS / "statement.json").read_text())
    expected = json.loads((VECTORS / "expected.json").read_text())
    assert normalize(statement).to_dict() == expected["normalized"]
