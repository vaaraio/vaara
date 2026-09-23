"""vaara.io/verify.html reaches the verdict its README paragraph promises.

The README says the page recomputes the DSSE pre-authentication encoding and
checks the Ed25519 signature with WebCrypto. Nothing ran that code until
2026-09-23, when it turned out to report SIGNATURE VERIFIES for an envelope
with no signatures at all: the loop over signatures never ran, so nothing
cleared the verdict. This runs the page's own verify() under Node.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "js" / "verify_html_harness.mjs"
PAGE = ROOT / "webpage" / "verify.html"

EXPECTED = {
    "good": True,
    "wrong_key": False,
    "forged_payload": False,
    "no_signatures": False,
    "no_key": False,
    "non_ascii_type": True,
}


@pytest.fixture(scope="module")
def verdicts() -> dict[str, bool]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    out = subprocess.run([node, str(HARNESS), str(PAGE)], capture_output=True,
                         text=True, timeout=60, check=True).stdout
    rows = [json.loads(line) for line in out.splitlines() if line.strip()]
    return {r["case"]: r["allOk"] for r in rows}


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_verdict(verdicts, case):
    assert verdicts[case] is EXPECTED[case]
