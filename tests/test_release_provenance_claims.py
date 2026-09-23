"""What the public pages say about release provenance matches the release job.

The README, the site and the OWASP mapping said releases carry SLSA Build
Level 3 provenance, verifiable with `slsa-verifier verify-artifact`. The
release job attests with `actions/attest-build-provenance` from its own
workflow, which is Build Level 2, and slsa-verifier rejects that builder as
an untrusted reusable workflow. Level 3 needs an isolated reusable builder
such as slsa-github-generator. These tests hold the text to the job.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = (ROOT / ".github" / "workflows" / "release.yml").read_text()
SURFACES = [
    "README.md",
    "llms.txt",
    "webpage/llms.txt",
    "webpage/index.html",
    "docs/OWASP_AGENTIC.md",
    "docs/signing-keys.md",
    ".github/SECURITY.md",
]
LEVEL3 = re.compile(r"(SLSA\s+)?Build\s+L(evel\s*)?3", re.IGNORECASE)


def _uses_an_isolated_builder() -> bool:
    return "slsa-framework/slsa-github-generator" in RELEASE


def test_the_release_job_attests_build_provenance():
    assert "actions/attest-build-provenance" in RELEASE


def test_no_public_surface_claims_level_3_without_an_isolated_builder():
    if _uses_an_isolated_builder():
        return
    offenders = [
        f"{rel}: {m.group(0)}"
        for rel in SURFACES
        if (ROOT / rel).exists()
        for m in LEVEL3.finditer((ROOT / rel).read_text())
    ]
    assert offenders == []


def test_the_documented_verifier_is_one_that_accepts_the_attestation():
    text = (ROOT / "README.md").read_text() + (ROOT / "docs" / "signing-keys.md").read_text()
    assert "gh attestation verify" in text
    if not _uses_an_isolated_builder():
        assert "slsa-verifier verify-artifact" not in text
