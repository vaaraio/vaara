# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The published adapter mapping table has to match the shipped mappings.

COMPLIANCE.md sells the adapter table as the artefact: "The adapter is
thin. The mapping is the artefact. A deployer can read the table, dispute
a row, and override mappings without touching adapter code." A deployer
who does that reads the counts and the category list, so both have to be
the real ones.

They were not. `_content_safety_articles.py` grew a GCP `virus_scan` ->
`malicious_file` row when the adapters were reconciled against their
upstream SDKs, and the doc kept saying 27 cloud rows and 68 total. The
`malicious_file` category was missing from the cloud table entirely, so a
deployer auditing what Vaara records for a Model Armor virus finding
would have concluded Vaara records nothing.

These tests fail the build when a mapping is added or removed without the
published table and its counts moving with it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from vaara.integrations._content_safety_articles import (
    AZURE_MAPPINGS, BEDROCK_MAPPINGS, GCP_MAPPINGS, GUARDRAILS_AI_MAPPINGS,
    LLM_GUARD_MAPPINGS, NEMO_MAPPINGS, REBUFF_MAPPINGS,
)

DOC = Path(__file__).resolve().parent.parent / "docs" / "COMPLIANCE.md"
ADAPTERS_DOC = DOC.parent / "adapters.md"

CLOUD = {"Bedrock": BEDROCK_MAPPINGS, "Azure": AZURE_MAPPINGS, "GCP": GCP_MAPPINGS}
OSS = {
    "NeMo": NEMO_MAPPINGS,
    "Guardrails AI": GUARDRAILS_AI_MAPPINGS,
    "LLM Guard": LLM_GUARD_MAPPINGS,
    "Rebuff": REBUFF_MAPPINGS,
}

CLOUD_TOTAL = sum(len(m) for m in CLOUD.values())
OSS_TOTAL = sum(len(m) for m in OSS.values())

# Table heading -> the heading that ends its section.
_TABLES = {
    "cloud": ("### Category to article mapping\n", "### Where the finding lands"),
    "oss": ("### Category to article mapping (OSS providers)", "### Where the finding lands"),
}


def _documented_categories(which: str) -> set[str]:
    """Vaara categories in the first column of one mapping table."""
    text = DOC.read_text(encoding="utf-8")
    start_heading, end_heading = _TABLES[which]
    start = text.index(start_heading)
    end = text.index(end_heading, start)
    section = text[start:end]
    return set(re.findall(r"^\|\s*`([a-z_]+)`\s*\|", section, re.MULTILINE))


def _real_categories(groups: dict) -> set[str]:
    return {m.vaara_category for group in groups.values() for m in group}


@pytest.mark.parametrize("which,groups", [("cloud", CLOUD), ("oss", OSS)])
def test_documented_categories_match_the_shipped_mappings(which, groups):
    documented = _documented_categories(which)
    real = _real_categories(groups)
    assert documented == real, (
        f"docs/COMPLIANCE.md {which} adapter table drifted from "
        f"_content_safety_articles.py.\n"
        f"  documented but not shipped: {sorted(documented - real)}\n"
        f"  shipped but not documented: {sorted(real - documented)}"
    )


def test_cloud_row_count_is_the_real_one():
    text = DOC.read_text(encoding="utf-8")
    match = re.search(r"(\d+) rows total across the three vendors", text)
    assert match, "COMPLIANCE.md no longer states a cloud row count"
    assert int(match.group(1)) == CLOUD_TOTAL


def test_oss_row_counts_are_the_real_ones():
    text = DOC.read_text(encoding="utf-8")
    match = re.search(
        r"(\d+) OSS rows total across the four vendors \((\d+) NeMo, "
        r"(\d+) Guardrails AI,\s*(\d+) LLM Guard, (\d+) Rebuff\)",
        text,
    )
    assert match, "COMPLIANCE.md no longer states the per-vendor OSS counts"
    total, nemo, gai, llm_guard, rebuff = (int(g) for g in match.groups())
    assert total == OSS_TOTAL
    assert nemo == len(NEMO_MAPPINGS)
    assert gai == len(GUARDRAILS_AI_MAPPINGS)
    assert llm_guard == len(LLM_GUARD_MAPPINGS)
    assert rebuff == len(REBUFF_MAPPINGS)


def test_combined_count_is_the_real_one_in_both_documents():
    """COMPLIANCE.md and adapters.md quote the same combined total."""
    combined = CLOUD_TOTAL + OSS_TOTAL

    compliance = re.search(
        r"covers (\d+) provider categories", DOC.read_text(encoding="utf-8")
    )
    assert compliance, "COMPLIANCE.md no longer states a combined count"
    assert int(compliance.group(1)) == combined

    adapters = re.search(
        r"normalises (\d+) provider categories",
        ADAPTERS_DOC.read_text(encoding="utf-8"),
    )
    assert adapters, "docs/adapters.md no longer states a combined count"
    assert int(adapters.group(1)) == combined
