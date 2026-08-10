# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""docs/COMPLIANCE.md must list every requirement the engine enforces.

COMPLIANCE.md is the article-by-article mapping a buyer or auditor
reads, and it claims outright that its table "matches the default
`ComplianceEngine` requirements". Nothing checked that, and it drifted:
the engine enforced Article 50(1) (AI system disclosure) and Article
26(10) (deployer logging) as CRITICAL requirements while neither
appeared in the table. The drift understated the product rather than
overstating it, which is the safer direction and still wrong.

These tests fail the build when a requirement is added to the engine
without a corresponding row, or when a row names an article the engine
does not actually enforce.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from vaara.compliance.engine import (
    ComplianceEngine, EvidenceStatus, EvidenceStrength,
)

DOC = Path(__file__).resolve().parent.parent / "docs" / "COMPLIANCE.md"

# Heading -> the heading that ends the section, so a table is read from
# exactly one section rather than the whole file.
_SECTIONS = {
    "EU_AI_ACT": ("## EU AI Act Article Mapping", "## EU AI Act Annex IV"),
    "DORA": ("## DORA Article Mapping", "## Cloud guardrail adapter pattern"),
}


def _table_articles(domain: str) -> set[str]:
    """Article labels in the bolded first column of one section's table."""
    text = DOC.read_text(encoding="utf-8")
    start_heading, end_heading = _SECTIONS[domain]
    start = text.index(start_heading)
    end = text.index(end_heading, start)
    section = text[start:end]
    return set(re.findall(r"^\|\s*\*\*([^*]+)\*\*\s*\|", section, re.MULTILINE))


def _engine_articles(domain: str) -> set[str]:
    articles = set()
    for requirement in ComplianceEngine().requirements:
        if domain in str(requirement.domain):
            # Engine stores "Article 50(1)"; the table column is "50(1)".
            articles.add(requirement.article.removeprefix("Article "))
    return articles


@pytest.mark.parametrize("domain", sorted(_SECTIONS))
def test_every_enforced_requirement_has_a_documented_row(domain):
    missing = _engine_articles(domain) - _table_articles(domain)
    assert not missing, (
        f"{domain} requirements enforced by ComplianceEngine but absent from "
        f"docs/COMPLIANCE.md: {sorted(missing)}. A buyer reading that table "
        f"would not know Vaara produces evidence for them."
    )


@pytest.mark.parametrize("domain", sorted(_SECTIONS))
def test_no_documented_row_claims_an_unenforced_article(domain):
    invented = _table_articles(domain) - _engine_articles(domain)
    assert not invented, (
        f"docs/COMPLIANCE.md lists {domain} articles the engine does not "
        f"enforce: {sorted(invented)}. Either add the requirement or drop "
        f"the row; an unbacked article claim is the kind a buyer checks."
    )


def test_critical_requirements_are_all_documented():
    """A critical requirement missing from the doc is the worst case."""
    critical = {
        r.article.removeprefix("Article ")
        for r in ComplianceEngine().requirements
        if r.is_critical
    }
    documented = _table_articles("EU_AI_ACT") | _table_articles("DORA")
    assert critical <= documented, sorted(critical - documented)


def test_article_50_disclosure_is_documented():
    """Regression: Article 50 is the transparency obligation in force."""
    assert "50(1)" in _table_articles("EU_AI_ACT")


def _documented_enum_values(enum_name: str) -> set[str]:
    """Values COMPLIANCE.md lists in the table under `enum_name`."""
    text = DOC.read_text()
    header = re.search(rf"\|\s*`{enum_name}`\s*\|[^\n]*\n\|[-\s|]+\n", text)
    assert header, f"no table found for {enum_name} in docs/COMPLIANCE.md"
    values = set()
    for line in text[header.end():].splitlines():
        if not line.startswith("|"):
            break
        cell = line.split("|")[1].strip()
        values.add(cell.strip("`"))
    return values


@pytest.mark.parametrize("enum_name,enum_cls", [
    ("EvidenceStatus", EvidenceStatus),
    ("EvidenceStrength", EvidenceStrength),
])
def test_documented_verdict_vocabulary_matches_the_engine(enum_name, enum_cls):
    """The words the report can print have to be the words the doc lists.

    COMPLIANCE.md described EvidenceStatus as "sufficient, insufficient,
    stale, error". Neither `stale` nor `error` has ever existed, and
    `evidence_partial` and `not_applicable`, both of which a report does
    print, were missing. EvidenceStrength omitted `absent`, which the
    README's own example output shows. An auditor reading the mapping
    document is reading it to learn what the verdicts mean, so inventing two
    and hiding three is the wrong direction to be wrong in.
    """
    documented = _documented_enum_values(enum_name)
    real = {member.value for member in enum_cls}
    assert documented == real, (
        f"docs/COMPLIANCE.md {enum_name} table drifted from the engine.\n"
        f"  documented but not real: {sorted(documented - real)}\n"
        f"  real but not documented: {sorted(real - documented)}"
    )
