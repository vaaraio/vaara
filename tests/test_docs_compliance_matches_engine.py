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

from vaara.compliance.engine import ComplianceEngine

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
