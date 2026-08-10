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


VERDICTS_DOC = DOC.parent / "VERDICTS.md"

#: The row VERDICTS.md leaves blank on purpose: Article 11(1) consumes no
#: runtime events, so its thresholds are printed as `n/a`.
_EXTERNAL_EVIDENCE_ROWS = {"11(1)"}


#: VERDICTS.md heading -> the heading that ends its threshold table. Both
#: domains carry an Article 12(1) and an Article 13(1), so the tables have to
#: be read one at a time or DORA silently overwrites the EU AI Act row.
_VERDICTS_SECTIONS = {
    "EU_AI_ACT": ("## EU AI Act per-article thresholds", "## DORA per-article thresholds"),
    "DORA": ("## DORA per-article thresholds", "## What an auditor sees"),
}


def _verdicts_threshold_rows(domain: str = "EU_AI_ACT") -> dict[str, tuple[int, float]]:
    """Article -> (min count, staleness hours) as VERDICTS.md prints them."""
    whole = VERDICTS_DOC.read_text(encoding="utf-8")
    start_heading, end_heading = _VERDICTS_SECTIONS[domain]
    start = whole.index(start_heading)
    text = whole[start:whole.index(end_heading, start)]
    rows: dict[str, tuple[int, float]] = {}
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) != 7 or not re.fullmatch(r"\d+\(\d+\)(\([a-z]\))?", cells[0]):
            continue
        if cells[0] in _EXTERNAL_EVIDENCE_ROWS:
            continue
        staleness = re.match(r"(\d+)\s*h", cells[3])
        assert staleness, f"unreadable staleness cell for {cells[0]}: {cells[3]!r}"
        rows[cells[0]] = (int(cells[2]), float(staleness.group(1)))
    return rows


@pytest.mark.parametrize("domain", sorted(_VERDICTS_SECTIONS))
def test_verdicts_documents_every_requirement_the_engine_ships(domain):
    """VERDICTS.md is the public reference for how the engine decides.

    Its EU AI Act table omitted Articles 26(10), 50(1) and 73(1) while the
    engine enforced all three, two of them as CRITICAL. The document states
    that the overall report status is the worst status across all critical
    articles, so a reader working from that table could not predict the
    verdict they would get.
    """
    documented = set(_verdicts_threshold_rows(domain)) | (
        _EXTERNAL_EVIDENCE_ROWS if domain == "EU_AI_ACT" else set()
    )
    real = _engine_articles(domain)
    missing = real - documented
    assert not missing, (
        f"docs/VERDICTS.md has no {domain} threshold row for {sorted(missing)}, "
        f"which the engine enforces"
    )
    invented = documented - real
    assert not invented, (
        f"docs/VERDICTS.md prints {domain} thresholds for {sorted(invented)}, "
        f"which the engine does not enforce"
    )


@pytest.mark.parametrize("domain", sorted(_VERDICTS_SECTIONS))
def test_verdicts_thresholds_are_the_engine_thresholds(domain):
    """Every number in the tables, not just the article list."""
    documented = _verdicts_threshold_rows(domain)
    wrong = {}
    for requirement in ComplianceEngine().requirements:
        if domain not in str(requirement.domain):
            continue
        article = requirement.article.removeprefix("Article ")
        if article not in documented:
            continue
        real = (requirement.min_evidence_count, float(requirement.staleness_hours))
        if documented[article] != real:
            wrong[article] = (documented[article], real)
    assert not wrong, (
        "docs/VERDICTS.md thresholds drifted from the engine "
        "(article: documented -> real): "
        + ", ".join(f"{a}: {d} -> {r}" for a, (d, r) in sorted(wrong.items()))
    )


def test_verdicts_strong_columns_are_twice_the_minimum():
    """The strong-count column is derived, so it can drift on its own."""
    text = VERDICTS_DOC.read_text(encoding="utf-8")
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) != 7 or not re.fullmatch(r"\d+\(\d+\)(\([a-z]\))?", cells[0]):
            continue
        if cells[0] in _EXTERNAL_EVIDENCE_ROWS:
            continue
        minimum, strong = int(cells[2]), int(cells[4])
        assert strong == 2 * minimum, (
            f"VERDICTS.md {cells[0]}: strong-count {strong} is not 2x {minimum}"
        )
        staleness = float(re.match(r"(\d+)", cells[3]).group(1))
        freshness = float(re.match(r"([\d.]+)", cells[5]).group(1))
        assert freshness == staleness / 4, (
            f"VERDICTS.md {cells[0]}: strong-freshness {freshness} is not "
            f"{staleness}/4"
        )


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
