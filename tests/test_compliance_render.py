"""Tests for Markdown / narrative / JSON renderers of ConformityReport."""

from __future__ import annotations

import json

from vaara.audit.trail import AuditTrail
from vaara.compliance.engine import create_default_engine
from vaara.compliance.render import (
    render_json,
    render_markdown,
    render_narrative,
)
from vaara.taxonomy.actions import (
    ActionCategory,
    ActionRequest,
    ActionType,
    BlastRadius,
    RegulatoryDomain,
    Reversibility,
    UrgencyClass,
)


def _populated_trail() -> AuditTrail:
    """Build a small trail with one full action lifecycle."""
    trail = AuditTrail()
    action_type = ActionType(
        name="data.read",
        category=ActionCategory.DATA,
        reversibility=Reversibility.FULLY,
        blast_radius=BlastRadius.LOCAL,
        urgency=UrgencyClass.DEFERRABLE,
        regulatory_domains=frozenset({RegulatoryDomain.EU_AI_ACT}),
    )
    req = ActionRequest(
        agent_id="a-1",
        tool_name="data.read",
        action_type=action_type,
        parameters={},
        confidence=0.9,
    )
    action_id = trail.record_action_requested(req)
    trail.record_risk_scored(
        action_id=action_id,
        agent_id="a-1",
        tool_name="data.read",
        assessment={"point_estimate": 0.2, "decision": "allow"},
        regulatory_domains=frozenset({RegulatoryDomain.EU_AI_ACT}),
    )
    trail.record_decision(
        action_id=action_id,
        agent_id="a-1",
        tool_name="data.read",
        decision="allow",
        reason="risk below threshold",
        risk_score=0.2,
        regulatory_domains=frozenset({RegulatoryDomain.EU_AI_ACT}),
    )
    return trail


def test_render_markdown_produces_structured_output():
    trail = _populated_trail()
    engine = create_default_engine()
    report = engine.assess(trail, system_name="TestSys", system_version="1.0")

    md = render_markdown(report)
    assert "# Article-level evidence report" in md
    assert "## System" in md
    assert "TestSys" in md
    assert "## Audit trail integrity" in md
    assert "intact" in md
    # Per-domain section header.
    assert "EU_AI_ACT" in md.upper() or "EU AI ACT" in md.upper()
    # Tables are present.
    assert "| Article | Title |" in md
    # Per-article sections.
    assert "Article 9(1)" in md
    # Trailing disclaimer.
    assert "deployer owns the conformity decision" in md


def test_render_json_is_strict_json():
    trail = _populated_trail()
    report = create_default_engine().assess(trail)
    text = render_json(report)
    parsed = json.loads(text)
    assert parsed["overall_status"]
    assert isinstance(parsed["articles"], list)


def test_render_narrative_matches_property():
    trail = _populated_trail()
    report = create_default_engine().assess(trail)
    assert render_narrative(report) == report.narrative


def test_render_markdown_flags_broken_chain():
    trail = _populated_trail()
    # Tamper a record to break the chain.
    trail._records[1].record_hash = "0" * 64
    report = create_default_engine().assess(trail)
    md = render_markdown(report)
    assert "**BROKEN**" in md
    assert "chain is broken" in md.lower()


def test_render_markdown_includes_summary_section():
    trail = _populated_trail()
    report = create_default_engine().assess(trail)
    md = render_markdown(report)
    assert "## Summary" in md
