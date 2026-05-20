"""Tests for the HTML article-coverage dashboard renderer."""

from __future__ import annotations

import re

from vaara.audit.trail import AuditTrail
from vaara.compliance.dashboard import render_html
from vaara.compliance.engine import create_default_engine
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
        agent_id="a-1", tool_name="data.read",
        action_type=action_type, parameters={}, confidence=0.9,
    )
    action_id = trail.record_action_requested(req)
    trail.record_risk_scored(
        action_id=action_id, agent_id="a-1", tool_name="data.read",
        assessment={"point_estimate": 0.2, "decision": "allow"},
        regulatory_domains=frozenset({RegulatoryDomain.EU_AI_ACT}),
    )
    trail.record_decision(
        action_id=action_id, agent_id="a-1", tool_name="data.read",
        decision="allow", reason="risk below threshold", risk_score=0.2,
        regulatory_domains=frozenset({RegulatoryDomain.EU_AI_ACT}),
    )
    return trail


def _report():
    trail = _populated_trail()
    engine = create_default_engine()
    return engine.assess(trail, system_name="TestSys", system_version="1.0")


def test_html_is_well_formed_doctype():
    out = render_html(_report())
    assert out.startswith("<!doctype html>")
    assert out.endswith("</html>")


def test_html_includes_system_and_status_block():
    out = render_html(_report())
    assert "Article-level evidence report" in out
    assert "TestSys" in out
    assert "Audit trail integrity" in out
    assert "intact" in out
    assert 'class="pill' in out


def test_html_escapes_user_supplied_strings():
    trail = _populated_trail()
    engine = create_default_engine()
    report = engine.assess(
        trail,
        system_name="<script>alert(1)</script>",
        system_version="v\"&'<>",
    )
    out = render_html(report)
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out
    assert "&amp;" in out
    assert "&quot;" in out


def test_html_renders_per_domain_table():
    out = render_html(_report())
    assert "EU_AI_ACT — article evidence" in out
    assert "<table>" in out
    assert "Article" in out and "Status" in out and "Records" in out


def test_html_renders_status_pills_for_known_statuses():
    out = render_html(_report())
    pill_classes = set(re.findall(r'pill (ok|warn|bad|na)', out))
    assert pill_classes  # at least one pill rendered


def test_html_renders_broken_chain_notice():
    trail = _populated_trail()
    # Tamper to break the chain.
    if trail._records:
        bad = trail._records[-1]
        object.__setattr__(bad, "previous_hash", "deadbeef")

    engine = create_default_engine()
    report = engine.assess(trail, system_name="Tampered", system_version="1.0")
    out = render_html(report)
    assert "chain-broken" in out
    assert "BROKEN" in out


def test_html_has_print_friendly_stylesheet():
    out = render_html(_report())
    assert "@media print" in out


def test_html_is_self_contained_no_external_assets():
    out = render_html(_report())
    # No external link/script tags.
    assert "<link " not in out
    assert "<script" not in out
    assert "http://" not in out
    assert "https://" not in out


def test_html_critical_gaps_section_when_present(monkeypatch):
    report = _report()
    object.__setattr__(report, "critical_gaps", ["chain integrity broken"])
    out = render_html(report)
    assert "Critical gaps" in out
    assert "chain integrity broken" in out


def test_html_surfaces_verdict_inputs_and_contributing_events():
    """v0.26 drill-down: dashboard must expose the threshold-vs-actual table
    and the contributing-events list under each runtime article."""
    out = render_html(_report())
    assert "Verdict inputs" in out
    assert "Verdict rationale" in out
    assert "Contributing events" in out
    # Threshold/observed columns and a known parameter row.
    assert "Threshold" in out
    assert "Observed" in out
    assert "Evidence record count" in out
