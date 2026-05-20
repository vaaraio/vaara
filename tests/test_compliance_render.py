"""Tests for Markdown / narrative / JSON / PDF renderers of ConformityReport."""

from __future__ import annotations

import json

import pytest

from vaara.audit.trail import AuditTrail
from vaara.compliance.engine import create_default_engine
from vaara.compliance.render import (
    render_json,
    render_markdown,
    render_narrative,
)

try:
    import reportlab  # noqa: F401

    from vaara.compliance.render import render_pdf

    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False
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


def test_render_markdown_surfaces_verdict_inputs_and_events():
    """v0.26 drill-down: markdown body must surface verdict thresholds and
    the contributing event rows so an auditor can trace status without
    re-running the engine."""
    trail = _populated_trail()
    report = create_default_engine().assess(trail)
    md = render_markdown(report)
    assert "**Verdict inputs:**" in md
    assert "| Parameter | Threshold | Observed |" in md
    assert "Evidence record count" in md
    assert "Strong-strength count" in md
    assert "**Verdict rationale:**" in md
    assert "**Contributing events**" in md
    # Drill-down values from the populated trail should appear.
    assert "point_estimate" in md or "decision" in md


def test_render_narrative_surfaces_verdict_reasoning():
    trail = _populated_trail()
    report = create_default_engine().assess(trail)
    narr = render_narrative(report)
    # Each runtime article narrative includes a "Why:" line and an "Event:"
    # line so a regulator reading plain text gets the rationale + receipts.
    assert "Why:" in narr
    assert "Event:" in narr


@pytest.mark.skipif(not _HAS_REPORTLAB, reason="reportlab not installed")
def test_render_pdf_includes_verdict_drill_down(tmp_path):
    """PDF flow must include verdict-inputs table + contributing-events table.
    Exact byte search is hopeless inside a compressed PDF, so this verifies
    the renderer doesn't crash on a populated trail with the new sections."""
    trail = _populated_trail()
    report = create_default_engine().assess(trail)
    out = tmp_path / "drill.pdf"
    size = render_pdf(report, out)
    assert size > 0
    assert out.read_bytes().startswith(b"%PDF-")


@pytest.mark.skipif(not _HAS_REPORTLAB, reason="reportlab not installed")
def test_render_pdf_produces_valid_pdf(tmp_path):
    trail = _populated_trail()
    report = create_default_engine().assess(
        trail, system_name="PDFSys", system_version="0.16.0",
    )
    out = tmp_path / "report.pdf"
    size = render_pdf(report, out)
    assert size > 0
    data = out.read_bytes()
    assert data.startswith(b"%PDF-")
    assert data.rstrip().endswith(b"%%EOF")
    assert size == len(data)


@pytest.mark.skipif(not _HAS_REPORTLAB, reason="reportlab not installed")
def test_render_pdf_escapes_html_metachars_in_system_name(tmp_path):
    """A hostile system_name must not inject reportlab markup."""
    trail = _populated_trail()
    report = create_default_engine().assess(
        trail,
        system_name="<script>alert('x')</script>",
        system_version="<b>1.0</b>",
    )
    out = tmp_path / "escaped.pdf"
    render_pdf(report, out)
    data = out.read_bytes()
    # PDF stream is compressed by default; just confirm valid magic and the
    # function did not raise during Paragraph parsing.
    assert data.startswith(b"%PDF-")


@pytest.mark.skipif(not _HAS_REPORTLAB, reason="reportlab not installed")
def test_render_pdf_handles_broken_chain(tmp_path):
    trail = _populated_trail()
    trail._records[1].record_hash = "0" * 64
    report = create_default_engine().assess(trail)
    out = tmp_path / "broken.pdf"
    size = render_pdf(report, out)
    assert size > 0
    assert out.read_bytes().startswith(b"%PDF-")


def test_render_pdf_raises_helpful_error_when_reportlab_missing(
    tmp_path, monkeypatch,
):
    """If reportlab is unimportable, render_pdf must point at the extra."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *a, **kw):
        if name.startswith("reportlab"):
            raise ImportError("simulated missing reportlab")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    from vaara.compliance.render import render_pdf as rp
    trail = _populated_trail()
    report = create_default_engine().assess(trail)
    with pytest.raises(ImportError, match=r"vaara\[pdf\]"):
        rp(report, tmp_path / "missing.pdf")
