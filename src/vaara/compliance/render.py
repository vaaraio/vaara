# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Auditor-facing renderers for ConformityReport.

Each rendering produces a deployer-shippable artefact:

- ``render_markdown`` — Markdown with per-domain sections, article tables,
  evidence status badges, and gap/recommendation lists. The canonical
  human-shipped format; reviewable in a PR, diffable in CI, attachable to
  a regulator submission as ``.md``.
- ``render_narrative`` — Plain-text narrative (existing
  ``ConformityReport.narrative`` property, re-exposed here for symmetry).
- ``render_json`` — Strict-JSON dict (existing ``ConformityReport.to_dict``,
  also re-exposed).
- ``render_pdf`` — Styled, single-file PDF suitable for attaching to a
  Notified-Body submission or internal-audit binder. Requires the ``pdf``
  extra (``pip install 'vaara[pdf]'``). The content mirrors the Markdown
  rendering; the PDF form is what auditors actually consume.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Union

from vaara.compliance.engine import (
    ArticleEvidence,
    ConformityReport,
    EvidenceStatus,
)


_STATUS_BADGE = {
    EvidenceStatus.EVIDENCE_SUFFICIENT: "[OK] sufficient",
    EvidenceStatus.EVIDENCE_PARTIAL: "[!!] partial",
    EvidenceStatus.EVIDENCE_INSUFFICIENT: "[XX] insufficient",
    EvidenceStatus.NOT_APPLICABLE: "[--] not applicable",
}


def render_narrative(report: ConformityReport) -> str:
    """Plain-text narrative — wraps the existing report.narrative property."""
    return report.narrative


def render_json(report: ConformityReport, *, indent: int = 2) -> str:
    """Strict-JSON serialization."""
    return json.dumps(report.to_dict(), indent=indent, sort_keys=False)


def render_markdown(report: ConformityReport) -> str:
    """Render the report as Markdown.

    Layout:
        # Article-level evidence report
        ## System / generation metadata
        ## Audit trail integrity
        ## Summary
        ## Critical gaps (if any)
        ## Per-domain article tables
        ## Detailed per-article sections
    """
    lines: list[str] = []
    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(report.generated_at))

    lines.append("# Article-level evidence report")
    lines.append("")
    lines.append(
        "> This is an evidence artefact assembled from the Vaara runtime "
        "audit trail. It is **not** a conformity determination. The "
        "deployer (and where applicable a Notified Body) owns the "
        "conformity verdict under the EU AI Act and other applicable law."
    )
    lines.append("")

    lines.append("## System")
    lines.append("")
    lines.append(f"- **Name:** {report.system_name}")
    lines.append(f"- **Version:** {report.system_version}")
    lines.append(f"- **Generated:** {ts}")
    lines.append(f"- **Overall evidence status:** `{report.overall_status.value}`")
    lines.append("")

    lines.append("## Audit trail integrity")
    lines.append("")
    chain_state = "intact" if report.trail_chain_intact else "**BROKEN**"
    lines.append(f"- **Trail size:** {report.trail_size} records")
    lines.append(f"- **Hash chain:** {chain_state}")
    if not report.trail_chain_intact:
        lines.append("")
        lines.append(
            "> The hash chain is broken. Every article below is reported as "
            "insufficient until the chain is reconstructed or re-verified."
        )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(report.summary or "_no summary provided_")
    lines.append("")

    if report.critical_gaps:
        lines.append("## Critical gaps")
        lines.append("")
        for gap in report.critical_gaps:
            lines.append(f"- {gap}")
        lines.append("")

    # Group by domain
    by_domain: dict[str, list[ArticleEvidence]] = {}
    for a in report.articles:
        by_domain.setdefault(a.requirement.domain.value, []).append(a)

    for domain in sorted(by_domain):
        articles = by_domain[domain]
        lines.append(f"## {domain.upper()} — article evidence")
        lines.append("")
        lines.append("| Article | Title | Status | Strength | Records |")
        lines.append("|---|---|---|---|---|")
        for art in articles:
            status_str = _STATUS_BADGE.get(art.status, art.status.value)
            lines.append(
                f"| `{art.requirement.article}` | "
                f"{art.requirement.title} | "
                f"{status_str} | "
                f"{art.strength.value} | "
                f"{art.evidence_count} |"
            )
        lines.append("")

        for art in articles:
            lines.append(
                f"### {domain.upper()} {art.requirement.article} — "
                f"{art.requirement.title}"
            )
            lines.append("")
            lines.append(
                f"- **Status:** {_STATUS_BADGE.get(art.status, art.status.value)}"
            )
            lines.append(f"- **Strength:** `{art.strength.value}`")
            lines.append(f"- **Evidence records:** {art.evidence_count}")
            if art.evidence_count > 0:
                freshest = art.freshest_evidence_age_hours
                oldest = art.oldest_evidence_age_hours
                if freshest is not None and freshest != float("inf"):
                    lines.append(
                        f"- **Freshest evidence age:** {freshest:.1f} hours"
                    )
                if oldest is not None and oldest != float("inf"):
                    lines.append(
                        f"- **Oldest evidence age:** {oldest:.1f} hours"
                    )
            if art.requirement.description:
                lines.append("")
                lines.append(f"> {art.requirement.description}")
            if art.gaps:
                lines.append("")
                lines.append("**Gaps:**")
                for gap in art.gaps:
                    lines.append(f"- {gap}")
            if art.recommendations:
                lines.append("")
                lines.append("**Recommendations:**")
                for rec in art.recommendations:
                    lines.append(f"- {rec}")
            if art.sample_record_ids:
                lines.append("")
                lines.append("**Sample audit record IDs:**")
                for rid in art.sample_record_ids[:5]:
                    lines.append(f"- `{rid}`")
            _md_append_verdict_inputs(lines, art)
            _md_append_contributing_events(lines, art)
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_Generated by Vaara. Article-level evidence is collected from the "
        "runtime audit trail; deployer owns the conformity decision._"
    )
    return "\n".join(lines)


def _md_append_verdict_inputs(lines: list[str], art: ArticleEvidence) -> None:
    """Append the per-article 'Verdict inputs' subsection to the markdown body.

    Surfaces threshold-vs-observed values and the verdict_reasons that the
    engine produced. Auditors reading the markdown can trace the status
    label back to the specific parameter that pushed it there.
    """
    vi = art.verdict_inputs or {}
    if not vi:
        return
    lines.append("")
    lines.append("**Verdict inputs:**")
    lines.append("")
    lines.append("| Parameter | Threshold | Observed |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Evidence record count | >= {vi.get('min_evidence_count', '?')} | "
        f"{vi.get('evidence_count_observed', '?')} |"
    )
    fresh = vi.get("freshest_evidence_age_hours")
    fresh_str = f"{fresh:.1f}h" if isinstance(fresh, (int, float)) else "n/a"
    lines.append(
        f"| Freshest evidence age | <= {vi.get('staleness_hours', '?')}h | "
        f"{fresh_str} |"
    )
    st = vi.get("strength_thresholds", {})
    lines.append(
        f"| Strong-strength count | >= {st.get('strong_min_count', '?')} | "
        f"{vi.get('evidence_count_observed', '?')} |"
    )
    lines.append(
        f"| Strong-strength freshness | < {st.get('strong_max_age_hours', '?')}h | "
        f"{fresh_str} |"
    )
    lines.append(
        f"| Future-timestamp records | 0 | "
        f"{vi.get('future_timestamp_count', 0)} |"
    )
    lines.append(
        f"| Chain integrity | intact | "
        f"{'intact' if vi.get('chain_intact', True) else 'BROKEN'} |"
    )
    reasons = vi.get("verdict_reasons") or []
    if reasons:
        lines.append("")
        lines.append("**Verdict rationale:**")
        for r in reasons:
            lines.append(f"- {r}")


def _md_append_contributing_events(lines: list[str], art: ArticleEvidence) -> None:
    """Append the per-article 'Contributing events' subsection to the markdown body."""
    events = art.contributing_events or []
    if not events:
        return
    lines.append("")
    lines.append("**Contributing events** (most recent first):")
    lines.append("")
    lines.append("| When | Event | Agent / tool | Drill-down |")
    lines.append("|---|---|---|---|")
    for ev in events:
        ts = ev.get("timestamp_iso") or "n/a"
        age = ev.get("age_hours")
        age_str = f" ({age:.2f}h)" if isinstance(age, (int, float)) else ""
        agent = ev.get("agent_id", "?")
        tool = ev.get("tool_name", "?")
        drill = ev.get("drill_down") or {}
        if drill:
            drill_bits = ", ".join(
                f"`{k}`=`{v}`" for k, v in drill.items()
            )
        else:
            drill_bits = "—"
        lines.append(
            f"| {ts}{age_str} | `{ev.get('event_type', '?')}` | "
            f"{agent} / {tool} | {drill_bits} |"
        )
    lines.append("")
    lines.append("Record IDs:")
    for ev in events:
        lines.append(
            f"- `{ev.get('record_id', '?')}` (action `{ev.get('action_id', '?')}`)"
        )


_PDF_STATUS_LABEL = {
    EvidenceStatus.EVIDENCE_SUFFICIENT: "sufficient",
    EvidenceStatus.EVIDENCE_PARTIAL: "partial",
    EvidenceStatus.EVIDENCE_INSUFFICIENT: "insufficient",
    EvidenceStatus.NOT_APPLICABLE: "not applicable",
}


def render_pdf(report: ConformityReport, path: Union[str, Path]) -> int:
    """Render the report as a single-file PDF.

    Returns the number of bytes written. Requires the ``pdf`` extra
    (``pip install 'vaara[pdf]'``); raises ``ImportError`` with a
    pointer to the extra if reportlab is missing.

    Layout mirrors the Markdown rendering: cover block, integrity,
    summary, critical gaps, per-domain article tables, per-article
    detail sections. One file per assessment, auditor-friendly and
    Notified-Body-attachable.
    """
    flow, doc = _pdf_build(report, path)
    doc.build(flow)
    return Path(path).expanduser().stat().st_size


def _pdf_build(report: ConformityReport, path: Union[str, Path]):
    """Construct the flowable list and SimpleDocTemplate for ``render_pdf``.

    Returns ``(flow, doc)``. Kept separate so the public entrypoint stays
    short enough to read at a glance and the heavy reportlab import is
    confined to one place.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise ImportError(
            "PDF export requires the 'pdf' extra: pip install 'vaara[pdf]'"
        ) from exc

    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body, h1, h2, h3 = (
        styles["BodyText"], styles["Heading1"],
        styles["Heading2"], styles["Heading3"],
    )
    small = ParagraphStyle(
        "small", parent=body, fontSize=8, leading=10, textColor=colors.grey,
    )
    disclaimer = ParagraphStyle(
        "disclaimer", parent=body, fontSize=9, leading=12, leftIndent=12,
        textColor=colors.HexColor("#444444"),
    )
    flow: list = []
    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(report.generated_at))
    flow.append(Paragraph("Article-level evidence report", h1))
    flow.append(Spacer(1, 0.3 * cm))
    flow.append(Paragraph(
        "This is an evidence artefact assembled from the Vaara runtime "
        "audit trail. It is <b>not</b> a conformity determination. The "
        "deployer (and where applicable a Notified Body) owns the "
        "conformity verdict under the EU AI Act and other applicable law.",
        disclaimer,
    ))
    flow.append(Spacer(1, 0.4 * cm))
    flow.append(Paragraph("System", h2))
    meta_table = Table(
        [
            ["Name", _pdf_escape(report.system_name)],
            ["Version", _pdf_escape(report.system_version)],
            ["Generated", ts],
            ["Overall evidence status", report.overall_status.value],
        ],
        colWidths=[5 * cm, 11 * cm],
    )
    meta_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.4, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.lightgrey),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    flow.append(meta_table)
    flow.append(Spacer(1, 0.4 * cm))
    flow.append(Paragraph("Audit trail integrity", h2))
    chain_state = "intact" if report.trail_chain_intact else "<b>BROKEN</b>"
    flow.append(Paragraph(
        f"Trail size: {report.trail_size} records. Hash chain: {chain_state}.",
        body,
    ))
    if not report.trail_chain_intact:
        flow.append(Spacer(1, 0.2 * cm))
        flow.append(Paragraph(
            "The hash chain is broken. Every article below is reported as "
            "insufficient until the chain is reconstructed or re-verified.",
            disclaimer,
        ))
    flow.append(Spacer(1, 0.4 * cm))
    flow.append(Paragraph("Summary", h2))
    flow.append(Paragraph(_pdf_escape(report.summary or ""), body))
    flow.append(Spacer(1, 0.4 * cm))
    if report.critical_gaps:
        flow.append(Paragraph("Critical gaps", h2))
        for gap in report.critical_gaps:
            flow.append(Paragraph("• " + _pdf_escape(gap), body))
        flow.append(Spacer(1, 0.4 * cm))
    _pdf_append_articles(flow, report, PageBreak, Paragraph, Spacer, Table,
                         TableStyle, colors, cm, body, h2, h3, small,
                         disclaimer)
    flow.append(Spacer(1, 0.5 * cm))
    flow.append(Paragraph(
        "Generated by Vaara. Article-level evidence is collected from the "
        "runtime audit trail; deployer owns the conformity decision.",
        small,
    ))
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        title=f"Vaara evidence report: {report.system_name}",
        author="Vaara",
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    return flow, doc


def _pdf_append_articles(
    flow, report, PageBreak, Paragraph, Spacer, Table, TableStyle, colors,
    cm, body, h2, h3, small, disclaimer,
) -> None:
    """Append per-domain article tables and per-article detail sections."""
    by_domain: dict[str, list[ArticleEvidence]] = {}
    for a in report.articles:
        by_domain.setdefault(a.requirement.domain.value, []).append(a)
    for domain in sorted(by_domain):
        articles = by_domain[domain]
        flow.append(PageBreak())
        flow.append(Paragraph(
            f"{domain.upper()} — article evidence", h2,
        ))
        flow.append(Spacer(1, 0.2 * cm))
        table_rows = [["Article", "Title", "Status", "Strength", "Records"]]
        for art in articles:
            table_rows.append([
                art.requirement.article,
                _pdf_escape(art.requirement.title),
                _PDF_STATUS_LABEL.get(art.status, art.status.value),
                art.strength.value,
                str(art.evidence_count),
            ])
        article_table = Table(
            table_rows,
            colWidths=[2.8 * cm, 6.5 * cm, 2.6 * cm, 2.2 * cm, 1.8 * cm],
            repeatRows=1,
        )
        article_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOX", (0, 0), (-1, -1), 0.4, colors.grey),
            ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        flow.append(article_table)
        flow.append(Spacer(1, 0.4 * cm))
        for art in articles:
            flow.append(Paragraph(
                f"{domain.upper()} {art.requirement.article} — "
                f"{_pdf_escape(art.requirement.title)}",
                h3,
            ))
            status_label = _PDF_STATUS_LABEL.get(art.status, art.status.value)
            flow.append(Paragraph(
                f"<b>Status:</b> {status_label} &nbsp; "
                f"<b>Strength:</b> {art.strength.value} &nbsp; "
                f"<b>Records:</b> {art.evidence_count}",
                body,
            ))
            if art.evidence_count > 0:
                age_bits = []
                if (art.freshest_evidence_age_hours is not None
                        and art.freshest_evidence_age_hours != float("inf")):
                    age_bits.append(
                        f"freshest {art.freshest_evidence_age_hours:.1f}h"
                    )
                if (art.oldest_evidence_age_hours is not None
                        and art.oldest_evidence_age_hours != float("inf")):
                    age_bits.append(
                        f"oldest {art.oldest_evidence_age_hours:.1f}h"
                    )
                if age_bits:
                    flow.append(Paragraph(
                        "<b>Evidence age:</b> " + ", ".join(age_bits),
                        body,
                    ))
            if art.requirement.description:
                flow.append(Spacer(1, 0.15 * cm))
                flow.append(Paragraph(
                    _pdf_escape(art.requirement.description), disclaimer,
                ))
            if art.gaps:
                flow.append(Spacer(1, 0.1 * cm))
                flow.append(Paragraph("<b>Gaps:</b>", body))
                for gap in art.gaps:
                    flow.append(Paragraph("• " + _pdf_escape(gap), body))
            if art.recommendations:
                flow.append(Spacer(1, 0.1 * cm))
                flow.append(Paragraph("<b>Recommendations:</b>", body))
                for rec in art.recommendations:
                    flow.append(Paragraph("• " + _pdf_escape(rec), body))
            if art.sample_record_ids:
                flow.append(Spacer(1, 0.1 * cm))
                flow.append(Paragraph(
                    "<b>Sample audit record IDs:</b> "
                    + ", ".join(
                        f"<font face='Courier' size='8'>{rid}</font>"
                        for rid in art.sample_record_ids[:5]
                    ),
                    small,
                ))
            _pdf_append_verdict_inputs(
                flow, art, Paragraph, Spacer, Table, TableStyle, colors,
                cm, body, small,
            )
            _pdf_append_contributing_events(
                flow, art, Paragraph, Spacer, Table, TableStyle, colors,
                cm, body, small,
            )
            flow.append(Spacer(1, 0.3 * cm))


def _pdf_append_verdict_inputs(
    flow, art, Paragraph, Spacer, Table, TableStyle, colors, cm, body, small,
) -> None:
    """Append per-article 'Verdict inputs' table + rationale to a PDF flow."""
    vi = art.verdict_inputs or {}
    if not vi:
        return
    flow.append(Spacer(1, 0.15 * cm))
    flow.append(Paragraph("<b>Verdict inputs:</b>", body))
    fresh = vi.get("freshest_evidence_age_hours")
    fresh_str = f"{fresh:.1f}h" if isinstance(fresh, (int, float)) else "n/a"
    st = vi.get("strength_thresholds", {})
    rows = [
        ["Parameter", "Threshold", "Observed"],
        ["Evidence record count",
         f">= {vi.get('min_evidence_count', '?')}",
         str(vi.get('evidence_count_observed', '?'))],
        ["Freshest evidence age",
         f"<= {vi.get('staleness_hours', '?')}h",
         fresh_str],
        ["Strong-strength count",
         f">= {st.get('strong_min_count', '?')}",
         str(vi.get('evidence_count_observed', '?'))],
        ["Strong-strength freshness",
         f"< {st.get('strong_max_age_hours', '?')}h",
         fresh_str],
        ["Future-timestamp records", "0",
         str(vi.get('future_timestamp_count', 0))],
        ["Chain integrity", "intact",
         "intact" if vi.get('chain_intact', True) else "BROKEN"],
    ]
    table = Table(rows, colWidths=[6 * cm, 4.5 * cm, 5 * cm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    flow.append(table)
    reasons = vi.get("verdict_reasons") or []
    if reasons:
        flow.append(Spacer(1, 0.1 * cm))
        flow.append(Paragraph("<b>Verdict rationale:</b>", body))
        for r in reasons:
            flow.append(Paragraph("• " + _pdf_escape(r), body))


def _pdf_append_contributing_events(
    flow, art, Paragraph, Spacer, Table, TableStyle, colors, cm, body, small,
) -> None:
    """Append per-article 'Contributing events' table to a PDF flow."""
    events = art.contributing_events or []
    if not events:
        return
    flow.append(Spacer(1, 0.15 * cm))
    flow.append(Paragraph(
        "<b>Contributing events</b> (most recent first):", body,
    ))
    rows = [["When", "Event", "Agent / tool", "Drill-down"]]
    for ev in events:
        ts = ev.get("timestamp_iso") or "n/a"
        age = ev.get("age_hours")
        age_str = f"\n({age:.2f}h)" if isinstance(age, (int, float)) else ""
        agent = _pdf_escape(ev.get("agent_id", "?"))
        tool = _pdf_escape(ev.get("tool_name", "?"))
        drill = ev.get("drill_down") or {}
        drill_str = _pdf_escape(
            ", ".join(f"{k}={v}" for k, v in drill.items()) or "—"
        )
        rows.append([
            ts + age_str,
            ev.get("event_type", "?"),
            f"{agent} / {tool}",
            drill_str,
        ])
    table = Table(
        rows,
        colWidths=[3.6 * cm, 3.2 * cm, 4.5 * cm, 4.2 * cm],
        repeatRows=1,
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.grey),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    flow.append(table)
    flow.append(Spacer(1, 0.1 * cm))
    rid_bits = "; ".join(
        f"<font face='Courier' size='7'>{ev.get('record_id', '?')}</font>"
        f" (action <font face='Courier' size='7'>{ev.get('action_id', '?')}</font>)"
        for ev in events
    )
    flow.append(Paragraph("<b>Record IDs:</b> " + rid_bits, small))


def _pdf_escape(value: object) -> str:
    """Escape user-controlled text so reportlab Paragraph cannot be tricked
    into rendering forged markup. `system_name`, gaps, recommendations, and
    article titles flow through Paragraph which parses a tiny HTML subset;
    a `<b>` smuggled by a hostile deployer name would otherwise render bold
    text in a regulator-facing PDF.
    """
    if value is None:
        return ""
    s = str(value)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


__all__ = [
    "render_markdown", "render_narrative", "render_json", "render_pdf",
]
