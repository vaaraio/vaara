"""Auditor-facing renderers for ConformityReport.

Each rendering produces a deployer-shippable artefact:

- ``render_markdown`` — Markdown with per-domain sections, article tables,
  evidence status badges, and gap/recommendation lists. The canonical
  human-shipped format; reviewable in a PR, diffable in CI, attachable to
  a regulator submission as `.md`.
- ``render_narrative`` — Plain-text narrative (existing
  `ConformityReport.narrative` property, re-exposed here for symmetry).
- ``render_json`` — Strict-JSON dict (existing `ConformityReport.to_dict`,
  also re-exposed).

PDF export is intentionally NOT in v1: a clean Markdown render can be piped
through `pandoc` or `weasyprint` by the deployer's own pipeline. Vaara
defines the article-evidence content; format conversion is downstream.
"""

from __future__ import annotations

import json
import time

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
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_Generated by Vaara. Article-level evidence is collected from the "
        "runtime audit trail; deployer owns the conformity decision._"
    )
    return "\n".join(lines)


__all__ = ["render_markdown", "render_narrative", "render_json"]
