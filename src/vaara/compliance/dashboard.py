"""Static HTML article-coverage dashboard renderer.

Produces a single self-contained HTML page from a ``ConformityReport``.
Embedded CSS keeps the output one-file portable — the auditor or
compliance officer can email it, attach it to a regulator submission, or
open it offline. No JavaScript, no external assets, no network calls.

The page lays out the same structure as ``render_markdown`` (system
metadata, audit-trail integrity, summary, critical gaps, per-domain
article tables, detailed per-article sections) with status badges
rendered as colored pills and a print-friendly stylesheet.
"""

from __future__ import annotations

import html
import time

from vaara.compliance.engine import (
    ArticleEvidence,
    ConformityReport,
    EvidenceStatus,
)


_STATUS_CLASS = {
    EvidenceStatus.EVIDENCE_SUFFICIENT: "ok",
    EvidenceStatus.EVIDENCE_PARTIAL: "warn",
    EvidenceStatus.EVIDENCE_INSUFFICIENT: "bad",
    EvidenceStatus.NOT_APPLICABLE: "na",
}

_STATUS_LABEL = {
    EvidenceStatus.EVIDENCE_SUFFICIENT: "sufficient",
    EvidenceStatus.EVIDENCE_PARTIAL: "partial",
    EvidenceStatus.EVIDENCE_INSUFFICIENT: "insufficient",
    EvidenceStatus.NOT_APPLICABLE: "not applicable",
}

_CSS = """
:root{--ok:#1f7a3a;--warn:#a86b00;--bad:#a32a2a;--na:#666;--bg:#fafafa;
--card:#fff;--border:#e0e0e0;--text:#1a1a1a;--muted:#666;}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
margin:0;padding:2rem 1rem;background:var(--bg);color:var(--text);line-height:1.5;}
.container{max-width:1100px;margin:0 auto;}
h1{margin-top:0;font-size:1.8rem}
h2{margin-top:2.5rem;font-size:1.3rem;border-bottom:1px solid var(--border);padding-bottom:0.4rem}
h3{margin-top:1.5rem;font-size:1.05rem}
.notice{background:#fff8e6;border-left:4px solid #d4a017;padding:0.75rem 1rem;margin:1rem 0;}
.card{background:var(--card);border:1px solid var(--border);border-radius:6px;padding:1rem 1.25rem;margin:0.75rem 0;}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:0.25rem 1rem;margin:0.5rem 0;}
.kv dt{font-weight:600;color:var(--muted)}
table{width:100%;border-collapse:collapse;margin:0.5rem 0;}
th,td{text-align:left;padding:0.5rem 0.6rem;border-bottom:1px solid var(--border);font-size:0.95rem;}
th{background:#f3f3f3;font-weight:600}
.pill{display:inline-block;padding:0.1rem 0.55rem;border-radius:999px;font-size:0.85rem;font-weight:600;color:#fff;}
.pill.ok{background:var(--ok)} .pill.warn{background:var(--warn)}
.pill.bad{background:var(--bad)} .pill.na{background:var(--na)}
.chain-broken{color:var(--bad);font-weight:700} .chain-intact{color:var(--ok)}
.gaps li{color:var(--bad)} .recs li{color:var(--warn)}
code{background:#f0f0f0;padding:0.05rem 0.3rem;border-radius:3px;font-size:0.9em;}
@media print{body{background:#fff;padding:0}.card{break-inside:avoid}}
""".strip()


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _pill(status: EvidenceStatus) -> str:
    cls = _STATUS_CLASS.get(status, "na")
    label = _STATUS_LABEL.get(status, status.value)
    return f'<span class="pill {cls}">{_esc(label)}</span>'


def _article_table_row(art: ArticleEvidence) -> str:
    return (
        "<tr>"
        f"<td><code>{_esc(art.requirement.article)}</code></td>"
        f"<td>{_esc(art.requirement.title)}</td>"
        f"<td>{_pill(art.status)}</td>"
        f"<td><code>{_esc(art.strength.value)}</code></td>"
        f"<td>{art.evidence_count}</td>"
        "</tr>"
    )


def _article_detail(art: ArticleEvidence, domain: str) -> str:
    out: list[str] = [
        '<div class="card">',
        f"<h3>{_esc(domain.upper())} {_esc(art.requirement.article)} — ",
        f"{_esc(art.requirement.title)}</h3>",
        '<dl class="kv">',
        f"<dt>Status</dt><dd>{_pill(art.status)}</dd>",
        f"<dt>Strength</dt><dd><code>{_esc(art.strength.value)}</code></dd>",
        f"<dt>Evidence records</dt><dd>{art.evidence_count}</dd>",
    ]
    if art.evidence_count > 0:
        fresh = art.freshest_evidence_age_hours
        old = art.oldest_evidence_age_hours
        if fresh is not None and fresh != float("inf"):
            out.append(f"<dt>Freshest evidence</dt><dd>{fresh:.1f} hours ago</dd>")
        if old is not None and old != float("inf"):
            out.append(f"<dt>Oldest evidence</dt><dd>{old:.1f} hours ago</dd>")
    out.append("</dl>")
    if art.requirement.description:
        out.append(f'<blockquote>{_esc(art.requirement.description)}</blockquote>')
    if art.gaps:
        out.append('<strong>Gaps</strong><ul class="gaps">')
        out.extend(f"<li>{_esc(g)}</li>" for g in art.gaps)
        out.append("</ul>")
    if art.recommendations:
        out.append('<strong>Recommendations</strong><ul class="recs">')
        out.extend(f"<li>{_esc(r)}</li>" for r in art.recommendations)
        out.append("</ul>")
    if art.sample_record_ids:
        out.append("<strong>Sample audit record IDs</strong><ul>")
        out.extend(f"<li><code>{_esc(rid)}</code></li>"
                   for rid in art.sample_record_ids[:5])
        out.append("</ul>")
    _html_append_verdict_inputs(out, art)
    _html_append_contributing_events(out, art)
    out.append("</div>")
    return "".join(out)


def _html_append_verdict_inputs(out: list[str], art: ArticleEvidence) -> None:
    """Render per-article 'Verdict inputs' threshold table + rationale list."""
    vi = art.verdict_inputs or {}
    if not vi:
        return
    fresh = vi.get("freshest_evidence_age_hours")
    fresh_str = f"{fresh:.1f}h" if isinstance(fresh, (int, float)) else "n/a"
    st = vi.get("strength_thresholds", {})
    chain = "intact" if vi.get("chain_intact", True) else "BROKEN"
    rows = [
        ("Evidence record count",
         f">= {vi.get('min_evidence_count', '?')}",
         vi.get("evidence_count_observed", "?")),
        ("Freshest evidence age",
         f"<= {vi.get('staleness_hours', '?')}h",
         fresh_str),
        ("Strong-strength count",
         f">= {st.get('strong_min_count', '?')}",
         vi.get("evidence_count_observed", "?")),
        ("Strong-strength freshness",
         f"< {st.get('strong_max_age_hours', '?')}h",
         fresh_str),
        ("Future-timestamp records", "0",
         vi.get("future_timestamp_count", 0)),
        ("Chain integrity", "intact", chain),
    ]
    out.append("<strong>Verdict inputs</strong>")
    out.append(
        "<table><thead><tr><th>Parameter</th><th>Threshold</th>"
        "<th>Observed</th></tr></thead><tbody>"
    )
    for name, threshold, observed in rows:
        out.append(
            f"<tr><td>{_esc(name)}</td><td>{_esc(threshold)}</td>"
            f"<td>{_esc(observed)}</td></tr>"
        )
    out.append("</tbody></table>")
    reasons = vi.get("verdict_reasons") or []
    if reasons:
        out.append("<strong>Verdict rationale</strong><ul>")
        out.extend(f"<li>{_esc(r)}</li>" for r in reasons)
        out.append("</ul>")


def _html_append_contributing_events(out: list[str], art: ArticleEvidence) -> None:
    """Render per-article 'Contributing events' table."""
    events = art.contributing_events or []
    if not events:
        return
    out.append(
        "<strong>Contributing events</strong> (most recent first)"
    )
    out.append(
        "<table><thead><tr><th>When</th><th>Event</th>"
        "<th>Agent / tool</th><th>Drill-down</th><th>Record</th></tr>"
        "</thead><tbody>"
    )
    for ev in events:
        ts = ev.get("timestamp_iso") or "n/a"
        age = ev.get("age_hours")
        age_str = f" ({age:.2f}h)" if isinstance(age, (int, float)) else ""
        agent = ev.get("agent_id", "?")
        tool = ev.get("tool_name", "?")
        drill = ev.get("drill_down") or {}
        if drill:
            drill_str = ", ".join(
                f"<code>{_esc(k)}</code>=<code>{_esc(v)}</code>"
                for k, v in drill.items()
            )
        else:
            drill_str = "—"
        out.append(
            "<tr>"
            f"<td>{_esc(ts)}{_esc(age_str)}</td>"
            f"<td><code>{_esc(ev.get('event_type', '?'))}</code></td>"
            f"<td>{_esc(agent)} / {_esc(tool)}</td>"
            f"<td>{drill_str}</td>"
            f"<td><code>{_esc(ev.get('record_id', '?'))}</code></td>"
            "</tr>"
        )
    out.append("</tbody></table>")


def render_html(report: ConformityReport) -> str:
    """Render the ConformityReport as a single self-contained HTML page."""
    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(report.generated_at))
    by_domain: dict[str, list[ArticleEvidence]] = {}
    for a in report.articles:
        by_domain.setdefault(a.requirement.domain.value, []).append(a)

    head = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Vaara compliance evidence — {_esc(report.system_name)}</title>"
        f"<style>{_CSS}</style></head><body><div class=\"container\">"
    )
    chain_cls = "chain-intact" if report.trail_chain_intact else "chain-broken"
    chain_lbl = "intact" if report.trail_chain_intact else "BROKEN"

    system_block = (
        "<h1>Article-level evidence report</h1>"
        '<div class="notice">This is an evidence artefact assembled from '
        "the Vaara runtime audit trail. It is <strong>not</strong> a "
        "conformity determination. The deployer (and where applicable a "
        "Notified Body) owns the conformity verdict under the EU AI Act "
        "and other applicable law.</div>"
        "<h2>System</h2>"
        f'<dl class="kv"><dt>Name</dt><dd>{_esc(report.system_name)}</dd>'
        f"<dt>Version</dt><dd>{_esc(report.system_version)}</dd>"
        f"<dt>Generated</dt><dd>{_esc(ts)}</dd>"
        f"<dt>Overall status</dt><dd>{_pill(report.overall_status)}</dd></dl>"
        "<h2>Audit trail integrity</h2>"
        f'<dl class="kv"><dt>Trail size</dt><dd>{report.trail_size} records</dd>'
        f'<dt>Hash chain</dt><dd class="{chain_cls}">{chain_lbl}</dd></dl>'
    )

    parts = [head, system_block]
    if not report.trail_chain_intact:
        parts.append(
            '<div class="notice">The hash chain is broken. Every article '
            "is reported as insufficient until the chain is reconstructed "
            "or re-verified.</div>"
        )
    parts.append("<h2>Summary</h2>")
    parts.append(f"<p>{_esc(report.summary or '(no summary)')}</p>")

    if report.critical_gaps:
        parts.append('<h2>Critical gaps</h2><ul class="gaps">')
        parts.extend(f"<li>{_esc(g)}</li>" for g in report.critical_gaps)
        parts.append("</ul>")

    for domain in sorted(by_domain):
        arts = by_domain[domain]
        parts.append(f"<h2>{_esc(domain.upper())} — article evidence</h2>")
        parts.append(
            "<table><thead><tr><th>Article</th><th>Title</th><th>Status</th>"
            "<th>Strength</th><th>Records</th></tr></thead><tbody>"
        )
        parts.extend(_article_table_row(a) for a in arts)
        parts.append("</tbody></table>")
        parts.extend(_article_detail(a, domain) for a in arts)

    parts.append(
        "<hr><p><em>Generated by Vaara. Article-level evidence is collected "
        "from the runtime audit trail; the deployer owns conformity "
        "determination.</em></p></div></body></html>"
    )
    return "".join(parts)
