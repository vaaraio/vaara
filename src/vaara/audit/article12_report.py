"""EU AI Act Article 12 record-keeping report — pure builders and renderers.

Turns an audit trail into a regulator-facing mapping against the Article 12
logging obligations. No crypto here: signing and zip assembly live in
``vaara.audit.article12_export``. These functions are pure and testable on
their own. Given the trail records and the signed-export manifest they
produce a report dict, its Markdown/HTML rendering, and the verify
instructions a regulator follows.

The report is a derived, human-readable view of the signed trail. It is
bound to that trail by the manifest ``trail_sha256`` it records, not by its
own signature: a regulator verifies the signature over trail+manifest, then
checks that the report names the same ``trail_sha256``. See
``docs/design/article12-export-spec.md``.

Schema version: ``vaara-article12/1.0``
"""

from __future__ import annotations

import html as _html
import time
from typing import Optional, Sequence

from vaara.audit.trail import AuditRecord, EventType

SCHEMA_VERSION = "vaara-article12/1.0"

# Honesty notes for the folded SEP-2828 attestations. These are the brand: a
# handoff is only corroborated when an eIDAS anchor predates the key's
# retirement, and only pinned against an out-of-band trusted DID document; an
# enforcement binding is never "attested" in this release. The eIDAS time
# anchor stays the only un-forgeable component of the package, and the report
# says so plainly rather than letting a folded sidecar read as proof.
HANDOFF_BASIS_NOTE = (
    "A handoff is verifiable when the record's signature binds to a key the "
    "archived DID document lists, inside that key's validity window, and the "
    "key was not revoked before issuance. It is corroborated only when an "
    "enclosed eIDAS time anchor predates the key's retirement and revocation. "
    "Producer identity is pinned only when checked against a DID document "
    "trusted out of band; otherwise it is self-asserted. Where a record is "
    "neither corroborated nor pinned, its authenticity rests on the signature "
    "against a self-asserted identity."
)
ENFORCEMENT_BASIS_NOTE = (
    "An enforcement binding proves a SEV-SNP attestation report carries this "
    "record's digest in REPORT_DATA. It is not attested in this release: the "
    "VCEK certificate chain to AMD's root is not validated "
    "(vcek_chain_basis=caller_supplied_unverified), and binding a report to a "
    "record does not prove the record came from the image's decision path "
    "(enforcement_logic_basis=not_established). A binding pins a launch "
    "measurement only when a vetted value was supplied. The eIDAS time anchor "
    "remains the only un-forgeable component of this package."
)


def _build_attestations_block(attestations: Optional[dict]) -> dict:
    """Shape the folded-attestation summary for the report dict.

    ``attestations`` is the pre-computed roll-up the export layer hands in
    (counts and basis flags from ``check_handoff_set`` /
    ``check_enforcement_set``); this module stays crypto-free. Returns a block
    that is always present so the rendered report can state plainly when no
    attestations were folded.
    """
    attestations = attestations or {}
    handoff = attestations.get("handoff")
    enforcement = attestations.get("enforcement")
    articles: list[str] = []
    if handoff:
        articles.append("26(6)")
    block: dict = {
        "folded": bool(handoff or enforcement),
        "articles": articles,
    }
    if handoff:
        block["handoff"] = {
            "present": True,
            "packages": handoff.get("total", 0),
            "verifiable": handoff.get("verifiable", 0),
            "corroborated": handoff.get("corroborated", 0),
            "producer_pinned": handoff.get("pinned", 0),
            "passed": handoff.get("passed", 0),
            "strict": bool(handoff.get("strict", False)),
            "pinning_gap": bool(handoff.get("pinning_gap", False)),
            "basis": HANDOFF_BASIS_NOTE,
        }
    else:
        block["handoff"] = {"present": False}
    if enforcement:
        block["enforcement"] = {
            "present": True,
            "bindings": enforcement.get("total", 0),
            "bound": enforcement.get("bound", 0),
            "measurement_pinned": enforcement.get("measurement_pinned", 0),
            "tier_counts": dict(enforcement.get("tier_counts", {})),
            "passed": enforcement.get("passed", 0),
            "strict": bool(enforcement.get("strict", False)),
            "pinning_gap": bool(enforcement.get("pinning_gap", False)),
            "basis": ENFORCEMENT_BASIS_NOTE,
        }
    else:
        block["enforcement"] = {"present": False}
    return block

# Static Article 12 / 26(5) obligation checklist. Each entry names a duty and
# the event types whose presence in the trail evidences it. The mapping is a
# structural aid, not a legal opinion: it shows which logs speak to which
# obligation. The semantic-correctness judgement stays human-owned (see the
# trust model). ``evidenced_by`` holds EventType values.
ARTICLE12_OBLIGATIONS: tuple[dict, ...] = (
    {
        "id": "art12_1",
        "article": "Article 12(1)",
        "title": "Automatic recording of events over the system lifetime",
        "evidenced_by": tuple(et.value for et in EventType),
    },
    {
        "id": "art12_2_a",
        "article": "Article 12(2)(a)",
        "title": (
            "Identifying situations that may present a risk (Article 79(1)) "
            "or a substantial modification"
        ),
        "evidenced_by": (
            EventType.ACTION_BLOCKED.value,
            EventType.POLICY_OVERRIDE.value,
            EventType.OUTCOME_RECORDED.value,
        ),
    },
    {
        "id": "art12_2_b",
        "article": "Article 12(2)(b)",
        "title": "Facilitating post-market monitoring (Article 72)",
        "evidenced_by": (
            EventType.OUTCOME_RECORDED.value,
            EventType.ACTION_EXECUTED.value,
        ),
    },
    {
        "id": "art12_2_c",
        "article": "Article 12(2)(c) / 26(5)",
        "title": "Monitoring of operation by the deployer",
        "evidenced_by": (
            EventType.DECISION_MADE.value,
            EventType.ACTION_EXECUTED.value,
            EventType.ESCALATION_SENT.value,
            EventType.ESCALATION_RESOLVED.value,
        ),
    },
    {
        "id": "art12_3_a",
        "article": "Article 12(3)(a)",
        "title": "Period of each use (start and end of operation)",
        "evidenced_by": (
            EventType.ACTION_REQUESTED.value,
            EventType.ACTION_EXECUTED.value,
        ),
    },
    {
        "id": "art19_logs",
        "article": "Article 19(1)",
        "title": (
            "Keeping the automatically generated logs under the deployer's "
            "control for the appropriate retention period"
        ),
        "evidenced_by": tuple(et.value for et in EventType),
    },
)


def _iso(epoch: Optional[float]) -> str:
    if epoch is None:
        return ""
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(epoch)))
    except (ValueError, OverflowError, OSError):
        return str(epoch)


def _in_period(ts: float, period: Optional[tuple]) -> bool:
    if period is None:
        return True
    start, end = period
    if start is not None and ts < start:
        return False
    if end is not None and ts > end:
        return False
    return True


def build_article12_report(
    records: Sequence[AuditRecord],
    manifest: dict,
    *,
    system_meta: Optional[dict] = None,
    period: Optional[tuple] = None,
    time_anchor: Optional[dict] = None,
    attestations: Optional[dict] = None,
) -> dict:
    """Build the Article 12 report dict from a trail and its export manifest.

    ``records`` is the full trail (whole-trail evidence). ``manifest`` is the
    dict written by ``export_signed`` / ``export_signed_threshold``.
    ``period`` is an optional ``(start_epoch, end_epoch)`` lens that narrows
    which records the *summary* counts. It never narrows the signed trail,
    which stays whole. ``system_meta`` carries operator-supplied identity the
    trail does not hold; absent fields render as "not provided".
    ``time_anchor`` is an optional :class:`~vaara.audit.timeanchor.TimeAnchor`
    serialised with ``to_dict()``: an external trusted timestamp over the
    signed trail head, evidencing Article 19 existence-in-time independently of
    the signing key. Its chain-head binding is checked by the caller.
    ``attestations`` is an optional pre-computed roll-up of folded SEP-2828
    evidence (cross-org handoff packages, confidential-VM enforcement
    bindings); the export layer verifies and counts them, this builder only
    shapes and renders the summary, with the honesty notes verbatim.
    """
    system_meta = dict(system_meta or {})
    in_scope = [r for r in records if _in_period(r.timestamp, period)]

    # Event histogram over the in-scope records, with per-type endpoints and
    # a few example record ids.
    histogram: dict[str, dict] = {}
    for r in in_scope:
        et = r.event_type.value
        h = histogram.setdefault(
            et, {"count": 0, "first": None, "last": None, "examples": []}
        )
        h["count"] += 1
        if h["first"] is None or r.timestamp < h["first"]:
            h["first"] = r.timestamp
        if h["last"] is None or r.timestamp > h["last"]:
            h["last"] = r.timestamp
        if len(h["examples"]) < 3:
            h["examples"].append(r.record_id)

    # Obligation mapping: for each duty, which evidencing event types are
    # present, the total count, and a few example record ids.
    obligations = []
    for ob in ARTICLE12_OBLIGATIONS:
        present = [et for et in ob["evidenced_by"] if et in histogram]
        count = sum(histogram[et]["count"] for et in present)
        examples: list[str] = []
        for et in present:
            for rid in histogram[et]["examples"]:
                if len(examples) < 3:
                    examples.append(rid)
        obligations.append({
            "id": ob["id"],
            "article": ob["article"],
            "title": ob["title"],
            "event_types_present": present,
            "record_count": count,
            "example_record_ids": examples,
            "status": "evidenced" if present else "no_matching_events",
        })

    # Records carrying explicit regulatory tags written into the chain at
    # record time, counted by referenced article.
    tagged = 0
    tagged_articles: dict[str, int] = {}
    for r in in_scope:
        if r.regulatory_articles:
            tagged += 1
            for a in r.regulatory_articles:
                key = f"{a.get('domain', '')} {a.get('article', '')}".strip()
                tagged_articles[key] = tagged_articles.get(key, 0) + 1

    # Integrity-relevant markers surfaced inline (custodian key lifecycle and
    # any fail-open anchor gaps).
    key_lifecycle = [
        {
            "timestamp": _iso(r.timestamp),
            "action": (r.data or {}).get("action"),
            "fingerprint": (r.data or {}).get("fingerprint"),
            "threshold_k": (r.data or {}).get("threshold_k"),
            "signers_n": (r.data or {}).get("signers_n"),
            "reason": (r.data or {}).get("reason", ""),
        }
        for r in in_scope if r.event_type == EventType.KEY_LIFECYCLE
    ]
    anchor_gaps = sum(1 for r in in_scope if r.event_type == EventType.ANCHOR_GAP)

    timestamps = [r.timestamp for r in in_scope]
    threshold = None
    if "threshold_k" in manifest:
        threshold = {
            "threshold_k": manifest.get("threshold_k"),
            "signers_n": manifest.get("signers_n"),
            "signer_fingerprints": manifest.get("signer_fingerprints", []),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "report_format": "article12",
        "generated_utc": _iso(time.time()),
        "regulation": {"framework": "EU AI Act", "articles": ["12", "19", "26(5)"]},
        "cover": {
            "system_name": system_meta.get("system_name", "not provided"),
            "provider": system_meta.get("provider", "not provided"),
            "deployer": system_meta.get("deployer", "not provided"),
            "intended_purpose": system_meta.get("intended_purpose", "not provided"),
            "risk_classification": system_meta.get(
                "risk_classification", "not provided"),
            "export_created_utc": manifest.get("created_utc", ""),
            "vaara_version": manifest.get("vaara_version", ""),
            "signer_fingerprint": manifest.get("signer_pubkey_fingerprint", ""),
            "signature_algorithm": manifest.get("signature_algorithm", ""),
            "threshold": threshold,
            "reporting_period": (
                {"start": _iso(period[0]), "end": _iso(period[1])}
                if period else None
            ),
        },
        "record_keeping_summary": {
            "records_in_trail": len(records),
            "records_in_scope": len(in_scope),
            "period_is_report_lens_only": period is not None,
            "first_event_utc": _iso(min(timestamps)) if timestamps else "",
            "last_event_utc": _iso(max(timestamps)) if timestamps else "",
            "chain_intact_at_export": bool(
                manifest.get("chain_intact_at_export", False)),
            "trail_sha256": manifest.get("trail_sha256", ""),
        },
        "obligation_mapping": obligations,
        "event_inventory": [
            {
                "event_type": et,
                "count": h["count"],
                "first_utc": _iso(h["first"]),
                "last_utc": _iso(h["last"]),
                "example_record_ids": h["examples"],
            }
            for et, h in sorted(histogram.items())
        ],
        "regulatory_tagging": {
            "records_with_tags": tagged,
            "articles_referenced": tagged_articles,
        },
        "integrity": {
            "chain_intact_at_export": bool(
                manifest.get("chain_intact_at_export", False)),
            "trail_sha256": manifest.get("trail_sha256", ""),
            "signature_algorithm": manifest.get("signature_algorithm", ""),
            "key_lifecycle_events": key_lifecycle,
            "anchor_gap_markers": anchor_gaps,
            "trust_model": (
                "A valid signature proves the integrity and provenance of "
                "these logs. It does not prove the truth of every recorded "
                "assertion."
            ),
        },
        "time_anchor": (
            {
                "anchored": True,
                "backend": time_anchor.get("backend", ""),
                "tsa_url": time_anchor.get("tsa_url", ""),
                "hash_algorithm": time_anchor.get("hash_algorithm", ""),
                "chain_position": time_anchor.get("chain_position"),
                "chain_head_hash": time_anchor.get("chain_head_hash", ""),
                "anchored_time": time_anchor.get("anchored_time", ""),
                "attests": (
                    "An authority outside Vaara's control timestamped the "
                    "signed trail head, proving these logs existed no later "
                    "than the attested time independently of the signing key. "
                    "Pinned to an eIDAS-qualified Time-Stamp Authority this is "
                    "EU-recognised evidence of existence-in-time over the "
                    "Article 19 retention window."
                ),
            }
            if time_anchor
            else {"anchored": False}
        ),
        "attestations": _build_attestations_block(attestations),
    }


def _md_cell(value) -> str:
    """Render a table cell, neutralizing Markdown table breakers.

    Cells can carry operator metadata or trail data (key-lifecycle reasons,
    record ids). A raw ``|`` would split a column and a newline would inject a
    row, so both are escaped/flattened before the value lands in the table.
    """
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _md_table(headers: Sequence[str], rows: Sequence[Sequence]) -> str:
    out = ["| " + " | ".join(_md_cell(h) for h in headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(_md_cell(c) for c in row) + " |")
    return "\n".join(out)


def render_report_md(report: dict) -> str:
    """Render the Article 12 report dict as Markdown."""
    c = report["cover"]
    s = report["record_keeping_summary"]
    integ = report["integrity"]
    out: list[str] = []
    out.append("# EU AI Act Article 12 record-keeping report")
    out.append("")
    out.append(
        f"Generated {report['generated_utc']} by Vaara {c['vaara_version']}."
    )
    out.append("")

    out.append("## System")
    out.append("")
    out.append(_md_table(["Field", "Value"], [
        ["System name", c["system_name"]],
        ["Provider", c["provider"]],
        ["Deployer", c["deployer"]],
        ["Intended purpose", c["intended_purpose"]],
        ["Risk classification", c["risk_classification"]],
        ["Export created (UTC)", c["export_created_utc"]],
        ["Signer fingerprint", c["signer_fingerprint"]],
        ["Signature algorithm", c["signature_algorithm"]],
    ]))
    if c.get("threshold"):
        t = c["threshold"]
        out.append("")
        out.append(
            f"Signed by {t['threshold_k']} of {t['signers_n']} named custodians."
        )
    if c.get("reporting_period"):
        p = c["reporting_period"]
        out.append("")
        out.append(
            f"Reporting period: {p['start']} to {p['end']}. The period narrows "
            "the counts below. The signed trail covers the whole record set, "
            "not only the period."
        )
    out.append("")

    out.append("## Record-keeping summary")
    out.append("")
    out.append(_md_table(["Field", "Value"], [
        ["Records in signed trail", s["records_in_trail"]],
        ["Records in reporting scope", s["records_in_scope"]],
        ["First event (UTC)", s["first_event_utc"] or "none"],
        ["Last event (UTC)", s["last_event_utc"] or "none"],
        ["Hash chain intact at export", s["chain_intact_at_export"]],
        ["Trail SHA-256", s["trail_sha256"]],
    ]))
    out.append("")

    out.append("## Article 12 obligation mapping")
    out.append("")
    out.append(_md_table(
        ["Obligation", "Title", "Status", "Records", "Event types present"],
        [
            [
                ob["article"], ob["title"], ob["status"], ob["record_count"],
                ", ".join(ob["event_types_present"]) or "none",
            ]
            for ob in report["obligation_mapping"]
        ],
    ))
    out.append("")

    out.append("## Event inventory")
    out.append("")
    out.append(_md_table(
        ["Event type", "Count", "First (UTC)", "Last (UTC)"],
        [
            [e["event_type"], e["count"], e["first_utc"], e["last_utc"]]
            for e in report["event_inventory"]
        ] or [["none", 0, "", ""]],
    ))
    out.append("")

    out.append("## Integrity")
    out.append("")
    out.append(f"Hash chain intact at export: {integ['chain_intact_at_export']}.")
    out.append(f"Signature algorithm: {integ['signature_algorithm']}.")
    out.append(f"Trail SHA-256: {integ['trail_sha256']}.")
    if integ["key_lifecycle_events"]:
        out.append("")
        out.append("Custodian key-lifecycle events:")
        out.append("")
        out.append(_md_table(
            ["When (UTC)", "Action", "Fingerprint", "k", "n", "Reason"],
            [
                [
                    k["timestamp"], k["action"], k["fingerprint"],
                    k["threshold_k"], k["signers_n"], k["reason"],
                ]
                for k in integ["key_lifecycle_events"]
            ],
        ))
    if integ["anchor_gap_markers"]:
        out.append("")
        out.append(
            f"Anchor-gap markers: {integ['anchor_gap_markers']} "
            "(auto-anchor attempts that failed open and were recorded)."
        )
    out.append("")
    out.append(integ["trust_model"])
    out.append("See docs/signing-keys.md and the trust model for the boundary.")
    out.append("")

    anchor = report.get("time_anchor") or {"anchored": False}
    out.append("## External time anchor (Article 19)")
    out.append("")
    if anchor.get("anchored"):
        out.append(_md_table(["Field", "Value"], [
            ["Anchored", "yes"],
            ["Backend", anchor.get("backend", "")],
            ["Time-Stamp Authority", anchor.get("tsa_url", "")],
            ["Hash algorithm", anchor.get("hash_algorithm", "")],
            ["Anchored chain position", anchor.get("chain_position", "")],
            ["Anchored chain head", anchor.get("chain_head_hash", "")],
            ["Attested time (UTC)", anchor.get("anchored_time", "")],
        ]))
        out.append("")
        out.append(anchor.get("attests", ""))
    else:
        out.append(
            "No external time anchor is included in this package. The signature "
            "proves integrity and provenance; on its own it does not prove when "
            "the logs existed. Add an RFC 3161 (eIDAS-qualified) time anchor "
            "with: vaara trail export-article12 --anchor-tsa <url>."
        )
    out.append("")

    _render_attestations_md(out, report.get("attestations") or {"folded": False})

    out.append("## How to verify")
    out.append("")
    out.append(verify_instructions_text(report))
    out.append("")
    return "\n".join(out)


def _render_attestations_md(out: list, att: dict) -> None:
    """Render the folded SEP-2828 attestations section, if any.

    Always emits the heading so the package is self-describing: when nothing
    was folded it states that plainly rather than leaving a regulator to wonder
    whether handoff or enforcement evidence was meant to be present.
    """
    out.append("## Cross-org handoff and enforcement evidence")
    out.append("")
    if not att.get("folded"):
        out.append(
            "No cross-org handoff or confidential-VM enforcement evidence is "
            "folded into this package. The Article 12 record-keeping above "
            "stands on its own. Attach handoff packages or enforcement bindings "
            "with: vaara trail export-article12 --handoffs <dir> "
            "--enforcements <dir>."
        )
        out.append("")
        return

    handoff = att.get("handoff") or {"present": False}
    if handoff.get("present"):
        strict = " [strict]" if handoff.get("strict") else ""
        out.append(f"### Cross-org handoff custody (Article 26(6)){strict}")
        out.append("")
        out.append(_md_table(["Field", "Value"], [
            ["Packages folded", handoff.get("packages", 0)],
            ["Verify under rotated-out key", handoff.get("passed", 0)],
            ["Verifiable", handoff.get("verifiable", 0)],
            ["Anchor-corroborated", handoff.get("corroborated", 0)],
            ["Producer pinned (out of band)", handoff.get("producer_pinned", 0)],
        ]))
        out.append("")
        if handoff.get("pinning_gap"):
            out.append(
                "Coverage note: no package pinned its producer identity against "
                "a trusted DID document, so every record's authenticity is "
                "self-asserted (advisory, does not gate)."
            )
            out.append("")
        out.append(handoff["basis"])
        out.append("")

    enforcement = att.get("enforcement") or {"present": False}
    if enforcement.get("present"):
        strict = " [strict]" if enforcement.get("strict") else ""
        out.append(f"### Where enforcement ran (confidential VM){strict}")
        out.append("")
        out.append(_md_table(["Field", "Value"], [
            ["Bindings folded", enforcement.get("bindings", 0)],
            ["Bound to a CVM report", enforcement.get("bound", 0)],
            ["Launch measurement pinned", enforcement.get("measurement_pinned", 0)],
        ]))
        tiers = enforcement.get("tier_counts") or {}
        tally = ", ".join(f"{t}: {n}" for t, n in tiers.items() if n)
        if tally:
            out.append("")
            out.append(f"Tiers: {tally}.")
        out.append("")
        if enforcement.get("pinning_gap"):
            out.append(
                "Coverage note: no binding pinned a launch measurement, so the "
                "set bound to a confidential VM but never to a vetted image "
                "(advisory, does not gate)."
            )
            out.append("")
        out.append(enforcement["basis"])
        out.append("")


def render_report_html(report: dict) -> str:
    """Render the report as a minimal standalone HTML document.

    Built from the Markdown body so the two views never drift. The Markdown
    is escaped and wrapped in a ``<pre>`` block: regulators get a portable,
    self-contained file without pulling in a Markdown renderer.
    """
    md = render_report_md(report)
    title = "EU AI Act Article 12 record-keeping report"
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        f"<title>{_html.escape(title)}</title>\n"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;}"
        "pre{white-space:pre-wrap;}</style>\n"
        "</head>\n<body>\n<pre>\n"
        f"{_html.escape(md)}\n"
        "</pre>\n</body>\n</html>\n"
    )


def verify_instructions_text(report: dict) -> str:
    """Plain-text steps a regulator follows to check the package."""
    s = report["record_keeping_summary"]
    threshold = report["cover"].get("threshold")
    anchor = report.get("time_anchor") or {"anchored": False}
    att = report.get("attestations") or {"folded": False}
    lines = [
        "How to verify this Article 12 package",
        "",
        "1. Check the signature over the signed trail and manifest:",
        "     python scripts/verify_vaara_trail.py <this_package>.zip",
        "   or, with Vaara installed:",
        "     vaara trail verify --zip <this_package>.zip",
        "",
        "2. Confirm this report describes the signed trail. The trail_sha256",
        "   recorded in this report and in article12_summary.json must equal",
        "   the trail_sha256 in manifest.json:",
        f"     {s['trail_sha256']}",
    ]
    step = 3
    if threshold:
        lines += [
            "",
            f"{step}. This package is threshold-signed: at least "
            f"{threshold['threshold_k']}",
            f"   of {threshold['signers_n']} named custodian signatures must "
            "verify.",
        ]
        step += 1
    if anchor.get("anchored"):
        lines += [
            "",
            f"{step}. Verify the external time anchor in time_anchor.json. It "
            "proves the signed",
            "   trail head existed no later than the attested time, "
            "independently of the",
            "   signing key. With Vaara installed:",
            "     vaara trail verify-anchor --zip <this_package>.zip",
            "   The anchor's chain_head_hash must equal the last record_hash "
            "in trail.jsonl,",
            "   and the RFC 3161 token must verify under a Time-Stamp Authority "
            "you trust",
            "   (pin an eIDAS-qualified TSA certificate to enforce that). "
            "Anchored head:",
            f"     {anchor.get('chain_head_hash', '')}",
        ]
        step += 1
    if att.get("folded"):
        handoff = att.get("handoff") or {"present": False}
        enforcement = att.get("enforcement") or {"present": False}
        lines += [
            "",
            f"{step}. Re-verify the folded SEP-2828 evidence from the bytes in "
            "evidence/.",
            "   These sidecars are verified at export and fail the export if "
            "they do not",
            "   verify; re-check them independently:",
        ]
        if handoff.get("present"):
            lines += [
                "     vaara verify-handoffs evidence/handoff",
                "   or the Vaara-free checker in "
                "tests/vectors/cross_org_handoff_v0/.",
            ]
        if enforcement.get("present"):
            lines += [
                "     vaara verify-enforcements evidence/enforcement",
                "   or the Vaara-free checker in "
                "tests/vectors/enforcement_attestation_v0/.",
            ]
        lines += [
            "   The roll-up in evidence/attestations_summary.json must match "
            "what you",
            "   reproduce. A handoff is not corroborated without a verified "
            "anchor, and an",
            "   enforcement binding is not attested: the VCEK chain to AMD's "
            "root is not",
            "   validated in this release. The eIDAS time anchor stays the only "
            "un-forgeable",
            "   component of this package.",
        ]
        step += 1
    lines += [
        "",
        "The signature covers trail.jsonl and manifest.json. This report and",
        "article12_summary.json are a derived, human-readable view bound to the",
        "signed trail by the trail_sha256 above. They are not themselves signed.",
        "The folded evidence/ sidecars are content-addressed SEP-2828 records, "
        "verified",
        "at export; they are not covered by the trail signature.",
    ]
    return "\n".join(lines)
