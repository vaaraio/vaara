# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""EU AI Act Article 50 transparency evidence: record it, then prove it.

Article 50 duties are behavioral: inform the person they are interacting
with AI (50(1)), mark synthetic content (50(2)), inform about emotion
recognition / biometric categorisation (50(3)), disclose deepfakes and
public-interest AI text (50(4)), all at the latest at the first
interaction and accessibly (50(5)). When challenged, the question is
retrospective: show that the disclosure happened. A banner screenshot is
a claim about today; a hash-chained record next to the agent's actions
is evidence about every session.

This module adds no new wire format and no new event type. A disclosure
is an ordinary action recorded through the existing interception
pipeline under the reserved tool name ``vaara.article50.disclosure``,
so it lands in the same signed, hash-chained trail as everything else.
``export_article50`` then writes the standard signed trail zip and folds
in a human-readable report plus a machine summary, mirroring
``export_article12``.

The report is honest about limits: a disclosure record proves the
operator's system logged the disclosure event at that moment inside a
tamper-evident chain. It does not prove pixels appeared on a screen or
that the wording met accessibility requirements. The report says so.
"""
from __future__ import annotations

import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

DISCLOSURE_TOOL = "vaara.article50.disclosure"

#: The Article 50 paragraphs a disclosure record may claim.
PARAGRAPHS = ("50(1)", "50(2)", "50(3)", "50(4)")

#: Profile tag for the Article 50(1) AI-agent disclosure shape (guidance
#: C(2026) 5054, para 31). Stored in the event parameters so a generic
#: disclosure and an agent-profile disclosure stay distinguishable.
AGENT_PROFILE = "art50-1-agent/v1"

#: The interaction steps para 31 names for agent disclosure: at the first
#: interaction and at "key steps (point of authorisation, reporting,
#: validation)", repeated at every new interaction.
KEY_STEPS = (
    "first_interaction",
    "authorisation",
    "reporting",
    "validation",
    "new_interaction",
)


def record_disclosure(
    trail,
    *,
    paragraph: str,
    statement: str,
    agent_id: str = "operator",
    session_id: str = "",
    channel: str = "",
    subject: str = "",
    notice_sha256: str = "",
) -> str:
    """Record one Article 50 disclosure event into ``trail``.

    ``paragraph`` names the obligation (one of :data:`PARAGRAPHS`).
    ``statement`` is what was disclosed, in the operator's words (for
    50(1) typically the notice text or its identifier). ``channel`` is
    where it surfaced (``"chat_ui"``, ``"api_response_header"``, ...),
    ``subject`` identifies what it applies to (a session, a piece of
    content), and ``notice_sha256`` optionally pins the exact notice
    bytes shown. Returns the ``action_id`` of the recorded event.

    The record flows through the normal interception pipeline, so it is
    chained, signable, and exportable like any other action. Call it at
    the moment the disclosure is made (50(5): at the latest at the first
    interaction), not retroactively.
    """
    if paragraph not in PARAGRAPHS:
        raise ValueError(
            f"paragraph must be one of {PARAGRAPHS}, got {paragraph!r}"
        )
    if not statement:
        raise ValueError("statement must not be empty")

    from vaara.pipeline import InterceptionPipeline

    pipeline = InterceptionPipeline(trail=trail, enforce=False)
    result = pipeline.intercept(
        agent_id=agent_id,
        tool_name=DISCLOSURE_TOOL,
        parameters={
            "article": paragraph,
            "statement": statement,
            "channel": channel,
            "subject": subject,
            "notice_sha256": notice_sha256,
        },
        session_id=session_id,
        context={"vaara_article50": True},
    )
    return result.action_id


def record_agent_disclosure(
    trail,
    *,
    statement: str,
    on_behalf_of: str,
    step: str,
    agent_id: str = "operator",
    session_id: str = "",
    channel: str = "",
    subject: str = "",
    notice_sha256: str = "",
    authority_ref: str = "",
    parent_action_id: Optional[str] = None,
) -> str:
    """Record an Article 50(1) AI-agent disclosure (guidance para 31).

    Para 31 requires an AI agent to disclose "both their artificial
    nature and the person on whose behalf they are acting", covering
    "the delegation of authority and accountability for the consequences
    of their actions", at key steps (authorisation, reporting,
    validation) and at every new interaction. This is the receipt of
    exactly that: ``statement`` is the disclosure of artificial nature
    in the operator's words, ``on_behalf_of`` names the principal the
    agent acts for, ``step`` is one of :data:`KEY_STEPS`, and
    ``authority_ref`` optionally pins the mandate (a grant id, contract
    reference, or eIDAS attestation identifier). ``parent_action_id``
    threads the disclosure into the delegation chain the trail already
    reconstructs, so "who authorised this agent" and "the agent
    disclosed itself" are one lineage.

    The event is an ordinary :data:`DISCLOSURE_TOOL` record with
    ``article="50(1)"`` and ``profile=art50-1-agent/v1`` — same wire,
    same chain, same exports. Returns the recorded ``action_id``.
    """
    if step not in KEY_STEPS:
        raise ValueError(f"step must be one of {KEY_STEPS}, got {step!r}")
    if not statement:
        raise ValueError("statement must not be empty")
    if not on_behalf_of:
        raise ValueError(
            "on_behalf_of must name the person on whose behalf the agent "
            "acts (guidance C(2026) 5054 para 31)"
        )

    from vaara.pipeline import InterceptionPipeline

    pipeline = InterceptionPipeline(trail=trail, enforce=False)
    result = pipeline.intercept(
        agent_id=agent_id,
        tool_name=DISCLOSURE_TOOL,
        parameters={
            "article": "50(1)",
            "profile": AGENT_PROFILE,
            "statement": statement,
            "on_behalf_of": on_behalf_of,
            "step": step,
            "authority_ref": authority_ref,
            "channel": channel,
            "subject": subject,
            "notice_sha256": notice_sha256,
        },
        session_id=session_id,
        parent_action_id=parent_action_id,
        context={"vaara_article50": True},
    )
    return result.action_id


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def find_disclosures(records) -> list[dict]:
    """Extract disclosure events from trail records, oldest first."""
    out = []
    for rec in records:
        if rec.tool_name != DISCLOSURE_TOOL:
            continue
        if rec.event_type.value != "action_requested":
            continue
        params = (rec.data or {}).get("parameters", {}) or {}
        out.append({
            "action_id": rec.action_id,
            "record_id": rec.record_id,
            "timestamp": rec.timestamp,
            "time_utc": _iso(rec.timestamp),
            "agent_id": rec.agent_id,
            "session_id": (rec.data or {}).get("session_id") or "",
            "article": params.get("article", ""),
            "statement": params.get("statement", ""),
            "channel": params.get("channel", ""),
            "subject": params.get("subject", ""),
            "notice_sha256": params.get("notice_sha256", ""),
            "profile": params.get("profile", ""),
            "on_behalf_of": params.get("on_behalf_of", ""),
            "step": params.get("step", ""),
            "authority_ref": params.get("authority_ref", ""),
        })
    return out


def build_article50_report(
    records,
    manifest: dict,
    *,
    system_meta: Optional[dict] = None,
    period: Optional[tuple] = None,
) -> dict:
    """Build the Article 50 summary as a JSON-serialisable dict.

    ``period`` optionally narrows the *summary counts* to (start_ts,
    end_ts); the signed trail itself always stays whole, matching the
    Article 12 report's behavior.
    """
    disclosures = find_disclosures(records)
    if period is not None:
        start, end = period
        disclosures = [
            d for d in disclosures
            if (start is None or d["timestamp"] >= start)
            and (end is None or d["timestamp"] <= end)
        ]

    by_paragraph = Counter(d["article"] for d in disclosures)

    # 50(1)/(5) coverage: of the sessions that saw agent activity, how
    # many carry a 50(1) disclosure, and did it come before the
    # session's first non-disclosure action?
    session_first_action: dict[str, float] = {}
    for rec in records:
        if rec.event_type.value != "action_requested":
            continue
        if rec.tool_name == DISCLOSURE_TOOL:
            continue
        sid = (rec.data or {}).get("session_id") or ""
        if not sid:
            continue
        if sid not in session_first_action or rec.timestamp < session_first_action[sid]:
            session_first_action[sid] = rec.timestamp

    first_disclosure_501: dict[str, float] = {}
    for d in disclosures:
        if d["article"] != "50(1)" or not d["session_id"]:
            continue
        sid = d["session_id"]
        if sid not in first_disclosure_501 or d["timestamp"] < first_disclosure_501[sid]:
            first_disclosure_501[sid] = d["timestamp"]

    covered, covered_before_first = [], []
    for sid, first_action in session_first_action.items():
        if sid in first_disclosure_501:
            covered.append(sid)
            if first_disclosure_501[sid] <= first_action:
                covered_before_first.append(sid)

    # Para 31 agent-profile coverage: of the agent-profile disclosures,
    # which key steps were disclosed at, per session, and did each one
    # name the principal?
    agent_events = [d for d in disclosures if d["profile"] == AGENT_PROFILE]
    by_step = Counter(d["step"] for d in agent_events)
    steps_by_session: dict[str, set] = {}
    for d in agent_events:
        if d["session_id"]:
            steps_by_session.setdefault(d["session_id"], set()).add(d["step"])

    return {
        "standard": "EU AI Act Article 50 transparency evidence",
        "generated_utc": _iso(records[-1].timestamp) if records else "",
        "system": system_meta or {},
        "trail": {
            "records": manifest.get("record_count", len(records)),
            "algorithm": manifest.get("signature_algorithm", ""),
            "signer_pubkey_fingerprint": manifest.get(
                "signer_pubkey_fingerprint", ""
            ),
        },
        "disclosures": {
            "total": len(disclosures),
            "by_paragraph": {p: by_paragraph.get(p, 0) for p in PARAGRAPHS},
            "events": disclosures,
        },
        "session_coverage_50_1": {
            "sessions_with_agent_activity": len(session_first_action),
            "sessions_with_50_1_disclosure": len(covered),
            "disclosed_at_or_before_first_action": len(covered_before_first),
        },
        "agent_disclosure_para31": {
            "profile": AGENT_PROFILE,
            "total": len(agent_events),
            "by_step": {s: by_step.get(s, 0) for s in KEY_STEPS},
            "named_principal": sum(
                1 for d in agent_events if d["on_behalf_of"]
            ),
            "carried_authority_ref": sum(
                1 for d in agent_events if d["authority_ref"]
            ),
            "sessions": {
                sid: sorted(steps) for sid, steps in steps_by_session.items()
            },
        },
        "what_this_proves": (
            "Each disclosure event above was recorded into a signed, "
            "hash-chained trail at the stated time, alongside the agent "
            "actions it accompanies. Editing, inserting, or deleting any "
            "of these records after the fact breaks the chain and the "
            "signature, which any party can check offline from this "
            "package alone."
        ),
        "what_this_does_not_prove": (
            "That the disclosure was rendered on a screen, read, or "
            "worded accessibly (Article 50(5) manner requirements), nor "
            "that machine-readable content marking (50(2)) survives "
            "downstream processing. Those need UI-level and content-level "
            "evidence respectively; this package proves the operator's "
            "system made and kept the record."
        ),
    }


def render_article50_md(report: dict) -> str:
    """Render the report dict as a regulator-readable Markdown document."""
    lines = [
        "# EU AI Act Article 50 transparency evidence",
        "",
        f"Signed trail: {report['trail']['records']} records, "
        f"{report['trail']['algorithm']}, signer fingerprint "
        f"`{report['trail']['signer_pubkey_fingerprint']}`.",
        "",
    ]
    system = report.get("system") or {}
    if system:
        lines += ["## System", ""]
        for key, value in system.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    disc = report["disclosures"]
    lines += [
        "## Disclosure events",
        "",
        f"Total recorded: {disc['total']}",
        "",
    ]
    for paragraph in PARAGRAPHS:
        lines.append(f"- Article {paragraph}: {disc['by_paragraph'][paragraph]}")
    lines.append("")

    if disc["events"]:
        lines += [
            "| time (UTC) | article | channel | session | statement |",
            "|---|---|---|---|---|",
        ]
        for event in disc["events"]:
            statement = event["statement"].replace("|", "\\|")
            if len(statement) > 80:
                statement = statement[:77] + "..."
            lines.append(
                f"| {event['time_utc']} | {event['article']} | "
                f"{event['channel']} | {event['session_id']} | {statement} |"
            )
        lines.append("")

    agent = report.get("agent_disclosure_para31") or {}
    if agent.get("total"):
        lines += [
            "## AI-agent disclosure (Article 50(1), guidance para 31)",
            "",
            f"Agent-profile disclosures recorded: {agent['total']} "
            f"(profile `{agent['profile']}`). "
            f"{agent['named_principal']} name the person on whose behalf "
            f"the agent acted; {agent['carried_authority_ref']} carry an "
            "authority reference (mandate / attestation).",
            "",
            "Disclosures by key step (para 31: authorisation, reporting, "
            "validation, and every new interaction):",
            "",
        ]
        for step in KEY_STEPS:
            lines.append(f"- {step}: {agent['by_step'].get(step, 0)}")
        lines.append("")
        if agent.get("sessions"):
            lines.append("Per session, the steps at which the agent disclosed:")
            lines.append("")
            for sid, steps in agent["sessions"].items():
                lines.append(f"- `{sid}`: {', '.join(steps)}")
            lines.append("")

    cov = report["session_coverage_50_1"]
    lines += [
        "## Article 50(1) session coverage",
        "",
        f"- Sessions with agent activity: {cov['sessions_with_agent_activity']}",
        f"- Sessions carrying a 50(1) disclosure: "
        f"{cov['sessions_with_50_1_disclosure']}",
        f"- Disclosed at or before the session's first action (50(5) "
        f"timing): {cov['disclosed_at_or_before_first_action']}",
        "",
        "## What this package proves",
        "",
        report["what_this_proves"],
        "",
        "## What this package does not prove",
        "",
        report["what_this_does_not_prove"],
        "",
        "## Verify",
        "",
        "`vaara trail verify --zip <this file> --pubkey <trusted key>` "
        "checks the signature and the hash chain offline. The disclosure "
        "events are ordinary records in `trail.jsonl`; re-derive this "
        "report from them without trusting the operator.",
        "",
    ]
    return "\n".join(lines)


def export_article50(
    trail,
    out_path: Union[str, Path],
    *,
    signer_key=None,
    signer=None,
    system_meta: Optional[dict] = None,
    period: Optional[tuple] = None,
    agent_id: str = "",
):
    """Write a signed Article 50 transparency evidence package.

    The zip is the standard signed trail export (same bytes an Article 12
    package signs) with ``article50_report.md`` and
    ``article50_summary.json`` folded in afterwards, built from the
    records that were actually signed.
    """
    from vaara.audit.article12_export import _records_from_zip
    from vaara.audit.export import export_signed

    out_path = Path(out_path)
    result = export_signed(
        trail, out_path, signer_key=signer_key, signer=signer,
        agent_id=agent_id,
    )

    records, manifest = _records_from_zip(out_path)
    report = build_article50_report(
        records, manifest, system_meta=system_meta, period=period,
    )

    with zipfile.ZipFile(out_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("article50_report.md", render_article50_md(report))
        zf.writestr(
            "article50_summary.json",
            json.dumps(report, indent=2, sort_keys=False).encode("utf-8"),
        )

    return result
