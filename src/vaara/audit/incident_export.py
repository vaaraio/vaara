"""Article 73 EU AI Act serious-incident report export.

INTERIM format pending publication of the Commission template promised by
Article 73 paragraph 7. The schema covers every field Article 73 references
plus Article 3(49) sub-category attribution that determines the reporting
deadline. When the template publishes, the schema_version bumps and any
new fields land additively. Consumers switch on ``schema_version`` to detect
format changes.

The export is a standalone JSON document. Audit records from the Vaara
trail are referenced by ``record_id`` as evidence. The report does not
duplicate their content. Pair with ``vaara.audit.prov_export`` (PROV-DM
export) for the full evidence bundle.

Reporting deadlines per Article 73:

* Paragraph 2 general:              15 days from awareness
* Paragraph 3 Article 3(49)(b):     2  days (widespread infringement / serious)
* Paragraph 4 death of a person:    10 days

Paragraph 5 supports an incomplete *initial* report followed by a
*complete* one. ``report_status`` selects between them; ``previous_report_id``
links a complete report back to its initial.

Schema version: ``vaara-incident/1.0``
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Iterable, Optional

from vaara.audit.trail import AuditRecord, EventType

SCHEMA_VERSION = "vaara-incident/1.0"

# Article 73 paragraph deadlines, keyed by Article 3(49) sub-category.
# ``death`` is paragraph 4's special case; ``b`` is paragraph 3; everything
# else falls under paragraph 2's general 15-day window.
_DEADLINE_DAYS = {"death": 10, "b": 2, "a": 15, "c": 15, "d": 15}

# Event types whose appearance in the trail can plausibly trigger an
# Article 73 report. Other event types describe normal operation and do
# not themselves indicate harm.
_INCIDENT_TRIGGER_EVENTS = frozenset({
    EventType.OUTCOME_RECORDED,
    EventType.ACTION_BLOCKED,
    EventType.POLICY_OVERRIDE,
})

_VALID_CAUSAL = ("established", "suspected", "reasonable_likelihood")
_VALID_REPORT_STATUS = ("initial", "complete")
_VALID_REPORTER_ROLE = ("provider", "deployer")


def _iso(epoch: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ",
                         time.gmtime(epoch if epoch is not None else time.time()))


def _validate_inputs(
    trigger: AuditRecord, severity_category: str,
    causal_status: str, reporter_role: str,
    report_status: str, previous_report_id: Optional[str],
) -> None:
    if severity_category not in _DEADLINE_DAYS:
        raise ValueError(
            f"severity_category must be one of {sorted(_DEADLINE_DAYS)}, "
            f"got {severity_category!r}"
        )
    if trigger.event_type not in _INCIDENT_TRIGGER_EVENTS:
        valid = sorted(et.value for et in _INCIDENT_TRIGGER_EVENTS)
        raise ValueError(
            f"trigger record event_type must be in {valid}, got "
            f"{trigger.event_type.value!r} for record_id {trigger.record_id!r}"
        )
    if causal_status not in _VALID_CAUSAL:
        raise ValueError(f"causal_link.status must be one of {list(_VALID_CAUSAL)}")
    if reporter_role not in _VALID_REPORTER_ROLE:
        raise ValueError(f"reporter.role must be one of {list(_VALID_REPORTER_ROLE)}")
    if report_status not in _VALID_REPORT_STATUS:
        raise ValueError(f"report_status must be one of {list(_VALID_REPORT_STATUS)}")
    if report_status == "complete" and not previous_report_id:
        raise ValueError("previous_report_id is required when report_status == 'complete'")


def build_incident_report(
    *,
    trigger_record: AuditRecord,
    evidence_records: Iterable[AuditRecord] = (),
    severity_category: str,
    severity_rationale: str,
    ai_system: dict,
    causal_link: dict,
    reporter: dict,
    recipient: dict,
    affected_persons: Optional[dict] = None,
    corrective_actions: Optional[list[dict]] = None,
    risk_assessment: str = "",
    investigation_status: str = "in_progress",
    system_unaltered_attestation: bool = True,
    report_status: str = "initial",
    previous_report_id: Optional[str] = None,
    evidence_bundle: Optional[dict] = None,
    incident_id: Optional[str] = None,
    report_timestamp: Optional[str] = None,
    extensions: Optional[dict] = None,
) -> dict:
    """Build a serialisable Article 73 incident report dict.

    See module docstring for the schema, deadline mapping, and field
    semantics. ``trigger_record`` is the audit event whose occurrence
    establishes the incident; ``evidence_records`` are additional records
    listed as evidence. The trigger's ``timestamp`` anchors
    ``incident_timeline.discovery_timestamp`` and the deadline computation.
    """
    _validate_inputs(
        trigger_record, severity_category,
        causal_link.get("status", ""), reporter.get("role", ""),
        report_status, previous_report_id,
    )

    deadline_days = _DEADLINE_DAYS[severity_category]
    discovery_iso = _iso(trigger_record.timestamp)
    report_due_iso = _iso(trigger_record.timestamp + deadline_days * 86400)

    record_ids = [trigger_record.record_id]
    for r in evidence_records:
        if r.record_id != trigger_record.record_id:
            record_ids.append(r.record_id)

    bundle = dict(evidence_bundle or {})
    bundle["audit_record_ids"] = record_ids

    return {
        "schema_version": SCHEMA_VERSION,
        "incident_id": incident_id or str(uuid.uuid4()),
        "report_status": report_status,
        "report_timestamp": report_timestamp or _iso(),
        "report_format": "article73",
        "previous_report_id": previous_report_id,
        "regulation": {
            "framework": "EU AI Act",
            "article": "73",
            "definition_reference": "Article 3(49)",
            "reporting_deadline_days": deadline_days,
            "applies_2019_1020": True,
            "commission_template_published": False,
        },
        "severity": {
            "article_3_49_subpoint": severity_category,
            "rationale": severity_rationale,
        },
        "ai_system": dict(ai_system),
        "causal_link": {
            "status": causal_link["status"],
            "description": causal_link.get("description", ""),
            "evidence_audit_records": record_ids,
        },
        "incident_timeline": {
            "discovery_timestamp": discovery_iso,
            "report_due_by": report_due_iso,
        },
        "affected_persons": affected_persons
            or {"count_estimated": 0, "harm_categories": []},
        "corrective_actions": list(corrective_actions or []),
        "risk_assessment": risk_assessment,
        "investigation_status": investigation_status,
        "system_unaltered_attestation": bool(system_unaltered_attestation),
        "reporter": dict(reporter),
        "recipient": dict(recipient),
        "evidence_bundle": bundle,
        "extensions": dict(extensions or {}),
    }


def write_incident_report(report: dict, out_path: Path) -> None:
    """Serialise an incident report to a JSON file (UTF-8, indent=2)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=False)
        f.write("\n")


def _resolve_trigger(
    records: list[AuditRecord], record_id: Optional[str],
) -> AuditRecord:
    """Pick the trigger record from a trail.

    With ``record_id`` set, locate exactly that record. Otherwise return
    the most recent trigger-eligible event. Raise if no candidate exists.
    """
    if record_id is not None:
        for r in records:
            if r.record_id == record_id:
                return r
        raise ValueError(f"trigger record_id {record_id!r} not found in trail")
    for r in reversed(records):
        if r.event_type in _INCIDENT_TRIGGER_EVENTS:
            return r
    valid = sorted(et.value for et in _INCIDENT_TRIGGER_EVENTS)
    raise ValueError(f"no trigger-eligible record (event_type in {valid}) in trail")


def build_from_trail(
    records: list[AuditRecord],
    *,
    incident_meta: dict,
    trigger_record_id: Optional[str] = None,
) -> dict:
    """Convenience builder that takes operator metadata as a single dict.

    ``incident_meta`` carries the operator-supplied fields (severity,
    ai_system, causal_link, reporter, recipient, plus the optional fields
    documented on ``build_incident_report``). The trail supplies trigger
    and evidence records.
    """
    trigger = _resolve_trigger(records, trigger_record_id)
    severity = incident_meta.get("severity") or {}
    return build_incident_report(
        trigger_record=trigger,
        evidence_records=records,
        severity_category=(
            severity.get("category")
            or severity.get("article_3_49_subpoint", "")
        ),
        severity_rationale=severity.get("rationale", ""),
        ai_system=incident_meta.get("ai_system") or {},
        causal_link=incident_meta.get("causal_link") or {},
        reporter=incident_meta.get("reporter") or {},
        recipient=incident_meta.get("recipient") or {},
        affected_persons=incident_meta.get("affected_persons"),
        corrective_actions=incident_meta.get("corrective_actions"),
        risk_assessment=incident_meta.get("risk_assessment", ""),
        investigation_status=incident_meta.get("investigation_status", "in_progress"),
        system_unaltered_attestation=incident_meta.get(
            "system_unaltered_attestation", True),
        report_status=incident_meta.get("report_status", "initial"),
        previous_report_id=incident_meta.get("previous_report_id"),
        evidence_bundle=incident_meta.get("evidence_bundle"),
        incident_id=incident_meta.get("incident_id"),
        report_timestamp=incident_meta.get("report_timestamp"),
        extensions=incident_meta.get("extensions"),
    )
