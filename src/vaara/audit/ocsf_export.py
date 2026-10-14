# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OCSF export for the Vaara audit trail.

Writes each decision on the trail as one OCSF 1.9.0 API Activity event
(class 6003) with the ``security_control`` and ``ai_operation`` profiles, as
JSON Lines, so a SIEM or security data lake that ingests OCSF (Splunk,
Microsoft Sentinel, Amazon Security Lake) takes Vaara's decisions without a
Vaara-specific parser.

The tool call is the API call: ``api.operation`` is the tool name and the
agent is ``ai_agent``. The decision is the security control's outcome:
allow is Allowed, deny is Denied and Blocked, escalate (held for a person) is
action Other "Escalate" with disposition Delayed.

Integrity stays with Vaara. An OCSF ``attestation`` is a signature over the
OCSF event's own canonical form, and a Vaara receipt signs the receipt, so the
export carries no ``attestation_list``. Each event names the trail record it
came from (``metadata.uid``, ``unmapped.vaara.record_hash``), which is what a
reader checks a receipt or the trail against.

Zero new runtime deps.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any, Iterable, Optional

from vaara.audit.trail import AuditRecord, EventType

OCSF_VERSION = "1.9.0"
CLASS_UID = 6003
CATEGORY_UID = 6
ACTIVITY_OTHER = 99
ACTIVITY_NAME = "Tool Call"

# decision -> (action_id, action, disposition_id, disposition)
_CONTROL = {
    "allow": (1, "Allowed", 1, "Allowed"),
    "deny": (2, "Denied", 2, "Blocked"),
    "escalate": (99, "Escalate", 14, "Delayed"),
}

# severity_id from the risk score: below the default allow line is Low or
# Informational, between the lines Medium, above the deny line High.
_SEVERITY = ((0.25, 1, "Informational"), (0.55, 2, "Low"),
             (0.85, 3, "Medium"), (float("inf"), 4, "High"))


def _severity(risk: float) -> tuple[int, str]:
    for bound, sid, name in _SEVERITY:
        if risk < bound:
            return sid, name
    return 4, "High"


def _risk(data: dict) -> float:
    try:
        value = float(data.get("risk_score", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, value))


def decision_event(
    record: AuditRecord,
    request: Optional[AuditRecord] = None,
    hostname: str = "",
    vaara_version: str = "",
) -> dict[str, Any]:
    """One OCSF API Activity event for one decision record.

    ``request`` is the action's request record when the trail has it; it
    supplies the session. ``hostname`` names the deciding host, which the
    trail does not record.
    """
    data = record.data or {}
    decision = str(data.get("decision", ""))
    action_id, action, disposition_id, disposition = _CONTROL.get(
        decision, (0, "Unknown", 0, "Unknown"))
    risk = _risk(data)
    severity_id, severity = _severity(risk)
    session_id = ""
    if request is not None:
        session_id = str((request.data or {}).get("session_id") or "")

    agent: dict[str, Any] = {"uid": record.agent_id, "name": record.agent_id}
    if session_id:
        agent["instance_uid"] = session_id
    actor: dict[str, Any] = (
        {"session": {"uid": session_id}} if session_id
        else {"application": {"uid": record.agent_id, "name": record.agent_id}})

    metadata: dict[str, Any] = {
        "version": OCSF_VERSION,
        "product": {"name": "Vaara", "vendor_name": "Vaara",
                    **({"version": vaara_version} if vaara_version else {})},
        "profiles": ["security_control", "ai_operation"],
        "uid": record.record_id,
        "original_event_uid": record.record_id,
        "correlation_uid": record.action_id,
        "log_name": "vaara-trail",
    }
    if record.tenant_id:
        metadata["tenant_uid"] = record.tenant_id

    policy: dict[str, Any] = {"is_applied": True}
    if data.get("policy_digest"):
        policy["uid"] = str(data["policy_digest"])
    policy["name"] = str(data.get("policy_id") or "vaara")

    event: dict[str, Any] = {
        "class_uid": CLASS_UID,
        "class_name": "API Activity",
        "category_uid": CATEGORY_UID,
        "category_name": "Application Activity",
        "activity_id": ACTIVITY_OTHER,
        "activity_name": ACTIVITY_NAME,
        "type_uid": CLASS_UID * 100 + ACTIVITY_OTHER,
        "type_name": f"API Activity: {ACTIVITY_NAME}",
        "time": int(round(record.timestamp * 1000)),
        "severity_id": severity_id,
        "severity": severity,
        "status_id": 2 if decision == "deny" else 0,
        "status": "Failure" if decision == "deny" else "Unknown",
        "message": str(data.get("reason", "")),
        "metadata": metadata,
        "actor": actor,
        "api": {"operation": record.tool_name},
        "src_endpoint": {"hostname": hostname} if hostname else {"name": "unknown"},
        "ai_agent": agent,
        "action_id": action_id,
        "action": action,
        "disposition_id": disposition_id,
        "disposition": disposition,
        "is_alert": decision in ("deny", "escalate"),
        "risk_score": int(round(risk * 100)),
        "policy": policy,
        "unmapped": {"vaara": _vaara_fields(record, data)},
    }
    if decision == "deny":
        event["status_detail"] = str(data.get("violation_type") or data.get("reason", ""))
    return event


def _vaara_fields(record: AuditRecord, data: dict) -> dict[str, Any]:
    """What the trail holds that OCSF has no attribute for."""
    out: dict[str, Any] = {
        "decision": str(data.get("decision", "")),
        "risk_score": _risk(data),
        "record_hash": record.record_hash,
        "previous_hash": record.previous_hash,
        "event_type": record.event_type.value,
    }
    for key in ("violation_type", "unavailable_input", "decision_detail",
                "policy_digest", "sequence_steps"):
        if data.get(key):
            out[key] = data[key]
    return out


def decision_events(
    records: Iterable[AuditRecord], hostname: str = "", vaara_version: str = "",
) -> list[dict[str, Any]]:
    """Every decision on the trail as an OCSF event, in trail order."""
    records = list(records)
    requested = {r.action_id: r for r in records
                if r.event_type == EventType.ACTION_REQUESTED}
    return [
        decision_event(r, requested.get(r.action_id), hostname, vaara_version)
        for r in records
        if r.event_type in (EventType.DECISION_MADE, EventType.ACTION_BLOCKED)
    ]


def write_ocsf_jsonl(
    records: Iterable[AuditRecord], out: Path, hostname: Optional[str] = None,
) -> int:
    """Write the trail's decisions as OCSF JSON Lines. Returns the count."""
    from vaara import __version__

    host = socket.gethostname() if hostname is None else hostname
    events = decision_events(records, host, __version__)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
    return len(events)
