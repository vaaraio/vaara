"""Tests for the Article 73 serious-incident report exporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaara.audit.incident_export import (
    SCHEMA_VERSION,
    build_from_trail,
    build_incident_report,
    write_incident_report,
)
from vaara.audit.trail import AuditRecord, EventType


def _rec(
    event_type: EventType,
    *,
    record_id: str,
    action_id: str = "act-1",
    timestamp: float = 1700000000.0,
    data: dict | None = None,
    record_hash: str = "",
) -> AuditRecord:
    return AuditRecord(
        record_id=record_id, action_id=action_id, event_type=event_type,
        timestamp=timestamp, agent_id="agent-007", tool_name="fs.write_file",
        data=data or {}, record_hash=record_hash,
    )


def _baseline_kwargs() -> dict:
    return dict(
        trigger_record=_rec(EventType.OUTCOME_RECORDED, record_id="r-trigger"),
        evidence_records=[
            _rec(EventType.RISK_SCORED, record_id="r-1"),
            _rec(EventType.DECISION_MADE, record_id="r-2"),
        ],
        severity_category="a",
        severity_rationale="serious harm to health",
        ai_system={
            "identifier": "vaara-demo-v0.8",
            "version": "0.8.0",
            "provider_organization": "Vaara",
        },
        causal_link={"status": "established", "description": "tool-call caused harm"},
        reporter={"role": "provider", "name": "Vaara", "contact_email": "hello@vaara.io"},
        recipient={"authority": "market_surveillance", "member_state": "FI"},
    )


def _trail_meta() -> dict:
    return {
        "severity": {"category": "a", "rationale": "x"},
        "ai_system": {"identifier": "x", "version": "0", "provider_organization": "x"},
        "causal_link": {"status": "established", "description": ""},
        "reporter": {"role": "provider", "name": "x", "contact_email": "x@x"},
        "recipient": {"authority": "market_surveillance", "member_state": "FI"},
    }


def test_minimal_build_has_expected_top_keys() -> None:
    rep = build_incident_report(**_baseline_kwargs())
    for key in (
        "schema_version", "incident_id", "report_status", "report_timestamp",
        "report_format", "regulation", "severity", "ai_system", "causal_link",
        "incident_timeline", "affected_persons", "corrective_actions",
        "risk_assessment", "investigation_status",
        "system_unaltered_attestation", "reporter", "recipient",
        "evidence_bundle", "extensions",
    ):
        assert key in rep, f"missing top-level key {key}"


def test_schema_version_stable() -> None:
    rep = build_incident_report(**_baseline_kwargs())
    assert rep["schema_version"] == SCHEMA_VERSION == "vaara-incident/1.0"


def test_default_deadline_general_15_days() -> None:
    rep = build_incident_report(**_baseline_kwargs())
    assert rep["regulation"]["reporting_deadline_days"] == 15


def test_deadline_death_10_days() -> None:
    kw = _baseline_kwargs()
    kw["severity_category"] = "death"
    rep = build_incident_report(**kw)
    assert rep["regulation"]["reporting_deadline_days"] == 10


def test_deadline_b_widespread_2_days() -> None:
    kw = _baseline_kwargs()
    kw["severity_category"] = "b"
    rep = build_incident_report(**kw)
    assert rep["regulation"]["reporting_deadline_days"] == 2


def test_rejects_unknown_severity_category() -> None:
    kw = _baseline_kwargs()
    kw["severity_category"] = "z"
    with pytest.raises(ValueError, match="severity_category"):
        build_incident_report(**kw)


def test_rejects_non_trigger_event_type() -> None:
    kw = _baseline_kwargs()
    kw["trigger_record"] = _rec(EventType.RISK_SCORED, record_id="r-bad")
    with pytest.raises(ValueError, match="trigger record event_type"):
        build_incident_report(**kw)


def test_rejects_unknown_causal_status() -> None:
    kw = _baseline_kwargs()
    kw["causal_link"] = {"status": "probably_maybe", "description": ""}
    with pytest.raises(ValueError, match="causal_link.status"):
        build_incident_report(**kw)


def test_rejects_unknown_reporter_role() -> None:
    kw = _baseline_kwargs()
    kw["reporter"] = {"role": "vendor", "name": "x", "contact_email": "x@x"}
    with pytest.raises(ValueError, match="reporter.role"):
        build_incident_report(**kw)


def test_complete_status_requires_previous_report_id() -> None:
    kw = _baseline_kwargs()
    kw["report_status"] = "complete"
    with pytest.raises(ValueError, match="previous_report_id"):
        build_incident_report(**kw)
    kw["previous_report_id"] = "incident-uuid-prev"
    rep = build_incident_report(**kw)
    assert rep["report_status"] == "complete"
    assert rep["previous_report_id"] == "incident-uuid-prev"


def test_evidence_audit_records_trigger_first_deduped() -> None:
    kw = _baseline_kwargs()
    kw["evidence_records"] = [
        kw["trigger_record"],
        _rec(EventType.RISK_SCORED, record_id="r-1"),
    ]
    rep = build_incident_report(**kw)
    ids = rep["causal_link"]["evidence_audit_records"]
    assert ids == ["r-trigger", "r-1"]
    assert rep["evidence_bundle"]["audit_record_ids"] == ids


def test_timeline_uses_trigger_timestamp_for_discovery_and_deadline() -> None:
    rep = build_incident_report(**_baseline_kwargs())
    assert rep["incident_timeline"]["discovery_timestamp"] == "2023-11-14T22:13:20Z"
    assert rep["incident_timeline"]["report_due_by"] == "2023-11-29T22:13:20Z"


def test_write_round_trips_json(tmp_path: Path) -> None:
    rep = build_incident_report(**_baseline_kwargs())
    out = tmp_path / "incident.json"
    write_incident_report(rep, out)
    with open(out, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == rep


def test_build_from_trail_picks_most_recent_trigger_eligible() -> None:
    records = [
        _rec(EventType.ACTION_REQUESTED, record_id="r-1", timestamp=1.0),
        _rec(EventType.RISK_SCORED, record_id="r-2", timestamp=2.0),
        _rec(EventType.ACTION_BLOCKED, record_id="r-3", timestamp=3.0),
        _rec(EventType.OUTCOME_RECORDED, record_id="r-4", timestamp=4.0),
        _rec(EventType.ACTION_EXECUTED, record_id="r-5", timestamp=5.0),
    ]
    rep = build_from_trail(records, incident_meta=_trail_meta())
    # r-5 is not trigger-eligible; latest eligible is r-4 (OUTCOME_RECORDED)
    assert rep["causal_link"]["evidence_audit_records"][0] == "r-4"


def test_build_from_trail_with_explicit_trigger_id() -> None:
    records = [
        _rec(EventType.ACTION_BLOCKED, record_id="r-3", timestamp=3.0),
        _rec(EventType.OUTCOME_RECORDED, record_id="r-4", timestamp=4.0),
    ]
    rep = build_from_trail(records, incident_meta=_trail_meta(),
                           trigger_record_id="r-3")
    assert rep["causal_link"]["evidence_audit_records"][0] == "r-3"


def test_build_from_trail_raises_when_no_eligible_trigger() -> None:
    records = [_rec(EventType.RISK_SCORED, record_id="r-1")]
    with pytest.raises(ValueError, match="no trigger-eligible record"):
        build_from_trail(records, incident_meta=_trail_meta())


def test_severity_accepts_either_category_or_subpoint_key() -> None:
    records = [_rec(EventType.OUTCOME_RECORDED, record_id="r-1")]
    base = _trail_meta()
    base.pop("severity")
    via_category = build_from_trail(
        records,
        incident_meta={**base, "severity": {"category": "death", "rationale": "x"}},
    )
    via_subpoint = build_from_trail(
        records,
        incident_meta={**base,
                       "severity": {"article_3_49_subpoint": "death", "rationale": "x"}},
    )
    assert via_category["regulation"]["reporting_deadline_days"] == 10
    assert via_subpoint["regulation"]["reporting_deadline_days"] == 10
