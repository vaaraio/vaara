"""Tests for the W3C PROV-DM audit-trail exporter."""

from __future__ import annotations

import json
import time
from pathlib import Path

from vaara.audit.prov_export import (
    AIACT_NS,
    PROV_CONTEXT,
    VAARA_NS,
    audit_to_prov_json,
    write_prov_json,
)
from vaara.audit.trail import AuditRecord, EventType


def _record(
    event_type: EventType,
    *,
    record_id: str,
    action_id: str = "act-1",
    agent_id: str = "agent-007",
    tool_name: str = "fs.write_file",
    timestamp: float = 1700000000.0,
    data: dict | None = None,
    regulatory_articles: list[dict] | None = None,
    record_hash: str = "",
    previous_hash: str = "",
) -> AuditRecord:
    return AuditRecord(
        record_id=record_id, action_id=action_id, event_type=event_type,
        timestamp=timestamp, agent_id=agent_id, tool_name=tool_name,
        data=data or {}, regulatory_articles=regulatory_articles or [],
        record_hash=record_hash, previous_hash=previous_hash,
    )


def _full_lifecycle_records() -> list[AuditRecord]:
    """A request → score → escalate → resolve → outcome lifecycle."""
    base_ts = 1700000000.0
    return [
        _record(
            EventType.ACTION_REQUESTED, record_id="r1", timestamp=base_ts,
            data={"parameters": {"path": "/etc/x"}, "agent_confidence": 0.8},
            regulatory_articles=[{"domain": "EU AI Act", "article": "Article 12(1)"}],
            record_hash="h1",
        ),
        _record(
            EventType.RISK_SCORED, record_id="r2", timestamp=base_ts + 1,
            data={"point_estimate": 0.62, "conformal_lower": 0.55, "conformal_upper": 0.69},
            regulatory_articles=[
                {"domain": "EU AI Act", "article": "Article 9(2)(a)"},
                {"domain": "EU AI Act", "article": "Article 9(7)"},
            ],
            record_hash="h2", previous_hash="h1",
        ),
        _record(
            EventType.DECISION_MADE, record_id="r3", timestamp=base_ts + 2,
            data={"decision": "escalate", "reason": "above threshold", "risk_score": 0.62},
            regulatory_articles=[{"domain": "EU AI Act", "article": "Article 14(1)"}],
            record_hash="h3", previous_hash="h2",
        ),
        _record(
            EventType.ESCALATION_SENT, record_id="r4", timestamp=base_ts + 3,
            data={"escalation_target": "ops", "risk_score": 0.62},
            record_hash="h4", previous_hash="h3",
        ),
        _record(
            EventType.ESCALATION_RESOLVED, record_id="r5", timestamp=base_ts + 4,
            data={"resolution": "deny", "reviewer": "operator-jane",
                  "justification": "needs CR approval"},
            record_hash="h5", previous_hash="h4",
        ),
        _record(
            EventType.OUTCOME_RECORDED, record_id="r6", timestamp=base_ts + 5,
            data={"outcome_severity": 0.0},
            regulatory_articles=[{"domain": "DORA", "article": "Article 13(1)"}],
            record_hash="h6", previous_hash="h5",
        ),
    ]


def test_top_level_doc_shape() -> None:
    doc = audit_to_prov_json(_full_lifecycle_records())
    assert doc["@context"] == PROV_CONTEXT
    assert doc["prefix"]["vaara"] == VAARA_NS
    assert doc["prefix"]["aiact"] == AIACT_NS
    assert "vaara:action/act-1" in doc["bundle"]


def test_bundle_has_expected_top_keys() -> None:
    bundle = audit_to_prov_json(_full_lifecycle_records())["bundle"]["vaara:action/act-1"]
    for key in ("agent", "entity", "activity", "wasGeneratedBy", "used",
                "wasInformedBy", "wasAssociatedWith", "wasAttributedTo"):
        assert key in bundle, f"missing {key}"


def test_agents_include_pipeline_caller_and_operator() -> None:
    agents = audit_to_prov_json(_full_lifecycle_records())["bundle"]["vaara:action/act-1"]["agent"]
    assert agents["vaara:agent/vaara-pipeline"]["prov:type"] == "prov:SoftwareAgent"
    assert agents["vaara:agent/agent-007"]["prov:type"] == "prov:SoftwareAgent"
    assert agents["vaara:agent/operator-jane"]["prov:type"] == "prov:Person"


def test_entities_emitted_per_event() -> None:
    entities = audit_to_prov_json(_full_lifecycle_records())["bundle"]["vaara:action/act-1"]["entity"]
    for ent in ("vaara:request/act-1", "vaara:score/act-1", "vaara:decision/act-1/v1",
                "vaara:operatorResponse/act-1", "vaara:outcome/act-1"):
        assert ent in entities
    assert entities["vaara:score/act-1"]["vaara:riskScore"] == 0.62
    assert entities["vaara:score/act-1"]["vaara:conformalInterval"] == [0.55, 0.69]


def test_activities_carry_regulatory_satisfies() -> None:
    activities = audit_to_prov_json(_full_lifecycle_records())["bundle"]["vaara:action/act-1"]["activity"]
    assert activities["vaara:event/r2"]["aiact:satisfies"] == ["Article 9(2)(a)", "Article 9(7)"]
    assert activities["vaara:event/r6"]["dora:satisfies"] == ["Article 13(1)"]


def test_was_informed_by_chains_activities_chronologically() -> None:
    wib = audit_to_prov_json(_full_lifecycle_records())["bundle"]["vaara:action/act-1"]["wasInformedBy"]
    pairs = {v["prov:informant"]: v["prov:informed"] for v in wib.values()}
    assert pairs["vaara:event/r1"] == "vaara:event/r2"
    assert pairs["vaara:event/r2"] == "vaara:event/r3"


def test_action_id_filter_and_empty_input() -> None:
    other = _record(EventType.ACTION_REQUESTED, record_id="o", action_id="act-other")
    doc = audit_to_prov_json(_full_lifecycle_records() + [other], action_id="act-1")
    assert "vaara:action/act-1" in doc["bundle"]
    assert "vaara:action/act-other" not in doc["bundle"]
    empty = audit_to_prov_json([])
    assert empty["@context"] == PROV_CONTEXT
    assert "bundle" not in empty


def test_chain_layer_links_records_by_was_derived_from() -> None:
    doc = audit_to_prov_json(_full_lifecycle_records(), include_chain=True)
    chain = doc["bundle"]["vaara:auditChain"]
    assert chain["entity"]["vaara:auditRecord/r1"]["vaara:recordHash"] == "h1"
    assert chain["entity"]["vaara:auditRecord/r6"]["vaara:previousHash"] == "h5"
    pairs = {v["prov:generatedEntity"]: v["prov:usedEntity"]
             for v in chain["wasDerivedFrom"].values()}
    assert pairs["vaara:auditRecord/r2"] == "vaara:auditRecord/r1"
    assert pairs["vaara:auditRecord/r6"] == "vaara:auditRecord/r5"


def test_include_chain_false_omits_chain_bundle() -> None:
    doc = audit_to_prov_json(_full_lifecycle_records(), include_chain=False)
    assert "vaara:auditChain" not in doc.get("bundle", {})


def test_policy_override_emits_revision_relation() -> None:
    base = _full_lifecycle_records()[:3]
    base.append(_record(
        EventType.POLICY_OVERRIDE, record_id="r-ov", timestamp=1700000010.0,
        data={"override_reason": "bad call", "overrider": "operator-jane",
              "original_decision": "escalate", "new_decision": "deny"},
        record_hash="h-ov", previous_hash="h3",
    ))
    bundle = audit_to_prov_json(base)["bundle"]["vaara:action/act-1"]
    assert "vaara:decision/act-1/v2" in bundle["entity"]
    rev = next(iter(bundle["wasDerivedFrom"].values()))
    assert rev["prov:generatedEntity"] == "vaara:decision/act-1/v2"
    assert rev["prov:usedEntity"] == "vaara:decision/act-1/v1"
    assert rev["prov:type"] == "prov:Revision"


def test_write_prov_json_round_trips(tmp_path: Path) -> None:
    out = tmp_path / "trail.prov.json"
    n = write_prov_json(_full_lifecycle_records(), out)
    assert n == 6
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["@context"] == PROV_CONTEXT
    assert "vaara:action/act-1" in loaded["bundle"]


def test_iso_timestamp_format() -> None:
    doc = audit_to_prov_json([
        _record(EventType.ACTION_REQUESTED, record_id="r1", timestamp=1700000000.0),
    ])
    activity = doc["bundle"]["vaara:action/act-1"]["activity"]["vaara:event/r1"]
    assert activity["prov:startTime"] == time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(1700000000.0)
    )
