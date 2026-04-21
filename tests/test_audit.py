"""Tests for the audit trail module."""

import json
import tempfile
from pathlib import Path

import pytest
from vaara.audit.trail import (
    AuditRecord,
    AuditTrail,
    EventType,
    EU_AI_ACT_MAPPINGS,
)
from vaara.taxonomy.actions import (
    ActionCategory,
    ActionRequest,
    ActionType,
    BlastRadius,
    RegulatoryDomain,
    Reversibility,
    UrgencyClass,
)


@pytest.fixture
def trail():
    return AuditTrail()


@pytest.fixture
def sample_request():
    at = ActionType(
        "tx.transfer", ActionCategory.FINANCIAL, Reversibility.IRREVERSIBLE,
        BlastRadius.SHARED, UrgencyClass.IRREVOCABLE,
        frozenset({RegulatoryDomain.MIFID2, RegulatoryDomain.DORA}),
    )
    return ActionRequest(
        agent_id="agent-007",
        tool_name="tx.transfer",
        action_type=at,
        parameters={"to": "0xabc", "amount": 1000},
        confidence=0.8,
    )


class TestAuditRecord:
    def test_compute_hash_deterministic(self):
        r = AuditRecord(
            record_id="r1", action_id="a1", event_type=EventType.ACTION_REQUESTED,
            timestamp=1000.0, agent_id="agent", tool_name="tool",
        )
        h1 = r.compute_hash()
        h2 = r.compute_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self):
        r1 = AuditRecord(
            record_id="r1", action_id="a1", event_type=EventType.ACTION_REQUESTED,
            timestamp=1000.0, agent_id="agent", tool_name="tool",
        )
        r2 = AuditRecord(
            record_id="r2", action_id="a1", event_type=EventType.ACTION_REQUESTED,
            timestamp=1000.0, agent_id="agent", tool_name="tool",
        )
        assert r1.compute_hash() != r2.compute_hash()

    def test_serialization_roundtrip(self):
        r = AuditRecord(
            record_id="r1", action_id="a1", event_type=EventType.RISK_SCORED,
            timestamp=1000.0, agent_id="agent", tool_name="tx.sign",
            data={"risk": 0.7},
        )
        d = r.to_dict()
        r2 = AuditRecord.from_dict(d)
        assert r2.record_id == "r1"
        assert r2.event_type == EventType.RISK_SCORED
        assert r2.data["risk"] == 0.7

    def test_narrative_generation(self):
        r = AuditRecord(
            record_id="r1", action_id="a1", event_type=EventType.ACTION_BLOCKED,
            timestamp=1000.0, agent_id="bad-agent", tool_name="tx.transfer",
            data={"risk_score": 0.9},
        )
        narrative = r.narrative
        assert "bad-agent" in narrative
        assert "BLOCKED" in narrative
        assert "tx.transfer" in narrative

    def test_narrative_defensive_on_bad_timestamp(self):
        """Loop 46: a single corrupt-timestamp record must not crash
        narrative rendering. AuditTrail.get_narrative() iterates all
        records; one bad timestamp would otherwise take down the entire
        regulator-facing narrative view."""
        import math
        for bad_ts in ("bogus", math.nan, math.inf, 1e20, None):
            r = AuditRecord(
                record_id="r", action_id="a",
                event_type=EventType.ACTION_REQUESTED,
                timestamp=bad_ts, agent_id="x", tool_name="y",
            )
            # Must not raise
            n = r.narrative
            assert "invalid-ts" in n or "Agent 'x'" in n


class TestAuditTrail:
    def test_record_action_returns_action_id(self, trail, sample_request):
        action_id = trail.record_action_requested(sample_request)
        assert action_id  # Non-empty string
        assert trail.size == 1

    def test_hash_chain_integrity(self, trail, sample_request):
        trail.record_action_requested(sample_request)
        trail.record_decision(
            action_id="a1", agent_id="agent-007", tool_name="tx.transfer",
            decision="deny", reason="too risky", risk_score=0.9,
        )
        assert trail.chain_intact

    def test_chain_detects_tampering(self, trail, sample_request):
        trail.record_action_requested(sample_request)
        trail.record_decision(
            action_id="a1", agent_id="agent-007", tool_name="tx.transfer",
            decision="allow", reason="ok", risk_score=0.2,
        )
        # Tamper with a record
        trail._records[0].data["tampered"] = True
        error = trail.verify_chain()
        assert error is not None
        assert "Hash mismatch" in error

    def test_action_trail_groups_events(self, trail, sample_request):
        action_id = trail.record_action_requested(sample_request)
        trail.record_risk_scored(
            action_id=action_id, agent_id="agent-007", tool_name="tx.transfer",
            assessment={"risk": 0.7},
        )
        trail.record_decision(
            action_id=action_id, agent_id="agent-007", tool_name="tx.transfer",
            decision="deny", reason="high risk", risk_score=0.7,
        )
        events = trail.get_action_trail(action_id)
        assert len(events) == 3
        assert events[0].event_type == EventType.ACTION_REQUESTED
        assert events[1].event_type == EventType.RISK_SCORED
        assert events[2].event_type == EventType.ACTION_BLOCKED

    def test_regulatory_articles_attached(self, trail, sample_request):
        action_id = trail.record_action_requested(sample_request)
        events = trail.get_action_trail(action_id)
        record = events[0]
        assert len(record.regulatory_articles) > 0
        domains = {a["domain"] for a in record.regulatory_articles}
        assert "eu_ai_act" in domains

    def test_get_blocked_actions(self, trail):
        trail.record_decision(
            action_id="a1", agent_id="agent", tool_name="danger",
            decision="deny", reason="blocked", risk_score=0.9,
        )
        blocked = trail.get_blocked_actions()
        assert len(blocked) == 1
        assert blocked[0].event_type == EventType.ACTION_BLOCKED

    def test_export_json(self, trail, sample_request):
        trail.record_action_requested(sample_request)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        count = trail.export_json(path)
        assert count == 1
        data = json.loads(path.read_text())
        assert len(data) == 1
        path.unlink()

    def test_export_jsonl(self, trail, sample_request):
        trail.record_action_requested(sample_request)
        trail.record_decision(
            action_id="a1", agent_id="agent-007", tool_name="tx.transfer",
            decision="allow", reason="ok", risk_score=0.2,
        )
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)
        count = trail.export_jsonl(path)
        assert count == 2
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        path.unlink()

    def test_narrative_output(self, trail, sample_request):
        action_id = trail.record_action_requested(sample_request)
        trail.record_decision(
            action_id=action_id, agent_id="agent-007", tool_name="tx.transfer",
            decision="deny", reason="too risky", risk_score=0.8,
        )
        narrative = trail.get_narrative(action_id)
        assert "agent-007" in narrative
        assert "tx.transfer" in narrative

    def test_on_record_callback(self, sample_request):
        captured = []
        trail = AuditTrail(on_record=lambda r: captured.append(r))
        trail.record_action_requested(sample_request)
        assert len(captured) == 1
        assert captured[0].event_type == EventType.ACTION_REQUESTED

    def test_outcome_recording(self, trail):
        trail.record_outcome(
            action_id="a1", agent_id="agent", tool_name="tx.swap",
            outcome_severity=0.3, description="Minor slippage",
        )
        outcomes = trail.get_records_by_type(EventType.OUTCOME_RECORDED)
        assert len(outcomes) == 1
        assert outcomes[0].data["outcome_severity"] == 0.3


class TestExecutionResultCap:
    """Loop 53: record_execution result dict capped to prevent audit-chain DoS."""

    def test_huge_result_dict_is_capped(self, trail):
        # 2MB payload under a single key
        huge_payload = "x" * (2 * 1024 * 1024)
        trail.record_execution(
            action_id="a1", agent_id="agent", tool_name="tx.swap",
            result={"payload": huge_payload, "status": "ok"},
        )
        records = trail.get_records_by_type(EventType.ACTION_EXECUTED)
        assert len(records) == 1
        summary = records[0].data["result_summary"]
        assert summary.get("_truncated") is True
        assert summary["_original_bytes"] > 2 * 1024 * 1024
        assert summary["_cap_bytes"] == AuditTrail._MAX_EXECUTION_RESULT_JSON_BYTES
        assert "payload" in summary["_keys"]
        # serialized record stays small
        assert len(json.dumps(records[0].data, default=str)) < 10 * 1024

    def test_small_result_dict_passes_through(self, trail):
        trail.record_execution(
            action_id="a2", agent_id="agent", tool_name="tx.swap",
            result={"tx_hash": "0xabc", "gas": 21000},
        )
        records = trail.get_records_by_type(EventType.ACTION_EXECUTED)
        assert records[0].data["result_summary"] == {"tx_hash": "0xabc", "gas": 21000}


class TestRequestDictCap:
    """Loop 54: record_action_requested caps parameters/context at trail API."""

    def _make_request(self, *, parameters=None, context=None):
        at = ActionType(
            "tx.swap", ActionCategory.FINANCIAL, Reversibility.IRREVERSIBLE,
            BlastRadius.SHARED, UrgencyClass.IRREVOCABLE,
            frozenset({RegulatoryDomain.EU_AI_ACT}),
        )
        return ActionRequest(
            agent_id="agent-1",
            tool_name="tx.swap",
            action_type=at,
            parameters=parameters or {},
            context=context or {},
        )

    def test_huge_parameters_dict_capped_at_trail_api(self, trail):
        huge = {"payload": "y" * (2 * 1024 * 1024), "k": "v"}
        req = self._make_request(parameters=huge)
        trail.record_action_requested(req)
        records = trail.get_records_by_type(EventType.ACTION_REQUESTED)
        params = records[0].data["parameters"]
        assert params.get("_truncated") is True
        assert params["_original_bytes"] > 2 * 1024 * 1024
        assert "payload" in params["_keys"]

    def test_huge_context_dict_capped_at_trail_api(self, trail):
        huge_ctx = {"trace": "z" * (2 * 1024 * 1024)}
        req = self._make_request(context=huge_ctx)
        trail.record_action_requested(req)
        records = trail.get_records_by_type(EventType.ACTION_REQUESTED)
        ctx = records[0].data["context"]
        assert ctx.get("_truncated") is True
        assert "trace" in ctx["_keys"]

    def test_small_request_dicts_pass_through(self, trail):
        req = self._make_request(parameters={"to": "0xabc"}, context={"chain": "base"})
        trail.record_action_requested(req)
        records = trail.get_records_by_type(EventType.ACTION_REQUESTED)
        assert records[0].data["parameters"] == {"to": "0xabc"}
        assert records[0].data["context"] == {"chain": "base"}


class TestRequestStringCap:
    """Loop 55: record_action_requested caps agent_id/tool_name/session_id/parent_action_id."""

    def _make(self, **overrides):
        at = ActionType(
            "tx.swap", ActionCategory.FINANCIAL, Reversibility.IRREVERSIBLE,
            BlastRadius.SHARED, UrgencyClass.IRREVOCABLE,
            frozenset({RegulatoryDomain.EU_AI_ACT}),
        )
        kwargs = dict(agent_id="agent-1", tool_name="tx.swap", action_type=at)
        kwargs.update(overrides)
        return ActionRequest(**kwargs)

    def test_huge_agent_id_capped(self, trail):
        trail.record_action_requested(self._make(agent_id="x" * 10000))
        rec = trail.get_records_by_type(EventType.ACTION_REQUESTED)[0]
        assert len(rec.agent_id) <= AuditTrail._MAX_AGENT_ID_LEN
        assert "TRUNCATED:10000B" in rec.agent_id

    def test_huge_tool_name_capped(self, trail):
        trail.record_action_requested(self._make(tool_name="y" * 20000))
        rec = trail.get_records_by_type(EventType.ACTION_REQUESTED)[0]
        assert len(rec.tool_name) <= AuditTrail._MAX_TOOL_NAME_LEN

    def test_huge_session_id_capped(self, trail):
        trail.record_action_requested(self._make(session_id="z" * 10000))
        rec = trail.get_records_by_type(EventType.ACTION_REQUESTED)[0]
        assert len(rec.data["session_id"]) <= AuditTrail._MAX_SESSION_ID_LEN

    def test_huge_parent_action_id_capped(self, trail):
        trail.record_action_requested(self._make(parent_action_id="w" * 10000))
        rec = trail.get_records_by_type(EventType.ACTION_REQUESTED)[0]
        assert len(rec.data["parent_action_id"]) <= AuditTrail._MAX_PARENT_ACTION_ID_LEN

    def test_default_session_and_parent_pass_through(self, trail):
        trail.record_action_requested(self._make())
        rec = trail.get_records_by_type(EventType.ACTION_REQUESTED)[0]
        assert rec.data["session_id"] == ""
        assert rec.data["parent_action_id"] is None


class TestRiskAssessmentCap:
    """Loop 56: record_risk_scored assessment dict capped at trail boundary."""

    def test_huge_signals_dict_capped(self, trail):
        huge = {"feature_x": "q" * (2 * 1024 * 1024), "other": 1.0}
        trail.record_risk_scored(
            action_id="a1", agent_id="agent", tool_name="tx.swap",
            assessment={"point_estimate": 0.5, "signals": huge},
        )
        rec = trail.get_records_by_type(EventType.RISK_SCORED)[0]
        # Either the whole assessment or signals dict gets replaced with marker
        serialized_size = len(json.dumps(rec.data, default=str))
        assert serialized_size < 100 * 1024  # much smaller than 2MB

    def test_huge_flat_assessment_capped(self, trail):
        huge_signals = {f"f{i}": "z" * 1000 for i in range(200)}  # ~200KB
        trail.record_risk_scored(
            action_id="a2", agent_id="agent", tool_name="tx.swap",
            assessment=huge_signals,
        )
        rec = trail.get_records_by_type(EventType.RISK_SCORED)[0]
        assert rec.data.get("_truncated") is True
        assert rec.data["_original_bytes"] > 100 * 1024

    def test_small_assessment_passes_through(self, trail):
        trail.record_risk_scored(
            action_id="a3", agent_id="agent", tool_name="tx.swap",
            assessment={"point_estimate": 0.4, "conformal_lower": 0.2,
                        "conformal_upper": 0.6},
        )
        rec = trail.get_records_by_type(EventType.RISK_SCORED)[0]
        assert rec.data["point_estimate"] == 0.4
        assert rec.data["conformal_lower"] == 0.2


class TestTrailBoundaryStringCaps:
    """Loop 57: outcome/escalation/escalation_resolved string fields capped at trail boundary."""

    def test_outcome_description_capped(self, trail):
        trail.record_outcome(
            action_id="a1", agent_id="a", tool_name="t",
            outcome_severity=0.3, description="d" * 20000,
        )
        rec = trail.get_records_by_type(EventType.OUTCOME_RECORDED)[0]
        assert len(rec.data["description"]) <= AuditTrail._MAX_OUTCOME_DESCRIPTION_LEN

    def test_outcome_agent_id_and_tool_name_capped(self, trail):
        trail.record_outcome(
            action_id="a2", agent_id="x" * 10000, tool_name="y" * 10000,
            outcome_severity=0.1,
        )
        rec = trail.get_records_by_type(EventType.OUTCOME_RECORDED)[0]
        assert len(rec.agent_id) <= AuditTrail._MAX_AGENT_ID_LEN
        assert len(rec.tool_name) <= AuditTrail._MAX_TOOL_NAME_LEN

    def test_escalation_target_capped(self, trail):
        trail.record_escalation(
            action_id="a3", agent_id="a", tool_name="t",
            escalation_target="human:" + "z" * 10000, risk_score=0.8,
        )
        rec = trail.get_records_by_type(EventType.ESCALATION_SENT)[0]
        assert len(rec.data["escalation_target"]) <= AuditTrail._MAX_ESCALATION_TARGET_LEN

    def test_escalation_resolved_strings_capped(self, trail):
        trail.record_escalation_resolved(
            action_id="a4", agent_id="a", tool_name="t",
            resolution="approve" * 1000,
            reviewer="reviewer@" + "x" * 10000,
            justification="j" * 50000,
        )
        rec = trail.get_records_by_type(EventType.ESCALATION_RESOLVED)[0]
        assert len(rec.data["resolution"]) <= AuditTrail._MAX_RESOLUTION_LEN
        assert len(rec.data["reviewer"]) <= AuditTrail._MAX_REVIEWER_LEN
        assert len(rec.data["justification"]) <= AuditTrail._MAX_JUSTIFICATION_LEN


class TestTrailBoundarySweep:
    """Loop 58: complete agent_id/tool_name + decision/reason sweep at trail boundary."""

    def test_risk_scored_agent_tool_capped(self, trail):
        trail.record_risk_scored(
            action_id="a1", agent_id="x" * 10000, tool_name="y" * 10000,
            assessment={"point_estimate": 0.5},
        )
        rec = trail.get_records_by_type(EventType.RISK_SCORED)[0]
        assert len(rec.agent_id) <= AuditTrail._MAX_AGENT_ID_LEN
        assert len(rec.tool_name) <= AuditTrail._MAX_TOOL_NAME_LEN

    def test_decision_reason_and_label_capped(self, trail):
        trail.record_decision(
            action_id="a2", agent_id="a", tool_name="t",
            decision="custom_decision_label_" * 10, reason="r" * 20000,
            risk_score=0.4,
        )
        rec = trail.get_records_by_type(EventType.DECISION_MADE)[0]
        assert len(rec.data["decision"]) <= AuditTrail._MAX_DECISION_LABEL_LEN
        assert len(rec.data["reason"]) <= AuditTrail._MAX_DECISION_REASON_LEN

    def test_decision_agent_tool_capped(self, trail):
        trail.record_decision(
            action_id="a3", agent_id="x" * 10000, tool_name="y" * 10000,
            decision="allow", reason="ok", risk_score=0.1,
        )
        rec = trail.get_records_by_type(EventType.DECISION_MADE)[0]
        assert len(rec.agent_id) <= AuditTrail._MAX_AGENT_ID_LEN
        assert len(rec.tool_name) <= AuditTrail._MAX_TOOL_NAME_LEN

    def test_execution_agent_tool_capped(self, trail):
        trail.record_execution(
            action_id="a4", agent_id="x" * 10000, tool_name="y" * 10000,
            result={"ok": True},
        )
        rec = trail.get_records_by_type(EventType.ACTION_EXECUTED)[0]
        assert len(rec.agent_id) <= AuditTrail._MAX_AGENT_ID_LEN
        assert len(rec.tool_name) <= AuditTrail._MAX_TOOL_NAME_LEN

    def test_policy_override_agent_tool_capped(self, trail):
        trail.record_policy_override(
            action_id="a5", agent_id="x" * 10000, tool_name="y" * 10000,
            override_reason="ok", overrider="op",
            original_decision="deny", new_decision="allow",
        )
        rec = trail.get_records_by_type(EventType.POLICY_OVERRIDE)[0]
        assert len(rec.agent_id) <= AuditTrail._MAX_AGENT_ID_LEN
        assert len(rec.tool_name) <= AuditTrail._MAX_TOOL_NAME_LEN


class TestRegulatoryMappings:
    def test_eu_ai_act_mappings_cover_key_events(self):
        assert EventType.ACTION_REQUESTED in EU_AI_ACT_MAPPINGS
        assert EventType.RISK_SCORED in EU_AI_ACT_MAPPINGS
        assert EventType.DECISION_MADE in EU_AI_ACT_MAPPINGS
        assert EventType.OUTCOME_RECORDED in EU_AI_ACT_MAPPINGS

    def test_all_mappings_have_required_fields(self):
        for event_type, articles in EU_AI_ACT_MAPPINGS.items():
            for article in articles:
                assert article.domain == RegulatoryDomain.EU_AI_ACT
                assert article.article  # Non-empty
                assert article.requirement  # Non-empty
                assert article.how_satisfied  # Non-empty
