"""Interception pipeline — the main entry point for the Vaara execution layer.

Wires the components together:

    ActionRequest → Registry (classify) → Scorer (risk) → Policy (decide)
                                                       → Audit (record)
                                                       → Execute or Block

Usage::

    from vaara.pipeline import InterceptionPipeline

    pipeline = InterceptionPipeline()

    # Agent wants to execute a tool
    result = pipeline.intercept(
        agent_id="agent-007",
        tool_name="tx.transfer",
        parameters={"to": "0x...", "amount": 1000},
        agent_confidence=0.8,
    )

    if result.allowed:
        # Execute the action
        actual_result = execute_tool(tool_name, parameters)
        # Report outcome for learning
        pipeline.report_outcome(result.action_id, outcome_severity=0.0)
    else:
        print(f"Blocked: {result.reason}")
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

from vaara._sanitize import json_safe
from vaara.audit.trail import AuditTrail
from vaara.compliance.engine import ComplianceEngine, ConformityReport
from vaara.scorer.adaptive import AdaptiveScorer
from vaara.taxonomy.actions import (
    ActionRegistry,
    ActionRequest,
    ActionType,
    create_default_registry,
)

logger = logging.getLogger(__name__)

# Cap on pending-outcome records. Callers that never invoke
# report_outcome (crashes, fire-and-forget loops, streaming aborts)
# would otherwise grow the dict unboundedly across a long-running
# agent session. Evicting the oldest entry keeps memory bounded;
# late report_outcome calls after eviction no-op with a warning.
_MAX_PENDING_OUTCOMES = 50_000


# Length caps on caller-supplied string fields at the pipeline ingress.
# Without caps, a misconfigured / malicious / buggy agent can submit
# a tool_name, agent_id, or session_id of arbitrary size. Each value
# is duplicated across 3–4 audit records per intercept (action_requested,
# risk_scored, decision_made, optionally escalation_sent), so a 50MB
# tool_name produces ~200MB of audit trail per single intercept. That
# poisons SHA256 hashing (O(n) per record), JSONL export, compliance
# report assembly, and any regulator-facing log reader.
# Real legitimate values are measured in bytes (tx.transfer, agent-007).
# The caps here are deliberately generous so unusual but valid inputs
# pass through; anything beyond is truncated with a marker so the
# governance invariant (every action produces an audit record) still
# holds, and the regulator sees evidence the input was oversized.
_MAX_AGENT_ID_LEN = 256
_MAX_TOOL_NAME_LEN = 512
_MAX_SESSION_ID_LEN = 256
_MAX_PARENT_ACTION_ID_LEN = 128
# JSON-serialized bytes. Real tool parameters are small (an address, an
# amount, a file path). A 50MB blob in params is either a bug or an
# attack. Over cap → replace entire dict with a truncation marker so
# the audit record is still produced and the hash chain stays intact.
_MAX_PARAMS_JSON_BYTES = 64 * 1024
_MAX_CONTEXT_JSON_BYTES = 64 * 1024
# Free-text fields at post-decision ingress. Same amplification
# concern as Loop 47: these strings land on the hash chain, narrative,
# export, and compliance report. Outcome description is bounded small
# because it's a short human note; reviewer/overrider identifiers are
# capped like agent_id; justification/override_reason are larger to
# accommodate legitimate human rationale but still bounded.
_MAX_DECISION_REASON_LEN = 4096
_MAX_OUTCOME_DESCRIPTION_LEN = 4096
_MAX_REVIEWER_LEN = 256
_MAX_JUSTIFICATION_LEN = 8192
_MAX_OVERRIDER_LEN = 256
_MAX_OVERRIDE_REASON_LEN = 8192
_MAX_DECISION_LABEL_LEN = 64


def _cap_str(value: Any, max_len: int, field_name: str) -> str:
    """Truncate a caller-supplied string to max_len with a TRUNCATED marker.

    Non-string inputs are coerced via str() — existing pipeline code
    already tolerates non-string leaks, so this preserves behaviour while
    adding the length cap. Returns a safe (capped) string so the audit
    chain still records the (oversized) attempt rather than rejecting.
    """
    if not isinstance(value, str):
        value = str(value) if value is not None else ""
    if len(value) <= max_len:
        return value
    logger.warning(
        "pipeline: %s length=%d exceeds cap=%d; truncating",
        field_name, len(value), max_len,
    )
    marker = f"...[TRUNCATED:{len(value)}B]"
    keep = max_len - len(marker)
    if keep <= 0:
        return value[:max_len]
    return value[:keep] + marker


def _json_safe_dict(d: Optional[dict]) -> dict:
    if not d:
        return {}
    return {str(k): json_safe(v) for k, v in d.items()}


def _cap_dict_bytes(d: dict, max_bytes: int, field_name: str) -> dict:
    """Cap a sanitised dict's JSON size. Over cap → single-key marker.

    Sanitisation (json_safe) runs first so this only sees values json.dumps
    can encode. We cheap-estimate by trying the dump and checking len —
    no full-walk traversal needed. If the caller passes a 50MB parameters
    blob, replace the whole dict with `{"_truncated": true, ...}` so the
    audit record still captures the shape of the attempt without poisoning
    the hash chain with MBs of attacker-controlled data.
    """
    if not d:
        return d
    try:
        import json as _json
        encoded = _json.dumps(d, default=str)
    except (TypeError, ValueError):
        # Shouldn't happen post-json_safe; err on the side of dropping.
        return {"_truncated": True, "_reason": "unserializable"}
    if len(encoded) <= max_bytes:
        return d
    logger.warning(
        "pipeline: %s JSON size=%d exceeds cap=%d; replacing with marker",
        field_name, len(encoded), max_bytes,
    )
    return {
        "_truncated": True,
        "_original_bytes": len(encoded),
        "_cap_bytes": max_bytes,
        "_keys": sorted(d.keys())[:20],  # Keep a hint of what was there
    }


@dataclass
class PipelineMetrics:
    """Live counters for operations dashboards and health checks."""
    total_intercepts: int = 0
    allowed: int = 0
    denied: int = 0
    escalated: int = 0
    scorer_failures: int = 0
    total_outcome_reports: int = 0
    _total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_intercepts == 0:
            return 0.0
        return round(self._total_latency_ms / self.total_intercepts, 2)

    def to_dict(self) -> dict:
        return {
            "total_intercepts": self.total_intercepts,
            "allowed": self.allowed,
            "denied": self.denied,
            "escalated": self.escalated,
            "scorer_failures": self.scorer_failures,
            "total_outcome_reports": self.total_outcome_reports,
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class InterceptionResult:
    """Result of intercepting an action request."""
    allowed: bool
    action_id: str              # For tracking outcome later
    decision: str               # "allow", "deny", "escalate"
    risk_score: float           # Point estimate
    risk_interval: tuple[float, float]  # Conformal (lower, upper)
    reason: str
    action_type: ActionType
    signals: dict[str, float]   # Contributing risk signals
    evaluation_ms: float


class InterceptionPipeline:
    """Main entry point — intercepts agent actions with risk scoring and audit.

    Thread-safe for read operations.  Write operations (intercept, report_outcome)
    should be serialized per agent_id in concurrent environments.
    """

    def __init__(
        self,
        registry: Optional[ActionRegistry] = None,
        scorer: Optional[AdaptiveScorer] = None,
        trail: Optional[AuditTrail] = None,
        compliance: Optional[ComplianceEngine] = None,
    ) -> None:
        self.registry = registry or create_default_registry()
        self.scorer = scorer or AdaptiveScorer()
        self.trail = trail or AuditTrail()
        self.compliance = compliance or ComplianceEngine()

        # Track action_id → (predicted_risk, signals) for outcome feedback.
        # OrderedDict + bounded FIFO eviction — see _MAX_PENDING_OUTCOMES.
        self._pending_outcomes: OrderedDict[
            str, tuple[float, dict[str, float]]
        ] = OrderedDict()
        self._pending_outcomes_lock = threading.Lock()
        self._metrics = PipelineMetrics()
        self._metrics_lock = threading.Lock()

    def intercept(
        self,
        agent_id: str,
        tool_name: str,
        parameters: Optional[dict] = None,
        context: Optional[dict] = None,
        agent_confidence: Optional[float] = None,
        session_id: str = "",
        parent_action_id: Optional[str] = None,
        sequence_position: int = 0,
    ) -> InterceptionResult:
        """Intercept an agent action request.

        This is the single function that agents/frameworks call.
        It classifies, scores, decides, and audits in one shot.

        Returns an InterceptionResult — check .allowed before executing.
        """
        start = time.monotonic()

        # Cap oversized caller-supplied strings BEFORE classify/audit so
        # a 50MB tool_name cannot land in the hash chain, export, or
        # compliance report. See _MAX_*_LEN constants for rationale.
        agent_id = _cap_str(agent_id, _MAX_AGENT_ID_LEN, "agent_id")
        tool_name = _cap_str(tool_name, _MAX_TOOL_NAME_LEN, "tool_name")
        session_id = _cap_str(session_id, _MAX_SESSION_ID_LEN, "session_id")
        if parent_action_id is not None:
            parent_action_id = _cap_str(
                parent_action_id, _MAX_PARENT_ACTION_ID_LEN, "parent_action_id"
            )

        # 1. Classify the action
        action_type = self.registry.classify(tool_name, parameters)

        # Sanitize parameters/context at the boundary so the audit-trail
        # hash chain can be computed with strict JSON. Integrations also
        # sanitize, but direct pipeline users can still pass datetimes,
        # bytes, dataclasses — json_safe keeps those from crashing the
        # audit trail without mutating the hash input downstream.
        safe_params = _cap_dict_bytes(
            _json_safe_dict(parameters), _MAX_PARAMS_JSON_BYTES, "parameters"
        )
        safe_context = _cap_dict_bytes(
            _json_safe_dict(context), _MAX_CONTEXT_JSON_BYTES, "context"
        )

        # 2. Build the request envelope
        request = ActionRequest(
            agent_id=agent_id,
            tool_name=tool_name,
            action_type=action_type,
            parameters=safe_params,
            context=safe_context,
            confidence=agent_confidence,
            session_id=session_id,
            parent_action_id=parent_action_id,
            sequence_position=sequence_position,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        # 3. Record the request in audit trail
        action_id = self.trail.record_action_requested(request)

        # 4. Score the risk. If the scorer raises — corrupt model bundle,
        # feature pipeline exception, any library bug — we've already
        # written an `action_requested` record at step 3 without a
        # matching `decision`. A regulator auditing Article 14 oversight
        # would then see an orphan request they cannot classify: allowed
        # silently, denied silently, or crashed silently are all
        # indistinguishable. Fail CLOSED: write a `decision=deny` with
        # an explicit "scorer failure" reason so the hash chain stays
        # consistent, then re-raise so the caller knows the governance
        # system broke rather than that the action is simply high-risk.
        policy_context = request.to_policy_context()
        try:
            scorer_result = self.scorer.evaluate(policy_context)
        except Exception as exc:
            with self._metrics_lock:
                self._metrics.scorer_failures += 1
            logger.exception(
                "Scorer evaluate failed for action_id=%s; recording fail-closed deny",
                action_id,
            )
            try:
                # Exception str() can be arbitrarily large — a plugin
                # scorer raising with a multi-megabyte payload would
                # otherwise balloon the fail-closed audit record.
                fail_reason = _cap_str(
                    f"scorer failure: {type(exc).__name__}: {exc}",
                    _MAX_DECISION_REASON_LEN,
                    "reason",
                )
                self.trail.record_decision(
                    action_id=action_id,
                    agent_id=agent_id,
                    tool_name=tool_name,
                    decision="deny",
                    reason=fail_reason,
                    risk_score=1.0,
                    regulatory_domains=action_type.regulatory_domains,
                )
            except Exception:
                logger.exception(
                    "Failed to record fail-closed deny for action_id=%s; "
                    "audit trail may be inconsistent",
                    action_id,
                )
            raise

        # Shape-validate the scorer result. A backend returning a
        # non-dict result, a non-dict raw_result, a scalar/None/wrong-
        # length conformal_interval, or non-finite point_estimate must
        # not crash the pipeline mid-intercept (same orphan-record risk
        # as the exception path above). Coerce any malformed field to
        # a safe neutral value and log a warning — fail-closed decision
        # is still honoured by the scorer's own decision string, which
        # defaults to "deny" at the .get() call below.
        if not isinstance(scorer_result, dict):
            logger.warning(
                "Scorer returned non-dict result (%r) for action_id=%s; "
                "treating as empty result",
                type(scorer_result).__name__, action_id,
            )
            scorer_result = {}

        raw = scorer_result.get("raw_result", {})
        if not isinstance(raw, dict):
            logger.warning(
                "Scorer raw_result is %r (not dict) for action_id=%s; "
                "using empty dict",
                type(raw).__name__, action_id,
            )
            raw = {}

        point_estimate = raw.get("point_estimate", 0.5)
        try:
            point_estimate = float(point_estimate)
        except (TypeError, ValueError):
            logger.warning(
                "Non-numeric point_estimate=%r for action_id=%s; defaulting 0.5",
                point_estimate, action_id,
            )
            point_estimate = 0.5
        import math as _math
        if not _math.isfinite(point_estimate):
            logger.warning(
                "Non-finite point_estimate=%s for action_id=%s; defaulting 0.5",
                point_estimate, action_id,
            )
            point_estimate = 0.5
        # Clamp defensively — RiskAssessment already does this for its own
        # path, but custom scorer backends bypass RiskAssessment.
        point_estimate = max(0.0, min(1.0, point_estimate))

        interval_raw = raw.get("conformal_interval", [0.2, 0.8])
        if (not isinstance(interval_raw, (list, tuple))
                or len(interval_raw) != 2):
            logger.warning(
                "Malformed conformal_interval=%r for action_id=%s; "
                "defaulting [0.2, 0.8]",
                interval_raw, action_id,
            )
            interval = [0.2, 0.8]
        else:
            try:
                lo = float(interval_raw[0])
                hi = float(interval_raw[1])
            except (TypeError, ValueError):
                logger.warning(
                    "Non-numeric conformal_interval=%r for action_id=%s; "
                    "defaulting [0.2, 0.8]",
                    interval_raw, action_id,
                )
                lo, hi = 0.2, 0.8
            if not _math.isfinite(lo) or not _math.isfinite(hi):
                logger.warning(
                    "Non-finite conformal_interval=%r for action_id=%s; "
                    "defaulting [0.2, 0.8]",
                    interval_raw, action_id,
                )
                lo, hi = 0.2, 0.8
            if lo > hi:
                lo, hi = hi, lo
            interval = [max(0.0, min(1.0, lo)), max(0.0, min(1.0, hi))]

        signals = raw.get("signals", {})
        if not isinstance(signals, dict):
            signals = {}

        # 5. Record risk scoring in audit
        self.trail.record_risk_scored(
            action_id=action_id,
            agent_id=agent_id,
            tool_name=tool_name,
            assessment={
                "point_estimate": point_estimate,
                "conformal_lower": interval[0],
                "conformal_upper": interval[1],
                "signals": signals,
                "calibration_size": raw.get("calibration_size", 0),
            },
            regulatory_domains=action_type.regulatory_domains,
        )

        # 6. Extract decision. `allowed` MUST be derived from the canonical
        # decision string — trusting the backend's independent `allowed`
        # field lets a malformed/buggy scorer return allowed=True with
        # action="deny" (or vice versa), so the caller executes an action
        # while the audit trail records ACTION_BLOCKED. That is the worst-
        # case governance failure: the record and the behaviour disagree.
        # Normalise to a known vocabulary; anything unrecognised fails
        # closed to "deny" — this preserves the pipeline's fail-closed
        # invariant established on the scorer-exception path above.
        raw_decision = scorer_result.get("action", "deny")
        if isinstance(raw_decision, str):
            decision_str = raw_decision.strip().lower()
        else:
            decision_str = "deny"
        if decision_str not in ("allow", "deny", "escalate"):
            logger.warning(
                "Scorer returned unknown action=%r for action_id=%s; "
                "failing closed to 'deny'",
                raw_decision, action_id,
            )
            decision_str = "deny"
        allowed = decision_str == "allow"
        reason = scorer_result.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)
        # Custom scorer backends are pluggable — treat their `reason`
        # output as a caller-controlled ingress point like any other
        # string that lands on the hash chain (see Loop 47/50 caps).
        reason = _cap_str(reason, _MAX_DECISION_REASON_LEN, "reason")

        # 7. Record the decision in audit
        self.trail.record_decision(
            action_id=action_id,
            agent_id=agent_id,
            tool_name=tool_name,
            decision=decision_str,
            reason=reason,
            risk_score=point_estimate,
            regulatory_domains=action_type.regulatory_domains,
        )

        # 8. If escalated, record escalation
        if decision_str == "escalate":
            self.trail.record_escalation(
                action_id=action_id,
                agent_id=agent_id,
                tool_name=tool_name,
                escalation_target="human_reviewer",
                risk_score=point_estimate,
            )

        # 9. Store for outcome feedback with bounded FIFO eviction
        with self._pending_outcomes_lock:
            self._pending_outcomes[action_id] = (point_estimate, signals)
            if len(self._pending_outcomes) > _MAX_PENDING_OUTCOMES:
                self._pending_outcomes.popitem(last=False)

        elapsed_ms = (time.monotonic() - start) * 1000

        with self._metrics_lock:
            self._metrics.total_intercepts += 1
            self._metrics._total_latency_ms += elapsed_ms
            if decision_str == "allow":
                self._metrics.allowed += 1
            elif decision_str == "deny":
                self._metrics.denied += 1
            else:
                self._metrics.escalated += 1

        logger.info(
            "Intercepted %s/%s: %s (risk=%.3f [%.3f,%.3f], %.1fms)",
            agent_id, tool_name, decision_str,
            point_estimate, interval[0], interval[1], elapsed_ms,
        )

        return InterceptionResult(
            allowed=allowed,
            action_id=action_id,
            decision=decision_str,
            risk_score=point_estimate,
            risk_interval=(interval[0], interval[1]),
            reason=reason,
            action_type=action_type,
            signals=signals,
            evaluation_ms=elapsed_ms,
        )

    def report_outcome(
        self,
        action_id: str,
        outcome_severity: float,
        description: str = "",
    ) -> None:
        """Report the actual outcome of an executed action.

        Call this after action execution with the observed severity.
        0.0 = completely safe, 1.0 = catastrophic.

        This closes the feedback loop:
        - Updates MWU expert weights in the scorer
        - Updates conformal calibration set
        - Records outcome in the audit trail
        """
        # Peek rather than pop: if the audit-trail lookup below fails
        # (transient backend error, partially-initialised trail on reload,
        # caller race), a pop-first ordering destroys the pending entry so
        # a retry with the same action_id becomes a silent no-op and the
        # MWU/conformal learning signal is lost permanently. Only remove
        # the entry once we've confirmed we can complete the outcome write.
        with self._pending_outcomes_lock:
            pending = self._pending_outcomes.get(action_id)
        if pending is None:
            logger.warning("No pending outcome for action_id=%s", action_id)
            return

        # Clamp severity to [0, 1]. MWU and conformal both assume this
        # range; out-of-range values would corrupt weight updates and
        # calibration residuals. Coerce to float up-front so a caller
        # that passes a string ("high"), bool, or NaN can't raise a
        # TypeError inside the comparison — or silently poison the
        # calibration set with a non-finite outcome.
        import math
        try:
            outcome_severity = float(outcome_severity)
        except (TypeError, ValueError):
            logger.warning(
                "Non-numeric outcome_severity=%r for action_id=%s; "
                "defaulting to 0.0",
                outcome_severity, action_id,
            )
            outcome_severity = 0.0
        if not math.isfinite(outcome_severity):
            logger.warning(
                "Non-finite outcome_severity=%s for action_id=%s; "
                "defaulting to 0.0",
                outcome_severity, action_id,
            )
            outcome_severity = 0.0
        if not (0.0 <= outcome_severity <= 1.0):
            logger.warning(
                "Clamping outcome_severity=%s to [0, 1] for action_id=%s",
                outcome_severity, action_id,
            )
            outcome_severity = max(0.0, min(1.0, outcome_severity))

        predicted_risk, signals = pending

        # Get agent_id and tool_name from the audit trail
        trail = self.trail.get_action_trail(action_id)
        if not trail:
            logger.warning("No audit trail for action_id=%s", action_id)
            return

        agent_id = trail[0].agent_id
        tool_name = trail[0].tool_name

        # Ordering matters for crash/error recovery:
        #   1. Trail write is authoritative for regulators — record it
        #      first so a later scorer failure never strands an outcome
        #      out of the audit log.
        #   2. Scorer update is best-effort — if it raises (corrupt
        #      bundle, malformed signals), log and continue; the audit
        #      record already exists and MWU divergence is advisory.
        #   3. Pop last so that a failure on step 1 leaves the pending
        #      entry in place for a retry. Popping earlier risks losing
        #      both the learning signal and the audit record.
        # Pre-pop-only-after-confirm (Loop 36) kept the pending entry
        # safe on trail-read failure; this adds the same discipline on
        # trail-WRITE failure and scorer-update failure.
        description = _cap_str(
            description, _MAX_OUTCOME_DESCRIPTION_LEN, "description"
        )

        try:
            self.trail.record_outcome(
                action_id=action_id,
                agent_id=agent_id,
                tool_name=tool_name,
                outcome_severity=outcome_severity,
                description=description,
            )
        except Exception:
            logger.exception(
                "trail.record_outcome failed for action_id=%s; "
                "pending entry preserved for retry",
                action_id,
            )
            return

        try:
            self.scorer.record_outcome(
                agent_id=agent_id,
                tool_name=tool_name,
                predicted_risk=predicted_risk,
                actual_outcome=outcome_severity,
                signals=signals,
            )
        except Exception:
            logger.exception(
                "scorer.record_outcome failed for action_id=%s; "
                "audit record already written — learning signal dropped",
                action_id,
            )

        # Both writes attempted — pop regardless. A second
        # report_outcome call after a successful trail write would
        # otherwise append a duplicate OUTCOME_RECORDED row, inflating
        # Article 61(1) post-market monitoring evidence.
        with self._pending_outcomes_lock:
            self._pending_outcomes.pop(action_id, None)
        with self._metrics_lock:
            self._metrics.total_outcome_reports += 1

    def resolve_escalation(
        self,
        action_id: str,
        resolution: str,
        reviewer: str,
        justification: str = "",
    ) -> None:
        """Record human resolution of an escalated action.

        Args:
            action_id: The action that was escalated.
            resolution: "allow" or "deny".
            reviewer: Who made the decision.
            justification: Why.
        """
        # Narrow the resolution to the two values downstream compliance
        # queries assume. Unchecked, a typo like "alllow" or a caller
        # who passes an Enum instance (e.g., Decision.ALLOW rather than
        # "allow") would be written to the audit trail verbatim and silently
        # misclassify the outcome during regulatory evidence assembly.
        if not isinstance(resolution, str):
            logger.warning(
                "resolve_escalation: non-string resolution=%r for action_id=%s; coercing via str()",
                resolution, action_id,
            )
            resolution = str(resolution)
        normalized = resolution.strip().lower()
        if normalized not in ("allow", "deny"):
            logger.warning(
                "resolve_escalation: unexpected resolution=%r for action_id=%s; "
                "expected 'allow' or 'deny'. Recording as-is.",
                resolution, action_id,
            )
        else:
            resolution = normalized

        trail = self.trail.get_action_trail(action_id)
        if not trail:
            logger.warning("No audit trail for action_id=%s", action_id)
            return

        # Require a prior ESCALATION_SENT and reject duplicate resolutions.
        # Without these guards an ESCALATION_RESOLVED record could appear
        # without a matching ESCALATION_SENT (e.g., caller misuse, race
        # where the initial decision was allow/deny), or a second
        # resolution could silently overwrite the first in auditor-facing
        # narratives. Both break Article 14(3)(a) evidence — the regulator
        # dashboard assumes 1-to-1 sent/resolved pairing.
        from vaara.audit.trail import EventType  # local import: avoid cycle
        has_escalation = any(
            r.event_type == EventType.ESCALATION_SENT for r in trail
        )
        already_resolved = any(
            r.event_type == EventType.ESCALATION_RESOLVED for r in trail
        )
        if not has_escalation:
            logger.warning(
                "resolve_escalation called for action_id=%s that was never "
                "escalated — ignoring to preserve trail integrity",
                action_id,
            )
            return
        if already_resolved:
            logger.warning(
                "resolve_escalation called twice for action_id=%s — ignoring "
                "the second resolution; record a POLICY_OVERRIDE instead to "
                "reverse a prior resolution",
                action_id,
            )
            return

        agent_id = trail[0].agent_id
        tool_name = trail[0].tool_name

        # Cap free-text fields: resolution is already normalised to
        # allow/deny above but a stray non-standard string gets a small
        # label cap so it can't balloon the audit record. reviewer and
        # justification are caller-controlled at the pipeline boundary
        # and reach the hash chain like every other field — same L47
        # amplification concern applies here.
        resolution = _cap_str(resolution, _MAX_DECISION_LABEL_LEN, "resolution")
        reviewer = _cap_str(reviewer, _MAX_REVIEWER_LEN, "reviewer")
        justification = _cap_str(
            justification, _MAX_JUSTIFICATION_LEN, "justification"
        )

        self.trail.record_escalation_resolved(
            action_id=action_id,
            agent_id=agent_id,
            tool_name=tool_name,
            resolution=resolution,
            reviewer=reviewer,
            justification=justification,
        )

    def run_compliance_assessment(
        self,
        system_name: str = "Vaara Execution Layer",
        system_version: str = "0.1.0",
    ) -> ConformityReport:
        """Run a compliance assessment against the current audit trail."""
        return self.compliance.assess(
            self.trail,
            system_name=system_name,
            system_version=system_version,
        )

    def status(self) -> dict[str, Any]:
        """Dashboard-friendly status snapshot.

        `trail_persistence_failures` surfaces the Loop 35 counter so health
        endpoints / Article 12(2) monitoring dashboards can observe live
        divergence between the in-memory chain and the persistent backend
        without parsing logs. A non-zero value means at least one on_record
        callback has raised since trail creation.
        """
        return {
            "scorer": self.scorer.status(),
            "trail_size": self.trail.size,
            "trail_chain_intact": self.trail.chain_intact,
            "trail_persistence_failures": self.trail.persistence_failures,
            "trail_skeleton_records": self.trail.skeleton_records,
            "pending_outcomes": len(self._pending_outcomes),
            "registered_action_types": len(self.registry.all_types),
            "metrics": self.metrics().to_dict(),
        }

    def metrics(self) -> PipelineMetrics:
        """Return a snapshot of runtime counters (thread-safe copy)."""
        with self._metrics_lock:
            import copy
            return copy.copy(self._metrics)
