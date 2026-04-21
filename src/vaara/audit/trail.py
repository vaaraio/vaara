"""Immutable audit trail with regulatory article mapping.

Every action that passes through the Vaara execution layer gets an audit
record — whether it was allowed, denied, or escalated.  Records are:

- **Immutable**: once written, the hash chain makes tampering detectable.
- **Regulation-mapped**: each record links to specific regulatory articles
  (EU AI Act, DORA, NIS2, MiFID II, GDPR) that the action is relevant to.
- **Machine-readable AND human-readable**: structured JSON for automation,
  narrative explanation for auditors and regulators.
- **Event-sourced**: outcomes are appended as follow-up events, never
  mutated in place.  The full decision→execution→outcome lifecycle is
  preserved.

This is the evidence base that the compliance engine reads to assemble
article-level evidence reports for a deployer's own conformity work.

EU AI Act Article 12 requires "automatic recording of events" for
high-risk AI systems.  Article 14 requires records sufficient for
human oversight.  This module satisfies both.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from vaara._sanitize import json_safe, strict_json_dumps
from vaara.taxonomy.actions import ActionRequest, RegulatoryDomain

logger = logging.getLogger(__name__)


# ── Event types ───────────────────────────────────────────────────────────

class EventType(str, Enum):
    """Lifecycle events in the audit trail."""
    ACTION_REQUESTED = "action_requested"   # Agent submitted an action
    RISK_SCORED = "risk_scored"             # Scorer produced assessment
    DECISION_MADE = "decision_made"         # Allow/deny/escalate decided
    ACTION_EXECUTED = "action_executed"      # Action was actually executed
    ACTION_BLOCKED = "action_blocked"       # Action was blocked
    ESCALATION_SENT = "escalation_sent"     # Sent to human for review
    ESCALATION_RESOLVED = "escalation_resolved"  # Human responded
    OUTCOME_RECORDED = "outcome_recorded"   # Post-execution outcome observed
    POLICY_OVERRIDE = "policy_override"     # Manual override of policy decision


# ── Regulatory article mappings ───────────────────────────────────────────

@dataclass(frozen=True)
class RegulatoryArticle:
    """A specific regulatory article that an audit record satisfies."""
    domain: RegulatoryDomain
    article: str           # e.g., "Article 12(1)" or "Article 9(2)(a)"
    requirement: str       # What the article requires
    how_satisfied: str     # How this audit record satisfies it


# Pre-built mappings — which record types satisfy which articles
EU_AI_ACT_MAPPINGS: dict[EventType, list[RegulatoryArticle]] = {
    EventType.ACTION_REQUESTED: [
        RegulatoryArticle(
            RegulatoryDomain.EU_AI_ACT, "Article 12(1)",
            "Automatic recording of events (logging capabilities)",
            "Every action request is recorded with full context before processing",
        ),
    ],
    EventType.RISK_SCORED: [
        RegulatoryArticle(
            RegulatoryDomain.EU_AI_ACT, "Article 9(2)(a)",
            "Risk management system shall identify and analyse known and reasonably foreseeable risks",
            "Adaptive risk scoring with conformal prediction intervals quantifies risk for every action",
        ),
        RegulatoryArticle(
            RegulatoryDomain.EU_AI_ACT, "Article 9(7)",
            "Testing shall be suitable to identify relevant risks",
            "Conformal calibration provides distribution-free coverage guarantees on risk estimates",
        ),
    ],
    EventType.DECISION_MADE: [
        RegulatoryArticle(
            RegulatoryDomain.EU_AI_ACT, "Article 14(1)",
            "High-risk AI systems shall be designed to be effectively overseen by natural persons",
            "Three-tier decision (allow/escalate/deny) ensures humans review borderline cases",
        ),
        RegulatoryArticle(
            RegulatoryDomain.EU_AI_ACT, "Article 14(4)(a)",
            "The human shall be able to fully understand the capacities and limitations of the system",
            "Decision records include risk scores, confidence intervals, and contributing signals",
        ),
    ],
    EventType.ACTION_BLOCKED: [
        RegulatoryArticle(
            RegulatoryDomain.EU_AI_ACT, "Article 9(4)(a)",
            "Appropriate risk management measures shall be adopted, including elimination or reduction of risks",
            "High-risk actions are automatically blocked with full audit trail",
        ),
    ],
    EventType.ESCALATION_SENT: [
        RegulatoryArticle(
            RegulatoryDomain.EU_AI_ACT, "Article 14(3)(a)",
            "Human oversight shall enable the human to properly monitor the AI system",
            "Borderline decisions are escalated with full context for human review",
        ),
    ],
    EventType.ESCALATION_RESOLVED: [
        RegulatoryArticle(
            RegulatoryDomain.EU_AI_ACT, "Article 14(4)(d)",
            "The human shall be able to decide not to use the system or to override the output",
            "Human override decisions are recorded with justification",
        ),
    ],
    EventType.OUTCOME_RECORDED: [
        RegulatoryArticle(
            RegulatoryDomain.EU_AI_ACT, "Article 9(2)(b)",
            "Appropriate measures to eliminate or reduce risks as far as possible through adequate design",
            "Post-execution outcomes feed back into adaptive scoring to improve future risk estimates",
        ),
        RegulatoryArticle(
            RegulatoryDomain.EU_AI_ACT, "Article 61(1)",
            "Post-market monitoring system proportionate to the nature of the AI technologies",
            "Continuous outcome tracking enables post-deployment monitoring of risk accuracy",
        ),
    ],
    EventType.POLICY_OVERRIDE: [
        RegulatoryArticle(
            RegulatoryDomain.EU_AI_ACT, "Article 14(4)(d)",
            "The human shall be able to decide not to use the system or to override the output",
            "Policy override records capture the human's decision to override the automated output, including reason, overrider identity, and prior/new decisions",
        ),
        RegulatoryArticle(
            RegulatoryDomain.EU_AI_ACT, "Article 12(1)",
            "Automatic recording of events (logging capabilities)",
            "Every policy override is recorded as an immutable audit event with hash-chained integrity",
        ),
    ],
}

DORA_MAPPINGS: dict[EventType, list[RegulatoryArticle]] = {
    EventType.ACTION_REQUESTED: [
        RegulatoryArticle(
            RegulatoryDomain.DORA, "Article 12(1)",
            "ICT-related incident detection, including automated alert mechanisms",
            "All agent actions are logged as events for incident detection",
        ),
    ],
    EventType.ACTION_BLOCKED: [
        RegulatoryArticle(
            RegulatoryDomain.DORA, "Article 10(1)",
            "ICT risk management framework shall include protection and prevention mechanisms",
            "Automated blocking of high-risk actions with full incident trail",
        ),
    ],
    EventType.OUTCOME_RECORDED: [
        RegulatoryArticle(
            RegulatoryDomain.DORA, "Article 13(1)",
            "Learning and evolving from ICT-related incidents",
            "Outcome data feeds back into risk scoring for continuous improvement",
        ),
    ],
}


# ── Audit Record ──────────────────────────────────────────────────────────

@dataclass
class AuditRecord:
    """A single immutable audit event in the trail.

    Each record is hash-chained to the previous record, forming a
    tamper-evident log.  Records are append-only — outcomes are added
    as new events referencing the original action_id, never mutated.
    """
    record_id: str
    action_id: str              # Groups all events for one action
    event_type: EventType
    timestamp: float            # Unix epoch (UTC)
    agent_id: str
    tool_name: str
    data: dict = field(default_factory=dict)
    regulatory_articles: list[dict] = field(default_factory=list)
    previous_hash: str = ""     # Hash chain — empty for first record
    record_hash: str = ""       # SHA-256 of this record's content

    def compute_hash(self) -> str:
        """Compute deterministic hash of this record's content.

        `regulatory_articles` is included because it encodes which EU AI Act /
        DORA / MiFID II obligations this event satisfies — it is the audit
        record's regulatory provenance. Excluding it would let an attacker
        with write access to the persistent store tamper with regulatory
        attribution post-facto while leaving `verify_chain()` clean, which
        is the opposite of what the hash chain is supposed to guarantee.
        allow_nan=False enforces RFC 8259 strictness on the hashable surface
        as a defence-in-depth partner to _sanitize.json_safe's non-finite
        scrub, so any non-finite slipped past sanitisation raises here
        rather than silently producing a non-portable hash input.
        """
        content = {
            "record_id": self.record_id,
            "action_id": self.action_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "data": self.data,
            "regulatory_articles": self.regulatory_articles,
            "previous_hash": self.previous_hash,
        }
        canonical = json.dumps(
            content, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict:
        """Serialize to dict for storage/export."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    @staticmethod
    def from_dict(d: dict) -> AuditRecord:
        """Deserialize from dict."""
        d = dict(d)
        d["event_type"] = EventType(d["event_type"])
        return AuditRecord(**d)

    @property
    def narrative(self) -> str:
        """Human-readable narrative of this event.

        All caller-supplied fields pass through `_narrative_str` so a
        malicious agent_id/reason/override_reason cannot forge a second
        log line via newline injection, smuggle ANSI escapes, or DoS
        the narrative with MB-scale strings. The raw record.data is
        untouched — the hash chain still covers original values.

        Timestamp formatting is defensive: a single corrupt record
        (non-numeric/NaN/overflow timestamp from JSONL reload via
        from_dict, direct AuditRecord construction, or future schema
        drift) must NOT crash AuditTrail.get_narrative() and take down
        the regulator-facing narrative view. Fall back to repr on any
        strftime/gmtime failure; the record is still rendered.
        """
        ts = _fmt_timestamp(self.timestamp)
        safe_agent = _narrative_str(self.agent_id, max_len=80)
        safe_tool = _narrative_str(self.tool_name, max_len=80)
        prefix = f"[{ts}] Agent '{safe_agent}'"

        narratives = {
            EventType.ACTION_REQUESTED: (
                f"{prefix} requested '{safe_tool}'"
                f" (params: {_summarize_params(self.data.get('parameters', {}))})"
            ),
            EventType.RISK_SCORED: (
                f"{prefix} action '{safe_tool}' scored "
                f"risk={_fmt_num(self.data.get('point_estimate'))} "
                f"[{_fmt_num(self.data.get('conformal_lower'))}, "
                f"{_fmt_num(self.data.get('conformal_upper'))}]"
            ),
            EventType.DECISION_MADE: (
                f"{prefix} action '{safe_tool}' → "
                f"{_narrative_str(self.data.get('decision', 'unknown'), max_len=32).upper()}"
                f" (reason: {_narrative_str(self.data.get('reason', 'none'))})"
            ),
            EventType.ACTION_EXECUTED: (
                f"{prefix} executed '{safe_tool}' successfully"
            ),
            EventType.ACTION_BLOCKED: (
                f"{prefix} BLOCKED from executing '{safe_tool}'"
                f" (risk={_fmt_num(self.data.get('risk_score'))})"
            ),
            EventType.ESCALATION_SENT: (
                f"{prefix} action '{safe_tool}' escalated for human review"
            ),
            EventType.ESCALATION_RESOLVED: (
                f"{prefix} escalation resolved: "
                f"{_narrative_str(self.data.get('resolution', 'unknown'))}"
                f" by {_narrative_str(self.data.get('reviewer', 'unknown'), max_len=80)}"
            ),
            EventType.OUTCOME_RECORDED: (
                f"{prefix} outcome for '{safe_tool}': "
                f"severity={_fmt_num(self.data.get('outcome_severity'))}"
            ),
            EventType.POLICY_OVERRIDE: (
                f"{prefix} policy overridden: "
                f"{_narrative_str(self.data.get('override_reason', 'no reason given'))}"
            ),
        }

        return narratives.get(self.event_type, f"{prefix} {self.event_type.value}")


def _fmt_timestamp(ts: Any) -> str:
    """Format a timestamp for narrative; never raises.

    Handles non-numeric (corrupt JSONL), NaN/inf (sanitizer bypass), and
    overflow (platform time_t limit) without crashing the narrative path.
    """
    try:
        ts_f = float(ts)
        if not math.isfinite(ts_f):
            return f"<invalid-ts:{ts!r}>"
        return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ts_f))
    except (TypeError, ValueError, OverflowError, OSError):
        return f"<invalid-ts:{ts!r}>"


def _fmt_num(val: Any, fmt: str = ".3f") -> str:
    """Format a number for narrative display, with fallback for missing values."""
    if val is None:
        return "?"
    try:
        return f"{float(val):{fmt}}"
    except (TypeError, ValueError):
        return str(val)


def _summarize_params(params: dict, max_len: int = 100) -> str:
    """Truncate parameter dict for narrative display."""
    s = str(params)
    if len(s) > max_len:
        return s[:max_len - 3] + "..."
    return s


# Maximum characters per caller-supplied field in a narrative line. Without
# this, an attacker who controls agent_id / override_reason / reason can
# (a) DoS a log reader by inserting MB-scale strings, or (b) smuggle a
# forged audit line via embedded newlines (the narrative is one line per
# record, so "\n[2099-...] Agent..." looks like a second record to anyone
# grepping the log). Control characters and ANSI escapes are stripped too
# so a malicious field can't paint a terminal or clear the screen.
_NARRATIVE_FIELD_MAX = 200


def _narrative_str(val: Any, max_len: int = _NARRATIVE_FIELD_MAX) -> str:
    """Render a caller-supplied value as a safe single-line narrative token.

    The underlying audit record still carries the raw value (the hash
    chain is over raw data, so tamper-evidence is preserved). Only the
    human-readable narrative is sanitized.

    Uses str.isprintable() to drop every char Python tags as
    "non-printable" — Cc (ASCII control, ANSI escape, NUL, \\n, \\r, \\t),
    Cf (format: bidi overrides U+202E, zero-width U+200B, BOM U+FEFF),
    Zl/Zp (Unicode line/paragraph separators U+2028/U+2029), and every
    Zs variant other than the regular ASCII space. This closes three
    historical injection paths:
      1. ASCII \\n/\\r — forged second audit line in grep output.
      2. Unicode line separators (U+0085/U+2028/U+2029) — same forgery
         through any log reader that honours str.splitlines().
      3. Bidi RLO (U+202E) — visually reordering "allow" into "deny"
         in bidi-rendering terminals / Slack / editors.
    Regular ASCII space ("\\x20") is Unicode-printable in Python and
    is preserved so narrative tokens stay human-readable.
    """
    if val is None:
        return ""
    s = str(val)
    s = "".join(ch if ch.isprintable() else " " for ch in s)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


# ── Audit Trail ───────────────────────────────────────────────────────────

class AuditTrail:
    """Append-only, hash-chained audit trail.

    All action lifecycle events are recorded here.  The trail is the
    single source of truth for compliance reporting.

    Storage backends are pluggable — the trail writes to an in-memory
    list and optionally to an on_record callback (for file/DB/streaming).
    """

    def __init__(
        self,
        on_record: Optional[Callable[[AuditRecord], None]] = None,
    ) -> None:
        """
        Args:
            on_record: Optional callback invoked on every new record.
                       Use this to pipe records to file, database, or stream.
        """
        self._records: list[AuditRecord] = []
        self._by_action: dict[str, list[AuditRecord]] = defaultdict(list)
        self._last_hash = ""
        self._on_record = on_record
        # Counts on_record callback failures so callers can detect
        # persistence divergence at runtime (e.g., DB gone, disk full).
        # Without this, a silent logger.error is the only signal and the
        # in-memory chain stays valid while the persistent store gaps —
        # surfaces only at next load_trail. Article 12(2) compliance asks
        # for active detection, not forensic-only.
        self._persistence_failures = 0
        # Count of rows loaded as skeletons during load_trail because
        # their data/regulatory JSON columns were corrupt. Parallel
        # signal to persistence_failures but for the READ path — an ops
        # dashboard polling only persistence_failures would miss silent
        # reload-time corruption that verify_chain surfaces via hash
        # mismatch. Settable by the backend after load_trail completes.
        self._skeleton_records = 0
        # Serializes _append so the hash chain stays correct when
        # multiple agent threads write concurrently. Without this, two
        # threads can read the same _last_hash and produce two records
        # pointing at the same predecessor — verify_chain then fails.
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def persistence_failures(self) -> int:
        """Number of on_record callback failures since trail creation."""
        return self._persistence_failures

    @property
    def skeleton_records(self) -> int:
        """Rows loaded as skeletons due to corrupt JSON columns during load_trail."""
        return self._skeleton_records

    @property
    def chain_intact(self) -> bool:
        """Verify the hash chain is unbroken."""
        return self.verify_chain() is None

    def verify_chain(self) -> Optional[str]:
        """Verify hash chain integrity.  Returns None if intact, error string if broken."""
        with self._lock:
            snapshot = list(self._records)
        prev_hash = ""
        for i, record in enumerate(snapshot):
            if record.previous_hash != prev_hash:
                return (
                    f"Chain broken at record {i} ({record.record_id}): "
                    f"expected previous_hash={prev_hash!r}, "
                    f"got {record.previous_hash!r}"
                )
            expected = record.compute_hash()
            if record.record_hash != expected:
                return (
                    f"Hash mismatch at record {i} ({record.record_id}): "
                    f"expected {expected!r}, got {record.record_hash!r}"
                )
            prev_hash = record.record_hash
        return None

    # ── Recording events ──────────────────────────────────────────

    def record_action_requested(self, request: ActionRequest) -> str:
        """Record that an agent requested an action.  Returns the action_id."""
        action_id = str(uuid.uuid4())

        articles = self._get_regulatory_articles(
            EventType.ACTION_REQUESTED,
            request.action_type.regulatory_domains,
        )

        safe_agent_id = self._cap_record_str(request.agent_id, self._MAX_AGENT_ID_LEN)
        safe_tool_name = self._cap_record_str(request.tool_name, self._MAX_TOOL_NAME_LEN)
        safe_session_id = (
            self._cap_record_str(request.session_id, self._MAX_SESSION_ID_LEN)
            if request.session_id is not None else None
        )
        safe_parent_action_id = (
            self._cap_record_str(request.parent_action_id, self._MAX_PARENT_ACTION_ID_LEN)
            if request.parent_action_id is not None else None
        )
        self._append(AuditRecord(
            record_id=str(uuid.uuid4()),
            action_id=action_id,
            event_type=EventType.ACTION_REQUESTED,
            timestamp=time.time(),
            agent_id=safe_agent_id,
            tool_name=safe_tool_name,
            data={
                "parameters": self._cap_record_dict_bytes(
                    request.parameters if isinstance(request.parameters, dict) else {},
                    self._MAX_REQUEST_PARAMETERS_JSON_BYTES,
                ),
                "context": self._cap_record_dict_bytes(
                    request.context if isinstance(request.context, dict) else {},
                    self._MAX_REQUEST_CONTEXT_JSON_BYTES,
                ),
                "action_category": request.action_type.category.value,
                "reversibility": request.action_type.reversibility.value,
                "blast_radius": request.action_type.blast_radius.value,
                "urgency": request.action_type.urgency.value,
                "base_risk_score": request.action_type.base_risk_score,
                "agent_confidence": request.confidence,
                "session_id": safe_session_id,
                "parent_action_id": safe_parent_action_id,
                "sequence_position": request.sequence_position,
            },
            regulatory_articles=articles,
        ))

        return action_id

    def record_risk_scored(
        self,
        action_id: str,
        agent_id: str,
        tool_name: str,
        assessment: dict,
        regulatory_domains: frozenset[RegulatoryDomain] = frozenset(),
    ) -> None:
        """Record the risk scoring result."""
        articles = self._get_regulatory_articles(
            EventType.RISK_SCORED, regulatory_domains,
        )

        safe_assessment = {str(k): json_safe(v) for k, v in (assessment or {}).items()}
        safe_assessment = self._cap_record_dict_bytes(
            safe_assessment, self._MAX_ASSESSMENT_JSON_BYTES,
        )

        self._append(AuditRecord(
            record_id=str(uuid.uuid4()),
            action_id=action_id,
            event_type=EventType.RISK_SCORED,
            timestamp=time.time(),
            agent_id=self._cap_record_str(agent_id, self._MAX_AGENT_ID_LEN),
            tool_name=self._cap_record_str(tool_name, self._MAX_TOOL_NAME_LEN),
            data=safe_assessment,
            regulatory_articles=articles,
        ))

    def record_decision(
        self,
        action_id: str,
        agent_id: str,
        tool_name: str,
        decision: str,
        reason: str,
        risk_score: float,
        regulatory_domains: frozenset[RegulatoryDomain] = frozenset(),
    ) -> None:
        """Record the allow/deny/escalate decision."""
        event_type = (
            EventType.ACTION_BLOCKED if decision == "deny"
            else EventType.DECISION_MADE
        )

        articles = self._get_regulatory_articles(
            event_type, regulatory_domains,
        )

        self._append(AuditRecord(
            record_id=str(uuid.uuid4()),
            action_id=action_id,
            event_type=event_type,
            timestamp=time.time(),
            agent_id=self._cap_record_str(agent_id, self._MAX_AGENT_ID_LEN),
            tool_name=self._cap_record_str(tool_name, self._MAX_TOOL_NAME_LEN),
            data={
                "decision": self._cap_record_str(decision, self._MAX_DECISION_LABEL_LEN),
                "reason": self._cap_record_str(reason, self._MAX_DECISION_REASON_LEN),
                "risk_score": risk_score,
            },
            regulatory_articles=articles,
        ))

    def record_execution(
        self,
        action_id: str,
        agent_id: str,
        tool_name: str,
        result: Optional[dict] = None,
    ) -> None:
        """Record that the action was executed."""
        # Sanitize caller-supplied result at the trail boundary; datetime/
        # bytes/Path values would otherwise crash compute_hash(). Tools
        # often return lists, strings, or primitive types — wrap non-dict
        # results in a {"value": ...} envelope so the audit record still
        # captures the shape without crashing on .items().
        if result is None:
            safe_result: dict = {}
        elif isinstance(result, dict):
            safe_result = {str(k): json_safe(v) for k, v in result.items()}
        else:
            logger.warning(
                "record_execution: non-dict result type=%r for action_id=%s; "
                "wrapping under 'value' key",
                type(result).__name__, action_id,
            )
            safe_result = {"value": json_safe(result)}
        self._append(AuditRecord(
            record_id=str(uuid.uuid4()),
            action_id=action_id,
            event_type=EventType.ACTION_EXECUTED,
            timestamp=time.time(),
            agent_id=self._cap_record_str(agent_id, self._MAX_AGENT_ID_LEN),
            tool_name=self._cap_record_str(tool_name, self._MAX_TOOL_NAME_LEN),
            data={"result_summary": self._cap_record_dict_bytes(
                safe_result, self._MAX_EXECUTION_RESULT_JSON_BYTES
            )},
        ))

    def record_escalation(
        self,
        action_id: str,
        agent_id: str,
        tool_name: str,
        escalation_target: str,
        risk_score: float,
    ) -> None:
        """Record that an action was escalated for human review."""
        articles = self._get_regulatory_articles(
            EventType.ESCALATION_SENT, frozenset({RegulatoryDomain.EU_AI_ACT}),
        )

        self._append(AuditRecord(
            record_id=str(uuid.uuid4()),
            action_id=action_id,
            event_type=EventType.ESCALATION_SENT,
            timestamp=time.time(),
            agent_id=self._cap_record_str(agent_id, self._MAX_AGENT_ID_LEN),
            tool_name=self._cap_record_str(tool_name, self._MAX_TOOL_NAME_LEN),
            data={
                "escalation_target": self._cap_record_str(
                    escalation_target, self._MAX_ESCALATION_TARGET_LEN,
                ),
                "risk_score": risk_score,
            },
            regulatory_articles=articles,
        ))

    def record_escalation_resolved(
        self,
        action_id: str,
        agent_id: str,
        tool_name: str,
        resolution: str,
        reviewer: str,
        justification: str = "",
    ) -> None:
        """Record human resolution of an escalation."""
        articles = self._get_regulatory_articles(
            EventType.ESCALATION_RESOLVED, frozenset({RegulatoryDomain.EU_AI_ACT}),
        )

        self._append(AuditRecord(
            record_id=str(uuid.uuid4()),
            action_id=action_id,
            event_type=EventType.ESCALATION_RESOLVED,
            timestamp=time.time(),
            agent_id=self._cap_record_str(agent_id, self._MAX_AGENT_ID_LEN),
            tool_name=self._cap_record_str(tool_name, self._MAX_TOOL_NAME_LEN),
            data={
                "resolution": self._cap_record_str(resolution, self._MAX_RESOLUTION_LEN),
                "reviewer": self._cap_record_str(reviewer, self._MAX_REVIEWER_LEN),
                "justification": self._cap_record_str(justification, self._MAX_JUSTIFICATION_LEN),
            },
            regulatory_articles=articles,
        ))

    def record_outcome(
        self,
        action_id: str,
        agent_id: str,
        tool_name: str,
        outcome_severity: float,
        description: str = "",
    ) -> None:
        """Record the actual outcome of an executed action.

        This closes the feedback loop — the outcome is used by the scorer
        to update its MWU weights and conformal calibration.
        """
        articles = self._get_regulatory_articles(
            EventType.OUTCOME_RECORDED,
            frozenset({RegulatoryDomain.EU_AI_ACT, RegulatoryDomain.DORA}),
        )

        self._append(AuditRecord(
            record_id=str(uuid.uuid4()),
            action_id=action_id,
            event_type=EventType.OUTCOME_RECORDED,
            timestamp=time.time(),
            agent_id=self._cap_record_str(agent_id, self._MAX_AGENT_ID_LEN),
            tool_name=self._cap_record_str(tool_name, self._MAX_TOOL_NAME_LEN),
            data={
                "outcome_severity": outcome_severity,
                "description": self._cap_record_str(
                    description, self._MAX_OUTCOME_DESCRIPTION_LEN,
                ),
            },
            regulatory_articles=articles,
        ))

    # Length caps for caller-controlled free-text fields on this direct
    # trail API. Mirrors the pipeline Loop 47 caps — record_policy_override
    # is a public surface that reaches the hash chain, narrative, export,
    # and compliance report without a pipeline wrapper. Same amplification
    # concern: a 50MB override_reason would balloon storage, hashing time,
    # and every regulator-facing view of this record.
    _MAX_OVERRIDER_LEN = 256
    _MAX_OVERRIDE_REASON_LEN = 8192
    _MAX_DECISION_LABEL_LEN = 64
    # record_execution is public API too — a tool returning a multi-MB
    # result dict would otherwise balloon ACTION_EXECUTED records and
    # every regulator view that iterates the trail. Same L47 pattern
    # applied at the trail boundary because no pipeline wrapper exists.
    _MAX_EXECUTION_RESULT_JSON_BYTES = 64 * 1024
    # record_risk_scored.assessment ingests scorer plugin output (signals +
    # interval + calibration). A malicious/buggy scorer returning a 10MB
    # signals dict would balloon every RISK_SCORED record on the hash chain.
    # Pipeline routes through here but applies no size cap; direct trail
    # callers also bypass. Mirror the L53 pattern.
    _MAX_ASSESSMENT_JSON_BYTES = 64 * 1024
    # record_action_requested ingests caller-controlled parameters/context
    # dicts on ActionRequest. Pipeline.intercept caps these before calling
    # this method, but direct trail API users bypass the pipeline. Mirror
    # the pipeline defense so the hash chain cannot be poisoned via either
    # ingress path.
    _MAX_REQUEST_PARAMETERS_JSON_BYTES = 64 * 1024
    _MAX_REQUEST_CONTEXT_JSON_BYTES = 64 * 1024
    # ActionRequest string fields — pipeline.intercept caps these before
    # constructing ActionRequest, but direct trail API callers bypass the
    # pipeline. Mirror the pipeline values so the hash chain cannot be
    # poisoned via either ingress path.
    _MAX_AGENT_ID_LEN = 256
    _MAX_TOOL_NAME_LEN = 512
    _MAX_SESSION_ID_LEN = 256
    _MAX_PARENT_ACTION_ID_LEN = 128
    # Remaining trail-boundary string caps (pipeline caps at its boundary
    # via the L50 constants but direct trail API callers bypass).
    _MAX_OUTCOME_DESCRIPTION_LEN = 4096
    _MAX_ESCALATION_TARGET_LEN = 256
    _MAX_RESOLUTION_LEN = 64
    _MAX_REVIEWER_LEN = 256
    _MAX_JUSTIFICATION_LEN = 8192
    # Reason on record_decision — pipeline caps at L52 but direct trail
    # callers bypass. Same cap as pipeline _MAX_DECISION_REASON_LEN.
    _MAX_DECISION_REASON_LEN = 4096

    @staticmethod
    def _cap_record_str(value, max_len: int) -> str:
        if not isinstance(value, str):
            value = str(value) if value is not None else ""
        if len(value) <= max_len:
            return value
        marker = f"...[TRUNCATED:{len(value)}B]"
        keep = max_len - len(marker)
        if keep <= 0:
            # marker itself exceeds max_len; plain truncation preserves the cap
            return value[:max_len]
        return value[:keep] + marker

    @staticmethod
    def _cap_record_dict_bytes(d: dict, max_bytes: int) -> dict:
        """Cap a sanitised dict's JSON size; over cap → single-key marker.

        Mirrors pipeline._cap_dict_bytes so a huge tool result does not
        poison the audit record. Records the original byte count and the
        first 20 keys as evidence the oversized input was attempted.
        """
        try:
            serialized = json.dumps(d, default=str)
        except (TypeError, ValueError):
            return {"_truncated": True, "_original_bytes": -1, "_cap_bytes": max_bytes}
        if len(serialized) <= max_bytes:
            return d
        return {
            "_truncated": True,
            "_original_bytes": len(serialized),
            "_cap_bytes": max_bytes,
            "_keys": list(d.keys())[:20] if isinstance(d, dict) else [],
        }

    def record_policy_override(
        self,
        action_id: str,
        agent_id: str,
        tool_name: str,
        override_reason: str,
        overrider: str,
        original_decision: str,
        new_decision: str,
    ) -> None:
        """Record a manual policy override.

        Policy overrides are explicitly mapped to EU AI Act Article 14(4)(d)
        ("decide not to use the system or to override the output"). Without
        regulatory article attribution on the record itself, an auditor
        enumerating `get_regulatory_evidence(EU_AI_ACT)` would see zero
        overrides even though Article 14(4)(d) evidence hinges on them.
        """
        articles = self._get_regulatory_articles(
            EventType.POLICY_OVERRIDE, frozenset({RegulatoryDomain.EU_AI_ACT}),
        )

        override_reason = self._cap_record_str(
            override_reason, self._MAX_OVERRIDE_REASON_LEN
        )
        overrider = self._cap_record_str(overrider, self._MAX_OVERRIDER_LEN)
        original_decision = self._cap_record_str(
            original_decision, self._MAX_DECISION_LABEL_LEN
        )
        new_decision = self._cap_record_str(
            new_decision, self._MAX_DECISION_LABEL_LEN
        )

        self._append(AuditRecord(
            record_id=str(uuid.uuid4()),
            action_id=action_id,
            event_type=EventType.POLICY_OVERRIDE,
            timestamp=time.time(),
            agent_id=self._cap_record_str(agent_id, self._MAX_AGENT_ID_LEN),
            tool_name=self._cap_record_str(tool_name, self._MAX_TOOL_NAME_LEN),
            data={
                "override_reason": override_reason,
                "overrider": overrider,
                "original_decision": original_decision,
                "new_decision": new_decision,
            },
            regulatory_articles=articles,
        ))

    # ── Querying ──────────────────────────────────────────────────

    def get_action_trail(self, action_id: str) -> list[AuditRecord]:
        """Get all events for a specific action, in order."""
        with self._lock:
            return list(self._by_action.get(action_id, []))

    def get_agent_records(
        self, agent_id: str, limit: int = 100
    ) -> list[AuditRecord]:
        """Get recent records for an agent."""
        with self._lock:
            snapshot = list(self._records)
        records = [r for r in snapshot if r.agent_id == agent_id]
        return records[-limit:]

    def get_records_by_type(
        self, event_type: EventType, limit: int = 100
    ) -> list[AuditRecord]:
        """Get recent records of a specific event type."""
        with self._lock:
            snapshot = list(self._records)
        records = [r for r in snapshot if r.event_type == event_type]
        return records[-limit:]

    def get_blocked_actions(self, limit: int = 50) -> list[AuditRecord]:
        """Get recently blocked actions — useful for compliance dashboards."""
        return self.get_records_by_type(EventType.ACTION_BLOCKED, limit)

    def get_regulatory_evidence(
        self, domain: RegulatoryDomain
    ) -> list[AuditRecord]:
        """Get all records relevant to a specific regulatory domain."""
        with self._lock:
            snapshot = list(self._records)
        results = []
        for record in snapshot:
            for article in record.regulatory_articles:
                if article.get("domain") == domain.value:
                    results.append(record)
                    break
        return results

    def get_narrative(
        self, action_id: Optional[str] = None, limit: int = 50
    ) -> str:
        """Generate human-readable narrative of the audit trail.

        If action_id is provided, narrates that action's lifecycle.
        Otherwise, narrates the most recent events.
        """
        if action_id:
            records = self.get_action_trail(action_id)
        else:
            with self._lock:
                records = self._records[-limit:]

        lines = []
        for record in records:
            lines.append(record.narrative)
            if record.regulatory_articles:
                domains = {a.get("domain", "") for a in record.regulatory_articles}
                lines.append(f"  Regulatory: {', '.join(sorted(domains))}")
        return "\n".join(lines)

    # ── Export ────────────────────────────────────────────────────

    def export_json(self, path: Path) -> int:
        """Export full trail to JSON file.  Returns record count."""
        # Snapshot under the lock so we never iterate _records while
        # _append is mutating it. Without this, free-threaded Python
        # (3.13t) has a data race; even under the GIL, the on-disk
        # file could miss or duplicate records appended mid-iteration.
        with self._lock:
            snapshot = list(self._records)
        records = [r.to_dict() for r in snapshot]
        # strict_json_dumps enforces RFC 8259 (no NaN/Infinity) + scrubs
        # non-finites at the wire boundary. Any legacy record loaded from
        # disk or built via direct AuditRecord construction that bypassed
        # _sanitize.json_safe at ingress is still rendered clean for the
        # regulator-facing export.
        path.write_text(
            strict_json_dumps(records, indent=2), encoding="utf-8"
        )
        return len(records)

    def export_jsonl(self, path: Path) -> int:
        """Export full trail as JSON Lines (one record per line)."""
        with self._lock:
            snapshot = list(self._records)
        # encoding="utf-8" guarantees cross-platform consistency for
        # non-ASCII agent names and reasons — platform default is not
        # portable. strict_json_dumps handles the non-finite scrub.
        with open(path, "w", encoding="utf-8") as f:
            for record in snapshot:
                f.write(strict_json_dumps(record.to_dict()) + "\n")
        return len(snapshot)

    # ── Internal ──────────────────────────────────────────────────

    def _append(self, record: AuditRecord) -> None:
        """Append a record to the trail with hash chaining."""
        # Sanitize caller-supplied `data` at the single choke point so any
        # record_* method (including direct AuditTrail users bypassing the
        # pipeline) cannot land a datetime/bytes/Path/UUID in data and
        # crash compute_hash — which would leave the chain half-appended.
        # json_safe is idempotent on JSON-native types, so records whose
        # data is already clean round-trip unchanged.
        if record.data:
            record.data = {str(k): json_safe(v) for k, v in record.data.items()}
        with self._lock:
            record.previous_hash = self._last_hash
            record.record_hash = record.compute_hash()
            self._last_hash = record.record_hash

            self._records.append(record)
            self._by_action[record.action_id].append(record)

            if self._on_record:
                try:
                    self._on_record(record)
                except Exception:
                    # logger.exception preserves the stack trace so ops can
                    # diagnose disk-full / locked-DB / permission errors
                    # instead of seeing a bare error line.
                    self._persistence_failures += 1
                    logger.exception(
                        "on_record callback failed for record %s "
                        "(persistent store now out of sync with in-memory "
                        "chain; failure count=%d)",
                        record.record_id,
                        self._persistence_failures,
                    )

    def _get_regulatory_articles(
        self,
        event_type: EventType,
        action_domains: frozenset[RegulatoryDomain],
    ) -> list[dict]:
        """Collect regulatory article mappings for an event type."""
        articles = []

        # EU AI Act mappings (always included for high-risk AI systems)
        for article in EU_AI_ACT_MAPPINGS.get(event_type, []):
            articles.append({
                "domain": article.domain.value,
                "article": article.article,
                "requirement": article.requirement,
                "how_satisfied": article.how_satisfied,
            })

        # DORA mappings (for financial actions)
        if RegulatoryDomain.DORA in action_domains:
            for article in DORA_MAPPINGS.get(event_type, []):
                articles.append({
                    "domain": article.domain.value,
                    "article": article.article,
                    "requirement": article.requirement,
                    "how_satisfied": article.how_satisfied,
                })

        return articles
