# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Immutable audit trail with regulatory article mapping.

Every action that passes through the Vaara execution layer gets an audit
record — whether it was allowed, denied, or escalated.  Records are:

- **Immutable**: once written, the hash chain makes tampering detectable.
- **Regulation-mapped**: each record links to specific regulatory articles
  (EU AI Act, DORA, NIS2, MiFID II, GDPR) that the action is relevant to.
- **Machine-readable AND human-readable**: structured JSON for automation,
  narrative explanation for auditors and regulators.
- **Event-sourced**: outcomes are appended as follow-up events, never
  mutated in place.  The full decision to execution to outcome lifecycle is
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
    ANCHOR_GAP = "anchor_gap"               # Auto-anchor attempt failed (fail-open marker)
    KEY_LIFECYCLE = "key_lifecycle"         # Signing-key custodian rotated/revoked/added


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
    EventType.KEY_LIFECYCLE: [
        RegulatoryArticle(
            RegulatoryDomain.EU_AI_ACT, "Article 15(1)",
            "High-risk AI systems shall achieve an appropriate level of accuracy, robustness and cybersecurity",
            "Custodian key rotation/revocation/addition is recorded as a hash-chained, externally time-anchored audit event, so the signing-key control set is itself tamper-evident",
        ),
        RegulatoryArticle(
            RegulatoryDomain.EU_AI_ACT, "Article 12(1)",
            "Automatic recording of events (logging capabilities)",
            "Every key-lifecycle change is captured as an immutable audit event pinned in the hash chain and the external time anchor",
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


# ── Transparency taxonomy (prEN ISO/IEC 12792 four-axis) ─────────────────
#
# v0.6 alignment with WG4's transparency taxonomy. Four axes per record:
# - system_operation: how the AI system worked at this event
# - data_usage:       what data was consumed
# - decision_making:  how the conclusion was reached
# - limitations:      known constraints (often None per-event; system-level)
#
# Defaults are filled by AuditRecord.__post_init__ from event_type. Callers
# may override per-record by passing explicit values at construction time.

TRANSPARENCY_DEFAULTS: dict[EventType, dict[str, str]] = {
    EventType.ACTION_REQUESTED: {
        "system_operation": "logging_intake",
        "data_usage": "tool_args+context",
        "decision_making": "n/a",
    },
    EventType.RISK_SCORED: {
        "system_operation": "scoring",
        "data_usage": "tool_args+history+policy",
        "decision_making": "heuristic_score",
    },
    EventType.DECISION_MADE: {
        "system_operation": "decision_threshold",
        "data_usage": "risk_score+thresholds",
        "decision_making": "threshold_match",
    },
    EventType.ACTION_EXECUTED: {
        "system_operation": "execution",
        "data_usage": "decision",
        "decision_making": "prior_decision",
    },
    EventType.ACTION_BLOCKED: {
        "system_operation": "blocking",
        "data_usage": "risk_score+thresholds",
        "decision_making": "threshold_match",
    },
    EventType.ESCALATION_SENT: {
        "system_operation": "escalation",
        "data_usage": "risk_score+context",
        "decision_making": "threshold_or_classifier_upgrade",
    },
    EventType.ESCALATION_RESOLVED: {
        "system_operation": "human_oversight",
        "data_usage": "operator_response",
        "decision_making": "human_decision",
    },
    EventType.OUTCOME_RECORDED: {
        "system_operation": "outcome_capture",
        "data_usage": "execution_result",
        "decision_making": "n/a",
    },
    EventType.POLICY_OVERRIDE: {
        "system_operation": "manual_override",
        "data_usage": "operator_decision",
        "decision_making": "human_decision",
    },
    EventType.ANCHOR_GAP: {
        "system_operation": "time_anchoring",
        "data_usage": "chain_head_digest",
        "decision_making": "n/a",
    },
    EventType.KEY_LIFECYCLE: {
        "system_operation": "key_custodian_management",
        "data_usage": "custodian_fingerprint+quorum",
        "decision_making": "operator_action",
    },
}


# ── Audit Record ──────────────────────────────────────────────────────────

# Hash-chain format version stamped on every newly appended record.
#   v1 (legacy): tenant_id is NOT part of compute_hash() — preserves
#       re-verification of pre-v0.47 trails written before tenant binding.
#   v2: tenant_id and chain_version ARE bound into the hash, so a record
#       cannot be silently re-attributed to another tenant (or downgraded
#       to v1 to strip the binding) without breaking the chain.
# Records loaded from storage keep their own stored chain_version; only
# AuditTrail._append stamps the current version on fresh records.
_CURRENT_CHAIN_VERSION = 2


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
    # v0.6: prEN ISO/IEC 12792 four-axis transparency tagging.
    # Defaults fill from TRANSPARENCY_DEFAULTS by event_type if None.
    system_operation: Optional[str] = None
    data_usage: Optional[str] = None
    decision_making: Optional[str] = None
    limitations: Optional[str] = None
    # v0.40: multi-tenant scoping. Empty string = single-tenant deployment.
    # Bound into compute_hash() from chain v2 (v0.47+); see chain_version.
    tenant_id: str = ""
    # Hash-chain format version (see _CURRENT_CHAIN_VERSION). Defaults to 1
    # so records deserialized from pre-v0.47 storage (which carry no
    # chain_version column/key) re-hash exactly as originally written.
    chain_version: int = 1

    def __post_init__(self) -> None:
        # Loaded-from-DB records carry a non-empty record_hash. Skip
        # default-fill for those — pre-v0.6 records on disk were written
        # before transparency tagging existed, and inventing defaults at
        # load time would falsely claim a classification the record did
        # not have.
        if self.record_hash:
            return
        defaults = TRANSPARENCY_DEFAULTS.get(self.event_type, {})
        if self.system_operation is None:
            self.system_operation = defaults.get("system_operation")
        if self.data_usage is None:
            self.data_usage = defaults.get("data_usage")
        if self.decision_making is None:
            self.decision_making = defaults.get("decision_making")
        # 'limitations' stays None unless explicitly set — system-level
        # constraints are recorded out-of-band, not per event.

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
        # Chain v2 (v0.47+): bind tenant_id into the tamper-evident surface
        # so a record cannot be silently re-attributed to another tenant.
        # chain_version is bound too, so a downgrade to v1 (which would drop
        # the tenant binding) also breaks the chain. v1 records omit both
        # keys and hash exactly as pre-v0.47 — old trails re-verify byte for
        # byte. The gate is >= 2 so a future v3 keeps binding these unless it
        # deliberately changes the scheme.
        if self.chain_version >= 2:
            content["tenant_id"] = self.tenant_id
            content["chain_version"] = self.chain_version
        # NOTE on transparency taxonomy (v0.6):
        # The four prEN ISO/IEC 12792 fields (system_operation, data_usage,
        # decision_making, limitations) are NOT included in the hash. They
        # are metadata annotations — not tamper-evident. This preserves
        # backward compatibility with pre-v0.6 records on disk: re-hashing
        # an old record after the schema bump produces the same hash as
        # before. v0.7+ may add a separate signed-bundle mechanism if a
        # compliance team requires tamper-evident transparency tagging.
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
                f"{prefix} action '{safe_tool}' to "
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
            EventType.KEY_LIFECYCLE: (
                f"{prefix} signing-key custodian "
                f"{_narrative_str(self.data.get('action', 'changed'), max_len=16)} "
                f"({_narrative_str(self.data.get('fingerprint', 'unknown'), max_len=64)})"
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
        # v0.40 multi-tenant: action_id -> tenant_id, seeded by
        # record_action_requested. Subsequent record_* calls (decision,
        # execution, escalation) look up the action_id so every record in
        # the lifecycle carries the same tenant scope without forcing
        # every caller to thread tenant_id through every method signature.
        self._tenant_for_action: dict[str, str] = {}
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
        # Guards _tenant_for_action. The map is read in _tenant_for and
        # mutated (len-check, evict, insert) in record_action_requested
        # from different agent threads. A bare dict left those racing:
        # a concurrent insert during the eviction's list() iteration could
        # raise "dictionary changed size during iteration", and a torn
        # read/insert could hand one tenant's lifecycle records another
        # tenant's scope. A dedicated lock keeps the map coherent without
        # widening the hash-chain lock or risking re-entrancy with _append.
        self._tenant_map_lock = threading.Lock()
        # v0.48 external time anchors over chain-head digests. Opt-in: stays
        # empty until anchor_head() is called. Each entry binds a chain
        # position + head hash to a third-party trusted timestamp, so the
        # chain's existence is provable against an external clock even if the
        # signing key is later compromised. See vaara.audit.timeanchor.
        self._anchors: list = []
        # v0.49 automatic cadence anchoring. Once enable_auto_anchor() sets a
        # client, the trail anchors its own head every _anchor_cadence records.
        # Fail-open: a failed anchor attempt records a chained ANCHOR_GAP marker
        # rather than raising, so a TSA outage is itself visible in the chain.
        # _anchor_lock guards the cadence counter without widening the chain
        # lock (the TSA round trip must happen off the chain lock).
        self._anchor_client: Any = None
        self._anchor_cadence = 0
        self._records_since_anchor = 0
        self._anchor_lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def anchors(self) -> list:
        """External time anchors recorded over this trail's chain heads."""
        return list(self._anchors)

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

    # Defense-in-depth cap for direct-trail callers that bypass the pipeline's
    # length cap on tenant_id. The HTTP boundary already caps at 256 via the
    # Pydantic schema, but the AuditTrail public API is reachable from
    # embedders that construct ActionRequest directly. A 50MB tenant_id would
    # otherwise balloon every record on the hash chain and the in-memory
    # action -> tenant map.
    _MAX_TENANT_ID_LEN = 256
    # Soft cap on the action -> tenant map. Long-running multi-tenant
    # deployments would otherwise leak memory at one entry per action,
    # because OUTCOME_RECORDED arrives well after ACTION_REQUESTED and the
    # map cannot be cleared at decision time. When the cap is reached the
    # oldest 1/8 of the map is evicted; subsequent lookups for evicted
    # actions fall back to "" tenant, which is the legacy single-tenant
    # contract — correct fail-soft behaviour.
    _MAX_ACTION_TENANT_MAP = 50_000

    def record_action_requested(self, request: ActionRequest) -> str:
        """Record that an agent requested an action.  Returns the action_id."""
        action_id = str(uuid.uuid4())
        tenant_id = getattr(request, "tenant_id", "") or ""
        if tenant_id:
            tenant_id = self._cap_record_str(tenant_id, self._MAX_TENANT_ID_LEN)
            with self._tenant_map_lock:
                if len(self._tenant_for_action) >= self._MAX_ACTION_TENANT_MAP:
                    evict = max(1, self._MAX_ACTION_TENANT_MAP // 8)
                    for stale in list(self._tenant_for_action)[:evict]:
                        self._tenant_for_action.pop(stale, None)
                self._tenant_for_action[action_id] = tenant_id

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
            tenant_id=tenant_id,
        ))

        return action_id

    def _tenant_for(self, action_id: str) -> str:
        """Resolve the tenant scope for an existing action lifecycle.

        Returns the tenant_id captured at record_action_requested time so
        every follow-up record (risk_scored, decision, execution,
        escalation, outcome) carries the same scope automatically.
        """
        with self._tenant_map_lock:
            return self._tenant_for_action.get(action_id, "")

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
            tenant_id=self._tenant_for(action_id),
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
            tenant_id=self._tenant_for(action_id),
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
            tenant_id=self._tenant_for(action_id),
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
            tenant_id=self._tenant_for(action_id),
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
            tenant_id=self._tenant_for(action_id),
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
            tenant_id=self._tenant_for(action_id),
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
        """Cap a sanitised dict's JSON size; over cap to single-key marker.

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
            tenant_id=self._tenant_for(action_id),
        ))

    _KEY_LIFECYCLE_ACTIONS = ("rotated", "revoked", "added")

    def record_key_lifecycle(
        self,
        action: str,
        fingerprint: str,
        *,
        threshold_k: Optional[int] = None,
        signers_n: Optional[int] = None,
        reason: str = "",
        actor: str = "",
        agent_id: str = "system",
        tenant_id: str = "",
    ) -> str:
        """Record a custodian key-lifecycle event (rotation, revocation, add).

        Written as an ordinary audit record, so it inherits the v0.47 hash
        chain and the v0.48 external time anchor. That is what makes
        "revoked before compromise" provable rather than asserted: a
        ``revoked`` marker anchored before a compromise window pins the
        revocation in time, and a compromised key can re-sign but cannot
        re-anchor past chain heads to a time source it does not control.
        :func:`vaara.audit.verify.verify_signed` surfaces these records so a
        reviewer sees the custodian set's history inline with the evidence.

        See the "Key lifecycle" section of
        ``docs/design/threshold-signing-spec.md``.

        Args:
            action: One of ``"rotated"``, ``"revoked"``, ``"added"``.
            fingerprint: The affected custodian public-key fingerprint
                (32-hex-char, as printed by ``vaara keygen`` and written to
                threshold exports).
            threshold_k: The quorum in force after this event, if it changed.
            signers_n: The authorized-set size after this event, if changed.
            reason: Free-text reason (e.g. ``"scheduled rotation"``,
                ``"suspected compromise"``).
            actor: Who performed the action (operator or custodian id).
            agent_id: Defaults to ``"system"`` — lifecycle events are
                operational, not agent-initiated.
            tenant_id: Tenant scope bound into the hash chain.

        Returns:
            The new record's ``record_id``.

        Raises:
            ValueError: If ``action`` is not a recognized lifecycle action,
                or ``fingerprint`` is empty.
        """
        if action not in self._KEY_LIFECYCLE_ACTIONS:
            raise ValueError(
                "record_key_lifecycle: action must be one of "
                f"{list(self._KEY_LIFECYCLE_ACTIONS)}, got {action!r}"
            )
        if not fingerprint:
            raise ValueError("record_key_lifecycle: fingerprint is required")

        articles = self._get_regulatory_articles(
            EventType.KEY_LIFECYCLE, frozenset({RegulatoryDomain.EU_AI_ACT}),
        )
        data: dict = {
            "action": action,
            "fingerprint": self._cap_record_str(fingerprint, 128),
            "reason": self._cap_record_str(reason, self._MAX_OVERRIDE_REASON_LEN),
            "actor": self._cap_record_str(actor, self._MAX_OVERRIDER_LEN),
        }
        if threshold_k is not None:
            data["threshold_k"] = int(threshold_k)
        if signers_n is not None:
            data["signers_n"] = int(signers_n)

        record_id = str(uuid.uuid4())
        self._append(AuditRecord(
            record_id=record_id,
            action_id=str(uuid.uuid4()),
            event_type=EventType.KEY_LIFECYCLE,
            timestamp=time.time(),
            agent_id=self._cap_record_str(agent_id, self._MAX_AGENT_ID_LEN),
            tool_name="key_lifecycle",
            data=data,
            regulatory_articles=articles,
            tenant_id=tenant_id,
        ))
        return record_id

    # ── Querying ──────────────────────────────────────────────────

    def snapshot(self) -> list[AuditRecord]:
        """Return a point-in-time list of every record, in append order.

        Read-only view for consumers that need the whole trail at once
        (e.g. delegation-chain reconstruction). Taken under the chain lock
        so it never races a concurrent ``_append``; the returned list is a
        fresh container, though the ``AuditRecord`` objects are shared.
        """
        with self._lock:
            return list(self._records)

    def get_action_trail(self, action_id: str) -> list[AuditRecord]:
        """Get all events for a specific action, in order."""
        with self._lock:
            return list(self._by_action.get(action_id, []))

    def get_action_chain_scoped(
        self, action_id: str, tenant_id: str = ""
    ) -> list[tuple[int, AuditRecord]]:
        """Tenant-scoped chain read for the reference server.

        Returns ``(chain_position, record)`` pairs for ``action_id``, but only
        the records whose ``tenant_id`` matches the caller's ``tenant_id``. A
        caller scoped to one tenant can never read another tenant's records,
        and the empty-string tenant (single-tenant deployments) only ever sees
        empty-tenant records. Positions are resolved in a single pass under the
        lock rather than an ``O(n)`` ``index()`` per record.

        Returns an empty list both when the action is unknown and when it
        belongs to a different tenant — the caller maps both to 404, so a
        cross-tenant probe cannot use the response to confirm an action_id
        exists for another tenant.
        """
        want = tenant_id or ""
        with self._lock:
            if action_id not in self._by_action:
                return []
            return [
                (pos, r)
                for pos, r in enumerate(self._records)
                if r.action_id == action_id and (r.tenant_id or "") == want
            ]

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

    def anchor_head(self, client: Any) -> Any:
        """Anchor the current chain head to an external time authority.

        ``client`` is a time-anchor backend such as
        ``vaara.audit.timeanchor.RFC3161TimeAnchorClient``. Reads the latest
        record's hash under the lock, obtains a trusted timestamp over it, and
        appends the resulting ``TimeAnchor`` to this trail's anchor list. The
        anchor proves the chain head existed no later than the attested time,
        attested by a party outside Vaara's trust boundary, which is what
        defeats post-hoc backdating after a signing-key compromise.

        Raises ``ValueError`` if the trail is empty (nothing to anchor) and
        ``TimeAnchorError`` if the authority cannot be reached or its token
        does not verify. The anchor is appended only after the token verifies.
        """
        with self._lock:
            if not self._records:
                raise ValueError("cannot anchor an empty trail")
            position = len(self._records) - 1
            head_hash = self._records[-1].record_hash
        anchor = client.anchor(position, head_hash)
        with self._lock:
            self._anchors.append(anchor)
        return anchor

    def enable_auto_anchor(self, client: Any, *, every_records: int = 32) -> None:
        """Anchor the chain head automatically every ``every_records`` records.

        ``client`` is a time-anchor backend (e.g.
        ``vaara.audit.timeanchor.RFC3161TimeAnchorClient``). After every
        ``every_records`` appended records the trail anchors its current head
        to that authority, so a deployment does not have to call
        :meth:`anchor_head` by hand. No TSA is configured by default; this is
        the opt-in that turns anchoring on.

        Fail-open: if the authority is unreachable or its token does not
        verify, the trail records a chained ``ANCHOR_GAP`` marker (carrying the
        reason and the head it tried to anchor) instead of raising, so the
        unanchored window is itself visible and tamper-evident in the chain.
        The TSA round trip runs off the hash-chain lock, so it does not block
        concurrent recording beyond the triggering append.

        Raises ``ValueError`` if ``every_records`` is not a positive integer.
        """
        if every_records < 1:
            raise ValueError("every_records must be a positive integer")
        with self._anchor_lock:
            self._anchor_client = client
            self._anchor_cadence = every_records
            self._records_since_anchor = 0

    # ── Internal ──────────────────────────────────────────────────

    def _append(self, record: AuditRecord) -> None:
        """Append a record, then anchor the head if the cadence is due.

        The chaining itself is in :meth:`_append_chained`; this wrapper adds
        the automatic-anchor trigger so the gap marker (which appends via
        ``_append_chained`` directly) cannot recurse back into anchoring.
        """
        self._append_chained(record)
        self._maybe_auto_anchor()

    def _maybe_auto_anchor(self) -> None:
        """Anchor the head when the per-record cadence is reached (fail-open)."""
        if self._anchor_client is None:
            return
        with self._anchor_lock:
            self._records_since_anchor += 1
            if self._records_since_anchor < self._anchor_cadence:
                return
            self._records_since_anchor = 0
            client = self._anchor_client
        with self._lock:
            if not self._records:
                return
            position = len(self._records) - 1
            head_hash = self._records[-1].record_hash
        try:
            anchor = client.anchor(position, head_hash)
        except Exception as exc:  # fail-open: never break recording on a TSA fault
            self._record_anchor_gap(position, head_hash, repr(exc), client)
            return
        with self._lock:
            self._anchors.append(anchor)

    def _record_anchor_gap(
        self, position: int, head_hash: str, reason: str, client: Any
    ) -> None:
        """Append a chained ANCHOR_GAP marker for a failed auto-anchor attempt."""
        logger.warning(
            "auto-anchor failed at chain position %d (%s); recording gap marker",
            position, reason,
        )
        self._append_chained(AuditRecord(
            record_id=str(uuid.uuid4()),
            action_id="anchor-gap",
            event_type=EventType.ANCHOR_GAP,
            timestamp=time.time(),
            agent_id="vaara",
            tool_name="timeanchor",
            data={
                "reason": self._cap_record_str(reason, 512),
                "attempted_chain_position": position,
                "chain_head_hash": head_hash,
                "tsa_url": getattr(client, "tsa_url", ""),
            },
        ))

    def _append_chained(self, record: AuditRecord) -> None:
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
            # Stamp the current chain format on every fresh record so its
            # tenant_id is bound into the hash. Records reloaded from storage
            # never pass through _append, so their stored version is left
            # intact and old trails keep re-verifying.
            record.chain_version = _CURRENT_CHAIN_VERSION
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
