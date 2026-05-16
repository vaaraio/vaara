"""Article 12 commit-prove receipt pair.

A receipt binds a gate-time commitment to its post-execution outcome
via two SHA-256 hashes:

- ``commit_hash`` covers (action_id, decision, risk_score, thresholds,
  decided_at). Proves the runtime committed to a specific decision before
  the action ran.
- ``outcome_hash`` covers (action_id, commit_hash, outcome_severity,
  outcome_payload, recorded_at). Proves a specific outcome belongs to
  that specific commitment by embedding the commit_hash.

The two hashes form an offline-verifiable chain of accountability for one
action. The full audit chain still protects integrity in aggregate; this
is a structured pairing for per-action handoff. Verification needs only
``hashlib.sha256`` — no key infrastructure, no external crypto libs.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from vaara.audit.trail import AuditRecord, AuditTrail, EventType


@dataclass(frozen=True)
class CommitPayload:
    """Canonical commit-side payload. Pre-action decision."""

    action_id: str
    decision: str
    risk_score: float
    threshold_allow: float
    threshold_deny: float
    decided_at: float

    def canonical_json(self) -> str:
        return _canonical_json(asdict(self))

    def hash(self) -> str:
        return _sha256_hex(self.canonical_json())


@dataclass(frozen=True)
class OutcomePayload:
    """Canonical outcome-side payload. Post-execution observation."""

    action_id: str
    commit_hash: str
    outcome_severity: float
    outcome_payload: dict = field(default_factory=dict)
    recorded_at: float = 0.0

    def canonical_json(self) -> str:
        return _canonical_json(asdict(self))

    def hash(self) -> str:
        return _sha256_hex(self.canonical_json())


@dataclass(frozen=True)
class Receipt:
    """Receipt pair binding a commitment to its outcome."""

    commit: CommitPayload
    outcome: Optional[OutcomePayload] = None

    @property
    def commit_hash(self) -> str:
        return self.commit.hash()

    @property
    def outcome_hash(self) -> Optional[str]:
        return self.outcome.hash() if self.outcome is not None else None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": "1.0",
            "commit": {
                "payload": asdict(self.commit),
                "hash": self.commit_hash,
            },
        }
        if self.outcome is not None:
            d["outcome"] = {
                "payload": asdict(self.outcome),
                "hash": self.outcome_hash,
            }
        return d


def verify_receipt(receipt: Receipt) -> bool:
    """Recompute hashes and verify the commit-outcome binding."""
    if receipt.commit.hash() != receipt.commit_hash:
        return False
    if receipt.outcome is None:
        return True
    if receipt.outcome.commit_hash != receipt.commit_hash:
        return False
    if receipt.outcome.hash() != receipt.outcome_hash:
        return False
    return True


def verify_receipt_dict(d: dict[str, Any]) -> bool:
    """Verify a serialized receipt (as produced by ``Receipt.to_dict``)."""
    try:
        commit_d = d["commit"]
        commit = CommitPayload(**commit_d["payload"])
        if commit.hash() != commit_d["hash"]:
            return False
        outcome_d = d.get("outcome")
        if outcome_d is None:
            return True
        outcome = OutcomePayload(**outcome_d["payload"])
        if outcome.commit_hash != commit_d["hash"]:
            return False
        if outcome.hash() != outcome_d["hash"]:
            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def extract_receipt(trail: AuditTrail, action_id: str) -> Optional[Receipt]:
    """Derive a receipt pair from an existing trail.

    Reads the per-action records and reconstructs the commit and (if
    present) outcome payloads. Returns None if no decision exists for the
    action.
    """
    records = trail._by_action.get(action_id, [])
    if not records:
        return None
    commit = _commit_from_records(records, action_id)
    if commit is None:
        return None
    outcome = _outcome_from_records(records, action_id, commit.hash())
    return Receipt(commit=commit, outcome=outcome)


def _commit_from_records(
    records: list[AuditRecord], action_id: str,
) -> Optional[CommitPayload]:
    decision_record: Optional[AuditRecord] = None
    risk_record: Optional[AuditRecord] = None
    for r in records:
        if r.event_type in (EventType.DECISION_MADE, EventType.ACTION_BLOCKED):
            decision_record = r
        if r.event_type == EventType.RISK_SCORED:
            risk_record = r
    if decision_record is None:
        return None
    data = decision_record.data or {}
    decision = str(data.get("decision", "")) or _event_to_decision(
        decision_record.event_type,
    )
    risk_score = _coerce_float(data.get("risk_score"))
    threshold_allow, threshold_deny = _thresholds_from_risk_record(risk_record)
    if risk_score is None and risk_record is not None:
        risk_score = _coerce_float((risk_record.data or {}).get("point_estimate"))
    if risk_score is None:
        risk_score = 0.0
    return CommitPayload(
        action_id=action_id,
        decision=decision,
        risk_score=float(risk_score),
        threshold_allow=threshold_allow,
        threshold_deny=threshold_deny,
        decided_at=float(decision_record.timestamp),
    )


def _outcome_from_records(
    records: list[AuditRecord], action_id: str, commit_hash: str,
) -> Optional[OutcomePayload]:
    for r in records:
        if r.event_type != EventType.OUTCOME_RECORDED:
            continue
        data = r.data or {}
        severity = _coerce_float(data.get("outcome_severity"))
        if severity is None:
            severity = _coerce_float(data.get("severity"))
        if severity is None:
            severity = 0.0
        return OutcomePayload(
            action_id=action_id,
            commit_hash=commit_hash,
            outcome_severity=float(severity),
            outcome_payload={
                k: v for k, v in data.items()
                if k not in ("outcome_severity", "severity")
            },
            recorded_at=float(r.timestamp),
        )
    return None


def _thresholds_from_risk_record(
    risk_record: Optional[AuditRecord],
) -> tuple[float, float]:
    if risk_record is None:
        return 0.4, 0.7
    data = risk_record.data or {}
    ta = _coerce_float(data.get("threshold_allow")) or 0.4
    td = _coerce_float(data.get("threshold_deny")) or 0.7
    return float(ta), float(td)


def _event_to_decision(event_type: EventType) -> str:
    return "deny" if event_type == EventType.ACTION_BLOCKED else "allow"


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _canonical_json(d: dict[str, Any]) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


__all__ = [
    "CommitPayload",
    "OutcomePayload",
    "Receipt",
    "extract_receipt",
    "verify_receipt",
    "verify_receipt_dict",
]
