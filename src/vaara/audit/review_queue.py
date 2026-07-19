# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Human-in-the-loop review queue.

EU AI Act Article 14 requires high-risk AI systems to be designed for
effective oversight by natural persons, and Article 14(4)(d) requires
the human to be able to decide not to use the system or override its
output. The pipeline emits an ``ESCALATION_SENT`` audit record when a
borderline score routes a decision to ``escalate``. Without a queue,
that record is the end of the story.

This module is the storage layer:

* Single SQLite table, separate file from the audit DB. Audit is
  append-only by contract; the queue needs ``UPDATE`` for status,
  claim, and resolution.
* Statuses: ``pending to claimed to resolved`` is the happy path.
  ``pending to expired`` is the stale-without-claim path. Resolved /
  expired are terminal.
* Resolution: ``allow``, ``deny``, ``abstain``. ``abstain`` means the
  reviewer declined to decide — keeps ``escalate`` as the final verdict.
* Loose coupling: ``enqueue`` does not write an audit record (the
  pipeline already wrote ``ESCALATION_SENT``). ``resolve`` accepts an
  optional ``trail`` and writes ``ESCALATION_RESOLVED`` when supplied.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

STATUS_PENDING = "pending"
STATUS_CLAIMED = "claimed"
STATUS_RESOLVED = "resolved"
STATUS_EXPIRED = "expired"
_VALID_STATUSES = frozenset({
    STATUS_PENDING, STATUS_CLAIMED, STATUS_RESOLVED, STATUS_EXPIRED,
})

RESOLUTION_ALLOW = "allow"
RESOLUTION_DENY = "deny"
RESOLUTION_ABSTAIN = "abstain"
_VALID_RESOLUTIONS = frozenset({
    RESOLUTION_ALLOW, RESOLUTION_DENY, RESOLUTION_ABSTAIN,
})

_MAX_REVIEWER_LEN = 256
_MAX_JUSTIFICATION_LEN = 8192
_MAX_REASON_LEN = 4096
_MAX_AGENT_ID_LEN = 256
_MAX_TOOL_NAME_LEN = 512
_MAX_ACTION_TYPE_LEN = 128
_MAX_BUCKET_CATEGORY_LEN = 128
_MAX_DATA_JSON_BYTES = 64 * 1024


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS review_queue_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_queue (
    queue_id          TEXT PRIMARY KEY,
    action_id         TEXT NOT NULL,
    agent_id          TEXT NOT NULL,
    tool_name         TEXT NOT NULL,
    action_type       TEXT NOT NULL DEFAULT '',
    risk_score        REAL NOT NULL,
    conformal_lower   REAL NOT NULL,
    conformal_upper   REAL NOT NULL,
    bucket_category   TEXT,
    reason            TEXT NOT NULL DEFAULT '',
    parameters        TEXT NOT NULL DEFAULT '{}',
    context           TEXT NOT NULL DEFAULT '{}',
    signals           TEXT NOT NULL DEFAULT '{}',
    status            TEXT NOT NULL,
    enqueued_at       REAL NOT NULL,
    claimed_at        REAL,
    claimed_by        TEXT,
    resolved_at       REAL,
    resolved_by       TEXT,
    resolution        TEXT,
    justification     TEXT
);

CREATE INDEX IF NOT EXISTS idx_rq_status      ON review_queue(status);
CREATE INDEX IF NOT EXISTS idx_rq_enqueued_at ON review_queue(enqueued_at);
CREATE INDEX IF NOT EXISTS idx_rq_action_id   ON review_queue(action_id);
CREATE INDEX IF NOT EXISTS idx_rq_agent_id    ON review_queue(agent_id);
"""


def _cap_str(value: Any, max_len: int) -> str:
    if not isinstance(value, str):
        value = "" if value is None else str(value)
    if len(value) <= max_len:
        return value
    marker = f"...[TRUNCATED:{len(value)}B]"
    keep = max_len - len(marker)
    if keep <= 0:
        return value[:max_len]
    return value[:keep] + marker


def _cap_json(d: Any, max_bytes: int) -> str:
    if d is None:
        return "{}"
    try:
        encoded = json.dumps(d, default=str)
    except (TypeError, ValueError):
        return json.dumps({"_unserialisable": True})
    if len(encoded) <= max_bytes:
        return encoded
    keys = list(d.keys())[:20] if isinstance(d, dict) else []
    return json.dumps({
        "_truncated": True,
        "_original_bytes": len(encoded),
        "_cap_bytes": max_bytes,
        "_keys": keys,
    })


def _loads(text: Any) -> Any:
    if not text:
        return {}
    if isinstance(text, (dict, list)):
        return text
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return {}


@dataclass
class ReviewItem:
    """One row from the review queue."""
    queue_id: str
    action_id: str
    agent_id: str
    tool_name: str
    action_type: str
    risk_score: float
    conformal_lower: float
    conformal_upper: float
    bucket_category: Optional[str]
    reason: str
    parameters: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)
    status: str = STATUS_PENDING
    enqueued_at: float = 0.0
    claimed_at: Optional[float] = None
    claimed_by: Optional[str] = None
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None
    resolution: Optional[str] = None
    justification: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "queue_id": self.queue_id, "action_id": self.action_id,
            "agent_id": self.agent_id, "tool_name": self.tool_name,
            "action_type": self.action_type, "risk_score": self.risk_score,
            "conformal_lower": self.conformal_lower,
            "conformal_upper": self.conformal_upper,
            "bucket_category": self.bucket_category, "reason": self.reason,
            "parameters": self.parameters, "context": self.context,
            "signals": self.signals, "status": self.status,
            "enqueued_at": self.enqueued_at, "claimed_at": self.claimed_at,
            "claimed_by": self.claimed_by, "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by, "resolution": self.resolution,
            "justification": self.justification,
        }

    @property
    def interval_width(self) -> float:
        return max(0.0, self.conformal_upper - self.conformal_lower)


class ReviewQueueError(Exception):
    """Generic queue error."""


class ItemNotFoundError(ReviewQueueError):
    """No item with the requested queue_id."""


class InvalidTransitionError(ReviewQueueError):
    """Tried to claim a non-pending item or resolve a terminal item."""


class ReviewQueue:
    """SQLite-backed human-in-the-loop review queue."""

    def __init__(self, db_path: str | Path) -> None:
        raw_path = Path(db_path)
        self._db_path = (
            raw_path if str(db_path) == ":memory:" else raw_path.resolve()
        )
        self._conn = sqlite3.connect(
            str(self._db_path),
            isolation_level=None,
            check_same_thread=False,
        )
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _init_schema(self) -> None:
        self._conn.executescript(SCHEMA_SQL)
        row = self._conn.execute(
            "SELECT value FROM review_queue_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO review_queue_meta (key, value) VALUES "
                "('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            return
        stored = int(row[0])
        if stored > SCHEMA_VERSION:
            raise RuntimeError(
                f"Review queue schema version {stored} is newer than this "
                f"vaara version (supports up to {SCHEMA_VERSION})."
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ReviewQueue":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def enqueue(
        self,
        *,
        action_id: str,
        agent_id: str,
        tool_name: str,
        risk_score: float,
        conformal_lower: float,
        conformal_upper: float,
        action_type: str = "",
        bucket_category: Optional[str] = None,
        reason: str = "",
        parameters: Optional[dict] = None,
        context: Optional[dict] = None,
        signals: Optional[dict] = None,
        queue_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> str:
        """Enqueue a borderline action for human review."""
        qid = queue_id or str(uuid.uuid4())
        ts = now if now is not None else time.time()
        params_json = _cap_json(parameters, _MAX_DATA_JSON_BYTES)
        context_json = _cap_json(context, _MAX_DATA_JSON_BYTES)
        signals_json = _cap_json(signals, _MAX_DATA_JSON_BYTES)
        with self._lock:
            self._conn.execute(
                """INSERT INTO review_queue (
                    queue_id, action_id, agent_id, tool_name, action_type,
                    risk_score, conformal_lower, conformal_upper,
                    bucket_category, reason, parameters, context, signals,
                    status, enqueued_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    qid,
                    _cap_str(action_id, _MAX_AGENT_ID_LEN),
                    _cap_str(agent_id, _MAX_AGENT_ID_LEN),
                    _cap_str(tool_name, _MAX_TOOL_NAME_LEN),
                    _cap_str(action_type, _MAX_ACTION_TYPE_LEN),
                    float(risk_score),
                    float(conformal_lower),
                    float(conformal_upper),
                    _cap_str(bucket_category, _MAX_BUCKET_CATEGORY_LEN)
                        if bucket_category else None,
                    _cap_str(reason, _MAX_REASON_LEN),
                    params_json,
                    context_json,
                    signals_json,
                    STATUS_PENDING,
                    ts,
                ),
            )
        return qid

    def claim(
        self,
        queue_id: str,
        *,
        reviewer: str,
        now: Optional[float] = None,
    ) -> ReviewItem:
        """Claim a pending item. Optimistic — loser of a race raises."""
        ts = now if now is not None else time.time()
        reviewer_capped = _cap_str(reviewer, _MAX_REVIEWER_LEN)
        with self._lock:
            cur = self._conn.execute(
                """UPDATE review_queue
                   SET status=?, claimed_at=?, claimed_by=?
                   WHERE queue_id=? AND status=?""",
                (STATUS_CLAIMED, ts, reviewer_capped, queue_id, STATUS_PENDING),
            )
            if cur.rowcount == 0:
                self._raise_for_failed_transition(queue_id, "claim")
        return self.get(queue_id)

    def resolve(
        self,
        queue_id: str,
        *,
        reviewer: str,
        resolution: str,
        justification: str = "",
        trail: Optional[Any] = None,
        now: Optional[float] = None,
    ) -> ReviewItem:
        """Resolve an item with ``allow``, ``deny``, or ``abstain``.

        Accepts items in either ``pending`` or ``claimed`` state — a
        reviewer can resolve without an explicit claim. When ``trail``
        is supplied, writes ``ESCALATION_RESOLVED`` so the Article
        14(4)(d) evidence row lands on the hash chain.
        """
        if resolution not in _VALID_RESOLUTIONS:
            raise ValueError(
                f"resolution must be one of {sorted(_VALID_RESOLUTIONS)}, "
                f"got {resolution!r}"
            )
        ts = now if now is not None else time.time()
        reviewer_capped = _cap_str(reviewer, _MAX_REVIEWER_LEN)
        justification_capped = _cap_str(justification, _MAX_JUSTIFICATION_LEN)
        with self._lock:
            cur = self._conn.execute(
                """UPDATE review_queue
                   SET status=?, resolved_at=?, resolved_by=?,
                       resolution=?, justification=?
                   WHERE queue_id=? AND status IN (?, ?)""",
                (
                    STATUS_RESOLVED, ts, reviewer_capped,
                    resolution, justification_capped,
                    queue_id, STATUS_PENDING, STATUS_CLAIMED,
                ),
            )
            if cur.rowcount == 0:
                self._raise_for_failed_transition(queue_id, "resolve")
        item = self.get(queue_id)
        if trail is not None:
            try:
                trail.record_escalation_resolved(
                    action_id=item.action_id,
                    agent_id=item.agent_id,
                    tool_name=item.tool_name,
                    resolution=resolution,
                    reviewer=reviewer_capped,
                    justification=justification_capped,
                )
            except Exception:
                logger.exception(
                    "review_queue: failed to write ESCALATION_RESOLVED "
                    "for queue_id=%s action_id=%s",
                    queue_id, item.action_id,
                )
        return item

    def expire_stale(
        self,
        *,
        timeout_seconds: float,
        now: Optional[float] = None,
        dry_run: bool = False,
    ) -> int:
        """Mark pending items older than ``timeout_seconds`` as expired.

        Only ``pending`` items are eligible — a claimed item is under
        active review and is left alone.
        """
        if timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be > 0, got {timeout_seconds}"
            )
        ts = now if now is not None else time.time()
        cutoff = ts - timeout_seconds
        with self._lock:
            if dry_run:
                row = self._conn.execute(
                    """SELECT COUNT(*) FROM review_queue
                       WHERE status=? AND enqueued_at < ?""",
                    (STATUS_PENDING, cutoff),
                ).fetchone()
                return int(row[0])
            cur = self._conn.execute(
                """UPDATE review_queue
                   SET status=?, resolved_at=?
                   WHERE status=? AND enqueued_at < ?""",
                (STATUS_EXPIRED, ts, STATUS_PENDING, cutoff),
            )
            return cur.rowcount

    def get(self, queue_id: str) -> ReviewItem:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM review_queue WHERE queue_id=?", (queue_id,)
            ).fetchone()
        if row is None:
            raise ItemNotFoundError(f"queue_id {queue_id!r} not found")
        return _row_to_item(row)

    def list_items(
        self,
        *,
        status: Optional[str] = STATUS_PENDING,
        limit: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> list[ReviewItem]:
        """List items. ``status=None`` lists across all statuses."""
        if status is not None and status not in _VALID_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(_VALID_STATUSES)} or None, "
                f"got {status!r}"
            )
        capped_limit = 1000 if limit is None else max(1, min(1000, int(limit)))
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        if agent_id:
            clauses.append("agent_id=?")
            params.append(agent_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT * FROM review_queue" + where +
            " ORDER BY enqueued_at ASC LIMIT ?"
        )
        params.append(capped_limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_item(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) FROM review_queue GROUP BY status"
            ).fetchall()
        out = {s: 0 for s in _VALID_STATUSES}
        for r in rows:
            out[r[0]] = int(r[1])
        return out

    def _raise_for_failed_transition(self, queue_id: str, op: str) -> None:
        row = self._conn.execute(
            "SELECT status FROM review_queue WHERE queue_id=?", (queue_id,)
        ).fetchone()
        if row is None:
            raise ItemNotFoundError(f"queue_id {queue_id!r} not found")
        raise InvalidTransitionError(
            f"cannot {op} item {queue_id!r}: current status is {row[0]!r}"
        )


def _row_to_item(row: sqlite3.Row) -> ReviewItem:
    return ReviewItem(
        queue_id=row["queue_id"],
        action_id=row["action_id"],
        agent_id=row["agent_id"],
        tool_name=row["tool_name"],
        action_type=row["action_type"] or "",
        risk_score=float(row["risk_score"]),
        conformal_lower=float(row["conformal_lower"]),
        conformal_upper=float(row["conformal_upper"]),
        bucket_category=row["bucket_category"],
        reason=row["reason"] or "",
        parameters=_loads(row["parameters"]),
        context=_loads(row["context"]),
        signals=_loads(row["signals"]),
        status=row["status"],
        enqueued_at=float(row["enqueued_at"]),
        claimed_at=float(row["claimed_at"]) if row["claimed_at"] is not None else None,
        claimed_by=row["claimed_by"],
        resolved_at=float(row["resolved_at"]) if row["resolved_at"] is not None else None,
        resolved_by=row["resolved_by"],
        resolution=row["resolution"],
        justification=row["justification"],
    )
