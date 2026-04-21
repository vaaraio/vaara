"""SQLite persistence backend for the audit trail.

The in-memory trail is fast but volatile.  This backend writes every record
to SQLite as it arrives (via on_record callback) and can reconstruct the
full trail from disk.

Design principles:
- **WAL mode** for concurrent read/write (readers never block writers)
- **Append-only** — no UPDATE or DELETE, matching the immutability guarantee
- **Hash chain verified on load** — detects on-disk tampering
- **Regulatory domain indexed** — fast compliance queries by regulation
- **JSON data column** — flexible schema for action-specific fields

EU AI Act Article 12(2): Logging capabilities shall allow for the recording
of events relevant to identify situations that may result in the AI system
posing a risk.  SQLite's ACID guarantees that no event is silently lost.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from vaara.audit.trail import AuditRecord, AuditTrail, EventType
from vaara.auth import APIKey, Role, _hash_key, generate_api_key

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2


def _scrub_nonfinite(obj: Any) -> Any:
    # Strict JSON (RFC 8259) forbids NaN/Infinity tokens. Python's default
    # json.dumps emits them; Go encoding/json, Rust serde_json, strict Node
    # JSON.parse, and most JSON-RPC validators then reject the audit export
    # — regulator-side evidence becomes unreadable. Mirrors the Loop 34
    # mcp_server pattern and the Loop 35 _sanitize.json_safe guard, applied
    # here as a second line of defence for legacy records or direct
    # AuditRecord callers that bypass pipeline sanitisation.
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _scrub_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_nonfinite(v) for v in obj]
    return obj


def _strict_json_dumps(obj: Any) -> str:
    return json.dumps(_scrub_nonfinite(obj), allow_nan=False, default=str)

# Schema v2 — full DDL for fresh databases.
# Migrations for v1→v2 upgrades are in _MIGRATIONS below.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS audit_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_records (
    record_id     TEXT PRIMARY KEY,
    action_id     TEXT NOT NULL,
    event_type    TEXT NOT NULL,
    timestamp     REAL NOT NULL,
    agent_id      TEXT NOT NULL,
    tool_name     TEXT NOT NULL,
    data          TEXT NOT NULL DEFAULT '{}',
    regulatory    TEXT NOT NULL DEFAULT '[]',
    previous_hash TEXT NOT NULL DEFAULT '',
    record_hash   TEXT NOT NULL DEFAULT '',
    seq           INTEGER NOT NULL,
    tenant_id     TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_action_id   ON audit_records(action_id);
CREATE INDEX IF NOT EXISTS idx_agent_id    ON audit_records(agent_id);
CREATE INDEX IF NOT EXISTS idx_event_type  ON audit_records(event_type);
CREATE INDEX IF NOT EXISTS idx_timestamp   ON audit_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_tool_name   ON audit_records(tool_name);
CREATE INDEX IF NOT EXISTS idx_tenant_id   ON audit_records(tenant_id);

-- GDPR Article 17 (Right to Erasure) redaction table.
CREATE TABLE IF NOT EXISTS gdpr_redactions (
    original_id  TEXT PRIMARY KEY,
    replacement  TEXT NOT NULL DEFAULT '[REDACTED:GDPR-Art17]',
    redacted_at  REAL NOT NULL
);

-- API key store for role-based access control.
-- Plaintext keys are never stored — only PBKDF2-SHA256 hashes.
CREATE TABLE IF NOT EXISTS api_keys (
    key_id       TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    role         TEXT NOT NULL,
    key_hash     TEXT NOT NULL UNIQUE,
    created_at   REAL NOT NULL,
    last_used_at REAL
);
"""

# Incremental migrations applied when opening a DB with stored version < SCHEMA_VERSION.
# Key = version being upgraded FROM (i.e., _MIGRATIONS[1] upgrades v1 → v2).
_MIGRATIONS: dict[int, str] = {
    1: """
    ALTER TABLE audit_records ADD COLUMN tenant_id TEXT NOT NULL DEFAULT '';
    CREATE INDEX IF NOT EXISTS idx_tenant_id ON audit_records(tenant_id);
    CREATE TABLE IF NOT EXISTS gdpr_redactions (
        original_id  TEXT PRIMARY KEY,
        replacement  TEXT NOT NULL DEFAULT '[REDACTED:GDPR-Art17]',
        redacted_at  REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS api_keys (
        key_id       TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        role         TEXT NOT NULL,
        key_hash     TEXT NOT NULL UNIQUE,
        created_at   REAL NOT NULL,
        last_used_at REAL
    );
    """,
}


class SQLiteAuditBackend:
    """Persistent audit trail backed by SQLite.

    Usage::

        backend = SQLiteAuditBackend("audit.db")
        trail = AuditTrail(on_record=backend.write_record)

        # On restart — reload the trail
        trail = backend.load_trail()

        # Compliance query — all DORA-relevant records
        records = backend.query_by_regulation("dora")

        # Export for external audit
        backend.export_jsonl(Path("audit_export.jsonl"))
    """

    def __init__(self, db_path: str | Path, tenant_id: str = "") -> None:
        """Open (or create) a Vaara audit database.

        Args:
            db_path:   Path to the SQLite file. Use ``":memory:"`` for tests.
            tenant_id: Tenant scope for all writes and queries from this
                       instance.  An empty string means "unscoped" (single-
                       tenant).  Multi-tenant deployments pass distinct
                       tenant_id values; queries only return records for
                       that tenant.
        """
        raw_path = Path(db_path)
        # ":memory:" is a special SQLite URI — don't try to resolve it.
        self._db_path = raw_path if str(db_path) == ":memory:" else raw_path.resolve()
        self._tenant_id = tenant_id

        # Warn loudly if the caller passed a path that traverses above its
        # apparent parent (e.g. "../../tmp/evil.db"). Resolving removes the
        # ".." segments; if the resolved path differs significantly from what
        # a naive caller might expect, that is a signal worth surfacing.
        if str(db_path) != ":memory:" and str(self._db_path) != str(raw_path.absolute()):
            logger.warning(
                "SQLiteAuditBackend: db_path %r resolved to %s — "
                "path traversal detected. Ensure this location is intentional.",
                str(db_path), self._db_path,
            )
        # check_same_thread=False lets us use this connection from any
        # thread (LangChain tool execution, web servers, async workers).
        # The Lock below serializes all access to the connection, which
        # is the SQLite python binding's thread-safety contract.
        self._conn = sqlite3.connect(
            str(self._db_path),
            isolation_level=None,  # Autocommit for WAL mode
            check_same_thread=False,
        )
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()
        # Load GDPR redaction map into memory for O(1) read-time substitution.
        self._redaction_cache: dict[str, str] = self._load_redaction_cache()

    def _init_schema(self) -> None:
        """Create tables if they don't exist, then run any pending migrations."""
        self._conn.executescript(SCHEMA_SQL)
        row = self._conn.execute(
            "SELECT value FROM audit_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            # Fresh database — already at current version via SCHEMA_SQL.
            self._conn.execute(
                "INSERT INTO audit_meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        else:
            stored = int(row[0])
            if stored > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Audit DB schema version {stored} is newer than this vaara "
                    f"version (supports up to {SCHEMA_VERSION}). Upgrade vaara."
                )
            if stored < SCHEMA_VERSION:
                self._run_migrations(stored, SCHEMA_VERSION)

    def _run_migrations(self, from_version: int, to_version: int) -> None:
        """Apply incremental schema migrations from_version up to to_version."""
        for v in range(from_version, to_version):
            sql = _MIGRATIONS.get(v)
            if sql:
                logger.info(
                    "Migrating audit DB schema v%d → v%d at %s",
                    v, v + 1, self._db_path,
                )
                # SQLite doesn't support transactional DDL for all statements
                # (e.g. ALTER TABLE). Execute each statement individually so
                # errors are reported at the failing statement, not the batch.
                for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
                    try:
                        self._conn.execute(stmt)
                    except Exception as exc:
                        # Some statements may fail if already applied (e.g.
                        # duplicate column from a partial migration). Log and
                        # continue — idempotency is more important than hard
                        # failure on a re-run.
                        logger.warning(
                            "Migration v%d stmt skipped (%s: %s): %.80s",
                            v, type(exc).__name__, exc, stmt,
                        )
        self._conn.execute(
            "UPDATE audit_meta SET value=? WHERE key='schema_version'",
            (str(to_version),),
        )
        logger.info("Audit DB schema migrated to v%d", to_version)

    def _get_max_seq(self) -> int:
        """Get the highest sequence number in the DB (diagnostic / tests)."""
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq), -1) FROM audit_records"
        ).fetchone()
        return row[0]

    # ── Write path ────────────────────────────────────────────────

    def write_record(self, record: AuditRecord) -> None:
        """Callback for AuditTrail.on_record — persists a single record.

        This is called synchronously for every audit event.  SQLite in WAL
        mode handles this efficiently. Serialized via self._lock so the
        underlying connection stays thread-safe.

        Seq is computed atomically inside the INSERT via subquery rather
        than from a cached in-process counter. Two processes sharing the
        same audit DB (e.g., pipeline + migration tool, MCP server +
        load_trail reader, multi-worker gunicorn) would otherwise each
        cache MAX(seq)+1 on open and then issue writes with colliding
        seq values, breaking ORDER BY seq ASC reconstruction in
        load_trail and the hash-chain integrity check below it. SQLite
        serializes writers at the transaction boundary, so the subquery
        reads MAX(seq) at the same point the INSERT takes effect.
        """
        with self._lock:
            self._conn.execute(
                """INSERT INTO audit_records
                   (record_id, action_id, event_type, timestamp, agent_id,
                    tool_name, data, regulatory, previous_hash, record_hash, seq,
                    tenant_id)
                   VALUES (
                     ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                     COALESCE((SELECT MAX(seq) FROM audit_records), -1) + 1,
                     ?
                   )""",
                (
                    record.record_id,
                    record.action_id,
                    record.event_type.value,
                    record.timestamp,
                    record.agent_id,
                    record.tool_name,
                    _strict_json_dumps(record.data),
                    _strict_json_dumps(record.regulatory_articles),
                    record.previous_hash,
                    record.record_hash,
                    self._tenant_id,
                ),
            )

    # ── Read path ─────────────────────────────────────────────────

    def _tenant_clause(self) -> tuple[str, list]:
        """Return (WHERE clause fragment, params) for tenant filtering."""
        if self._tenant_id:
            return "tenant_id = ?", [self._tenant_id]
        return "1=1", []

    def load_trail(self, strict: bool = False) -> AuditTrail:
        """Reconstruct a full AuditTrail from the database.

        Verifies the hash chain on load. If the chain is broken, a log
        ERROR is emitted; with ``strict=True`` a ``RuntimeError`` is
        raised instead so callers that cannot safely proceed on a
        compromised trail (e.g., regulator-facing inspection, forensic
        export) fail loud rather than silently consuming tampered
        evidence.

        A single row with a corrupt ``data``/``regulatory`` JSON column
        no longer DoSes the entire load — that row is logged and loaded
        with empty dicts; the subsequent hash-chain verification will
        flag it via hash mismatch so corruption remains detectable.
        """
        t_clause, t_params = self._tenant_clause()
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM audit_records WHERE {t_clause} ORDER BY seq ASC",
                t_params,
            ).fetchall()

        trail = AuditTrail(on_record=self.write_record)

        corrupt_rows = 0
        for row in rows:
            try:
                record = self._row_to_record(row)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                # Single-row corruption must not destroy the rest of the
                # audit reload. Reconstruct a "skeleton" record with empty
                # dicts and the raw hashes preserved; verify_chain() will
                # then flag the row via hash mismatch (original data was
                # part of the hash input), so corruption stays visible.
                corrupt_rows += 1
                logger.error(
                    "load_trail: corrupt record row seq=%s record_id=%s (%s: %s) — "
                    "loading skeleton; hash-chain verification will flag it",
                    row[10], row[0], type(exc).__name__, exc,
                )
                try:
                    skeleton_event_type = EventType(row[2])
                except (ValueError, KeyError):
                    skeleton_event_type = EventType.ACTION_REQUESTED
                record = AuditRecord(
                    record_id=row[0],
                    action_id=row[1],
                    event_type=skeleton_event_type,
                    timestamp=row[3] if isinstance(row[3], (int, float)) else 0.0,
                    agent_id=row[4] or "",
                    tool_name=row[5] or "",
                    data={"_corrupt": True, "_error": str(exc)},
                    regulatory_articles=[],
                    previous_hash=row[8] or "",
                    record_hash=row[9] or "",
                )
            # Inject directly into the trail (bypass on_record to avoid re-write)
            trail._records.append(record)
            trail._by_action[record.action_id].append(record)
            trail._last_hash = record.record_hash

        # Expose the skeleton-row count on the returned trail so callers
        # can observe reload-time corruption without parsing logs. This
        # is the read-side analogue of AuditTrail.persistence_failures
        # (write side). Regulator-facing dashboards / Article 12(2)
        # monitoring should surface both.
        trail._skeleton_records = corrupt_rows

        chain_error = trail.verify_chain()
        if chain_error:
            logger.error("AUDIT CHAIN INTEGRITY FAILURE: %s", chain_error)
            if strict:
                raise RuntimeError(
                    f"load_trail(strict=True) refused to return compromised "
                    f"trail: {chain_error}"
                )

        logger.info(
            "Loaded %d audit records from %s (corrupt_rows=%d)",
            len(rows), self._db_path, corrupt_rows,
        )
        return trail

    def count(self) -> int:
        """Total records (scoped to tenant_id if set)."""
        t_clause, t_params = self._tenant_clause()
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM audit_records WHERE {t_clause}", t_params
            ).fetchone()
        return row[0]

    def _safe_row_to_record(self, row: tuple) -> Optional[AuditRecord]:
        """Convert a DB row to AuditRecord, returning None on corrupt rows."""
        try:
            return self._row_to_record(row)
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            logger.error(
                "query: corrupt record row seq=%s record_id=%s (%s: %s) — skipped",
                row[10] if len(row) > 10 else "?", row[0] if row else "?",
                type(exc).__name__, exc,
            )
            return None

    def query_by_action(self, action_id: str) -> list[AuditRecord]:
        t_clause, t_params = self._tenant_clause()
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM audit_records WHERE action_id=? AND {t_clause} ORDER BY seq ASC",
                [action_id] + t_params,
            ).fetchall()
        return [r for r in (self._safe_row_to_record(row) for row in rows) if r is not None]

    def query_by_agent(self, agent_id: str, limit: int = 100) -> list[AuditRecord]:
        t_clause, t_params = self._tenant_clause()
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM audit_records WHERE agent_id=? AND {t_clause} "
                "ORDER BY seq DESC LIMIT ?",
                [agent_id] + t_params + [limit],
            ).fetchall()
        return [r for r in (self._safe_row_to_record(row) for row in reversed(rows)) if r is not None]

    def query_by_event_type(self, event_type: EventType, limit: int = 100) -> list[AuditRecord]:
        t_clause, t_params = self._tenant_clause()
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM audit_records WHERE event_type=? AND {t_clause} "
                "ORDER BY seq DESC LIMIT ?",
                [event_type.value] + t_params + [limit],
            ).fetchall()
        return [r for r in (self._safe_row_to_record(row) for row in reversed(rows)) if r is not None]

    def query_by_regulation(self, domain: str, limit: int = 500) -> list[AuditRecord]:
        escaped = domain.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        t_clause, t_params = self._tenant_clause()
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM audit_records WHERE regulatory LIKE ? ESCAPE '\\' "
                f"AND {t_clause} ORDER BY seq DESC LIMIT ?",
                [f'%"{escaped}"%'] + t_params + [limit],
            ).fetchall()
        return [r for r in (self._safe_row_to_record(row) for row in reversed(rows)) if r is not None]

    def query_time_range(
        self, start_ts: float, end_ts: Optional[float] = None, limit: int = 1000,
    ) -> list[AuditRecord]:
        if end_ts is None:
            end_ts = time.time()
        t_clause, t_params = self._tenant_clause()
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM audit_records "
                f"WHERE timestamp >= ? AND timestamp <= ? AND {t_clause} "
                "ORDER BY seq ASC LIMIT ?",
                [start_ts, end_ts] + t_params + [limit],
            ).fetchall()
        return [r for r in (self._safe_row_to_record(row) for row in rows) if r is not None]

    def query_blocked(self, limit: int = 50) -> list[AuditRecord]:
        """Get recently blocked actions."""
        return self.query_by_event_type(EventType.ACTION_BLOCKED, limit)

    # ── Statistics ────────────────────────────────────────────────

    def stats(self) -> dict:
        """Database statistics for dashboards (scoped to tenant_id if set)."""
        t_clause, t_params = self._tenant_clause()
        with self._lock:
            rows = self._conn.execute(
                f"SELECT event_type, COUNT(*) FROM audit_records WHERE {t_clause} GROUP BY event_type",
                t_params,
            ).fetchall()
            agent_rows = self._conn.execute(
                f"SELECT COUNT(DISTINCT agent_id) FROM audit_records WHERE {t_clause}", t_params
            ).fetchone()
            time_rows = self._conn.execute(
                f"SELECT MIN(timestamp), MAX(timestamp) FROM audit_records WHERE {t_clause}", t_params
            ).fetchone()
            total_row = self._conn.execute(
                f"SELECT COUNT(*) FROM audit_records WHERE {t_clause}", t_params
            ).fetchone()
        by_type = {row[0]: row[1] for row in rows}

        return {
            "total_records": total_row[0],
            "by_event_type": by_type,
            "unique_agents": agent_rows[0],
            "time_range": {
                "earliest": time_rows[0],
                "latest": time_rows[1],
            },
            "db_path": str(self._db_path),
            "db_size_bytes": self._db_path.stat().st_size if self._db_path.exists() else 0,
        }

    # ── Export ────────────────────────────────────────────────────

    def export_jsonl(self, path: Path, limit: int = 0) -> int:
        """Export records as JSON Lines.  Returns count exported."""
        if not isinstance(limit, int) or limit < 0:
            raise ValueError(f"export_jsonl limit must be a non-negative int, got {limit!r}")
        t_clause, t_params = self._tenant_clause()
        query = f"SELECT * FROM audit_records WHERE {t_clause} ORDER BY seq ASC"
        params = t_params[:]
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        # encoding="utf-8" is explicit — audit exports routinely carry
        # non-ASCII agent names, reasons, and justifications, and the
        # platform default (esp. non-POSIX locales) is not guaranteed UTF-8.
        exported = 0
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                record = self._safe_row_to_record(row)
                if record is None:
                    continue
                f.write(_strict_json_dumps(record.to_dict()) + "\n")
                exported += 1
        return exported

    # ── GDPR Article 17 ───────────────────────────────────────────

    def _load_redaction_cache(self) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT original_id, replacement FROM gdpr_redactions"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def redact_agent_pii(
        self,
        original_id: str,
        replacement: str = "[REDACTED:GDPR-Art17]",
    ) -> int:
        """Redact all occurrences of an agent_id from read results (GDPR Art. 17).

        The stored records and hash chain are NOT modified — append-only
        immutability is preserved for regulatory evidence integrity. Redaction
        is applied at read time: every call to query_*, load_trail, and
        export_jsonl substitutes ``replacement`` wherever ``original_id``
        appears as agent_id.

        **Hash chain note:** after redaction, chain verification will report
        a mismatch for redacted records because agent_id contributes to the
        hash input. This is expected and correct — the erasure event is itself
        compliance evidence. Surface ``trail.chain_intact=False`` in the
        conformity report with a note about GDPR redaction.

        Returns the number of records that will be affected by the redaction.
        """
        if not isinstance(original_id, str) or not original_id:
            raise ValueError("original_id must be a non-empty string")
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM audit_records WHERE agent_id = ?",
                (original_id,),
            ).fetchone()[0]
            self._conn.execute(
                "INSERT OR REPLACE INTO gdpr_redactions "
                "(original_id, replacement, redacted_at) VALUES (?, ?, ?)",
                (original_id, replacement, time.time()),
            )
            self._redaction_cache[original_id] = replacement
        logger.info(
            "GDPR Art.17 redaction: agent_id %r → %r affects %d records",
            original_id, replacement, count,
        )
        return count

    def list_redactions(self) -> list[dict]:
        """Return all active GDPR redactions (original_id redacted, not shown)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT replacement, redacted_at FROM gdpr_redactions"
            ).fetchall()
        return [{"replacement": r[0], "redacted_at": r[1]} for r in rows]

    # ── Internal ──────────────────────────────────────────────────

    def _row_to_record(self, row: tuple) -> AuditRecord:
        """Convert a database row to an AuditRecord, applying GDPR redactions."""
        agent_id = row[4]
        if self._redaction_cache and agent_id in self._redaction_cache:
            agent_id = self._redaction_cache[agent_id]
        return AuditRecord(
            record_id=row[0],
            action_id=row[1],
            event_type=EventType(row[2]),
            timestamp=row[3],
            agent_id=agent_id,
            tool_name=row[5],
            data=json.loads(row[6]),
            regulatory_articles=json.loads(row[7]),
            previous_hash=row[8],
            record_hash=row[9],
        )

    # ── Backup ────────────────────────────────────────────────────

    def backup(self, dest_path: str | Path) -> None:
        """Hot backup to dest_path using SQLite's online backup API.

        Safe to call while the database is in use — no write lock is held
        during the copy. The destination is a fully consistent SQLite file.
        """
        dest = Path(dest_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest_conn = sqlite3.connect(str(dest))
        try:
            with self._lock:
                self._conn.backup(dest_conn)
        finally:
            dest_conn.close()
        logger.info("Audit DB backed up to %s", dest)

    def checkpoint_wal(self) -> tuple[int, int]:
        """Force a WAL checkpoint and return (pages_total, pages_checkpointed).

        Call before a backup or before handing the DB file to a reader that
        does not understand WAL mode (e.g. a legacy SQLite 3.6 client).
        """
        with self._lock:
            row = self._conn.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
        return (row[1], row[2]) if row else (0, 0)

    # ── API Key management ────────────────────────────────────────

    def create_api_key(self, name: str, role: "Role") -> str:
        """Create a new API key with the given role. Returns the plaintext key (shown once)."""
        from vaara.auth import Role as _Role
        import uuid as _uuid
        plaintext = generate_api_key()
        key_id = str(_uuid.uuid4())
        key_hash = _hash_key(plaintext)
        role_value = role.value if isinstance(role, _Role) else str(role)
        with self._lock:
            self._conn.execute(
                "INSERT INTO api_keys (key_id, name, role, key_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key_id, name, role_value, key_hash, time.time()),
            )
        logger.info("API key created: name=%r role=%s key_id=%s", name, role_value, key_id)
        return plaintext

    def authenticate_api_key(self, plaintext_key: str) -> Optional["APIKey"]:
        """Validate a plaintext key and return its APIKey metadata, or None."""
        if not isinstance(plaintext_key, str) or not plaintext_key:
            return None
        key_hash = _hash_key(plaintext_key)
        with self._lock:
            row = self._conn.execute(
                "SELECT key_id, name, role, created_at, last_used_at "
                "FROM api_keys WHERE key_hash=?",
                (key_hash,),
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE api_keys SET last_used_at=? WHERE key_hash=?",
                    (time.time(), key_hash),
                )
        if not row:
            return None
        try:
            role = Role(row[2])
        except ValueError:
            role = Role.READER
        return APIKey(
            key_id=row[0], name=row[1], role=role,
            created_at=row[3], last_used_at=row[4],
        )

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key by key_id. Returns True if a key was removed."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM api_keys WHERE key_id=?", (key_id,)
            )
        if cursor.rowcount:
            logger.info("API key revoked: key_id=%s", key_id)
        return bool(cursor.rowcount)

    def list_api_keys(self) -> list["APIKey"]:
        """Return metadata for all active API keys (no hashes or plaintext)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT key_id, name, role, created_at, last_used_at FROM api_keys"
            ).fetchall()
        result = []
        for row in rows:
            try:
                role = Role(row[2])
            except ValueError:
                role = Role.READER
            result.append(APIKey(
                key_id=row[0], name=row[1], role=role,
                created_at=row[3], last_used_at=row[4],
            ))
        return result

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
