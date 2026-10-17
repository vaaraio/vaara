# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Erase one record's content and keep the chain verifiable.

GDPR Art. 17 against an append-only trail. ``erase_record`` appends a
RECORD_ERASED record naming the target by id and stored ``record_hash``, then
overwrites the target's ``data`` with a tombstone pointing at that record. The
hash is kept, so the next record still chains to it. Both chain walks report
the slot as erased: link verified, content not checkable. A tombstone that no
later erasure record names is a hash mismatch, as any rewrite is.

Only ``data`` is erased. ``agent_id``, ``tool_name``, the timestamp and the
regulatory tags stay, because they say that something happened and when.

The erasure record is appended first. A run stopped between the two steps
leaves a declared erasure over a record that still verifies, and running the
command again finishes the tombstone without a second declaration.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from vaara.audit.trail import (
    ERASED_DATA_KEY,
    SYSTEM_EVENT_TYPES,
    AuditTrail,
    EventType,
    erasure_pointer,
)


class ErasureRefused(ValueError):
    """The record cannot be erased; the message says why."""


def _existing_erasure(conn: sqlite3.Connection, record_id: str) -> Optional[str]:
    for erasure_id, data in conn.execute(
        "SELECT record_id, data FROM audit_records WHERE event_type = ? ORDER BY seq",
        (EventType.RECORD_ERASED.value,),
    ):
        try:
            named = json.loads(data).get("erased_record_id")
        except (ValueError, AttributeError):
            continue
        if named == record_id:
            return erasure_id
    return None


def erase_record(db: str | Path, record_id: str, *, reason: str) -> dict:
    """Erase ``record_id``'s data in the trail at ``db``. Returns a report."""
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    if not reason or not reason.strip():
        raise ErasureRefused("an erasure needs a reason; it is recorded on the chain")
    backend = SQLiteAuditBackend(Path(db).expanduser())
    try:
        with backend._lock:
            row = backend._conn.execute(
                "SELECT * FROM audit_records WHERE record_id = ?", (record_id,)
            ).fetchone()
            existing = _existing_erasure(backend._conn, record_id) if row else None
        if row is None:
            raise ErasureRefused(f"no record with id {record_id!r} in {db}")
        target = backend._row_to_record(row)
        seq = row[10]
        if erasure_pointer(target) is not None:
            raise ErasureRefused(f"record {record_id} is already erased")
        known = target.event_type.value in {e.value for e in EventType}
        if target.event_type in SYSTEM_EVENT_TYPES or not known:
            raise ErasureRefused(
                f"record {record_id} is a {target.event_type.value} record: the trail's "
                "own evidence about itself is not erasable"
            )
        if existing is None:
            erasure = AuditTrail(on_record=backend.write_record).record_erasure(
                target, seq, reason.strip(),
            )
            existing = erasure.record_id
        scrubbed = backend.erase_record_data(
            record_id, {ERASED_DATA_KEY: {"erasure_record_id": existing}},
        )
    finally:
        backend.close()
    return {
        "erased_record_id": record_id,
        "erased_seq": seq,
        "erased_record_hash": target.record_hash,
        "erasure_record_id": existing,
        "wal_scrubbed": scrubbed,
    }
