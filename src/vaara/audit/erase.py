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

With ``legal_ground`` and ``erased_by`` the erasure record also says on
which ground and by whom; with ``sign_key`` the eraser signs it with their
own EC P-256 or RSA key (``vaara.record-erasure/v1``), so a reader holding
that public key can tell a lawful erasure from a hole without trusting the
operator. The record never carries the key.

The erasure record is appended first. A run stopped between the two steps
leaves a declared erasure over a record that still verifies, and running the
command again finishes the tombstone without a second declaration.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

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


def _signer(private_key: Any):
    from cryptography.hazmat.primitives.asymmetric import ec, rsa

    from vaara.attestation._attest_signing import sign_es256, sign_rs256
    from vaara.audit.verify import ERASURE_RECORD_TYPE, erasure_payload

    if isinstance(private_key, ec.EllipticCurvePrivateKey) and isinstance(
            private_key.curve, ec.SECP256R1):
        alg, sign = "ES256", sign_es256
    elif isinstance(private_key, rsa.RSAPrivateKey):
        alg, sign = "RS256", sign_rs256
    else:
        raise ErasureRefused("an eraser signs with an EC P-256 or RSA private key")

    def _sign(data: dict) -> dict:
        return {"type": ERASURE_RECORD_TYPE, "alg": alg,
                "signature": sign(erasure_payload(data), private_key=private_key)}
    return _sign


def erase_record(
    db: str | Path,
    record_id: str,
    *,
    reason: str,
    legal_ground: Optional[str] = None,
    erased_by: Optional[str] = None,
    sign_key: Any = None,
) -> dict:
    """Erase ``record_id``'s data in the trail at ``db``. Returns a report."""
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    if not reason or not reason.strip():
        raise ErasureRefused("an erasure needs a reason; it is recorded on the chain")
    legal_ground = legal_ground.strip() if legal_ground else None
    erased_by = erased_by.strip() if erased_by else None
    if sign_key is not None and not erased_by:
        raise ErasureRefused("a signed erasure needs --erased-by, the person whose key signs it")
    sign = _signer(sign_key) if sign_key is not None else None
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
        declared = existing is None
        if existing is None:
            erasure = AuditTrail(on_record=backend.write_record).record_erasure(
                target, seq, reason.strip(),
                legal_ground=legal_ground, erased_by=erased_by, sign=sign,
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
        "legal_ground": legal_ground or "",
        "erased_by": erased_by or "",
        "signed": sign is not None and declared,
    }
