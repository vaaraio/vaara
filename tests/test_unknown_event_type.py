"""A record with an event type this Vaara does not know still loads and verifies.

On 13.10.2026 a newer build appended a `repair_recovered` record to a trail
that a 2.12.1 install also read. 2.12.1 had no such EventType, so it loaded
the row as a corrupt skeleton, the skeleton failed its hash, and every
reader on the box reported the chain broken over a record that was intact.
Every new event type would do the same to every older reader. These tests
pin the rule: a stored record keeps the type it was written with, hashes as
written, and matches none of the types this build knows. Input validation
is unchanged: ``EventType("...")`` still refuses an unknown value.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import AuditRecord, EventType, stored_event_type

from tests.test_declared_gaps import _build, _calls

FUTURE = "from_a_newer_vaara"


def _append_future_record(path: Path) -> str:
    """Append a well-formed record of an unknown type, hashed by hand."""
    conn = sqlite3.connect(path)
    prev = conn.execute(
        "SELECT seq, record_hash, chain_version FROM audit_records ORDER BY seq DESC LIMIT 1"
    ).fetchone()
    record_id = str(uuid.uuid4())
    content = {
        "record_id": record_id, "action_id": "act-future", "event_type": FUTURE,
        "timestamp": 1791880000.0, "agent_id": "future-agent", "tool_name": "",
        "data": {"note": "written by a newer build"}, "regulatory_articles": [],
        "previous_hash": prev[1],
    }
    if (prev[2] or 1) >= 2:
        content["tenant_id"] = ""
        content["chain_version"] = prev[2]
    record_hash = hashlib.sha256(json.dumps(
        content, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    conn.execute(
        "INSERT INTO audit_records (record_id, action_id, event_type, timestamp, agent_id, "
        "tool_name, data, regulatory, previous_hash, record_hash, seq, tenant_id, chain_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?)",
        (record_id, "act-future", FUTURE, 1791880000.0, "future-agent", "",
         json.dumps(content["data"]), "[]", prev[1], record_hash, prev[0] + 1, prev[2] or 1),
    )
    conn.execute(
        "UPDATE audit_meta SET value = ? WHERE key = 'chain_head'",
        (json.dumps({"hash": record_hash, "seq": prev[0] + 1}, sort_keys=True),),
    )
    conn.commit()
    conn.close()
    return record_id


def test_an_unknown_event_type_loads_and_the_chain_verifies(tmp_path: Path):
    path = _build(tmp_path)
    record_id = _append_future_record(path)
    backend = SQLiteAuditBackend(path)
    try:
        trail = backend.load_trail()
        assert trail._skeleton_records == 0
        assert trail.verify_chain_verdict().state == "intact"
        assert backend.verify_chain_streaming_verdict().state == "intact"
        (record,) = [r for r in trail._records if r.record_id == record_id]
        assert record.event_type.value == FUTURE
        assert record.event_type not in list(EventType)
        assert record.data == {"note": "written by a newer build"}
        # The trail keeps recording on top of it.
        _calls(backend, 1, "after")
        assert backend.load_trail().verify_chain_verdict().state == "intact"
    finally:
        backend.close()


def test_an_exported_record_of_an_unknown_type_round_trips(tmp_path: Path):
    path = _build(tmp_path)
    record_id = _append_future_record(path)
    backend = SQLiteAuditBackend(path)
    try:
        (record,) = [r for r in backend.load_trail()._records if r.record_id == record_id]
    finally:
        backend.close()
    again = AuditRecord.from_dict(json.loads(json.dumps(record.to_dict())))
    assert again.event_type.value == FUTURE
    assert again.compute_hash() == again.record_hash


def test_an_unknown_type_survives_copy_and_pickle():
    import copy
    import pickle

    unknown = stored_event_type(FUTURE)
    for again in (copy.copy(unknown), copy.deepcopy(unknown),
                  pickle.loads(pickle.dumps(unknown))):
        assert again.value == FUTURE
    assert pickle.loads(pickle.dumps(EventType.DECISION_MADE)) is EventType.DECISION_MADE
    assert copy.deepcopy(EventType.DECISION_MADE) is EventType.DECISION_MADE


def test_input_validation_still_refuses_an_unknown_type():
    with pytest.raises(ValueError):
        EventType(FUTURE)


def test_an_unknown_type_matches_no_known_member():
    unknown = stored_event_type(FUTURE)
    assert all(unknown != member for member in EventType)
    assert stored_event_type("decision_made") is EventType.DECISION_MADE
    with pytest.raises(ValueError):
        stored_event_type("")
