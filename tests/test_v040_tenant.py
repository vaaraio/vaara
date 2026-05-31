"""v0.40 tenant_id end-to-end propagation tests."""

from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient

    from vaara.server import create_app
except ImportError:
    pytest.skip(
        "server extra not installed (pip install 'vaara[server]')",
        allow_module_level=True,
    )

from vaara.audit.trail import AuditRecord, AuditTrail, EventType
from vaara.taxonomy.actions import (
    ActionCategory,
    ActionRequest,
    ActionType,
    BlastRadius,
    Reversibility,
)


def _action_type(name: str = "t") -> ActionType:
    return ActionType(
        name=name,
        category=ActionCategory.DATA,
        reversibility=Reversibility.FULLY,
        blast_radius=BlastRadius.LOCAL,
    )


def _minimal_policy() -> dict:
    return {
        "version": 1,
        "thresholds_default": {"escalate": 0.5, "deny": 0.9},
    }


def test_score_body_tenant_id_lands_in_action_info():
    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/v1/score",
        json={"tool_name": "search", "agent_id": "a", "tenant_id": "tenant-a"},
    )
    assert resp.status_code == 200
    info = app.state.vaara.lookup_action(resp.json()["action_id"])
    assert info is not None
    assert info.tenant_id == "tenant-a"


def test_score_header_tenant_id_used_when_body_empty():
    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/v1/score",
        json={"tool_name": "search", "agent_id": "a"},
        headers={"X-Vaara-Tenant": "tenant-b"},
    )
    assert resp.status_code == 200
    info = app.state.vaara.lookup_action(resp.json()["action_id"])
    assert info.tenant_id == "tenant-b"


def test_score_body_tenant_wins_over_header():
    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/v1/score",
        json={"tool_name": "search", "agent_id": "a", "tenant_id": "body-tenant"},
        headers={"X-Vaara-Tenant": "header-tenant"},
    )
    info = app.state.vaara.lookup_action(resp.json()["action_id"])
    assert info.tenant_id == "body-tenant"


def test_audit_event_writes_with_tenant_from_action_lookup():
    app = create_app()
    client = TestClient(app)
    score = client.post(
        "/v1/score",
        json={"tool_name": "search", "agent_id": "a", "tenant_id": "tenant-c"},
    )
    action_id = score.json()["action_id"]
    resp = client.post(
        "/v1/audit/events",
        json={
            "event_type": "action_executed",
            "action_id": action_id,
            "agent_id": "a",
            "tool_name": "search",
            "payload": {"result": "ok"},
        },
    )
    assert resp.status_code == 201
    records = [
        r for r in app.state.vaara.audit._records if r.action_id == action_id
    ]
    assert any(r.tenant_id == "tenant-c" for r in records)


def test_audit_trail_tenant_propagates_to_followup_records():
    trail = AuditTrail()
    action_type = _action_type()
    req = ActionRequest(
        agent_id="a", tool_name="t", action_type=action_type,
        parameters={}, tenant_id="tenant-d",
    )
    action_id = trail.record_action_requested(req)
    trail.record_decision(
        action_id=action_id, agent_id="a", tool_name="t",
        decision="allow", reason="ok", risk_score=0.1,
    )
    trail.record_execution(
        action_id=action_id, agent_id="a", tool_name="t", result={"ok": True},
    )
    records = trail.get_action_trail(action_id)
    assert len(records) == 3
    assert all(r.tenant_id == "tenant-d" for r in records)


def test_audit_trail_tenant_map_evicts_under_pressure():
    trail = AuditTrail()
    trail._MAX_ACTION_TENANT_MAP = 100
    action_type = _action_type()
    last_ids: list[str] = []
    for i in range(150):
        req = ActionRequest(
            agent_id="a", tool_name="t", action_type=action_type,
            tenant_id=f"t{i}",
        )
        last_ids.append(trail.record_action_requested(req))
    assert len(trail._tenant_for_action) <= 100
    assert trail._tenant_for_action.get(last_ids[-1]) == "t149"
    assert last_ids[0] not in trail._tenant_for_action


def test_audit_trail_caps_tenant_id_length():
    trail = AuditTrail()
    action_type = _action_type()
    huge = "x" * 10_000
    req = ActionRequest(
        agent_id="a", tool_name="t", action_type=action_type,
        tenant_id=huge,
    )
    action_id = trail.record_action_requested(req)
    record = trail.get_action_trail(action_id)[0]
    assert len(record.tenant_id) <= trail._MAX_TENANT_ID_LEN


def test_concurrent_multi_tenant_lifecycles_preserve_scope_and_chain():
    """Many agent threads, each its own tenant, run full lifecycles at once.

    Proves the multi-tenant claim the registry entry rests on: under real
    contention the hash chain stays intact AND every action's follow-up
    records carry the tenant captured at request time. Before the
    _tenant_map_lock fix this could tear the action -> tenant map.
    """
    import threading

    trail = AuditTrail()
    action_type = _action_type()
    n_threads, per_thread = 16, 40
    barrier = threading.Barrier(n_threads)
    results: dict[str, list[str]] = {}
    results_lock = threading.Lock()

    def worker(t: int) -> None:
        tenant = f"tenant-{t}"
        ids: list[str] = []
        barrier.wait()  # release all threads together for maximum contention
        for _ in range(per_thread):
            req = ActionRequest(
                agent_id=f"a{t}", tool_name="t", action_type=action_type,
                tenant_id=tenant,
            )
            aid = trail.record_action_requested(req)
            trail.record_decision(
                action_id=aid, agent_id=f"a{t}", tool_name="t",
                decision="allow", reason="ok", risk_score=0.1,
            )
            trail.record_execution(
                action_id=aid, agent_id=f"a{t}", tool_name="t", result={"ok": True},
            )
            ids.append(aid)
        with results_lock:
            results[tenant] = ids

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # Chain integrity held across all concurrent appends.
    assert trail.verify_chain() is None
    assert trail.size == n_threads * per_thread * 3
    # Every lifecycle kept its own tenant scope, no cross-tenant bleed.
    for tenant, ids in results.items():
        for aid in ids:
            recs = trail.get_action_trail(aid)
            assert len(recs) == 3
            assert all(r.tenant_id == tenant for r in recs)


def test_concurrent_tenant_map_eviction_is_threadsafe():
    """Eviction under the cap must not race with concurrent reads/writes.

    The eviction path iterates list(self._tenant_for_action) while other
    threads insert; an unguarded dict raised "dictionary changed size
    during iteration". With the lock this completes cleanly.
    """
    import threading

    trail = AuditTrail()
    trail._MAX_ACTION_TENANT_MAP = 200
    action_type = _action_type()
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(t: int) -> None:
        try:
            barrier.wait()
            for i in range(300):  # well past the cap to force repeated eviction
                req = ActionRequest(
                    agent_id="a", tool_name="t", action_type=action_type,
                    tenant_id=f"t{t}-{i}",
                )
                aid = trail.record_action_requested(req)
                trail._tenant_for(aid)  # concurrent read against the evicting writer
        except BaseException as exc:  # noqa: BLE001 — surface any race to the assert
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, f"tenant-map race raised: {errors[0]!r}"
    assert len(trail._tenant_for_action) <= trail._MAX_ACTION_TENANT_MAP
    assert trail.verify_chain() is None


def test_v1_record_hash_excludes_tenant_id():
    """Chain v1 (legacy default) leaves tenant_id out of the hash, so pre-v0.47
    trails written before tenant binding re-verify byte for byte."""
    rec_no = AuditRecord(
        record_id="r1", action_id="a1", event_type=EventType.ACTION_REQUESTED,
        timestamp=1.0, agent_id="a", tool_name="t",
    )
    rec_with = AuditRecord(
        record_id="r1", action_id="a1", event_type=EventType.ACTION_REQUESTED,
        timestamp=1.0, agent_id="a", tool_name="t", tenant_id="tenant-x",
    )
    assert rec_no.chain_version == 1
    assert rec_no.compute_hash() == rec_with.compute_hash()


def test_v2_record_hash_binds_tenant_id():
    """Chain v2 binds tenant_id into the hash, so two records differing only
    in tenant_id hash differently."""
    rec_a = AuditRecord(
        record_id="r1", action_id="a1", event_type=EventType.ACTION_REQUESTED,
        timestamp=1.0, agent_id="a", tool_name="t",
        tenant_id="tenant-a", chain_version=2,
    )
    rec_b = AuditRecord(
        record_id="r1", action_id="a1", event_type=EventType.ACTION_REQUESTED,
        timestamp=1.0, agent_id="a", tool_name="t",
        tenant_id="tenant-b", chain_version=2,
    )
    assert rec_a.compute_hash() != rec_b.compute_hash()


def test_v1_hash_is_byte_stable_across_versions():
    """The exact v1 hash must never change — a fixed expected digest guards
    against any future edit silently breaking pre-v0.47 chain re-verification."""
    import hashlib
    import json

    rec = AuditRecord(
        record_id="r1", action_id="a1", event_type=EventType.ACTION_REQUESTED,
        timestamp=1.0, agent_id="a", tool_name="t", tenant_id="ignored-in-v1",
    )
    expected = hashlib.sha256(
        json.dumps(
            {
                "record_id": "r1", "action_id": "a1",
                "event_type": "action_requested", "timestamp": 1.0,
                "agent_id": "a", "tool_name": "t", "data": {},
                "regulatory_articles": [], "previous_hash": "",
            },
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
    ).hexdigest()
    assert rec.compute_hash() == expected


def test_trail_stamps_v2_and_detects_tenant_reattribution():
    """A live trail stamps chain v2 on every record, and re-attributing a
    record to another tenant breaks verify_chain (the v0.47 fix)."""
    trail = AuditTrail()
    aid = trail.record_action_requested(
        ActionRequest(
            action_type=_action_type(), tool_name="secret_tool",
            agent_id="agent-a", parameters={}, tenant_id="tenant-a",
        )
    )
    rec = trail._records[0]
    assert rec.chain_version == 2
    assert rec.tenant_id == "tenant-a"
    assert trail.verify_chain() is None

    rec.tenant_id = "tenant-b"  # simulate store-write re-attribution
    assert trail.verify_chain() is not None
    assert aid  # silence unused


def test_v2_records_roundtrip_through_sqlite(tmp_path):
    """v2 records persist their chain_version + tenant_id, and the chain
    re-verifies on reload (catches the stored-vs-hashed-tenant landmine)."""
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    db = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(db)
    trail = AuditTrail(on_record=backend.write_record)
    try:
        for tid in ("tenant-a", "tenant-b", ""):
            trail.record_action_requested(
                ActionRequest(
                    action_type=_action_type(), tool_name="t",
                    agent_id="agent", parameters={}, tenant_id=tid,
                )
            )
    finally:
        backend.close()

    reopened = SQLiteAuditBackend(db)
    try:
        reloaded = reopened.load_trail(strict=True)  # raises if chain broken
    finally:
        reopened.close()
    assert len(reloaded._records) == 3
    assert [r.chain_version for r in reloaded._records] == [2, 2, 2]
    assert [r.tenant_id for r in reloaded._records] == ["tenant-a", "tenant-b", ""]
    assert reloaded.verify_chain() is None


def test_signed_export_roundtrips_v2_chain(tmp_path):
    """The standalone verifier (verify._verify_chain_bytes) must accept a v2
    chain — keeps it in lockstep with AuditRecord.compute_hash."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from vaara.audit.export import export_signed
    from vaara.audit.verify import verify_signed

    trail = AuditTrail()
    for tid in ("tenant-a", "tenant-b"):
        trail.record_action_requested(
            ActionRequest(
                action_type=_action_type(), tool_name="t",
                agent_id="agent", parameters={}, tenant_id=tid,
            )
        )
    key = Ed25519PrivateKey.generate()
    out = tmp_path / "trail.zip"
    export_signed(trail, out, key)

    result = verify_signed(out)
    assert result.ok, result.errors
