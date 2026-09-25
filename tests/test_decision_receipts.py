# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The engine emits a signed decision receipt for every decision it records."""

from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3
import stat
from pathlib import Path

import pytest

pytest.importorskip("cryptography")
rfc8785 = pytest.importorskip("rfc8785")

from cryptography.exceptions import InvalidSignature  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature  # noqa: E402

from vaara.audit.sqlite_backend import SQLiteAuditBackend  # noqa: E402

dr = importlib.import_module("vaara.audit.decision_receipts")


def _trail(tmp_path: Path):
    db = tmp_path / "trail" / "audit.db"
    db.parent.mkdir()
    return db, SQLiteAuditBackend(db).load_trail()


def _record(trail, decision: str, tool: str = "read_file") -> str:
    action_id = f"a-{decision}-{tool}"
    trail.record_decision(
        action_id=action_id, agent_id="agent-1", tool_name=tool,
        decision=decision, reason=f"test {decision}", risk_score=0.4321,
    )
    return action_id


def _hashes(db: Path) -> dict[str, str]:
    with sqlite3.connect(db) as con:
        return dict(con.execute("SELECT record_id, record_hash FROM audit_records"))


def _files(db: Path) -> list[Path]:
    return sorted((db.parent / "receipts").rglob("*.json"))


def test_every_decision_leaves_a_verifying_receipt(tmp_path):
    db, trail = _trail(tmp_path)
    for d in ("allow", "escalate", "deny"):
        _record(trail, d)
    files = _files(db)
    assert len(files) == 3
    checks = [dr.verify_receipt_file(f, trail_hashes=_hashes(db)) for f in files]
    assert all(c.ok for c in checks), [c.detail for c in checks]
    assert sorted(c.decision for c in checks) == ["allow", "block", "escalate"]


def test_only_decisions_mint(tmp_path):
    db, trail = _trail(tmp_path)
    trail.record_escalation(
        action_id="a1", agent_id="agent-1", tool_name="t",
        escalation_target="human_reviewer", risk_score=0.5,
    )
    assert _files(db) == []


def test_tampered_evidence_fails(tmp_path):
    db, trail = _trail(tmp_path)
    _record(trail, "deny")
    f = _files(db)[0]
    body = json.loads(f.read_text())
    body["evidence"]["decision"] = "allow"
    f.write_text(json.dumps(body))
    c = dr.verify_receipt_file(f)
    assert c.signature_ok and not c.evidence_ok and not c.ok


def test_tampered_envelope_fails_signature(tmp_path):
    db, trail = _trail(tmp_path)
    _record(trail, "deny")
    f = _files(db)[0]
    body = json.loads(f.read_text())
    body["receipt"]["decisionDerived"]["decision"] = "allow"
    f.write_text(json.dumps(body))
    assert not dr.verify_receipt_file(f).signature_ok


def test_receipt_for_a_record_the_trail_does_not_hold_fails(tmp_path):
    db, trail = _trail(tmp_path)
    _record(trail, "allow")
    f = _files(db)[0]
    rid = json.loads(f.read_text())["evidence"]["recordId"]
    hashes = _hashes(db)
    hashes[rid] = "0" * 64
    c = dr.verify_receipt_file(f, trail_hashes=hashes)
    assert c.trail_ok is False and not c.ok
    c = dr.verify_receipt_file(f, trail_hashes={})
    assert c.trail_ok is False and c.detail == "record not in trail"


def test_receipts_off(tmp_path, monkeypatch):
    monkeypatch.setenv("VAARA_RECEIPTS", "0")
    db, trail = _trail(tmp_path)
    _record(trail, "allow")
    assert not (db.parent / "receipts").exists()


def test_a_failing_sink_does_not_stop_the_decision(tmp_path, monkeypatch):
    db, trail = _trail(tmp_path)

    def boom(self, record):
        raise RuntimeError("disk full")

    monkeypatch.setattr(dr.DecisionReceiptSink, "mint", boom)
    _record(trail, "deny")
    assert trail._receipt_sink.failures == 1
    events = [r.event_type.value for r in trail._records]
    assert events[-1] == "action_blocked"


def test_key_is_private(tmp_path):
    db, trail = _trail(tmp_path)
    _record(trail, "allow")
    key = db.parent / "keys" / "receipt-es256.pem"
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert stat.S_IMODE(key.parent.stat().st_mode) == 0o700


def test_verifies_without_vaara(tmp_path):
    """What the app does: JCS, SHA-256 and P-256 only, no vaara code."""
    db, trail = _trail(tmp_path)
    _record(trail, "escalate")
    body = json.loads(_files(db)[0].read_text())
    env, evidence = body["receipt"], body["evidence"]
    signed = {k: env[k] for k in ("version", "alg", "backLink", "decisionDerived", "issuerAsserted")}
    pub = serialization.load_pem_public_key(
        (db.parent / "receipts" / "issuer-es256.pub.pem").read_bytes()
    )
    raw = bytes.fromhex(env["signature"])
    der = encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
    try:
        pub.verify(der, rfc8785.dumps(signed), ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:  # pragma: no cover
        pytest.fail("signature does not verify")
    digest = "sha256:" + hashlib.sha256(rfc8785.dumps(evidence)).hexdigest()
    assert env["decisionDerived"]["evidenceRef"]["digest"] == digest
    assert evidence["recordHash"] == "sha256:" + _hashes(db)[evidence["recordId"]]


def test_one_key_across_trail_loads(tmp_path):
    db, trail = _trail(tmp_path)
    _record(trail, "allow")
    trail2 = SQLiteAuditBackend(db).load_trail()
    _record(trail2, "deny", tool="shell")
    kids = {json.loads(f.read_text())["receipt"]["issuerAsserted"]["secretVersion"] for f in _files(db)}
    assert len(kids) == 1
    assert all(dr.verify_receipt_file(f, trail_hashes=_hashes(db)).ok for f in _files(db))


VECTORS = Path(__file__).parent / "vectors" / "trail_decision_v0"


@pytest.mark.parametrize(
    "name", sorted(json.loads((VECTORS / "expected.json").read_text()))
)
def test_vectors(name):
    want = json.loads((VECTORS / "expected.json").read_text())[name]
    hashes = json.loads((VECTORS / "trail-hashes.json").read_text())
    c = dr.verify_receipt_file(
        VECTORS / name,
        public_key_pem=(VECTORS / dr.PUBKEY_NAME).read_bytes(),
        trail_hashes=hashes,
    )
    assert (c.signature_ok, c.evidence_ok, c.trail_ok) == (
        want["signature"], want["evidence"], want["trail"]
    )
