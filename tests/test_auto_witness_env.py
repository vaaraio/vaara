# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The documented head-publishing and anchoring cadence can be switched on.

docs/PRIOR_ART.md says the cadence at which trail heads reach a public log
"is configured and readable". ``AuditTrail.enable_auto_publish`` and
``enable_auto_anchor`` existed, and no command, flag, variable or surface
called either. Now every surface that opens a trail through
``SQLiteAuditBackend.load_trail`` reads ``VAARA_AUTO_PUBLISH_EVERY`` and
``VAARA_AUTO_ANCHOR_EVERY``: off unless set, a disclosure to stderr once
per process, and a ``witness_configured`` record on the trail that states
the cadence, so a verifier reads the bound instead of guessing it.
"""

from __future__ import annotations

import sqlite3

import pytest

from vaara.audit import witness
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.audit.trail import EventType


class _Publication:
    def __init__(self, head):
        self.head = head

    def to_dict(self):
        return {"log_url": "https://log.example", "log_index": 7, "entry_url": "e", "head": self.head}


class _Anchor:
    def __init__(self, position, head_hash):
        self.chain_position, self.chain_head_hash = position, head_hash
        self.authority, self.method = "TSA Example", "rfc3161"

    def to_dict(self):
        return {"chain_position": self.chain_position, "chain_head_hash": self.chain_head_hash}


class _TSA:
    def __init__(self):
        self.calls = []

    def anchor(self, position, head_hash):
        self.calls.append((position, head_hash))
        return _Anchor(position, head_hash)


@pytest.fixture
def quiet(monkeypatch):
    for name in (witness.PUBLISH_EVERY_ENV, witness.PUBLISH_KEY_ENV, witness.PUBLISH_LOG_ENV,
                 witness.ANCHOR_EVERY_ENV, witness.ANCHOR_TSA_ENV):
        monkeypatch.delenv(name, raising=False)
    witness._disclosed.clear()
    yield
    witness._disclosed.clear()


def _append(trail, n):
    for i in range(n):
        trail.record_agent_seen(agent="a", state="governed", detail="", pid=i)


def _records(trail, kind):
    return [r for r in trail._records if r.event_type == kind]


# ── Off by default ───────────────────────────────────────────────────────

def test_nothing_is_witnessed_unless_asked(quiet, tmp_path, monkeypatch):
    published = []
    monkeypatch.setattr("vaara.attestation.rekor_log.publish_head",
                        lambda head, signer, log_url: published.append(head) or _Publication(head))
    trail = SQLiteAuditBackend(str(tmp_path / "a.db")).load_trail()
    _append(trail, 20)
    assert published == []
    assert _records(trail, EventType.WITNESS_CONFIGURED) == []


# ── Publishing ───────────────────────────────────────────────────────────

def test_publish_every_env_publishes_the_head_at_the_stated_cadence(quiet, tmp_path, monkeypatch, capsys):
    published = []
    monkeypatch.setattr("vaara.attestation.rekor_log.publish_head",
                        lambda head, signer, log_url: published.append((head, log_url)) or _Publication(head))
    monkeypatch.setenv(witness.PUBLISH_EVERY_ENV, "4")
    monkeypatch.setenv(witness.PUBLISH_KEY_ENV, str(tmp_path / "key.pem"))
    monkeypatch.setenv(witness.PUBLISH_LOG_ENV, "https://log.example")

    trail = SQLiteAuditBackend(str(tmp_path / "a.db")).load_trail()
    _append(trail, 9)

    assert len(published) == 2, published
    assert all(log == "https://log.example" for _, log in published)
    assert (tmp_path / "key.pem").is_file(), "a publishing key was generated at the named path"
    err = capsys.readouterr().err
    assert "every 4 records" in err.lower() and "https://log.example" in err
    assert "permanent" in err and "cannot be withdrawn" in err


def test_the_cadence_is_readable_on_the_trail(quiet, tmp_path, monkeypatch):
    monkeypatch.setattr("vaara.attestation.rekor_log.publish_head",
                        lambda head, signer, log_url: _Publication(head))
    monkeypatch.setenv(witness.PUBLISH_EVERY_ENV, "256")
    monkeypatch.setenv(witness.PUBLISH_KEY_ENV, str(tmp_path / "key.pem"))
    trail = SQLiteAuditBackend(str(tmp_path / "a.db")).load_trail()
    records = _records(trail, EventType.WITNESS_CONFIGURED)
    assert len(records) == 1
    data = records[0].data
    assert data["kind"] == "publish"
    assert data["every_records"] == 256
    assert data["target"] == witness.DEFAULT_PUBLISH_LOG
    assert len(data["key_fingerprint"]) >= 16
    assert data["source"] == witness.PUBLISH_EVERY_ENV


def test_the_same_configuration_is_recorded_once_not_per_process(quiet, tmp_path, monkeypatch):
    monkeypatch.setattr("vaara.attestation.rekor_log.publish_head",
                        lambda head, signer, log_url: _Publication(head))
    monkeypatch.setenv(witness.PUBLISH_EVERY_ENV, "64")
    monkeypatch.setenv(witness.PUBLISH_KEY_ENV, str(tmp_path / "key.pem"))
    db = str(tmp_path / "a.db")
    for _ in range(3):
        witness._disclosed.clear()
        SQLiteAuditBackend(db).load_trail()
    trail = SQLiteAuditBackend(db).load_trail()
    assert len(_records(trail, EventType.WITNESS_CONFIGURED)) == 1

    monkeypatch.setenv(witness.PUBLISH_EVERY_ENV, "128")
    trail = SQLiteAuditBackend(db).load_trail()
    assert [r.data["every_records"] for r in _records(trail, EventType.WITNESS_CONFIGURED)] == [64, 128]


def test_the_disclosure_is_printed_once_per_process(quiet, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("vaara.attestation.rekor_log.publish_head",
                        lambda head, signer, log_url: _Publication(head))
    monkeypatch.setenv(witness.PUBLISH_EVERY_ENV, "64")
    monkeypatch.setenv(witness.PUBLISH_KEY_ENV, str(tmp_path / "key.pem"))
    db = str(tmp_path / "a.db")
    SQLiteAuditBackend(db).load_trail()
    SQLiteAuditBackend(db).load_trail()
    assert capsys.readouterr().err.lower().count("every 64 records") == 1


def test_a_bad_cadence_is_said_and_stays_off(quiet, tmp_path, monkeypatch, capsys):
    published = []
    monkeypatch.setattr("vaara.attestation.rekor_log.publish_head",
                        lambda head, signer, log_url: published.append(head) or _Publication(head))
    monkeypatch.setenv(witness.PUBLISH_EVERY_ENV, "soon")
    monkeypatch.setenv(witness.PUBLISH_KEY_ENV, str(tmp_path / "key.pem"))
    trail = SQLiteAuditBackend(str(tmp_path / "a.db")).load_trail()
    _append(trail, 5)
    assert published == []
    assert "VAARA_AUTO_PUBLISH_EVERY" in capsys.readouterr().err
    assert not (tmp_path / "key.pem").exists()


def test_an_existing_key_is_used_not_replaced(quiet, tmp_path, monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    key = tmp_path / "key.pem"
    key.write_bytes(ec.generate_private_key(ec.SECP256R1()).private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    before = key.read_bytes()
    monkeypatch.setattr("vaara.attestation.rekor_log.publish_head",
                        lambda head, signer, log_url: _Publication(head))
    monkeypatch.setenv(witness.PUBLISH_EVERY_ENV, "8")
    monkeypatch.setenv(witness.PUBLISH_KEY_ENV, str(key))
    SQLiteAuditBackend(str(tmp_path / "a.db")).load_trail()
    assert key.read_bytes() == before


# ── Anchoring ────────────────────────────────────────────────────────────

def test_anchor_every_env_anchors_the_head_at_the_stated_cadence(quiet, tmp_path, monkeypatch, capsys):
    tsa = _TSA()
    monkeypatch.setattr(witness, "_anchor_client", lambda url: tsa)
    monkeypatch.setenv(witness.ANCHOR_EVERY_ENV, "3")
    monkeypatch.setenv(witness.ANCHOR_TSA_ENV, "https://tsa.example/tsr")
    trail = SQLiteAuditBackend(str(tmp_path / "a.db")).load_trail()
    _append(trail, 7)
    assert len(tsa.calls) == 2, tsa.calls
    records = _records(trail, EventType.WITNESS_CONFIGURED)
    assert [r.data["kind"] for r in records] == ["anchor"]
    assert records[0].data["target"] == "https://tsa.example/tsr"
    assert records[0].data["every_records"] == 3
    assert "every 3 records" in capsys.readouterr().err.lower()


def test_anchoring_without_an_authority_is_said_and_stays_off(quiet, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(witness, "_configured_tsa", lambda: None)
    monkeypatch.setenv(witness.ANCHOR_EVERY_ENV, "3")
    trail = SQLiteAuditBackend(str(tmp_path / "a.db")).load_trail()
    _append(trail, 4)
    assert _records(trail, EventType.WITNESS_CONFIGURED) == []
    assert "VAARA_ANCHOR_TSA" in capsys.readouterr().err


def test_the_operators_chosen_authority_is_the_default(quiet, tmp_path, monkeypatch):
    tsa = _TSA()
    seen = []

    def client(url):
        seen.append(url)
        return tsa

    monkeypatch.setattr(witness, "_anchor_client", client)
    monkeypatch.setattr(witness, "_configured_tsa", lambda: "https://chosen.example/tsr")
    monkeypatch.setenv(witness.ANCHOR_EVERY_ENV, "2")
    SQLiteAuditBackend(str(tmp_path / "a.db")).load_trail()
    assert seen == ["https://chosen.example/tsr"]


# ── The record stands on its own ─────────────────────────────────────────

def test_the_record_is_on_the_hash_chain(quiet, tmp_path, monkeypatch):
    monkeypatch.setattr("vaara.attestation.rekor_log.publish_head",
                        lambda head, signer, log_url: _Publication(head))
    monkeypatch.setenv(witness.PUBLISH_EVERY_ENV, "16")
    monkeypatch.setenv(witness.PUBLISH_KEY_ENV, str(tmp_path / "key.pem"))
    db = str(tmp_path / "a.db")
    SQLiteAuditBackend(db).load_trail()
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT event_type, record_hash, previous_hash FROM audit_records ORDER BY seq").fetchall()
    conn.close()
    assert rows and rows[0][0] == "witness_configured" and rows[0][1]
    backend = SQLiteAuditBackend(db)
    assert backend.verify_chain_streaming() is None
