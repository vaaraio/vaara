"""A rotated trail says where it begins, and every verifier can read that.

`vaara trail rotate` archives the whole trail, then purges what is older than
the retention window. The purge used to leave the first surviving record
linked to a record that is no longer there, so the live trail failed its own
verification (an export of it broke at record 0) and the next rotation refused
to run ("exported chain did not verify at export time"). Rotation now appends a
signed `trail_rotated` record before the purge: the hash the first survivor
links to, how many records went, and the digest of the archive that holds them.
A trail whose first record links to such a record begins at a rotation seam; a
first record that links to nothing stays a chain break.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from vaara.audit import sealed_archive as sa  # noqa: E402
from vaara.audit.export import export_signed  # noqa: E402
from vaara.audit.rotate import rotate  # noqa: E402
from vaara.audit.sqlite_backend import SQLiteAuditBackend  # noqa: E402
from vaara.audit.trail import AuditTrail  # noqa: E402
from vaara.audit.verify import (  # noqa: E402
    ROTATION_RECORD_TYPE,
    describe_rotation_seams,
    rotation_payload,
    verify_signed,
)

DAY = 86400.0
SEAM = "trail_rotated"


# --- helpers --------------------------------------------------------------------


def _key(tmp_path: Path, name: str = "signer") -> tuple[Ed25519PrivateKey, Path, Path]:
    key = Ed25519PrivateKey.generate()
    priv = tmp_path / f"{name}.pem"
    priv.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    pub = tmp_path / f"{name}.pub.pem"
    pub.write_bytes(key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    return key, priv, pub


def _seed(db: Path, monkeypatch, *, old: int = 2, new: int = 2) -> None:
    """``old`` records 10 days old, then ``new`` written now, all chained."""
    backend = SQLiteAuditBackend(db)
    trail = AuditTrail(on_record=backend.write_record)
    real = time.time
    monkeypatch.setattr("vaara.audit.trail.time.time", lambda: real() - 10 * DAY)
    for i in range(old):
        trail.record_decision(f"old{i + 1}", "agent-x", "read_file", "allow", "low risk", 0.1)
    monkeypatch.setattr("vaara.audit.trail.time.time", real)
    for i in range(new):
        trail.record_decision(f"new{i + 1}", "agent-x", "read_file", "allow", "low risk", 0.1)
    backend.close()


def _live(db: Path) -> list:
    backend = SQLiteAuditBackend(db)
    try:
        return list(backend.load_trail()._records)
    finally:
        backend.close()


def _export_live(db: Path, out: Path, priv: Path):
    backend = SQLiteAuditBackend(db)
    try:
        return export_signed(backend.load_trail(), out, signer_key=priv)
    finally:
        backend.close()


def _zip_records(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        return [json.loads(ln) for ln in zf.read("trail.jsonl").splitlines() if ln.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signer(key: Ed25519PrivateKey):
    def sign(data: dict) -> dict:
        return {"type": ROTATION_RECORD_TYPE, "alg": "Ed25519",
                "signature": key.sign(rotation_payload(data)).hex()}
    return sign


def _fingerprint(key: Ed25519PrivateKey) -> str:
    raw = key.public_key().public_bytes(serialization.Encoding.Raw,
                                        serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:32]


def _trail_of(records) -> AuditTrail:
    """A trail holding exactly these records, as a reload from a purged store does."""
    trail = AuditTrail()
    for rec in records:
        trail._index_record(rec)
        trail._last_hash = rec.record_hash
    return trail


def _chain(key: Ed25519PrivateKey, *, names=("a", "b", "c"), sign=None, **overrides):
    """In memory: len(names) decisions, then a seam naming the second as the last
    one archived. Returns (all records, the seam)."""
    trail = AuditTrail()
    for name in names:
        trail.record_decision(name, "agent-x", "read_file", "allow", "low risk", 0.1)
    fields = dict(
        last_hash=trail._records[1].record_hash,
        purged_records=2,
        archive_sha256="ab" * 32,
        archive_records=len(names),
        archive_kind="zip",
        boundary_utc="2026-10-20T00:00:00Z",
        signer_fingerprint=_fingerprint(key),
    )
    fields.update(overrides)
    seam = trail.record_rotation(sign=sign or _signer(key), **fields)
    return list(trail._records), seam


def _survivors_zip(tmp_path, key_path, records, name="live.zip"):
    out = tmp_path / name
    export_signed(_trail_of(records), out, signer_key=key_path)
    return out


# --- rotate writes the record ---------------------------------------------------


def test_rotate_appends_a_signed_seam_record_after_the_archive(tmp_path, monkeypatch):
    key, priv, _ = _key(tmp_path)
    db, out = tmp_path / "audit.db", tmp_path / "archive.zip"
    _seed(db, monkeypatch)

    result = rotate(db, out, priv, 7)

    assert result.ok and result.purged_records == 2, result.errors
    live = _live(db)
    assert [r.event_type.value for r in live] == ["decision_made", "decision_made", SEAM]
    seam = live[-1]
    archived = _zip_records(out)
    manifest = verify_signed(out).manifest
    data = seam.data
    assert data["last_hash"] == archived[1]["record_hash"]
    assert live[0].previous_hash == data["last_hash"]
    assert data["purged_records"] == 2
    assert data["archive_records"] == 4
    assert data["archive_kind"] == "zip"
    assert data["archive_sha256"] == _sha(out)
    assert data["signer_fingerprint"] == manifest["signer_pubkey_fingerprint"]
    assert data["boundary_utc"].endswith("Z") and data["rotated_at"].endswith("Z")
    assert data["rotation_signature"]["type"] == ROTATION_RECORD_TYPE
    assert data["rotation_signature"]["alg"] == "Ed25519"
    key.public_key().verify(bytes.fromhex(data["rotation_signature"]["signature"]),
                            rotation_payload(data))
    assert (seam.agent_id, seam.tool_name) == ("vaara", "trail.rotate")


def test_an_export_of_the_rotated_trail_verifies_and_lists_the_seam(tmp_path, monkeypatch):
    _, priv, _ = _key(tmp_path)
    db, out = tmp_path / "audit.db", tmp_path / "archive.zip"
    _seed(db, monkeypatch)
    rotate(db, out, priv, 7)

    exported = _export_live(db, tmp_path / "live.zip", priv)
    result = verify_signed(tmp_path / "live.zip")

    assert exported.chain_intact
    assert result.ok, result.errors
    (seam,) = result.rotation_seams
    assert seam["state"] == "signed" and seam["begins_here"] is True
    assert seam["purged_records"] == 2 and seam["archive_sha256"] == _sha(out)
    assert describe_rotation_seams(result) == [
        f"begins at a rotation seam: 2 records archived as {_sha(out)}"
    ]


def test_a_second_rotation_runs(tmp_path, monkeypatch):
    _, priv, _ = _key(tmp_path)
    db = tmp_path / "audit.db"
    _seed(db, monkeypatch)
    assert rotate(db, tmp_path / "a1.zip", priv, 7).ok

    real = time.time
    monkeypatch.setattr("vaara.audit.trail.time.time", lambda: real() + 8 * DAY)
    second = rotate(db, tmp_path / "a2.zip", priv, 7)

    assert second.ok, second.errors
    assert second.purged_records == 3  # the two new records and the first seam
    (only,) = _live(db)
    assert only.event_type.value == SEAM
    assert only.previous_hash == only.data["last_hash"]
    # The second archive starts at the first seam and still verifies.
    assert verify_signed(tmp_path / "a2.zip").ok
    _export_live(db, tmp_path / "live.zip", priv)
    assert verify_signed(tmp_path / "live.zip").ok


def test_nothing_to_purge_writes_no_seam(tmp_path, monkeypatch):
    _, priv, _ = _key(tmp_path)
    db = tmp_path / "audit.db"
    _seed(db, monkeypatch)

    result = rotate(db, tmp_path / "archive.zip", priv, 30)

    assert result.ok and result.purged_records == 0
    assert len(_live(db)) == 4
    assert SEAM not in [r.event_type.value for r in _live(db)]


def test_a_dry_run_writes_no_seam(tmp_path, monkeypatch):
    _, priv, _ = _key(tmp_path)
    db = tmp_path / "audit.db"
    _seed(db, monkeypatch)

    result = rotate(db, tmp_path / "archive.zip", priv, 7, dry_run=True)

    assert result.ok and result.purged_records == 2
    assert len(_live(db)) == 4


def test_a_failed_export_writes_no_seam(tmp_path, monkeypatch):
    db = tmp_path / "audit.db"
    _seed(db, monkeypatch)
    bad = tmp_path / "bad.pem"
    bad.write_text("not a key")

    result = rotate(db, tmp_path / "archive.zip", bad, 7)

    assert not result.ok and len(_live(db)) == 4


def test_a_store_that_refuses_the_seam_skips_the_purge(tmp_path, monkeypatch):
    _, priv, _ = _key(tmp_path)
    db, out = tmp_path / "audit.db", tmp_path / "archive.zip"
    _seed(db, monkeypatch)

    def refuse(self, record, stamp):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(SQLiteAuditBackend, "append_record", refuse)
    result = rotate(db, out, priv, 7)
    monkeypatch.undo()

    assert not result.ok and result.purged_records == 0
    assert any("rotation record" in e and "purge skipped" in e for e in result.errors)
    assert out.exists() and len(_live(db)) == 4


def test_a_clock_that_stepped_back_still_leaves_an_exact_seam(tmp_path, monkeypatch):
    """A record written later with an old timestamp is not purged out of order:
    the purge takes the unbroken prefix, and the seam names the hash after it."""
    _, priv, _ = _key(tmp_path)
    db = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(db)
    trail = AuditTrail(on_record=backend.write_record)
    real = time.time
    monkeypatch.setattr("vaara.audit.trail.time.time", lambda: real() - 10 * DAY)
    trail.record_decision("old1", "agent-x", "read_file", "allow", "low risk", 0.1)
    monkeypatch.setattr("vaara.audit.trail.time.time", real)
    trail.record_decision("new1", "agent-x", "read_file", "allow", "low risk", 0.1)
    monkeypatch.setattr("vaara.audit.trail.time.time", lambda: real() - 10 * DAY)
    trail.record_decision("old2", "agent-x", "read_file", "allow", "low risk", 0.1)
    monkeypatch.setattr("vaara.audit.trail.time.time", real)
    backend.close()

    result = rotate(db, tmp_path / "archive.zip", priv, 7)

    assert result.ok and result.purged_records == 1, result.errors
    live = _live(db)
    assert [r.action_id for r in live[:2]] == ["new1", "old2"]
    assert live[0].previous_hash == live[-1].data["last_hash"]
    _export_live(db, tmp_path / "live.zip", priv)
    assert verify_signed(tmp_path / "live.zip").ok


def test_an_encrypted_rotation_names_the_sealed_file(tmp_path, monkeypatch):
    _, priv, _ = _key(tmp_path)
    rpriv, rpub = tmp_path / "dpo.key", tmp_path / "dpo.pub"
    sa.generate_recipient_key(rpriv, rpub)
    db, out = tmp_path / "audit.db", tmp_path / "archive.zip"
    _seed(db, monkeypatch)

    result = rotate(db, out, priv, 7, encrypt_to=rpub)

    assert result.ok, result.errors
    seam = _live(db)[-1]
    assert seam.data["archive_kind"] == "sealed"
    assert seam.data["archive_sha256"] == _sha(result.sealed_path)
    assert not out.exists()  # only the sealed file is kept


# --- the live walkers -----------------------------------------------------------


def test_the_live_walkers_accept_the_seam(tmp_path, monkeypatch):
    _, priv, _ = _key(tmp_path)
    db, out = tmp_path / "audit.db", tmp_path / "archive.zip"
    _seed(db, monkeypatch)
    rotate(db, out, priv, 7)

    backend = SQLiteAuditBackend(db)
    try:
        trail = backend.load_trail()
        verdicts = [trail.verify_chain_verdict(), backend.verify_chain_streaming_verdict()]
        for verdict in verdicts:
            assert verdict.state == "intact", verdict.summary()
            (seam,) = verdict.seams
            assert seam.purged_records == 2 and seam.archive_sha256 == _sha(out)
        assert trail.verify_chain() is None
        assert backend.verify_chain_streaming() is None
    finally:
        backend.close()


def test_a_plain_purge_is_still_a_chain_break(tmp_path, monkeypatch):
    """The old command leaves no record of what it removed, so nothing accounts
    for the first survivor's link."""
    _, priv, _ = _key(tmp_path)
    db = tmp_path / "audit.db"
    _seed(db, monkeypatch)
    backend = SQLiteAuditBackend(db)
    try:
        assert backend.purge_older_than(7 * 86400) == 2
        trail = backend.load_trail()
        assert trail.verify_chain_verdict().state == "broken"
        assert backend.verify_chain_streaming_verdict().state == "broken"
        out = tmp_path / "live.zip"
        export_signed(trail, out, signer_key=priv)
    finally:
        backend.close()
    result = verify_signed(out)
    assert not result.ok
    assert any("record 0: previous_hash mismatch" in e for e in result.errors)


def test_a_seam_naming_another_hash_is_still_a_break(tmp_path):
    key, priv, _ = _key(tmp_path)
    records, _ = _chain(key, last_hash="f" * 64)
    trail = _trail_of(records[2:])

    verdict = trail.verify_chain_verdict()

    assert verdict.state == "broken"
    assert "Chain broken at record 0" in verdict.error


def test_a_first_record_that_links_to_a_seam_outside_the_trail_is_a_break(tmp_path):
    """The seam has to be in the trail it accounts for."""
    key, priv, _ = _key(tmp_path)
    records, _ = _chain(key)
    trail = _trail_of([records[2]])  # c, without the seam that names b

    assert trail.verify_chain_verdict().state == "broken"


# --- the export verifier --------------------------------------------------------


def test_the_export_verifier_accepts_a_signed_seam_and_lists_it(tmp_path):
    key, priv, _ = _key(tmp_path)
    records, seam = _chain(key)
    out = _survivors_zip(tmp_path, priv, records[2:])

    result = verify_signed(out)

    assert result.ok, result.errors
    (listed,) = result.rotation_seams
    assert listed["index"] == 1 and listed["record_id"] == seam.record_id
    assert listed["state"] == "signed" and listed["begins_here"] is True
    assert listed["last_hash"] == records[1].record_hash


def test_an_unmatched_first_record_fails_with_the_chain_break(tmp_path):
    key, priv, _ = _key(tmp_path)
    records, _ = _chain(key, last_hash="f" * 64)
    out = _survivors_zip(tmp_path, priv, records[2:])

    result = verify_signed(out)

    assert not result.ok
    assert any("record 0: previous_hash mismatch" in e for e in result.errors)
    assert result.rotation_seams[0]["begins_here"] is False


def test_an_unsigned_seam_does_not_account_for_the_start(tmp_path):
    key, priv, _ = _key(tmp_path)

    def no_signature(data):
        return None

    records, _ = _chain(key, sign=no_signature)
    out = _survivors_zip(tmp_path, priv, records[2:])

    result = verify_signed(out)

    assert not result.ok
    assert result.rotation_seams[0]["state"] == "unsigned"
    assert any("record 0: previous_hash mismatch" in e and "unsigned" in e
               for e in result.errors)


def test_a_seam_signed_by_another_key_fails_unless_that_key_is_given(tmp_path):
    key, priv, _ = _key(tmp_path)
    other, _, other_pub = _key(tmp_path, "other")
    records, _ = _chain(key, sign=_signer(other))
    out = _survivors_zip(tmp_path, priv, records[2:])

    wrong = verify_signed(out)
    assert not wrong.ok
    assert wrong.rotation_seams[0]["state"] == "bad_signature"
    assert any("rotation record" in e and "bad_signature" in e for e in wrong.errors)

    pinned = verify_signed(out, rotation_key=other_pub)
    assert pinned.ok, pinned.errors
    assert pinned.rotation_seams[0]["state"] == "signed"


def test_a_signature_over_other_fields_fails(tmp_path):
    key, priv, _ = _key(tmp_path)

    def sign_a_different_count(data):
        forged = dict(data, purged_records=data["purged_records"] + 1)
        return {"type": ROTATION_RECORD_TYPE, "alg": "Ed25519",
                "signature": key.sign(rotation_payload(forged)).hex()}

    records, _ = _chain(key, sign=sign_a_different_count)
    out = _survivors_zip(tmp_path, priv, records[2:])

    result = verify_signed(out)

    assert not result.ok and result.rotation_seams[0]["state"] == "bad_signature"


def test_a_malformed_signature_block_fails(tmp_path):
    key, priv, _ = _key(tmp_path)
    records, _ = _chain(key, sign=lambda data: {"type": ROTATION_RECORD_TYPE, "alg": "Ed25519"})
    out = _survivors_zip(tmp_path, priv, records[2:])

    result = verify_signed(out)

    assert not result.ok and result.rotation_seams[0]["state"] == "malformed"


def test_a_seam_that_does_not_begin_the_trail_is_listed_and_judged(tmp_path):
    """A rotation whose purge never ran leaves a seam in the middle of a trail
    that still begins at genesis. It is listed, and its signature still counts."""
    key, priv, _ = _key(tmp_path)
    records, seam = _chain(key)
    out = _survivors_zip(tmp_path, priv, records)  # whole chain, seam at the tip

    result = verify_signed(out)

    assert result.ok, result.errors
    (listed,) = result.rotation_seams
    assert listed["begins_here"] is False and listed["state"] == "signed"
    assert describe_rotation_seams(result) == [
        f"rotation record 3: 2 records archived as {'ab' * 32} (this trail does not begin there)"
    ]


# --- the archive ----------------------------------------------------------------


def test_the_archive_is_checked_against_its_digest(tmp_path, monkeypatch):
    _, priv, _ = _key(tmp_path)
    db, out = tmp_path / "audit.db", tmp_path / "archive.zip"
    _seed(db, monkeypatch)
    rotate(db, out, priv, 7)
    _export_live(db, tmp_path / "live.zip", priv)

    result = verify_signed(tmp_path / "live.zip", archive=out)

    assert result.ok, result.errors
    assert result.rotation_seams[0]["archive_check"] == "verified"
    assert describe_rotation_seams(result)[0].endswith(
        "(archive checked: digest matches, 4 records verify, holds the record this trail links to)"
    )


def test_a_different_archive_fails_the_digest(tmp_path, monkeypatch):
    key, priv, _ = _key(tmp_path)
    db, out = tmp_path / "audit.db", tmp_path / "archive.zip"
    _seed(db, monkeypatch)
    rotate(db, out, priv, 7)
    _export_live(db, tmp_path / "live.zip", priv)
    elsewhere = _survivors_zip(tmp_path, priv, _chain(key)[0], "elsewhere.zip")

    result = verify_signed(tmp_path / "live.zip", archive=elsewhere)

    assert not result.ok
    assert result.rotation_seams[0]["archive_check"] == "digest_mismatch"
    assert any("archive" in e and "does not match" in e for e in result.errors)


def test_a_sealed_archive_is_checked_by_digest_only(tmp_path, monkeypatch):
    _, priv, _ = _key(tmp_path)
    rpriv, rpub = tmp_path / "dpo.key", tmp_path / "dpo.pub"
    sa.generate_recipient_key(rpriv, rpub)
    db, out = tmp_path / "audit.db", tmp_path / "archive.zip"
    _seed(db, monkeypatch)
    result = rotate(db, out, priv, 7, encrypt_to=rpub)
    _export_live(db, tmp_path / "live.zip", priv)

    checked = verify_signed(tmp_path / "live.zip", archive=result.sealed_path)

    assert checked.ok, checked.errors
    assert checked.rotation_seams[0]["archive_check"] == "digest_only"
    assert "contents not checked" in describe_rotation_seams(checked)[0]


# --- the commands ---------------------------------------------------------------


def test_the_cli_rotates_and_verifies_across_the_seam(tmp_path, monkeypatch, capsys):
    from vaara.cli import main

    _, priv, pub = _key(tmp_path)
    db, out = tmp_path / "audit.db", tmp_path / "archive.zip"
    _seed(db, monkeypatch)

    assert main(["trail", "rotate", "--db", str(db), "--out", str(out), "--key", str(priv),
                 "--retention-days", "7", "--all-tenants"]) == 0
    rotated = capsys.readouterr().out
    assert "Purged 2" in rotated and "begins at a rotation seam" in rotated

    live = tmp_path / "live.zip"
    assert main(["trail", "export", "--db", str(db), "--out", str(live), "--key", str(priv)]) == 0
    capsys.readouterr()

    assert main(["trail", "verify", "--zip", str(live), "--pubkey", str(pub),
                 "--archive", str(out)]) == 0
    verified = capsys.readouterr().out
    assert "Verification: OK" in verified
    assert f"begins at a rotation seam: 2 records archived as {_sha(out)}" in verified
    assert "archive checked: digest matches" in verified


def test_the_cli_fails_a_trail_with_no_seam_and_says_why(tmp_path, monkeypatch, capsys):
    from vaara.cli import main

    _, priv, pub = _key(tmp_path)
    db = tmp_path / "audit.db"
    _seed(db, monkeypatch)
    backend = SQLiteAuditBackend(db)
    backend.purge_older_than(7 * 86400)
    backend.close()
    live = tmp_path / "live.zip"
    # The export is written, and its own exit code says the chain is not intact.
    assert main(["trail", "export", "--db", str(db), "--out", str(live), "--key", str(priv)]) == 1
    capsys.readouterr()

    assert main(["trail", "verify", "--zip", str(live), "--pubkey", str(pub)]) == 1
    err = capsys.readouterr().err
    assert "Verification: FAILED" in err
    assert "no rotation record in the trail names that hash" in err


def test_the_cli_refuses_an_archive_that_is_not_the_one(tmp_path, monkeypatch, capsys):
    from vaara.cli import main

    key, priv, pub = _key(tmp_path)
    db, out = tmp_path / "audit.db", tmp_path / "archive.zip"
    _seed(db, monkeypatch)
    rotate(db, out, priv, 7)
    live = tmp_path / "live.zip"
    _export_live(db, live, priv)
    elsewhere = _survivors_zip(tmp_path, priv, _chain(key)[0], "elsewhere.zip")

    assert main(["trail", "verify", "--zip", str(live), "--pubkey", str(pub),
                 "--archive", str(elsewhere)]) == 1
    assert "does not match rotation record" in capsys.readouterr().err
    assert main(["trail", "verify", "--zip", str(live), "--archive",
                 str(tmp_path / "nope.zip")]) == 2


def test_vaara_audit_verify_reports_the_seam(tmp_path, monkeypatch, capsys):
    from vaara.audit_cli import main

    _, priv, pub = _key(tmp_path)
    db, out = tmp_path / "audit.db", tmp_path / "archive.zip"
    _seed(db, monkeypatch)
    rotate(db, out, priv, 7)
    live = tmp_path / "live.zip"
    _export_live(db, live, priv)

    assert main(["verify", str(live), "--pubkey", str(pub), "--archive", str(out),
                 "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    (seam,) = payload["rotation_seams"]
    assert seam["begins_here"] is True and seam["archive_check"] == "verified"
