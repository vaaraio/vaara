"""A signed erasure record says who erased what, on which legal ground.

`vaara trail erase` already appends a record_erased record naming the slot by
id and stored hash. That record is signed only by the trail, so it says the
operator erased something. With ``--legal-ground``, ``--erased-by`` and
``--sign-key`` the record also carries the ground, the person, the time, and
that person's own ES256 or RS256 signature over all of it. A reader holding
the eraser's public key can then tell a lawful erasure from a hole, without
trusting the operator. The key is never carried in the record.

An erasure without the new options is byte-for-byte the record it was before.
"""
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec, rsa  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from vaara.audit.erase import ErasureRefused, erase_record  # noqa: E402
from vaara.audit.sqlite_backend import SQLiteAuditBackend  # noqa: E402
from vaara.audit.trail import EventType  # noqa: E402
from vaara.audit.verify import (  # noqa: E402
    ERASURE_RECORD_TYPE,
    check_erasures,
    erasure_payload,
    verify_signed,
)
from vaara.pipeline import InterceptionPipeline  # noqa: E402

SECRET = "/home/matti-meikalainen/terveystiedot.txt"
GROUND = "GDPR Art. 17(1)(a)"
DPO = "did:web:example.fi:dpo"


def _build(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(path)
    pipeline = InterceptionPipeline(trail=backend.load_trail())
    for i in range(3):
        pipeline.intercept(agent_id="erase-test", tool_name="file.read",
                           parameters={"path": f"/srv/plain/{i}.txt"})
    pipeline.intercept(agent_id="erase-test", tool_name="file.read",
                       parameters={"path": SECRET})
    pipeline.intercept(agent_id="erase-test", tool_name="file.read",
                       parameters={"path": "/srv/after.txt"})
    target = next(r for r in backend.load_trail()._records
                  if SECRET in json.dumps(r.data))
    backend.close()
    return path, target.record_id


def _erasure_records(path: Path) -> list:
    backend = SQLiteAuditBackend(path)
    try:
        return [r for r in backend.load_trail()._records
                if r.event_type == EventType.RECORD_ERASED]
    finally:
        backend.close()


def _as_dicts(records: list) -> list[dict]:
    return [{"record_id": r.record_id, "event_type": r.event_type.value,
             "data": r.data} for r in records]


def _pem_public(key) -> bytes:
    return key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _pem_private(key) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def _export(path: Path, tmp_path: Path) -> Path:
    from vaara.audit.export import export_signed

    backend = SQLiteAuditBackend(path)
    try:
        out = tmp_path / "trail.zip"
        export_signed(backend.load_trail(), out_path=out,
                      signer_key=Ed25519PrivateKey.generate())
    finally:
        backend.close()
    return out


def test_an_unsigned_erasure_keeps_the_old_record_shape(tmp_path: Path):
    path, record_id = _build(tmp_path)

    erase_record(path, record_id, reason="dsr")

    (rec,) = _erasure_records(path)
    assert set(rec.data) == {"erased_record_id", "erased_record_hash", "erased_seq",
                             "erased_event_type", "erased_action_id", "reason"}
    (row,) = check_erasures(_as_dicts([rec]))
    assert row["state"] == "unsigned"
    assert row["legal_ground"] == "" and row["erased_by"] == ""


@pytest.mark.parametrize("make_key", [
    lambda: ec.generate_private_key(ec.SECP256R1()),
    lambda: rsa.generate_private_key(public_exponent=65537, key_size=2048),
], ids=["ES256", "RS256"])
def test_a_signed_erasure_verifies_under_the_erasers_key(tmp_path: Path, make_key):
    path, record_id = _build(tmp_path)
    key = make_key()

    report = erase_record(path, record_id, reason="data subject request",
                          legal_ground=GROUND, erased_by=DPO, sign_key=key)

    assert report["signed"] is True
    (rec,) = _erasure_records(path)
    assert rec.data["legal_ground"] == GROUND
    assert rec.data["erased_by"] == DPO
    assert rec.data["erased_at"].endswith("Z")
    block = rec.data["eraser_signature"]
    assert set(block) == {"type", "alg", "signature"}
    assert block["type"] == ERASURE_RECORD_TYPE
    assert "BEGIN" not in json.dumps(rec.data)  # no key in the record
    (row,) = check_erasures(_as_dicts([rec]), public_key=_pem_public(key))
    assert row["state"] == "signed"
    assert row["erased_record_id"] == record_id
    assert row["legal_ground"] == GROUND and row["erased_by"] == DPO


def test_without_a_key_a_signature_is_unpinned_not_signed(tmp_path: Path):
    path, record_id = _build(tmp_path)
    erase_record(path, record_id, reason="dsr", legal_ground=GROUND, erased_by=DPO,
                 sign_key=ec.generate_private_key(ec.SECP256R1()))

    (row,) = check_erasures(_as_dicts(_erasure_records(path)))

    assert row["state"] == "unpinned"


def test_another_key_is_a_bad_signature(tmp_path: Path):
    path, record_id = _build(tmp_path)
    erase_record(path, record_id, reason="dsr", legal_ground=GROUND, erased_by=DPO,
                 sign_key=ec.generate_private_key(ec.SECP256R1()))
    other = ec.generate_private_key(ec.SECP256R1())

    (row,) = check_erasures(_as_dicts(_erasure_records(path)),
                            public_key=_pem_public(other))

    assert row["state"] == "bad_signature"


@pytest.mark.parametrize("field,value", [
    ("legal_ground", "GDPR Art. 17(1)(d)"),
    ("erased_by", "did:web:example.fi:intern"),
    ("reason", "other"),
    ("erased_record_hash", "0" * 64),
    ("erased_at", "2020-01-01T00:00:00Z"),
])
def test_every_signed_field_is_bound(tmp_path: Path, field, value):
    path, record_id = _build(tmp_path)
    key = ec.generate_private_key(ec.SECP256R1())
    erase_record(path, record_id, reason="dsr", legal_ground=GROUND, erased_by=DPO,
                 sign_key=key)
    (rec,) = _as_dicts(_erasure_records(path))
    rec["data"][field] = value

    (row,) = check_erasures([rec], public_key=_pem_public(key))

    assert row["state"] == "bad_signature"


def test_a_block_that_is_not_the_closed_shape_is_malformed(tmp_path: Path):
    path, record_id = _build(tmp_path)
    key = ec.generate_private_key(ec.SECP256R1())
    erase_record(path, record_id, reason="dsr", legal_ground=GROUND, erased_by=DPO,
                 sign_key=key)
    (rec,) = _as_dicts(_erasure_records(path))
    rec["data"]["eraser_signature"]["publicKey"] = "x"

    (row,) = check_erasures([rec], public_key=_pem_public(key))

    assert row["state"] == "malformed"


def test_with_a_key_given_an_unsigned_erasure_is_a_finding(tmp_path: Path):
    path, record_id = _build(tmp_path)
    erase_record(path, record_id, reason="dsr")
    key = ec.generate_private_key(ec.SECP256R1())

    vr = verify_signed(_export(path, tmp_path), erasure_key=_pem_public(key))

    assert not vr.ok
    assert any("erasure" in e and "unsigned" in e for e in vr.errors)


def test_a_signing_key_needs_a_named_eraser(tmp_path: Path):
    path, record_id = _build(tmp_path)

    with pytest.raises(ErasureRefused, match="erased-by"):
        erase_record(path, record_id, reason="dsr",
                     sign_key=ec.generate_private_key(ec.SECP256R1()))


def test_the_payload_is_rfc8785(tmp_path: Path):
    rfc8785 = pytest.importorskip("rfc8785")
    data = {"erased_record_id": "id-ä", "erased_record_hash": "ab",
            "erased_seq": 7, "erased_event_type": "action_requested",
            "erased_action_id": "a\"\\\n\x01/€", "reason": "tab\there",
            "legal_ground": "GDPR Art. 17", "erased_by": "Matti Meikäläinen",
            "erased_at": "2026-10-21T14:00:00Z"}
    expected = rfc8785.dumps({
        "type": ERASURE_RECORD_TYPE, "erasedRecordId": "id-ä",
        "erasedRecordHash": "ab", "erasedSeq": 7,
        "erasedEventType": "action_requested", "erasedActionId": "a\"\\\n\x01/€",
        "reason": "tab\there", "legalGround": "GDPR Art. 17",
        "erasedBy": "Matti Meikäläinen", "erasedAt": "2026-10-21T14:00:00Z",
    })
    assert erasure_payload(data) == expected
    data["erased_seq"] = None
    assert b'"erasedSeq":null' in erasure_payload(data)


def test_the_export_verifier_reports_each_erasure(tmp_path: Path):
    path, record_id = _build(tmp_path)
    key = ec.generate_private_key(ec.SECP256R1())
    erase_record(path, record_id, reason="dsr", legal_ground=GROUND, erased_by=DPO,
                 sign_key=key)
    zp = _export(path, tmp_path)

    plain = verify_signed(zp)
    pinned = verify_signed(zp, erasure_key=_pem_public(key))
    wrong = verify_signed(zp, erasure_key=_pem_public(ec.generate_private_key(ec.SECP256R1())))

    assert plain.ok and [e["state"] for e in plain.erasures] == ["unpinned"]
    assert pinned.ok and [e["state"] for e in pinned.erasures] == ["signed"]
    assert pinned.erasures[0]["legal_ground"] == GROUND
    assert not wrong.ok and [e["state"] for e in wrong.erasures] == ["bad_signature"]
    with zipfile.ZipFile(zp) as z:
        assert SECRET not in z.read("trail.jsonl").decode()


def test_cli_erase_signed_then_verify(tmp_path: Path):
    path, record_id = _build(tmp_path)
    key = ec.generate_private_key(ec.SECP256R1())
    priv, pub = tmp_path / "dpo.pem", tmp_path / "dpo.pub.pem"
    priv.write_bytes(_pem_private(key))
    pub.write_bytes(_pem_public(key))

    out = subprocess.run(
        [sys.executable, "-m", "vaara.cli", "trail", "erase", "--db", str(path),
         "--record-id", record_id, "--reason", "dsr", "--legal-ground", GROUND,
         "--erased-by", DPO, "--sign-key", str(priv), "--format", "json"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["signed"] is True

    zp = _export(path, tmp_path)
    ver = subprocess.run(
        [sys.executable, "-m", "vaara.cli", "trail", "verify", "--zip", str(zp),
         "--erasure-key", str(pub)],
        capture_output=True, text=True,
    )
    assert ver.returncode == 0, ver.stderr
    assert "signed by " + DPO in ver.stdout and GROUND in ver.stdout


def test_vaara_audit_verify_takes_the_erasure_key(tmp_path: Path):
    path, record_id = _build(tmp_path)
    key = ec.generate_private_key(ec.SECP256R1())
    erase_record(path, record_id, reason="dsr", legal_ground=GROUND, erased_by=DPO,
                 sign_key=key)
    pub = tmp_path / "dpo.pub.pem"
    pub.write_bytes(_pem_public(key))
    zp = _export(path, tmp_path)

    ver = subprocess.run(
        [sys.executable, "-m", "vaara.audit_cli", "verify", str(zp),
         "--erasure-key", str(pub), "--json"],
        capture_output=True, text=True,
    )

    assert ver.returncode == 0, ver.stderr
    (row,) = json.loads(ver.stdout)["erasures"]
    assert row["state"] == "signed" and row["erased_by"] == DPO
