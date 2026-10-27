"""The standalone verifier reads an erased trail the way `vaara trail verify` does.

`vaara trail erase` keeps the erased slot's stored hash, so the chain links
still verify, and a later `record_erased` record names the slot. The library
verifier accounts for that. `scripts/verify_vaara_trail.py` imports no Vaara
code and used to call every erased slot a tampered record, so an auditor
holding only the script could not check a trail that had honoured a GDPR
Art. 17 request. It now gives the same verdicts: the slot reads as erased,
the eraser's signature is judged under `--erasure-key`, and a tombstone no
erasure record accounts for is still a hash mismatch.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric import ec, rsa  # noqa: E402

from tests.test_erasure_record_signed import (  # noqa: E402
    DPO,
    GROUND,
    SECRET,
    _as_dicts,
    _build,
    _erasure_records,
    _export,
    _pem_public,
)
from vaara.audit import verify as lib  # noqa: E402
from vaara.audit.erase import erase_record  # noqa: E402
from vaara.audit.sqlite_backend import SQLiteAuditBackend  # noqa: E402
from vaara.audit.trail import ERASED_DATA_KEY  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_vaara_trail.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_vaara_trail_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


script = _load_script()


def test_the_script_still_imports_no_vaara_code():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names]
    imported += [node.module or "" for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom)]

    assert not [name for name in imported if name.split(".")[0] == "vaara"]


def _run(zip_path: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(zip_path), *extra],
                          capture_output=True, text=True)


def _keys():
    return [
        pytest.param(lambda: ec.generate_private_key(ec.SECP256R1()), id="ES256"),
        pytest.param(lambda: rsa.generate_private_key(public_exponent=65537,
                                                      key_size=2048), id="RS256"),
    ]


def _pub_file(tmp_path: Path, key, name: str = "eraser.pub.pem") -> Path:
    path = tmp_path / name
    path.write_bytes(_pem_public(key))
    return path


# --- the whole script, on a real erased export ---------------------------------


def test_an_unsigned_erasure_verifies_and_the_slot_reads_as_erased(tmp_path: Path):
    path, record_id = _build(tmp_path)
    erase_record(path, record_id, reason="data subject request")
    zp = _export(path, tmp_path)

    run = _run(zp)

    assert run.returncode == 0, run.stderr
    (slot,) = lib.verify_signed(zp).erased
    assert (f"  record {slot['index']} ({slot['record_id']}) erased: link verified, "
            f"content not checkable (erasure record {slot['erased_by_index']})"
            ) in run.stdout.splitlines()
    assert "eraser not signed" in run.stdout
    assert SECRET not in run.stdout


@pytest.mark.parametrize("make_key", _keys())
def test_a_signed_erasure_is_checked_under_the_erasure_key(tmp_path: Path, make_key):
    path, record_id = _build(tmp_path)
    key = make_key()
    erase_record(path, record_id, reason="dsr", legal_ground=GROUND, erased_by=DPO,
                 sign_key=key)
    zp = _export(path, tmp_path)

    run = _run(zp, "--erasure-key", str(_pub_file(tmp_path, key)))

    assert run.returncode == 0, run.stderr
    assert f"signed by {DPO} under the given key, ground: {GROUND}" in run.stdout


def test_without_an_erasure_key_a_signature_is_reported_unchecked(tmp_path: Path):
    path, record_id = _build(tmp_path)
    erase_record(path, record_id, reason="dsr", legal_ground=GROUND, erased_by=DPO,
                 sign_key=ec.generate_private_key(ec.SECP256R1()))
    zp = _export(path, tmp_path)

    run = _run(zp)

    assert run.returncode == 0, run.stderr
    assert (f"signed as {DPO}, not checked (pass --erasure-key to check whose key)"
            in run.stdout)


def test_another_erasure_key_fails_the_export(tmp_path: Path):
    path, record_id = _build(tmp_path)
    erase_record(path, record_id, reason="dsr", legal_ground=GROUND, erased_by=DPO,
                 sign_key=ec.generate_private_key(ec.SECP256R1()))
    zp = _export(path, tmp_path)
    other = ec.generate_private_key(ec.SECP256R1())

    run = _run(zp, "--erasure-key", str(_pub_file(tmp_path, other)))

    assert run.returncode == 1
    assert "Verification: FAILED" in run.stderr
    assert f"erasure of {record_id} is bad_signature" in run.stderr


def test_with_a_key_given_an_unsigned_erasure_is_a_finding(tmp_path: Path):
    path, record_id = _build(tmp_path)
    erase_record(path, record_id, reason="dsr")
    zp = _export(path, tmp_path)
    key = ec.generate_private_key(ec.SECP256R1())

    run = _run(zp, "--erasure-key", str(_pub_file(tmp_path, key)))

    assert run.returncode == 1
    assert f"erasure of {record_id} is unsigned" in run.stderr


def test_an_unreadable_erasure_key_stops_with_exit_2(tmp_path: Path):
    path, record_id = _build(tmp_path)
    erase_record(path, record_id, reason="dsr")
    zp = _export(path, tmp_path)
    bad = tmp_path / "not-a-key.pem"
    bad.write_text("not a key", encoding="utf-8")

    run = _run(zp, "--erasure-key", str(bad))

    assert run.returncode == 2
    assert "--erasure-key: erasure key could not be parsed" in run.stderr


def test_a_tombstone_no_erasure_record_names_is_a_hash_mismatch(tmp_path: Path):
    path, record_id = _build(tmp_path)
    backend = SQLiteAuditBackend(path)
    try:
        backend.erase_record_data(record_id, {ERASED_DATA_KEY: {"erasure_record_id": "x"}})
    finally:
        backend.close()
    zp = _export(path, tmp_path)

    run = _run(zp)

    assert run.returncode == 1
    assert "erasure tombstone that no later record_erased record accounts for" in run.stderr


# --- the chain walk and the eraser check, line for line against the library ----


def _record(prev: str, record_id: str, event_type: str, data: dict) -> dict:
    rec = {"record_id": record_id, "action_id": "act", "event_type": event_type,
           "timestamp": 1760000000.0, "agent_id": "agent", "tool_name": "file.read",
           "data": data, "regulatory_articles": [], "previous_hash": prev}
    canonical = json.dumps(rec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    rec["record_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    return rec


def _chain(*, tombstone: bool = True, erasure: str = "named",
           tamper_plain: bool = False) -> bytes:
    """Four records: the erased one at index 1, the erasure record at index 3."""
    r0 = _record("", "r0", "action_requested", {"n": 0})
    r1 = _record(r0["record_hash"], "r1", "action_requested", {"path": SECRET})
    r2 = _record(r1["record_hash"], "r2", "action_requested", {"n": 2})
    records = [r0, r1, r2]
    if erasure != "none":
        named_hash = r1["record_hash"] if erasure == "named" else "0" * 64
        named_id = "r1" if erasure != "other-id" else "r9"
        records.append(_record(r2["record_hash"], "r3", "record_erased", {
            "erased_record_id": named_id, "erased_record_hash": named_hash,
            "erased_seq": 1, "erased_event_type": "action_requested",
            "erased_action_id": "act", "reason": "dsr"}))
    if tombstone:
        r1["data"] = {ERASED_DATA_KEY: {"erasure_record_id": "r3"}}
    if tamper_plain:
        r1["data"] = {"path": "/elsewhere"}
    return b"\n".join(json.dumps(r).encode() for r in records) + b"\n"


@pytest.mark.parametrize("kwargs", [
    {},
    {"tombstone": False, "erasure": "none"},
    {"erasure": "none"},
    {"erasure": "wrong-hash"},
    {"erasure": "other-id"},
    {"tombstone": False, "tamper_plain": True},
], ids=["accounted", "no-erasure-no-tombstone", "tombstone-alone", "wrong-hash",
        "other-id", "plain-tamper"])
def test_the_chain_walk_gives_the_librarys_verdict(kwargs):
    blob = _chain(**kwargs)

    assert script._verify_chain(blob) == lib._verify_chain_bytes(blob)


def test_the_accounted_chain_lists_the_erased_slot():
    error, erased = script._verify_chain(_chain())

    assert error is None
    assert erased == [{"index": 1, "record_id": "r1",
                       "record_hash": json.loads(_chain().splitlines()[1])["record_hash"],
                       "erased_by_index": 3}]


@pytest.mark.parametrize("data", [
    {"erased_record_id": "id-ä", "erased_record_hash": "ab", "erased_seq": 7,
     "erased_event_type": "action_requested", "erased_action_id": "a\"\\\n\x01/€",
     "reason": "tab\there", "legal_ground": "GDPR Art. 17",
     "erased_by": "Matti Meikäläinen", "erased_at": "2026-10-21T14:00:00Z"},
    {"erased_record_id": "x", "erased_record_hash": "y", "erased_seq": None,
     "reason": "r"},
], ids=["escapes-and-unicode", "nulls"])
def test_the_signed_payload_is_the_librarys_bytes(data):
    assert script.erasure_payload(data) == lib.erasure_payload(data)


def test_the_signed_payload_refuses_what_the_library_refuses():
    bad = {"erased_record_id": "x", "erased_seq": True}

    with pytest.raises(ValueError):
        lib.erasure_payload(bad)
    with pytest.raises(ValueError):
        script.erasure_payload(bad)


@pytest.mark.parametrize("make_key", _keys())
def test_every_eraser_state_matches_the_library(tmp_path: Path, make_key):
    path, record_id = _build(tmp_path)
    key = make_key()
    erase_record(path, record_id, reason="dsr", legal_ground=GROUND, erased_by=DPO,
                 sign_key=key)
    signed = _as_dicts(_erasure_records(path))
    unsigned = [{**signed[0], "data": {k: v for k, v in signed[0]["data"].items()
                                       if k != "eraser_signature"}}]
    malformed = [{**signed[0], "data": {**signed[0]["data"],
                                        "eraser_signature": {"type": "x"}}}]
    wrong = _pem_public(ec.generate_private_key(ec.SECP256R1()))

    for records in (signed, unsigned, malformed):
        for pub in (None, _pem_public(key), wrong):
            assert (script.check_erasures(records, pub)
                    == lib.check_erasures(records, pub))
    assert [r["state"] for r in script.check_erasures(signed, _pem_public(key))] == ["signed"]
    assert [r["state"] for r in script.check_erasures(signed, wrong)] == ["bad_signature"]
    assert [r["state"] for r in script.check_erasures(signed)] == ["unpinned"]
    assert [r["state"] for r in script.check_erasures(unsigned)] == ["unsigned"]
    assert [r["state"] for r in script.check_erasures(malformed, wrong)] == ["malformed"]
