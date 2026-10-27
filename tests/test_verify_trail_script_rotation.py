"""The standalone verifier reads a rotation seam the way the library does.

A trail that begins at a retention rotation carries a signed `trail_rotated`
record naming the hash its first record links to. `vaara trail verify` accepts
the export when that record is in it and its signature checks out, and fails a
first record that links to nothing. `scripts/verify_vaara_trail.py` imports no
Vaara code, so an auditor holding only the script used to get a chain break on
every rotated trail. It now gives the same verdicts and prints the same lines.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from tests.test_trail_rotation_seam import (  # noqa: E402
    _chain,
    _export_live,
    _key,
    _seed,
    _sha,
    _signer,
    _survivors_zip,
    _zip_records,
)
from vaara.audit import sealed_archive as sa  # noqa: E402
from vaara.audit import verify as lib  # noqa: E402
from vaara.audit.rotate import rotate  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_vaara_trail.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("verify_vaara_trail_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


script = _load_script()


def _run(zip_path: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(zip_path), *extra],
                          capture_output=True, text=True)


def _rotated(tmp_path, monkeypatch, **kwargs):
    key, priv, pub = _key(tmp_path)
    db, archive = tmp_path / "audit.db", tmp_path / "archive.zip"
    _seed(db, monkeypatch)
    result = rotate(db, archive, priv, 7, **kwargs)
    assert result.ok, result.errors
    live = tmp_path / "live.zip"
    _export_live(db, live, priv)
    return live, archive, pub, result


# --- the whole script, on a real rotation ---------------------------------------


def test_a_rotated_trail_verifies_and_the_script_prints_the_library_lines(tmp_path, monkeypatch):
    live, archive, _, _ = _rotated(tmp_path, monkeypatch)

    run = _run(live)

    assert run.returncode == 0, run.stderr
    expected = lib.describe_rotation_seams(lib.verify_signed(live))
    assert expected == [f"begins at a rotation seam: 2 records archived as {_sha(archive)}"]
    assert f"  {expected[0]}" in run.stdout.splitlines()


def test_the_script_checks_the_archive(tmp_path, monkeypatch):
    live, archive, pub, _ = _rotated(tmp_path, monkeypatch)

    run = _run(live, "--pubkey", str(pub), "--archive", str(archive))

    assert run.returncode == 0, run.stderr
    expected = lib.describe_rotation_seams(
        lib.verify_signed(live, public_key=pub, archive=archive))
    assert "archive checked: digest matches" in expected[0]
    assert f"  {expected[0]}" in run.stdout.splitlines()


def test_the_script_fails_an_archive_that_is_not_the_one(tmp_path, monkeypatch):
    live, _, _, _ = _rotated(tmp_path, monkeypatch)
    key, priv, _ = _key(tmp_path, "other")
    elsewhere = _survivors_zip(tmp_path, priv, _chain(key)[0], "elsewhere.zip")

    run = _run(live, "--archive", str(elsewhere))

    assert run.returncode == 1
    assert "does not match rotation record" in run.stderr
    assert _run(live, "--archive", str(tmp_path / "nope.zip")).returncode == 2


def test_the_script_checks_a_sealed_archive_by_digest_only(tmp_path, monkeypatch):
    rpriv, rpub = tmp_path / "dpo.key", tmp_path / "dpo.pub"
    sa.generate_recipient_key(rpriv, rpub)
    live, _, _, result = _rotated(tmp_path, monkeypatch, encrypt_to=rpub)

    run = _run(live, "--archive", str(result.sealed_path))

    assert run.returncode == 0, run.stderr
    assert "archive checked: digest matches, contents not checked: sealed" in run.stdout


def test_a_purged_trail_with_no_rotation_record_fails_the_script(tmp_path, monkeypatch):
    from vaara.audit.export import export_signed
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    _, priv, _ = _key(tmp_path)
    db = tmp_path / "audit.db"
    _seed(db, monkeypatch)
    backend = SQLiteAuditBackend(db)
    try:
        backend.purge_older_than(7 * 86400)
        export_signed(backend.load_trail(), tmp_path / "live.zip", signer_key=priv)
    finally:
        backend.close()

    run = _run(tmp_path / "live.zip")

    assert run.returncode == 1
    assert "record 0: previous_hash mismatch" in run.stderr
    assert "no rotation record in the trail names that hash" in run.stderr


# --- the same verdicts as the library, one forged export at a time -------------


def _forge(tmp_path, kind):
    key, priv, pub = _key(tmp_path)
    other, _, other_pub = _key(tmp_path, "other")
    sign = _signer(key)
    overrides: dict = {}
    keep = slice(2, None)
    if kind == "unsigned":
        sign = lambda data: None  # noqa: E731
    elif kind == "other key":
        sign = _signer(other)
    elif kind == "no signature field":
        sign = lambda data: {"type": lib.ROTATION_RECORD_TYPE, "alg": "Ed25519"}  # noqa: E731
    elif kind == "wrong hash":
        overrides["last_hash"] = "f" * 64
    elif kind == "zero purged":
        overrides["purged_records"] = 0
    elif kind == "unknown archive kind":
        overrides["archive_kind"] = "tape"
    elif kind == "not the start":
        keep = slice(0, None)
    records, _ = _chain(key, sign=sign, **overrides)
    return _survivors_zip(tmp_path, priv, records[keep], f"{kind.replace(' ', '-')}.zip"), other_pub


@pytest.mark.parametrize("kind", [
    "signed", "unsigned", "other key", "no signature field", "wrong hash",
    "zero purged", "unknown archive kind", "not the start",
])
def test_script_and_library_agree_on_each_export(tmp_path, kind):
    out, other_pub = _forge(tmp_path, kind)

    ok, errors, _ = script.verify(out, None)
    library = lib.verify_signed(out)

    assert ok == library.ok, (errors, library.errors)
    from_script = script.check_rotations(_zip_records(out), lib._load_public_key(
        zipfile.ZipFile(out).read("signer_pubkey.pem")))
    assert [r["state"] for r in from_script] == [r["state"] for r in library.rotation_seams]
    assert [r["begins_here"] for r in library.rotation_seams] == \
        [s["state"] in ("signed", "unpinned") and s["last_hash"] == _zip_records(out)[0]["previous_hash"]
         for s in from_script]


def test_a_rotation_key_settles_a_seam_signed_by_another_key(tmp_path):
    out, other_pub = _forge(tmp_path, "other key")

    assert _run(out).returncode == 1
    run = _run(out, "--rotation-key", str(other_pub))

    assert run.returncode == 0, run.stderr
    assert "begins at a rotation seam" in run.stdout
    assert _run(out, "--rotation-key", str(tmp_path / "nope.pem")).returncode == 2
