# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gate the vaara.trail-rotation/v0 vectors: the library and an independent checker agree.

The checker imports no Vaara and decides every case from the records, the
public keys and the archive bytes. The library test packs the same records
into a signed export and runs ``verify_signed`` on it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from vaara.audit.verify import verify_signed  # noqa: E402

VECTORS = Path(__file__).resolve().parent / "vectors" / "trail_rotation_v0"
DOC = json.loads((VECTORS / "cases.json").read_text(encoding="utf-8"))


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, "-I", str(VECTORS / "_check_independent.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _private(name: str) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(DOC["signers"][name]["seed_hex"]))


def _public_pem(name: str) -> bytes:
    return _private(name).public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)


def _export(records: list[dict], path: Path) -> None:
    """The records as a signed export, the way ``export_signed`` lays one out."""
    key = _private(DOC["export_signer"])
    trail = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode()
    raw = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    manifest = json.dumps({
        "schema_version": "1",
        "record_count": len(records),
        "first_hash": records[0]["record_hash"],
        "last_hash": records[-1]["record_hash"],
        "trail_sha256": hashlib.sha256(trail).hexdigest(),
        "signer_pubkey_fingerprint": hashlib.sha256(raw).hexdigest()[:32],
        "signature_algorithm": "Ed25519",
    }, sort_keys=True, indent=2).encode()
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("trail.jsonl", trail)
        zf.writestr("manifest.json", manifest)
        zf.writestr("trail.sig", key.sign(hashlib.sha256(trail + manifest).digest()))
        zf.writestr("signer_pubkey.pem", _public_pem(DOC["export_signer"]))


@pytest.mark.parametrize("case", DOC["cases"], ids=[c["name"] for c in DOC["cases"]])
def test_the_library_reproduces_every_committed_case(case, tmp_path):
    export = tmp_path / "trail.zip"
    _export(case["records"], export)
    archive = None
    if case["archive"] is not None:
        archive = tmp_path / "archive.bin"
        archive.write_bytes(bytes.fromhex(
            DOC["archive_hex" if case["archive"] == "match" else "altered_archive_hex"]))

    result = verify_signed(
        export,
        rotation_key=_public_pem(case["rotation_key"]) if case["rotation_key"] else None,
        archive=archive,
    )

    expected = case["expected"]
    assert ("pass" if result.ok else "fail") == expected["verdict"], result.errors
    chain = "break" if any("hash chain check failed" in e for e in result.errors) else "ok"
    assert chain == expected["chain"], result.errors
    assert [{"index": s["index"], "state": s["state"], "begins_here": s["begins_here"]}
            for s in result.rotation_seams] == expected["seams"]
    assert any(s["begins_here"] for s in result.rotation_seams) == expected["begins_at_seam"]
    assert next((s["archive_check"] for s in result.rotation_seams if s["begins_here"]),
                None) == expected["archive_check"]
