"""`vaara trail verify` and the standalone script agree on the revocation registry.

An export may carry `revocation.json` and pin its digest in the signed manifest
(`manifest.revocation.registry_sha256`). `scripts/verify_vaara_trail.py` has
always checked the three ways that can go wrong: the file does not match the
pinned digest, the manifest declares a registry and the file is missing, the
file is there and the manifest declares none. `vaara.audit.verify.verify_signed`,
the reader behind `vaara trail verify`, read neither, so one zip printed
`Verification: OK` on the CLI and FAILED offline. Both now return the same
errors, in the same words.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from tests.test_verifier_member_refusal import (  # noqa: E402
    _export, _records, _run, _swap, lib, script,
)

_SIG_ALG = '"signature_algorithm":"Ed25519"'
_REGISTRY = b'{"revoked":[],"as_of":"2026-09-03T00:00:00Z"}'
_DIGEST = "sha256:" + hashlib.sha256(_REGISTRY).hexdigest()

DIGEST_MISMATCH = (
    "revocation.json digest does not match manifest.revocation.registry_sha256 (tampered)"
)
FILE_MISSING = "manifest declares revocation but revocation.json is missing"
BLOCK_MISSING = "revocation.json present but manifest has no revocation block"
NOT_AN_OBJECT = "manifest revocation block is not an object"


def _with_registry(tmp_path: Path, *, block, file: bytes | None) -> Path:
    """A signed export whose manifest carries `block` as its revocation entry
    (None = no entry) and whose zip holds `file` as revocation.json (None = absent)."""
    mutate = None
    if block is not None:
        entry = json.dumps(block, separators=(",", ":"))
        mutate = _swap(_SIG_ALG, f'{_SIG_ALG},"revocation":{entry}')
    zp = _export(tmp_path, _records(3), mutate_manifest=mutate)
    if file is not None:
        with zipfile.ZipFile(zp, "a") as zf:
            zf.writestr("revocation.json", file)
    return zp


def _verdicts(zp: Path):
    result = lib.verify_signed(zp)
    ok, script_errors, _ = script.verify(zp, None)
    return result, ok, script_errors


def test_a_registry_that_matches_its_pinned_digest_verifies(tmp_path: Path):
    zp = _with_registry(tmp_path, block={"registry_sha256": _DIGEST}, file=_REGISTRY)

    result, ok, script_errors = _verdicts(zp)

    assert result.ok and ok
    assert result.errors == [] and script_errors == []
    assert _run(zp).returncode == 0


def test_an_export_that_pins_no_registry_is_unchanged(tmp_path: Path):
    zp = _with_registry(tmp_path, block=None, file=None)

    result, ok, script_errors = _verdicts(zp)

    assert result.ok and ok
    assert result.errors == [] and script_errors == []


@pytest.mark.parametrize("pinned", ["sha256:00", "", None, 7, _DIGEST.upper()],
                         ids=["wrong", "empty", "absent", "a number", "other case"])
def test_a_registry_that_does_not_match_the_pinned_digest_fails_both(
        tmp_path: Path, pinned):
    block = {} if pinned is None else {"registry_sha256": pinned}
    zp = _with_registry(tmp_path, block=block, file=_REGISTRY)

    result, ok, script_errors = _verdicts(zp)

    assert not result.ok and not ok
    assert result.errors == [DIGEST_MISMATCH]
    assert script_errors == result.errors


def test_a_registry_edited_after_signing_fails_both(tmp_path: Path):
    zp = _with_registry(tmp_path, block={"registry_sha256": _DIGEST},
                        file=_REGISTRY.replace(b"[]", b'["key-1"]'))

    result, ok, script_errors = _verdicts(zp)

    assert not result.ok and not ok
    assert result.errors == [DIGEST_MISMATCH]
    assert script_errors == result.errors


def test_a_declared_registry_with_no_file_fails_both(tmp_path: Path):
    zp = _with_registry(tmp_path, block={"registry_sha256": _DIGEST}, file=None)

    result, ok, script_errors = _verdicts(zp)

    assert not result.ok and not ok
    assert result.errors == [FILE_MISSING]
    assert script_errors == result.errors


def test_a_registry_file_with_no_declaration_fails_both(tmp_path: Path):
    zp = _with_registry(tmp_path, block=None, file=_REGISTRY)

    result, ok, script_errors = _verdicts(zp)

    assert not result.ok and not ok
    assert result.errors == [BLOCK_MISSING]
    assert script_errors == result.errors


@pytest.mark.parametrize("block", [[], "sha256:00", 7, True],
                         ids=["an array", "a string", "a number", "a boolean"])
def test_a_revocation_entry_that_is_not_an_object_fails_both(tmp_path: Path, block):
    zp = _with_registry(tmp_path, block=block, file=_REGISTRY)

    result, ok, script_errors = _verdicts(zp)

    assert not result.ok and not ok
    assert result.errors == [NOT_AN_OBJECT]
    assert script_errors == result.errors


def test_the_cli_path_reports_the_finding_and_exits_nonzero(tmp_path: Path):
    zp = _with_registry(tmp_path, block={"registry_sha256": "sha256:00"}, file=_REGISTRY)

    run = _run(zp)

    assert run.returncode == 1
    assert f"  - {DIGEST_MISMATCH}" in run.stderr
    assert "Verification: OK" not in run.stdout


# --- the exporter's own bundle, not a hand-built one ------------------------------


def _real_export(tmp_path: Path) -> Path:
    pytest.importorskip("rfc8785")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from vaara.attestation import RevocationRegistry
    from vaara.audit.export import export_signed
    from vaara.audit.signer import Ed25519Signer
    from vaara.audit.trail import AuditTrail
    from vaara.taxonomy.actions import ActionRequest, create_default_registry

    tx = create_default_registry().get("tx.transfer")
    trail = AuditTrail()
    trail.record_action_requested(ActionRequest(
        agent_id="agent-0", tool_name="send_funds", action_type=tx,
        parameters={"to": "0xabc", "amount": 1},
    ))
    out = tmp_path / "real.zip"
    export_signed(trail, out, signer=Ed25519Signer(Ed25519PrivateKey.generate()),
                  revocation=RevocationRegistry((), as_of="2026-09-03T00:00:00Z"))
    return out


def test_an_export_that_pins_a_real_registry_verifies_in_both(tmp_path: Path):
    zp = _real_export(tmp_path)
    with zipfile.ZipFile(zp) as zf:
        assert "revocation.json" in zf.namelist()

    result, ok, script_errors = _verdicts(zp)

    assert result.ok and ok, (result.errors, script_errors)


def test_a_real_export_whose_registry_file_is_swapped_fails_in_both(tmp_path: Path):
    zp = _real_export(tmp_path)
    swapped = tmp_path / "swapped.zip"
    with zipfile.ZipFile(zp) as src, zipfile.ZipFile(swapped, "w") as dst:
        for name in src.namelist():
            body = src.read(name)
            if name == "revocation.json":
                body = b'{"revoked":["someone-else"]}'
            dst.writestr(name, body)

    result, ok, script_errors = _verdicts(swapped)

    assert not result.ok and not ok
    assert result.errors == [DIGEST_MISMATCH]
    assert script_errors == result.errors
