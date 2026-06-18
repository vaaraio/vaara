"""The gateway shim refuses calls that lack a valid, bound credential."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from vaara.credential import CredentialGateway  # noqa: E402

KEY = b"x" * 32
TOOL = "read_file"
ARGS = {"path": "/tmp/x"}


def _emitter(receipts_dir: Path) -> Any:
    from vaara.integrations._mcp_attest import build_attest_emitter

    return build_attest_emitter(
        signing_key_path=_keyfile(receipts_dir),
        receipts_dir=receipts_dir,
        upstream_commands={"default": ["echo"]},
    )


def _keyfile(d: Path) -> Path:
    p = d.parent / "attest.key"
    p.write_bytes(KEY)
    return p


def _mint_through_emitter(receipts_dir: Path, *, tenant: str = "t1") -> dict:
    """Emit a real attestation + bound grant; return the credential wire dict."""
    em = _emitter(receipts_dir)
    att, counter = em.emit_attestation(
        tool_name=TOOL, arguments=ARGS, upstream_name="default", tenant_id=tenant
    )
    cred = em.emit_grant(
        attestation=att, counter=counter, tool_name=TOOL,
        upstream_name="default", tenant_id=tenant,
    )
    return cred.to_dict()


def _gateway(receipts_dir: Path, tenant: str = "t1") -> CredentialGateway:
    return CredentialGateway(
        verifying_material=KEY, receipts_dir=receipts_dir, expected_tenant=tenant
    )


def test_no_meta_is_missing_credential(tmp_path: Path):
    d = tmp_path / "r"
    d.mkdir()
    gw = _gateway(d)
    v = gw.authorize({"name": TOOL}, tool_name=TOOL, arguments=ARGS)
    assert not v.ok and v.reason == "missing_credential"


def test_valid_minted_grant_is_ok(tmp_path: Path):
    d = tmp_path / "r"
    d.mkdir()
    cred = _mint_through_emitter(d)
    gw = _gateway(d)
    params = {"_meta": {"vaara/credential": cred}}
    v = gw.authorize(params, tool_name=TOOL, arguments=ARGS)
    assert v.ok and v.reason == "ok"


def test_mutated_args_is_scope_mismatch(tmp_path: Path):
    d = tmp_path / "r"
    d.mkdir()
    cred = _mint_through_emitter(d)
    gw = _gateway(d)
    params = {"_meta": {"vaara/credential": cred}}
    v = gw.authorize(params, tool_name=TOOL, arguments={"path": "/etc/shadow"})
    assert not v.ok and v.reason == "scope_mismatch"


def test_malformed_credential_refused(tmp_path: Path):
    d = tmp_path / "r"
    d.mkdir()
    cred = _mint_through_emitter(d)
    cred["rogue"] = 1
    gw = _gateway(d)
    params = {"_meta": {"vaara/credential": cred}}
    v = gw.authorize(params, tool_name=TOOL, arguments=ARGS)
    assert not v.ok and v.reason == "malformed"


def test_unknown_binding_when_attest_file_absent(tmp_path: Path):
    # Mint in one dir, point the gateway at an empty dir: the attestation
    # digest is not known there, so the bound grant is refused (fail-closed).
    mint_dir = tmp_path / "r"
    mint_dir.mkdir()
    empty_dir = tmp_path / "e"
    empty_dir.mkdir()
    cred = _mint_through_emitter(mint_dir)
    gw = _gateway(empty_dir)
    params = {"_meta": {"vaara/credential": cred}}
    v = gw.authorize(params, tool_name=TOOL, arguments=ARGS)
    assert not v.ok and v.reason == "binding_unknown"
