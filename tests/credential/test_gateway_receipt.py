"""The gateway mints a signed receipt for every grant-bound decision.

Opt-in: only a gateway given a :class:`ReceiptSigner` mints. A refused call mints
too, leaving a portable proof of the non-action. A missing or malformed
credential refuses with no receipt, since there is no grant to bind a proof to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from vaara.attestation.decision import verify_decision_signature  # noqa: E402
from vaara.credential import CredentialGateway, ReceiptSigner  # noqa: E402

KEY = b"x" * 32
TOOL = "read_file"
ARGS = {"path": "/tmp/x"}


def _emitter(receipts_dir: Path) -> Any:
    from vaara.integrations._mcp_attest import build_attest_emitter

    keyfile = receipts_dir.parent / "attest.key"
    keyfile.write_bytes(KEY)
    return build_attest_emitter(
        signing_key_path=keyfile,
        receipts_dir=receipts_dir,
        upstream_commands={"default": ["echo"]},
    )


def _mint_credential(receipts_dir: Path, *, tenant: str = "t1") -> dict:
    em = _emitter(receipts_dir)
    att, counter = em.emit_attestation(
        tool_name=TOOL, arguments=ARGS, upstream_name="default", tenant_id=tenant
    )
    cred = em.emit_grant(
        attestation=att, counter=counter, tool_name=TOOL,
        upstream_name="default", tenant_id=tenant,
    )
    return cred.to_dict()


def _signer() -> ReceiptSigner:
    return ReceiptSigner(
        signing_material=KEY,
        iss="vaara-mcp-proxy",
        sub="t1/upstream",
        secret_version="k1",
        alg="HS256",
    )


def _gateway(receipts_dir: Path, *, signer: bool = True) -> CredentialGateway:
    return CredentialGateway(
        verifying_material=KEY,
        receipts_dir=receipts_dir,
        expected_tenant="t1",
        signer=_signer() if signer else None,
    )


def test_allow_mints_a_verifiable_receipt(tmp_path: Path):
    d = tmp_path / "r"
    d.mkdir()
    cred = _mint_credential(d)
    gw = _gateway(d)
    params = {"_meta": {"vaara/credential": cred}}

    verdict, auth = gw.authorize_and_receipt(params, tool_name=TOOL, arguments=ARGS)
    assert verdict.ok
    assert auth is not None
    assert auth.record.decision_derived.decision == "allow"
    assert auth.evidence["verdict"] == "allow"
    assert verify_decision_signature(auth.record, verifying_material=KEY)


def test_refusal_leaves_a_signed_proof(tmp_path: Path):
    d = tmp_path / "r"
    d.mkdir()
    cred = _mint_credential(d)
    gw = _gateway(d)
    # Mutated args break the exact-args commitment: a Phase A scope_mismatch deny.
    params = {"_meta": {"vaara/credential": cred}}

    verdict, auth = gw.authorize_and_receipt(
        params, tool_name=TOOL, arguments={"path": "/etc/shadow"}
    )
    assert not verdict.ok and verdict.reason == "scope_mismatch"
    assert auth is not None
    assert auth.record.decision_derived.decision == "block"
    assert auth.evidence["verdict"] == "deny"
    assert auth.evidence["reason"] == "scope_mismatch"
    assert verify_decision_signature(auth.record, verifying_material=KEY)


def test_no_signer_means_no_receipt(tmp_path: Path):
    d = tmp_path / "r"
    d.mkdir()
    cred = _mint_credential(d)
    gw = _gateway(d, signer=False)
    params = {"_meta": {"vaara/credential": cred}}

    verdict, auth = gw.authorize_and_receipt(params, tool_name=TOOL, arguments=ARGS)
    assert verdict.ok
    assert auth is None


def test_missing_credential_refuses_without_a_receipt(tmp_path: Path):
    d = tmp_path / "r"
    d.mkdir()
    gw = _gateway(d)

    verdict, auth = gw.authorize_and_receipt(
        {"name": TOOL}, tool_name=TOOL, arguments=ARGS
    )
    assert not verdict.ok and verdict.reason == "missing_credential"
    assert auth is None
