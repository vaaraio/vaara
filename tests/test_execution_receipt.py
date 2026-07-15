"""Execution-receipt round-trip, back-link, and result-commitment tests."""

from __future__ import annotations

import dataclasses
import importlib.util

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives.asymmetric import ec, rsa  # noqa: E402

from vaara.attestation.receipt import (  # noqa: E402
    OutcomeDerived,
    emit_receipt,
    make_back_link,
    make_result_digest,
    parse_receipt,
    verify_back_link,
    verify_receipt_signature,
)
from vaara.attestation.tool_call_attestation import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
    verify_args_commitment,
)

HS_SECRET = b"\x42" * 32
RESULT = {"deleted": True, "path": "/archive/2024-Q3.md"}


def _attestation():
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="delete_file",
        server_fingerprint="sha256:" + "1" * 64,
        args=make_args_digest({"path": "/archive/2024-Q3.md"}),
    ),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="archive obsolete report"),
        payload_derived=payload,
        iss="issuer://test",
        sub="agent:archiver",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )


def _outcome(status="executed", commit_result=True):
    commitment = make_result_digest(RESULT) if commit_result else None
    return OutcomeDerived(
        status=status,
        completed_at="2026-05-29T10:00:00Z",
        result_commitment=commitment,
    )


def _emit(att, **overrides):
    kwargs = dict(
        back_link=make_back_link(att),
        outcome_derived=_outcome(),
        iss="issuer://test",
        sub="agent:archiver",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )
    kwargs.update(overrides)
    return emit_receipt(**kwargs)


def test_hs256_round_trip():
    r = _emit(_attestation())
    assert r.alg == "HS256"
    assert r.version == 1
    assert r.signature
    assert verify_receipt_signature(r, verifying_material=HS_SECRET) is True


def test_es256_round_trip():
    priv = ec.generate_private_key(ec.SECP256R1())
    r = _emit(_attestation(), alg="ES256", signing_material=priv)
    assert len(r.signature) == 128
    assert verify_receipt_signature(r, verifying_material=priv.public_key()) is True


def test_rs256_round_trip():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    r = _emit(_attestation(), alg="RS256", signing_material=priv)
    assert verify_receipt_signature(r, verifying_material=priv.public_key()) is True


def test_wrong_secret_fails_signature():
    r = _emit(_attestation())
    assert verify_receipt_signature(r, verifying_material=b"\x00" * 32) is False


def test_tampered_outcome_fails_signature():
    att = _attestation()
    r = _emit(att)
    tampered = dataclasses.replace(r, outcome_derived=_outcome(status="refused"))
    assert verify_receipt_signature(tampered, verifying_material=HS_SECRET) is False


def test_back_link_valid():
    att = _attestation()
    r = _emit(att)
    assert verify_back_link(r, attestation=att).ok is True


def test_back_link_rejects_other_attestation():
    att = _attestation()
    r = _emit(att)
    other = _attestation()  # fresh nonce, different signature
    res = verify_back_link(r, attestation=other)
    assert res.ok is False
    assert res.reason == "back_link_mismatch"


def test_back_link_tampered_digest():
    att = _attestation()
    r = _emit(att)
    bad = dataclasses.replace(
        r.back_link, attestation_digest="sha256:" + "0" * 64
    )
    r = dataclasses.replace(r, back_link=bad)
    assert verify_back_link(r, attestation=att).ok is False


def test_back_link_wrong_nonce():
    att = _attestation()
    r = _emit(att)
    bad = dataclasses.replace(r.back_link, attestation_nonce="not-the-nonce")
    r = dataclasses.replace(r, back_link=bad)
    assert verify_back_link(r, attestation=att).ok is False


def test_result_commitment_binds_runtime_result():
    r = _emit(_attestation())
    res = verify_args_commitment(
        r.outcome_derived.result_commitment, runtime_arguments=RESULT
    )
    assert res.ok is True


def test_result_commitment_rejects_other_result():
    r = _emit(_attestation())
    res = verify_args_commitment(
        r.outcome_derived.result_commitment,
        runtime_arguments={"deleted": False},
    )
    assert res.ok is False
    assert res.reason == "args_commitment_mismatch"


def test_refused_has_no_result_commitment():
    att = _attestation()
    r = _emit(att, outcome_derived=_outcome(status="refused", commit_result=False))
    assert r.outcome_derived.result_commitment is None
    assert verify_receipt_signature(r, verifying_material=HS_SECRET) is True
    assert "resultCommitment" not in r.to_dict()["outcomeDerived"]


def test_wire_round_trip():
    att = _attestation()
    r = _emit(att)
    reparsed = parse_receipt(r.to_dict())
    assert reparsed == r
    assert verify_receipt_signature(reparsed, verifying_material=HS_SECRET) is True
    assert verify_back_link(reparsed, attestation=att).ok is True


def test_parse_rejects_invalid_status():
    att = _attestation()
    d = _emit(att).to_dict()
    d["outcomeDerived"]["status"] = "exploded"
    with pytest.raises(Exception):
        parse_receipt(d)


def test_emit_rejects_bad_digest_prefix():
    att = _attestation()
    bl = dataclasses.replace(make_back_link(att), attestation_digest="deadbeef")
    with pytest.raises(Exception):
        _emit(att, back_link=bl)
