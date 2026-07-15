"""SEP-2787 v2 envelope round-trip, tampering, and signing tests."""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives.asymmetric import ec, rsa  # noqa: E402

from vaara.attestation.tool_call_attestation import (  # noqa: E402
    ArgsRef,
    Attestation,
    AttestationError,
    IssuerAsserted,
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    canonical_json,
    emit_attestation,
    make_args_digest,
    make_args_projection,
    verify_attestation,
)

HS_SECRET = b"\x42" * 32


def _planner() -> PlannerDeclared:
    return PlannerDeclared(
        intent="archive obsolete report per Q4 retention policy",
        requested_capability="filesystem.delete",
    )


def _payload(args=None) -> PayloadDerived:
    args = args or make_args_digest({"path": "/archive/2024-Q3.md"})
    return PayloadDerived(tool_calls=(ToolCallBinding(
        name="delete_file",
        server_fingerprint="sha256:" + "1" * 64,
        args=args,
    ),))


def _emit(**overrides) -> Attestation:
    kwargs = dict(
        planner_declared=_planner(),
        payload_derived=_payload(),
        iss="issuer://test",
        sub="agent:archiver",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )
    kwargs.update(overrides)
    return emit_attestation(**kwargs)


def test_hs256_round_trip():
    env = _emit()
    assert env.alg == "HS256"
    assert env.version == 1
    assert env.signature
    assert verify_attestation(env, verifying_material=HS_SECRET) is True


def test_es256_round_trip():
    priv = ec.generate_private_key(ec.SECP256R1())
    env = _emit(alg="ES256", signing_material=priv)
    assert len(env.signature) == 128
    assert verify_attestation(env, verifying_material=priv.public_key()) is True


def test_rs256_round_trip():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    env = _emit(alg="RS256", signing_material=priv)
    assert verify_attestation(env, verifying_material=priv.public_key()) is True


def test_tampered_planner_fails():
    env = _emit()
    tampered_planner = PlannerDeclared(
        intent=env.planner_declared.intent,
        requested_capability="filesystem.WRITE",
    )
    tampered = Attestation(
        version=env.version, alg=env.alg,
        planner_declared=tampered_planner,
        issuer_asserted=env.issuer_asserted,
        payload_derived=env.payload_derived,
        signature=env.signature,
    )
    assert verify_attestation(tampered, verifying_material=HS_SECRET) is False


def test_tampered_issuer_fails():
    env = _emit()
    bad = IssuerAsserted(
        iss=env.issuer_asserted.iss, sub="agent:HIJACKED",
        iat=env.issuer_asserted.iat,
        exp_seconds=env.issuer_asserted.exp_seconds,
        nonce=env.issuer_asserted.nonce,
        secret_version=env.issuer_asserted.secret_version,
        alg=env.issuer_asserted.alg,
    )
    tampered = Attestation(
        version=env.version, alg=env.alg,
        planner_declared=env.planner_declared,
        issuer_asserted=bad,
        payload_derived=env.payload_derived,
        signature=env.signature,
    )
    assert verify_attestation(tampered, verifying_material=HS_SECRET) is False


def test_tampered_payload_fails():
    env = _emit()
    other = make_args_digest({"path": "/etc/passwd"})
    bad = PayloadDerived(tool_calls=(ToolCallBinding(
        name=env.payload_derived.tool_calls[0].name,
        server_fingerprint=env.payload_derived.tool_calls[0].server_fingerprint,
        args=other,
    ),))
    tampered = Attestation(
        version=env.version, alg=env.alg,
        planner_declared=env.planner_declared,
        issuer_asserted=env.issuer_asserted,
        payload_derived=bad,
        signature=env.signature,
    )
    assert verify_attestation(tampered, verifying_material=HS_SECRET) is False


def test_args_round_trip_two_shapes():
    hash_only = make_args_digest({"a": 1, "b": [1, 2, 3]})
    ref = ArgsRef(ref="ipfs://Qm...", digest="sha256:" + "0" * 64)
    projection = make_args_projection({"redacted_user_id": "u-001"})
    for args in (hash_only, ref, projection):
        env = _emit(payload_derived=_payload(args=args))
        assert verify_attestation(env, verifying_material=HS_SECRET)


def test_canonical_json_rejects_floats():
    with pytest.raises(AttestationError, match="float"):
        canonical_json({"rate": 0.5})


def test_canonical_json_sorts_keys():
    out_a = canonical_json({"b": 1, "a": 2})
    out_b = canonical_json({"a": 2, "b": 1})
    assert out_a == out_b == b'{"a":2,"b":1}'


def test_canonical_json_unicode_handling():
    assert b'"name":"Sirkkavaara"' in canonical_json({"name": "Sirkkavaara"})


def test_ttl_expired_returns_false():
    env = _emit(exp_seconds=60)
    assert verify_attestation(
        env, verifying_material=HS_SECRET, now=9_999_999_999.0,
    ) is False


def test_ttl_clock_skew_tolerance():
    iat = "2026-05-26T12:00:00Z"
    iat_epoch = datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    env = _emit(exp_seconds=60, iat=iat)
    assert verify_attestation(
        env, verifying_material=HS_SECRET, now=iat_epoch + 60 + 15,
    ) is True
    assert verify_attestation(
        env, verifying_material=HS_SECRET, now=iat_epoch + 60 + 31,
    ) is False


def test_future_dated_iat_rejected():
    # An attestation stamped in the future must not verify. Without the
    # lower bound, a forged or clock-wrong issuer could set iat far ahead
    # to keep the TTL window (iat + exp + skew) live indefinitely.
    iat = "2026-05-26T12:00:00Z"
    iat_epoch = datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    env = _emit(exp_seconds=60, iat=iat)
    # Verifier's clock is well before issuance: rejected.
    assert verify_attestation(
        env, verifying_material=HS_SECRET, now=iat_epoch - 3600,
    ) is False


def test_future_dated_iat_within_skew_accepted():
    # Forward drift up to clock_skew_seconds is tolerated symmetrically with
    # the trailing-edge skew, so a slightly-fast issuer clock still verifies.
    iat = "2026-05-26T12:00:00Z"
    iat_epoch = datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    env = _emit(exp_seconds=60, iat=iat)
    assert verify_attestation(
        env, verifying_material=HS_SECRET, now=iat_epoch - 15, clock_skew_seconds=30,
    ) is True
    assert verify_attestation(
        env, verifying_material=HS_SECRET, now=iat_epoch - 31, clock_skew_seconds=30,
    ) is False


def test_cross_alg_verification_fails():
    env = _emit()
    priv = ec.generate_private_key(ec.SECP256R1())
    assert verify_attestation(env, verifying_material=priv.public_key()) is False
