"""SEP-2787 proposed-shape envelope tests."""

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

from vaara.attestation.sep2787 import (  # noqa: E402
    ArgsRef,
    Attestation,
    AttestationError,
    IssuerAsserted,
    PlannerDeclared,
    ToolCallBinding,
    canonical_json,
    emit_attestation,
    make_args_digest,
    make_args_projection,
    verify_args_commitment,
    verify_attestation,
)


HS_SECRET = b"\x42" * 32


def _planner(args=None) -> PlannerDeclared:
    args = args or make_args_digest({"path": "/archive/2024-Q3.md"})
    return PlannerDeclared(
        intent="archive obsolete report per Q4 retention policy",
        tool_calls=(
            ToolCallBinding(
                name="delete_file",
                server_fingerprint="sha256:1111111111111111111111111111111111111111111111111111111111111111",
                args=args,
            ),
        ),
        requested_capability="filesystem.delete",
    )


def _emit_attestation(**overrides) -> Attestation:
    kwargs = dict(
        planner_declared=_planner(),
        iss="issuer://test",
        sub="agent:archiver",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )
    kwargs.update(overrides)
    return emit_attestation(**kwargs)


def test_hs256_emit_and_verify_round_trip():
    env = _emit_attestation()
    assert env.alg == "HS256"
    assert env.version == 1
    assert env.signature
    assert verify_attestation(env, verifying_material=HS_SECRET) is True


def test_es256_emit_and_verify_round_trip():
    priv = ec.generate_private_key(ec.SECP256R1())
    env = _emit_attestation(alg="ES256", signing_material=priv)
    assert len(env.signature) == 128
    assert verify_attestation(env, verifying_material=priv.public_key()) is True


def test_rs256_emit_and_verify_round_trip():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    env = _emit_attestation(alg="RS256", signing_material=priv)
    assert verify_attestation(env, verifying_material=priv.public_key()) is True


def test_tampered_planner_block_fails_verification():
    env = _emit_attestation()
    tampered_planner = PlannerDeclared(
        intent="archive obsolete report per Q4 retention policy",
        tool_calls=env.planner_declared.tool_calls,
        requested_capability="filesystem.WRITE",
    )
    tampered = Attestation(
        version=env.version,
        alg=env.alg,
        planner_declared=tampered_planner,
        issuer_asserted=env.issuer_asserted,
        payload_derived=env.payload_derived,
        signature=env.signature,
    )
    assert verify_attestation(tampered, verifying_material=HS_SECRET) is False


def test_tampered_issuer_block_fails_verification():
    env = _emit_attestation()
    tampered_issuer = IssuerAsserted(
        iss=env.issuer_asserted.iss,
        sub="agent:HIJACKED",
        iat=env.issuer_asserted.iat,
        exp_seconds=env.issuer_asserted.exp_seconds,
        nonce=env.issuer_asserted.nonce,
        secret_version=env.issuer_asserted.secret_version,
        alg=env.issuer_asserted.alg,
    )
    tampered = Attestation(
        version=env.version,
        alg=env.alg,
        planner_declared=env.planner_declared,
        issuer_asserted=tampered_issuer,
        payload_derived=env.payload_derived,
        signature=env.signature,
    )
    assert verify_attestation(tampered, verifying_material=HS_SECRET) is False


def test_args_digest_round_trip_three_shapes():
    d = make_args_digest({"a": 1, "b": [1, 2, 3]})
    r = ArgsRef(ref="ipfs://Qm...", digest="sha256:" + "0" * 64)
    p = make_args_projection({"redacted_user_id": "u-001"})
    for args in (d, r, p):
        env = _emit_attestation(planner_declared=_planner(args=args))
        assert verify_attestation(env, verifying_material=HS_SECRET)
        assert env.payload_derived[0].kind == args.kind


def test_canonical_json_rejects_floats():
    with pytest.raises(AttestationError, match="float"):
        canonical_json({"rate": 0.5})


def test_canonical_json_sorts_keys():
    out_a = canonical_json({"b": 1, "a": 2})
    out_b = canonical_json({"a": 2, "b": 1})
    assert out_a == out_b
    assert out_a == b'{"a":2,"b":1}'


def test_canonical_json_unicode_handling():
    out = canonical_json({"name": "Sirkkavaara"})
    assert b'"name":"Sirkkavaara"' in out


def test_make_args_digest_is_deterministic():
    one = make_args_digest({"x": 1, "y": [1, 2]})
    two = make_args_digest({"y": [1, 2], "x": 1})
    assert one.digest == two.digest


def test_ttl_expired_returns_false():
    env = _emit_attestation(exp_seconds=60)
    far_future = 9_999_999_999.0
    assert verify_attestation(
        env, verifying_material=HS_SECRET, now=far_future,
    ) is False


def test_ttl_clock_skew_tolerance_window():
    iat = "2026-05-26T12:00:00Z"
    iat_epoch = datetime(
        2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc,
    ).timestamp()
    env = _emit_attestation(exp_seconds=60, iat=iat)
    # default clock_skew_seconds=30: iat + 60 + 15 still inside window
    assert verify_attestation(
        env, verifying_material=HS_SECRET, now=iat_epoch + 60 + 15,
    ) is True
    # iat + 60 + 31 is past the default skew window
    assert verify_attestation(
        env, verifying_material=HS_SECRET, now=iat_epoch + 60 + 31,
    ) is False


def test_emit_rejects_empty_intent():
    bad = PlannerDeclared(
        intent="   ",
        tool_calls=_planner().tool_calls,
    )
    with pytest.raises(AttestationError, match="intent"):
        _emit_attestation(planner_declared=bad)


def test_emit_rejects_unsupported_alg():
    with pytest.raises(AttestationError, match="unsupported alg"):
        _emit_attestation(alg="HS512")


def test_hs256_rejects_non_bytes_secret():
    with pytest.raises(AttestationError, match="bytes shared_secret"):
        _emit_attestation(signing_material="not-bytes")


def test_to_dict_round_trip_shape():
    env = _emit_attestation()
    d = env.to_dict()
    assert set(d) == {
        "version", "alg", "planner_declared",
        "issuer_asserted", "payload_derived", "signature",
    }
    assert d["planner_declared"]["tool_calls"][0]["args"]["kind"] == "digest"
    assert d["payload_derived"][0]["kind"] == "digest"


def test_cross_alg_verification_fails():
    env = _emit_attestation()
    priv = ec.generate_private_key(ec.SECP256R1())
    assert verify_attestation(env, verifying_material=priv.public_key()) is False


# --- Step 5: argument commitment verification ---


def test_args_commitment_digest_matching_runtime_args_ok():
    runtime = {"path": "/archive/2024-Q3.md"}
    args = make_args_digest(runtime)
    result = verify_args_commitment(args, runtime_arguments=runtime)
    assert result.ok is True
    assert result.reason is None


def test_args_commitment_digest_mismatch_rejects():
    args = make_args_digest({"path": "/archive/2024-Q3.md"})
    result = verify_args_commitment(
        args, runtime_arguments={"path": "/etc/passwd"},
    )
    assert result.ok is False
    assert result.reason == "args_commitment_mismatch"


def test_args_commitment_digest_key_reorder_still_matches():
    args = make_args_digest({"a": 1, "b": [1, 2, 3]})
    result = verify_args_commitment(
        args, runtime_arguments={"b": [1, 2, 3], "a": 1},
    )
    assert result.ok is True


def test_args_commitment_ref_no_resolver_rejects():
    args = ArgsRef(ref="ipfs://Qm...", digest="sha256:" + "0" * 64)
    result = verify_args_commitment(args, runtime_arguments={"x": 1})
    assert result.ok is False
    assert result.reason == "args_commitment_mismatch"


def test_args_commitment_ref_resolver_content_matches():
    runtime = {"path": "/archive/2024-Q3.md"}
    canonical = canonical_json(runtime)
    import hashlib as _h
    digest = "sha256:" + _h.sha256(canonical).hexdigest()
    args = ArgsRef(ref="memory://q3", digest=digest)
    result = verify_args_commitment(
        args,
        runtime_arguments=runtime,
        ref_resolver=lambda _ref: canonical,
    )
    assert result.ok is True


def test_args_commitment_ref_digest_mismatch_rejects():
    runtime = {"path": "/archive/2024-Q3.md"}
    args = ArgsRef(ref="memory://q3", digest="sha256:" + "0" * 64)
    result = verify_args_commitment(
        args,
        runtime_arguments=runtime,
        ref_resolver=lambda _ref: canonical_json(runtime),
    )
    assert result.ok is False
    assert result.reason == "args_commitment_mismatch"


def test_args_commitment_ref_content_does_not_match_runtime_rejects():
    referenced = {"path": "/archive/2024-Q3.md"}
    runtime = {"path": "/etc/passwd"}
    canonical = canonical_json(referenced)
    import hashlib as _h
    digest = "sha256:" + _h.sha256(canonical).hexdigest()
    args = ArgsRef(ref="memory://other", digest=digest)
    result = verify_args_commitment(
        args,
        runtime_arguments=runtime,
        ref_resolver=lambda _ref: canonical,
    )
    assert result.ok is False
    assert result.reason == "args_commitment_mismatch"


def test_args_commitment_ref_resolver_raising_rejects():
    args = ArgsRef(ref="memory://oops", digest="sha256:" + "0" * 64)
    def _broken(_ref):
        raise RuntimeError("offline")
    result = verify_args_commitment(
        args, runtime_arguments={"x": 1}, ref_resolver=_broken,
    )
    assert result.ok is False
    assert result.reason == "args_commitment_mismatch"


def test_args_commitment_identity_projection_marked_match():
    runtime = {"path": "/archive/2024-Q3.md"}
    args = make_args_projection(runtime)
    result = verify_args_commitment(args, runtime_arguments=runtime)
    assert result.ok is True
    assert result.projection_match is True


def test_args_commitment_redacted_projection_ok_but_not_identity():
    redacted = {"redacted_user_id": "u-001"}
    args = make_args_projection(redacted)
    result = verify_args_commitment(
        args, runtime_arguments={"path": "/archive/2024-Q3.md", "user_id": "u-001"},
    )
    assert result.ok is True
    assert result.projection_match is False


def test_args_commitment_projection_with_tampered_digest_rejects():
    from vaara.attestation.sep2787 import ArgsProjection
    args = ArgsProjection(
        projection={"redacted_user_id": "u-001"},
        projection_digest="sha256:" + "0" * 64,
    )
    result = verify_args_commitment(args, runtime_arguments={"x": 1})
    assert result.ok is False
    assert result.reason == "args_commitment_mismatch"
