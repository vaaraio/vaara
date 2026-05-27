"""SEP-2787 wire round-trip: emit, serialise to JSON bytes, parse back, verify.

Covers the path a third-party consumer of the published v0 test
vectors actually takes:

    emit_attestation(...) -> Attestation
    -> canonical_json(env.to_dict()) -> bytes
    -> json.loads(bytes) -> dict
    -> parse_attestation(dict) -> Attestation
    -> verify_attestation(parsed, verifying_material=...) -> True

The Python in-memory round-trip tests in test_attestation_sep2787.py
verify the emit/verify pair but do not exercise the JSON boundary.
If to_dict() ever produces a shape that, when JCS-canonicalised by a
wire consumer, does not byte-match what was signed, in-memory tests
still pass but the published vectors break. These tests close that gap.
"""

from __future__ import annotations

import importlib.util
import json

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
    AttestationError,
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    canonical_json,
    emit_attestation,
    make_args_digest,
    parse_attestation,
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


def _emit(**overrides):
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


def _wire(env):
    """Serialise the envelope to canonical JSON bytes, then parse back."""
    wire_bytes = canonical_json(env.to_dict())
    parsed_dict = json.loads(wire_bytes)
    return parse_attestation(parsed_dict)


def test_wire_round_trip_hs256():
    env = _emit()
    parsed = _wire(env)
    assert verify_attestation(parsed, verifying_material=HS_SECRET) is True


def test_wire_round_trip_es256():
    priv = ec.generate_private_key(ec.SECP256R1())
    env = _emit(alg="ES256", signing_material=priv)
    parsed = _wire(env)
    assert verify_attestation(parsed, verifying_material=priv.public_key()) is True


def test_wire_round_trip_rs256():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    env = _emit(alg="RS256", signing_material=priv)
    parsed = _wire(env)
    assert verify_attestation(parsed, verifying_material=priv.public_key()) is True


def test_wire_round_trip_args_ref():
    """ArgsRef goes through the wire intact."""
    ref_args = ArgsRef(
        ref="cid://blob/sha256/" + "a" * 64,
        digest="sha256:" + "a" * 64,
    )
    env = _emit(payload_derived=_payload(args=ref_args))
    parsed = _wire(env)
    assert verify_attestation(parsed, verifying_material=HS_SECRET) is True
    binding = parsed.payload_derived.tool_calls[0]
    assert isinstance(binding.args, ArgsRef)
    assert binding.args.ref == ref_args.ref
    assert binding.args.digest == ref_args.digest


def test_wire_round_trip_args_projection():
    """ArgsProjection (hash-only-identity shape) goes through the wire intact."""
    env = _emit()
    parsed = _wire(env)
    binding = parsed.payload_derived.tool_calls[0]
    from vaara.attestation.sep2787 import ArgsProjection
    assert isinstance(binding.args, ArgsProjection)
    assert binding.args.projection_digest.startswith("sha256:")


def test_parse_rejects_missing_signature():
    env = _emit()
    d = env.to_dict()
    del d["signature"]
    with pytest.raises(AttestationError, match="signature"):
        parse_attestation(d)


def test_parse_rejects_missing_planner_declared():
    env = _emit()
    d = env.to_dict()
    del d["plannerDeclared"]
    with pytest.raises(AttestationError, match="plannerDeclared"):
        parse_attestation(d)


def test_parse_rejects_unsupported_alg():
    env = _emit()
    d = env.to_dict()
    d["alg"] = "HS512"
    with pytest.raises(AttestationError, match="unsupported alg"):
        parse_attestation(d)


def test_parse_rejects_missing_tool_calls():
    env = _emit()
    d = env.to_dict()
    del d["payloadDerived"]["toolCalls"]
    with pytest.raises(AttestationError, match="toolCalls"):
        parse_attestation(d)


def test_parse_rejects_args_without_discriminator():
    env = _emit()
    d = env.to_dict()
    d["payloadDerived"]["toolCalls"][0]["args"] = {"unknown": "field"}
    with pytest.raises(AttestationError, match="missing both"):
        parse_attestation(d)


def test_parse_rejects_args_projection_without_digest():
    env = _emit()
    d = env.to_dict()
    d["payloadDerived"]["toolCalls"][0]["args"] = {"projection": "{}"}
    with pytest.raises(AttestationError, match="projectionDigest"):
        parse_attestation(d)


def test_wire_round_trip_preserves_optional_requested_capability_absent():
    """If plannerDeclared has no requestedCapability, it stays absent through the wire."""
    planner = PlannerDeclared(intent="read-only audit query", requested_capability=None)
    env = _emit(planner_declared=planner)
    wire_bytes = canonical_json(env.to_dict())
    assert b"requestedCapability" not in wire_bytes
    parsed = _wire(env)
    assert parsed.planner_declared.requested_capability is None
    assert verify_attestation(parsed, verifying_material=HS_SECRET) is True


def test_wire_bytes_byte_identical_round_trip():
    """Re-emitting the parsed envelope produces the same wire bytes."""
    env = _emit()
    first = canonical_json(env.to_dict())
    parsed = parse_attestation(json.loads(first))
    second = canonical_json(parsed.to_dict())
    assert first == second
