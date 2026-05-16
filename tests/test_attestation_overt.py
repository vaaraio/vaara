"""OVERT 1.0 Protocol Profile 1.0 Base Envelope emission tests."""

from __future__ import annotations

import pytest

try:
    import cbor2  # noqa: F401
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    from vaara.attestation.overt import (
        BaseEnvelope,
        EnvelopeError,
        canonical_cbor,
        emit_base_envelope,
        encoder_binary_identity,
        make_request_commitment,
        verify_base_envelope,
    )
except ImportError:
    pytest.skip(
        "attestation extra not installed (pip install 'vaara[attestation]')",
        allow_module_level=True,
    )


@pytest.fixture
def signing_key():
    return Ed25519PrivateKey.generate()


@pytest.fixture
def pub_raw(signing_key):
    return signing_key.public_key().public_bytes_raw()


def _arbiter_id() -> bytes:
    return b"\x00" * 16


def _emit_default(signing_key, **overrides):
    kwargs = dict(
        signing_key=signing_key,
        request_commitment=make_request_commitment(
            b"action-payload", operator_key=b"\x42" * 32,
        ),
        encoder_binary_identity=encoder_binary_identity(
            arbiter_version="vaara/0.11.0", policy_hash=b"\xaa" * 32,
        ),
        non_content_metadata={"action_class": "data.read", "decision": "allow"},
        monotonic_counter=1,
        arbiter_instance_identifier=_arbiter_id(),
    )
    kwargs.update(overrides)
    return emit_base_envelope(**kwargs)


def test_emit_produces_signed_envelope(signing_key, pub_raw):
    env = _emit_default(signing_key)
    assert isinstance(env, BaseEnvelope)
    assert len(env.signature) == 64
    assert len(env.blinded_identifier) == 32
    assert env.monotonic_counter == 1
    assert env.non_content_metadata["decision"] == "allow"
    assert verify_base_envelope(env, pub_raw) is True


def test_envelope_verification_rejects_tampered_metadata(signing_key, pub_raw):
    env = _emit_default(signing_key)
    tampered = BaseEnvelope(
        blinded_identifier=env.blinded_identifier,
        request_commitment=env.request_commitment,
        encoder_binary_identity=env.encoder_binary_identity,
        non_content_metadata={**env.non_content_metadata, "decision": "deny"},
        monotonic_counter=env.monotonic_counter,
        nanosecond_timestamp=env.nanosecond_timestamp,
        key_identifier=env.key_identifier,
        arbiter_instance_identifier=env.arbiter_instance_identifier,
        signature=env.signature,
    )
    assert verify_base_envelope(tampered, pub_raw) is False


def test_envelope_verification_rejects_wrong_key(signing_key):
    env = _emit_default(signing_key)
    wrong_pub = Ed25519PrivateKey.generate().public_key().public_bytes_raw()
    assert verify_base_envelope(env, wrong_pub) is False


def test_envelope_verification_rejects_malformed_pubkey(signing_key):
    env = _emit_default(signing_key)
    # Wrong-length raw bytes raise ValueError inside cryptography;
    # verify must surface that as a False return, not an exception.
    assert verify_base_envelope(env, b"too-short") is False


def test_canonical_cbor_rejects_ieee754_floats():
    with pytest.raises(EnvelopeError) as exc:
        canonical_cbor({"rate": 0.42})
    assert "float" in str(exc.value).lower()


def test_canonical_cbor_rejects_floats_in_nested_structure():
    with pytest.raises(EnvelopeError):
        canonical_cbor({"a": {"b": [1, 2, 3.14]}})


def test_canonical_cbor_accepts_decimal_strings_for_rates():
    blob = canonical_cbor({"rate": "0.42", "ci_lower": "0.30", "ci_upper": "0.55"})
    assert isinstance(blob, bytes)
    assert len(blob) > 0


def test_canonical_cbor_is_deterministic():
    a = canonical_cbor({"b": 2, "a": 1})
    b = canonical_cbor({"a": 1, "b": 2})
    assert a == b


def test_emit_rejects_floats_in_metadata(signing_key):
    with pytest.raises(EnvelopeError):
        _emit_default(
            signing_key,
            non_content_metadata={"risk_score": 0.42},
        )


def test_emit_rejects_bad_arbiter_id_length(signing_key):
    with pytest.raises(EnvelopeError):
        _emit_default(signing_key, arbiter_instance_identifier=b"\x00" * 8)


def test_request_commitment_is_keyed_and_deterministic():
    c1 = make_request_commitment(b"payload", operator_key=b"k" * 32)
    c2 = make_request_commitment(b"payload", operator_key=b"k" * 32)
    c3 = make_request_commitment(b"payload", operator_key=b"j" * 32)
    assert c1 == c2
    assert c1 != c3
    assert len(c1) == 32


def test_encoder_binary_identity_varies_on_inputs():
    a = encoder_binary_identity(arbiter_version="vaara/0.11.0", policy_hash=b"\x00" * 32)
    b = encoder_binary_identity(arbiter_version="vaara/0.12.0", policy_hash=b"\x00" * 32)
    c = encoder_binary_identity(arbiter_version="vaara/0.11.0", policy_hash=b"\x01" * 32)
    assert len({a, b, c}) == 3
    assert len(a) == 32


def test_envelope_to_dict_is_json_serializable(signing_key):
    import json
    env = _emit_default(signing_key)
    d = env.to_dict()
    json.dumps(d)  # raises if any non-JSON value sneaks through


def test_monotonic_counter_is_in_signed_payload(signing_key, pub_raw):
    env = _emit_default(signing_key, monotonic_counter=1)
    tampered = BaseEnvelope(
        blinded_identifier=env.blinded_identifier,
        request_commitment=env.request_commitment,
        encoder_binary_identity=env.encoder_binary_identity,
        non_content_metadata=env.non_content_metadata,
        monotonic_counter=2,  # only field changed
        nanosecond_timestamp=env.nanosecond_timestamp,
        key_identifier=env.key_identifier,
        arbiter_instance_identifier=env.arbiter_instance_identifier,
        signature=env.signature,
    )
    assert verify_base_envelope(tampered, pub_raw) is False
