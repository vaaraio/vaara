"""OVERT Phase 3 IAP + transparency-log tests."""

from __future__ import annotations

import uuid

import pytest

try:
    import cbor2  # noqa: F401
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
except ImportError:
    pytest.skip(
        "attestation extra not installed (pip install 'vaara[attestation]')",
        allow_module_level=True,
    )

from vaara.attestation.iap import (
    IAPError,
    Phase3Attestation,
    emit_phase3_attestation,
    envelope_to_canonical_cbor,
    verify_phase3_attestation,
)
from vaara.attestation.overt import (
    emit_base_envelope,
    encoder_binary_identity,
    make_request_commitment,
)
from vaara.attestation.transparency_log import (
    InProcessTransparencyLog,
    TransparencyLogError,
    verify_inclusion,
)


def _make_envelope(arbiter_key, counter=1):
    return emit_base_envelope(
        signing_key=arbiter_key,
        request_commitment=make_request_commitment(
            b"req-content", operator_key=b"k" * 32,
        ),
        encoder_binary_identity=encoder_binary_identity(
            arbiter_version="vaara/test", policy_hash=b"\x00" * 32,
        ),
        non_content_metadata={"decision": "allow"},
        monotonic_counter=counter,
        arbiter_instance_identifier=uuid.uuid4().bytes,
    )


def _raw_pub(key):
    return key.public_key().public_bytes_raw()


def test_emit_and_verify_roundtrip():
    arbiter = Ed25519PrivateKey.generate()
    notary = Ed25519PrivateKey.generate()
    log = InProcessTransparencyLog()
    env = _make_envelope(arbiter)
    att = emit_phase3_attestation(
        envelope=env, notary_signing_key=notary,
        transparency_log=log, iap_identifier="vaara-iap-test/1.0",
    )
    assert isinstance(att, Phase3Attestation)
    assert att.log_index == 0 and att.log_tree_size == 1
    assert verify_phase3_attestation(
        attestation=att,
        notary_public_key_raw=_raw_pub(notary),
        expected_log_root=log.root_hash,
        arbiter_public_key_raw=_raw_pub(arbiter),
    )


def test_rejects_arbiter_acting_as_notary():
    arbiter = Ed25519PrivateKey.generate()
    log = InProcessTransparencyLog()
    env = _make_envelope(arbiter)
    with pytest.raises(IAPError, match="structural independence"):
        emit_phase3_attestation(
            envelope=env, notary_signing_key=arbiter,
            transparency_log=log, iap_identifier="vaara-iap-test/1.0",
        )


def test_verify_rejects_wrong_notary_pubkey():
    arbiter = Ed25519PrivateKey.generate()
    notary = Ed25519PrivateKey.generate()
    other = Ed25519PrivateKey.generate()
    log = InProcessTransparencyLog()
    att = emit_phase3_attestation(
        envelope=_make_envelope(arbiter), notary_signing_key=notary,
        transparency_log=log, iap_identifier="iap-x",
    )
    assert not verify_phase3_attestation(
        attestation=att, notary_public_key_raw=_raw_pub(other),
    )


def test_verify_rejects_tampered_envelope_cbor():
    arbiter = Ed25519PrivateKey.generate()
    notary = Ed25519PrivateKey.generate()
    log = InProcessTransparencyLog()
    att = emit_phase3_attestation(
        envelope=_make_envelope(arbiter), notary_signing_key=notary,
        transparency_log=log, iap_identifier="iap-x",
    )
    bad = Phase3Attestation(
        envelope_cbor=att.envelope_cbor[:-1] + b"\xff",
        notary_signature=att.notary_signature,
        notary_key_identifier=att.notary_key_identifier,
        log_index=att.log_index,
        log_tree_size=att.log_tree_size,
        log_root_at_append=att.log_root_at_append,
        inclusion_proof_siblings=att.inclusion_proof_siblings,
        iap_identifier=att.iap_identifier,
        attestation_timestamp_ns=att.attestation_timestamp_ns,
    )
    assert not verify_phase3_attestation(
        attestation=bad, notary_public_key_raw=_raw_pub(notary),
    )


def test_inclusion_proof_round_trip_many_entries():
    log = InProcessTransparencyLog()
    arbiter = Ed25519PrivateKey.generate()
    notary = Ed25519PrivateKey.generate()
    attestations = []
    for i in range(11):
        attestations.append(emit_phase3_attestation(
            envelope=_make_envelope(arbiter, counter=i + 1),
            notary_signing_key=notary, transparency_log=log,
            iap_identifier="iap-x",
        ))
    root_after_all = log.root_hash
    for att in attestations:
        proof = log.inclusion_proof(att.log_index)
        assert verify_inclusion(
            leaf_data=att.envelope_cbor, proof=proof,
            expected_root=root_after_all,
        )


def test_log_rejects_non_bytes_leaf():
    log = InProcessTransparencyLog()
    with pytest.raises(TransparencyLogError):
        log.append("not-bytes")  # type: ignore[arg-type]


def test_log_inclusion_proof_bad_index_raises():
    log = InProcessTransparencyLog()
    log.append(b"x")
    with pytest.raises(TransparencyLogError):
        log.inclusion_proof(5)


def test_verify_rejects_inclusion_proof_against_wrong_root():
    arbiter = Ed25519PrivateKey.generate()
    notary = Ed25519PrivateKey.generate()
    log = InProcessTransparencyLog()
    att = emit_phase3_attestation(
        envelope=_make_envelope(arbiter), notary_signing_key=notary,
        transparency_log=log, iap_identifier="iap-x",
    )
    assert not verify_phase3_attestation(
        attestation=att, notary_public_key_raw=_raw_pub(notary),
        expected_log_root=b"\x00" * 32,
    )


def test_verify_rejects_inner_signature_when_arbiter_key_supplied():
    arbiter = Ed25519PrivateKey.generate()
    other = Ed25519PrivateKey.generate()
    notary = Ed25519PrivateKey.generate()
    log = InProcessTransparencyLog()
    att = emit_phase3_attestation(
        envelope=_make_envelope(arbiter), notary_signing_key=notary,
        transparency_log=log, iap_identifier="iap-x",
    )
    assert not verify_phase3_attestation(
        attestation=att, notary_public_key_raw=_raw_pub(notary),
        arbiter_public_key_raw=_raw_pub(other),
    )


def test_envelope_to_canonical_cbor_includes_signature():
    import cbor2
    arbiter = Ed25519PrivateKey.generate()
    blob = envelope_to_canonical_cbor(_make_envelope(arbiter))
    decoded = cbor2.loads(blob)
    assert set(decoded.keys()) == {
        "blinded_identifier", "request_commitment",
        "encoder_binary_identity", "non_content_metadata",
        "monotonic_counter", "nanosecond_timestamp",
        "key_identifier", "arbiter_instance_identifier",
        "signature",
    }


def test_to_dict_is_jsonable():
    import json
    arbiter = Ed25519PrivateKey.generate()
    notary = Ed25519PrivateKey.generate()
    log = InProcessTransparencyLog()
    att = emit_phase3_attestation(
        envelope=_make_envelope(arbiter), notary_signing_key=notary,
        transparency_log=log, iap_identifier="iap-x",
    )
    body = json.dumps(att.to_dict())
    assert "envelope_cbor" in body and "log_root_at_append" in body
