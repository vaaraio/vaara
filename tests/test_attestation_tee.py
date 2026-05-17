"""Hardware TEE attestation (AMD SEV-SNP) tests."""

from __future__ import annotations

import pytest

try:
    import cbor2  # noqa: F401
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives import serialization

    from vaara.attestation.overt import (
        emit_base_envelope,
        encoder_binary_identity,
        make_request_commitment,
    )
    from vaara.attestation.tee import (
        MockSEVSNPAttester,
        SEVSNPHostAttester,
        SEV_SNP_REPORT_DATA_SIZE,
        SEV_SNP_REPORT_SIZE,
        SIGNATURE_ALGO_ECDSA_P384_SHA384,
        TEEAttestationError,
        bind_overt_envelope_to_report_data,
        parse_sev_snp_report,
        verify_envelope_binding,
        verify_sev_snp_report_signature,
    )
except ImportError:
    pytest.skip(
        "attestation extra not installed (pip install 'vaara[attestation]')",
        allow_module_level=True,
    )


def _emit_overt(seed: bytes = b"action-payload", counter: int = 1):
    signing_key = Ed25519PrivateKey.generate()
    return emit_base_envelope(
        signing_key=signing_key,
        request_commitment=make_request_commitment(
            seed, operator_key=b"\x42" * 32,
        ),
        encoder_binary_identity=encoder_binary_identity(
            arbiter_version="vaara/0.18.0", policy_hash=b"\xaa" * 32,
        ),
        non_content_metadata={"action_class": "data.read", "decision": "allow"},
        monotonic_counter=counter,
        arbiter_instance_identifier=b"\x00" * 16,
    )


@pytest.fixture
def overt_envelope():
    return _emit_overt()


@pytest.fixture
def vcek_key():
    return ec.generate_private_key(ec.SECP384R1())


@pytest.fixture
def vcek_pem(vcek_key):
    return vcek_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def test_bind_overt_envelope_to_report_data_is_64_bytes(overt_envelope):
    assert len(bind_overt_envelope_to_report_data(overt_envelope)) == SEV_SNP_REPORT_DATA_SIZE


def test_bind_is_deterministic(overt_envelope):
    a = bind_overt_envelope_to_report_data(overt_envelope)
    b = bind_overt_envelope_to_report_data(overt_envelope)
    assert a == b


def test_bind_differs_across_envelopes(overt_envelope):
    other = _emit_overt(seed=b"other-payload", counter=2)
    assert bind_overt_envelope_to_report_data(overt_envelope) != \
        bind_overt_envelope_to_report_data(other)


def test_mock_attester_emits_well_formed_report(vcek_key, overt_envelope):
    attester = MockSEVSNPAttester(vcek_key)
    report_data = bind_overt_envelope_to_report_data(overt_envelope)
    blob = attester.emit(report_data)
    assert len(blob) == SEV_SNP_REPORT_SIZE


def test_parse_round_trip(vcek_key, overt_envelope):
    attester = MockSEVSNPAttester(vcek_key, measurement=b"\xCC" * 48)
    report_data = bind_overt_envelope_to_report_data(overt_envelope)
    report = parse_sev_snp_report(attester.emit(report_data))
    assert report.report_data == report_data
    assert report.measurement == b"\xCC" * 48
    assert report.signature_algo == SIGNATURE_ALGO_ECDSA_P384_SHA384


def test_parse_rejects_wrong_size():
    with pytest.raises(TEEAttestationError, match="must be 1184 bytes"):
        parse_sev_snp_report(b"\x00" * 100)


def test_verify_signature_on_mock_report(vcek_key, vcek_pem, overt_envelope):
    attester = MockSEVSNPAttester(vcek_key)
    blob = attester.emit(bind_overt_envelope_to_report_data(overt_envelope))
    report = parse_sev_snp_report(blob)
    assert verify_sev_snp_report_signature(report, vcek_pem) is True


def test_verify_signature_fails_on_tamper(vcek_key, vcek_pem, overt_envelope):
    attester = MockSEVSNPAttester(vcek_key)
    blob = bytearray(
        attester.emit(bind_overt_envelope_to_report_data(overt_envelope))
    )
    blob[0x100] ^= 0x01
    report = parse_sev_snp_report(bytes(blob))
    assert verify_sev_snp_report_signature(report, vcek_pem) is False


def test_verify_signature_fails_on_wrong_vcek(vcek_key, overt_envelope):
    other_pem = ec.generate_private_key(ec.SECP384R1()).public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    attester = MockSEVSNPAttester(vcek_key)
    blob = attester.emit(bind_overt_envelope_to_report_data(overt_envelope))
    report = parse_sev_snp_report(blob)
    assert verify_sev_snp_report_signature(report, other_pem) is False


def test_verify_envelope_binding(vcek_key, overt_envelope):
    attester = MockSEVSNPAttester(vcek_key)
    report = parse_sev_snp_report(
        attester.emit(bind_overt_envelope_to_report_data(overt_envelope))
    )
    assert verify_envelope_binding(report, overt_envelope) is True


def test_verify_envelope_binding_fails_on_wrong_envelope(vcek_key, overt_envelope):
    attester = MockSEVSNPAttester(vcek_key)
    other = _emit_overt(seed=b"unrelated", counter=99)
    report = parse_sev_snp_report(
        attester.emit(bind_overt_envelope_to_report_data(overt_envelope))
    )
    assert verify_envelope_binding(report, other) is False


def test_mock_attester_rejects_wrong_report_data_length(vcek_key):
    attester = MockSEVSNPAttester(vcek_key)
    with pytest.raises(TEEAttestationError, match="exactly 64 bytes"):
        attester.emit(b"\x00" * 32)


def test_mock_attester_rejects_non_ec_key():
    with pytest.raises(TEEAttestationError, match="EC private key"):
        MockSEVSNPAttester(Ed25519PrivateKey.generate())


def test_mock_attester_rejects_wrong_curve():
    with pytest.raises(TEEAttestationError, match="secp384r1"):
        MockSEVSNPAttester(ec.generate_private_key(ec.SECP256R1()))


def test_host_attester_raises_clearly_on_non_snp_host():
    attester = SEVSNPHostAttester(device="/dev/does-not-exist")
    with pytest.raises(TEEAttestationError, match="not present"):
        attester.emit(b"\x00" * 64)


def test_verify_rejects_unsupported_algo(vcek_key, vcek_pem, overt_envelope):
    attester = MockSEVSNPAttester(vcek_key)
    blob = bytearray(
        attester.emit(bind_overt_envelope_to_report_data(overt_envelope))
    )
    blob[0x034:0x038] = b"\x00\x00\x00\x00"
    report = parse_sev_snp_report(bytes(blob))
    with pytest.raises(TEEAttestationError, match="Unsupported"):
        verify_sev_snp_report_signature(report, vcek_pem)
