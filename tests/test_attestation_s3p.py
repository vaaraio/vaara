"""Tests for vaara.attestation.s3p — OVERT 1.0 S3P emitter."""

from __future__ import annotations

import os
import uuid

import pytest

try:
    import cbor2  # noqa: F401
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    from vaara.attestation.s3p import (
        ConformalExtension,
        S3PAttestation,
        S3PError,
        clopper_pearson_ci,
        emit_s3p_attestation,
        make_epoch_nonce_commitment,
        regularized_incomplete_beta,
        verify_s3p_attestation,
    )
except ImportError:
    pytest.skip(
        "attestation extra not installed (pip install 'vaara[attestation]')",
        allow_module_level=True,
    )


# ── Clopper-Pearson sanity ────────────────────────────────────────────────

def test_regularized_incomplete_beta_endpoints():
    assert regularized_incomplete_beta(0.0, 2.0, 5.0) == 0.0
    assert regularized_incomplete_beta(1.0, 2.0, 5.0) == 1.0


def test_regularized_incomplete_beta_uniform_case():
    # Beta(1, 1) is uniform, so I_x(1, 1) = x.
    for x in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert regularized_incomplete_beta(x, 1.0, 1.0) == pytest.approx(
            x, abs=1e-12,
        )


def test_clopper_pearson_zero_violations():
    lo, hi = clopper_pearson_ci(0, 20, 0.95)
    assert lo == 0.0
    # k=0, n=20, 95% CI upper = 1 - 0.025^(1/20).
    expected_upper = 1.0 - 0.025 ** (1.0 / 20.0)
    assert hi == pytest.approx(expected_upper, abs=1e-6)


def test_clopper_pearson_all_violations():
    lo, hi = clopper_pearson_ci(20, 20, 0.95)
    assert hi == 1.0
    expected_lower = 0.025 ** (1.0 / 20.0)
    assert lo == pytest.approx(expected_lower, abs=1e-6)


def test_clopper_pearson_known_value():
    # 2/20, 95% — R binom.test gives ~[0.01234, 0.31698].
    lo, hi = clopper_pearson_ci(2, 20, 0.95)
    assert lo == pytest.approx(0.01234, abs=1e-3)
    assert hi == pytest.approx(0.31698, abs=1e-3)


def test_clopper_pearson_covers_observed_rate():
    for k, n in [(1, 10), (5, 20), (50, 100), (123, 500)]:
        lo, hi = clopper_pearson_ci(k, n, 0.95)
        assert lo <= k / n <= hi


def test_clopper_pearson_invalid_inputs():
    with pytest.raises(S3PError):
        clopper_pearson_ci(0, 0, 0.95)
    with pytest.raises(S3PError):
        clopper_pearson_ci(-1, 10, 0.95)
    with pytest.raises(S3PError):
        clopper_pearson_ci(11, 10, 0.95)
    with pytest.raises(S3PError):
        clopper_pearson_ci(5, 10, 0.0)
    with pytest.raises(S3PError):
        clopper_pearson_ci(5, 10, 1.0)


# ── Epoch nonce commitment ────────────────────────────────────────────────

def test_epoch_nonce_commitment_binds_epoch():
    nonce = os.urandom(32)
    key = os.urandom(32)
    c1 = make_epoch_nonce_commitment(nonce, epoch=1, operator_key=key)
    c2 = make_epoch_nonce_commitment(nonce, epoch=2, operator_key=key)
    assert c1 != c2


def test_epoch_nonce_commitment_short_nonce_rejected():
    with pytest.raises(S3PError):
        make_epoch_nonce_commitment(b"too-short", epoch=1, operator_key=b"k")


# ── S3P attestation emission and verification ────────────────────────────

def _make_arbiter():
    sk = Ed25519PrivateKey.generate()
    pub_raw = sk.public_key().public_bytes_raw()
    arbiter_id = uuid.uuid4().bytes
    operator_key = os.urandom(32)
    nonce = os.urandom(32)
    commitment = make_epoch_nonce_commitment(
        nonce, epoch=42, operator_key=operator_key,
    )
    return sk, pub_raw, arbiter_id, commitment


def test_emit_and_verify_roundtrip():
    sk, pub_raw, arbiter_id, commitment = _make_arbiter()
    att = emit_s3p_attestation(
        signing_key=sk,
        epoch=42,
        violation_type="tool.unauthorized",
        n_total=10000,
        n_sampled=200,
        n_violations=3,
        sampling_rate=0.02,
        sampling_threshold=0.05,
        epoch_nonce_commitment=commitment,
        arbiter_instance_identifier=arbiter_id,
    )
    assert verify_s3p_attestation(att, pub_raw) is True


def test_status_compliant_when_ci_below_threshold():
    sk, _, arbiter_id, commitment = _make_arbiter()
    att = emit_s3p_attestation(
        signing_key=sk,
        epoch=1,
        violation_type="tool.unauthorized",
        n_total=10000,
        n_sampled=500,
        n_violations=2,
        sampling_rate=0.05,
        sampling_threshold=0.10,
        epoch_nonce_commitment=commitment,
        arbiter_instance_identifier=arbiter_id,
    )
    assert att.status == "compliant"


def test_status_threshold_exceeded_when_ci_lower_above_threshold():
    sk, _, arbiter_id, commitment = _make_arbiter()
    att = emit_s3p_attestation(
        signing_key=sk,
        epoch=1,
        violation_type="tool.unauthorized",
        n_total=200,
        n_sampled=100,
        n_violations=50,
        sampling_rate=0.5,
        sampling_threshold=0.10,
        epoch_nonce_commitment=commitment,
        arbiter_instance_identifier=arbiter_id,
    )
    assert att.status == "threshold_exceeded"


def test_status_insufficient_sample_when_n_sampled_zero():
    sk, _, arbiter_id, commitment = _make_arbiter()
    att = emit_s3p_attestation(
        signing_key=sk,
        epoch=1,
        violation_type="tool.unauthorized",
        n_total=100,
        n_sampled=0,
        n_violations=0,
        sampling_rate=0.0,
        sampling_threshold=0.05,
        epoch_nonce_commitment=commitment,
        arbiter_instance_identifier=arbiter_id,
    )
    assert att.status == "insufficient_sample"


def test_rates_serialised_as_decimal_strings():
    sk, _, arbiter_id, commitment = _make_arbiter()
    att = emit_s3p_attestation(
        signing_key=sk,
        epoch=1,
        violation_type="tool.unauthorized",
        n_total=1000,
        n_sampled=100,
        n_violations=5,
        sampling_rate=0.1,
        sampling_threshold=0.05,
        epoch_nonce_commitment=commitment,
        arbiter_instance_identifier=arbiter_id,
    )
    for field_value in (
        att.sampling_rate,
        att.observed_rate,
        att.confidence_level,
        att.ci_lower,
        att.ci_upper,
        att.sampling_threshold,
    ):
        assert isinstance(field_value, str)


def test_conformal_extension_attaches():
    sk, pub_raw, arbiter_id, commitment = _make_arbiter()
    ext = ConformalExtension(
        conformal_alpha="0.10",
        mean_upper_bound="0.34",
        fraction_over_threshold="0.07",
        n_calibration_points=512,
    )
    att = emit_s3p_attestation(
        signing_key=sk,
        epoch=1,
        violation_type="tool.unauthorized",
        n_total=10000,
        n_sampled=200,
        n_violations=3,
        sampling_rate=0.02,
        sampling_threshold=0.05,
        epoch_nonce_commitment=commitment,
        arbiter_instance_identifier=arbiter_id,
        conformal_extension=ext,
    )
    assert verify_s3p_attestation(att, pub_raw) is True
    payload = att.extension["vaara_conformal_aggregate"]
    assert payload["conformal_alpha"] == "0.10"
    assert payload["mean_upper_bound"] == "0.34"
    assert payload["n_calibration_points"] == 512


def test_tampered_field_rejected():
    sk, pub_raw, arbiter_id, commitment = _make_arbiter()
    att = emit_s3p_attestation(
        signing_key=sk,
        epoch=1,
        violation_type="tool.unauthorized",
        n_total=1000,
        n_sampled=100,
        n_violations=5,
        sampling_rate=0.1,
        sampling_threshold=0.05,
        epoch_nonce_commitment=commitment,
        arbiter_instance_identifier=arbiter_id,
    )
    tampered = S3PAttestation(**{**att.__dict__, "n_violations": 0})
    assert verify_s3p_attestation(tampered, pub_raw) is False


def test_wrong_public_key_rejected():
    sk, _, arbiter_id, commitment = _make_arbiter()
    other_pub_raw = (
        Ed25519PrivateKey.generate().public_key().public_bytes_raw()
    )
    att = emit_s3p_attestation(
        signing_key=sk,
        epoch=1,
        violation_type="tool.unauthorized",
        n_total=1000,
        n_sampled=100,
        n_violations=5,
        sampling_rate=0.1,
        sampling_threshold=0.05,
        epoch_nonce_commitment=commitment,
        arbiter_instance_identifier=arbiter_id,
    )
    assert verify_s3p_attestation(att, other_pub_raw) is False


def test_sampling_rate_out_of_range_rejected():
    sk, _, arbiter_id, commitment = _make_arbiter()
    with pytest.raises(S3PError):
        emit_s3p_attestation(
            signing_key=sk,
            epoch=1,
            violation_type="x",
            n_total=100,
            n_sampled=10,
            n_violations=0,
            sampling_rate=1.5,
            sampling_threshold=0.05,
            epoch_nonce_commitment=commitment,
            arbiter_instance_identifier=arbiter_id,
        )
    with pytest.raises(S3PError):
        emit_s3p_attestation(
            signing_key=sk,
            epoch=1,
            violation_type="x",
            n_total=100,
            n_sampled=10,
            n_violations=0,
            sampling_rate=0.1,
            sampling_threshold=-0.01,
            epoch_nonce_commitment=commitment,
            arbiter_instance_identifier=arbiter_id,
        )


def test_malformed_pubkey_returns_false():
    sk, _, arbiter_id, commitment = _make_arbiter()
    att = emit_s3p_attestation(
        signing_key=sk,
        epoch=1,
        violation_type="x",
        n_total=100,
        n_sampled=10,
        n_violations=1,
        sampling_rate=0.1,
        sampling_threshold=0.5,
        epoch_nonce_commitment=commitment,
        arbiter_instance_identifier=arbiter_id,
    )
    # Wrong-length raw bytes raise ValueError inside cryptography;
    # verify must surface that as a False return, not an exception.
    assert verify_s3p_attestation(att, b"too-short") is False


def test_invalid_counts_rejected():
    sk, _, arbiter_id, commitment = _make_arbiter()
    with pytest.raises(S3PError):
        emit_s3p_attestation(
            signing_key=sk,
            epoch=1,
            violation_type="x",
            n_total=10,
            n_sampled=20,
            n_violations=0,
            sampling_rate=0.5,
            sampling_threshold=0.05,
            epoch_nonce_commitment=commitment,
            arbiter_instance_identifier=arbiter_id,
        )
    with pytest.raises(S3PError):
        emit_s3p_attestation(
            signing_key=sk,
            epoch=1,
            violation_type="x",
            n_total=100,
            n_sampled=50,
            n_violations=60,
            sampling_rate=0.5,
            sampling_threshold=0.05,
            epoch_nonce_commitment=commitment,
            arbiter_instance_identifier=arbiter_id,
        )
