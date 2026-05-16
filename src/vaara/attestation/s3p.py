"""OVERT 1.0 MEA-2 Statistical Safety Signal Protocol (S3P) emitter.

S3P (OVERT Section 9, control MEA-2) is the normative auditor-reproducible
measurement method defined by the OVERT 1.0 standard. An S3P attestation
reports the rate of safety violations observed in a deterministically
sampled subset of an epoch's traffic, with an exact confidence interval
that requires no distributional assumptions.

This module:

1. Computes exact Clopper-Pearson binomial confidence intervals per
   MEA-2.4. Pure-Python implementation via the regularized incomplete
   beta function; no scipy dependency.
2. Emits S3P attestations with the closed schema specified in MEA-2.6.
3. Canonically CBOR-encodes per Protocol Profile 1.0 (RFC 8949
   Section 4.2, IEEE-754 floats rejected, decimal-string rates).
4. Ed25519-signs the canonical encoding.
5. Optionally attaches a Vaara-specific conformal aggregate extension
   (proposed Protocol Profile extension; non-conflicting with the closed
   schema because the extension lives in a separate field).

Vaara's wedge for S3P: the standard MEA-2 method counts a binary
violation event per sampled request. Vaara's adaptive scorer also
produces a continuous risk score with conformal prediction intervals
(per-action upper / lower bounds with non-parametric coverage
guarantees). The ConformalExtension reports aggregate statistics over
those per-action intervals as a complementary signal an auditor can
verify independently.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional


class S3PError(RuntimeError):
    """Raised when S3P attestation construction or verification fails."""


_S3P_REQUEST_PREFIX = b"vaara/overt-pp1/s3p/v1\x00"

_VALID_STATUSES = frozenset(
    {"compliant", "threshold_exceeded", "insufficient_sample"}
)


def _decimal_str(value: float | Decimal, *, places: int = 12) -> str:
    """Stable decimal string for a rate or probability.

    Protocol Profile 1.0 forbids IEEE-754 floats in the canonical
    encoding. Rates and probabilities are emitted as decimal strings.
    Twelve decimal places is enough for Clopper-Pearson bounds at
    realistic sample sizes.
    """
    if isinstance(value, Decimal):
        dec = value
    else:
        if not math.isfinite(float(value)):
            raise S3PError(f"non-finite rate: {value!r}")
        dec = Decimal(repr(float(value)))
    quant = Decimal(10) ** -places
    quantised = dec.quantize(quant)
    text = format(quantised, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


@dataclass(frozen=True)
class ConformalExtension:
    """Vaara-specific proposed Protocol Profile extension.

    Aggregate statistics over Vaara's per-action conformal prediction
    intervals across the sampled epoch. Rides in a single ``extension``
    slot in the S3P attestation's signed metadata, so a standard OVERT
    verifier ignores it and a Vaara-aware verifier can cross-check the
    extension against the per-action receipts.

    All numeric fields are decimal strings per Protocol Profile 1.0
    numeric rules.
    """

    conformal_alpha: str
    mean_upper_bound: str
    fraction_over_threshold: str
    n_calibration_points: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "conformal_alpha": self.conformal_alpha,
            "mean_upper_bound": self.mean_upper_bound,
            "fraction_over_threshold": self.fraction_over_threshold,
            "n_calibration_points": int(self.n_calibration_points),
        }


@dataclass(frozen=True)
class S3PAttestation:
    """OVERT 1.0 S3P attestation per MEA-2.6.

    Closed schema with the 14 normative fields listed in MEA-2.6 plus the
    signature, key_identifier, arbiter_instance_identifier, and
    nanosecond_timestamp fields needed for verification. The
    ``extension`` field is a Protocol-Profile-defined map for additive
    signals; Vaara emits the ConformalExtension here when available.
    """

    epoch: int
    violation_type: str
    n_total: int
    n_sampled: int
    sampling_rate: str
    n_violations: int
    observed_rate: str
    confidence_level: str
    ci_lower: str
    ci_upper: str
    sampling_threshold: str
    epoch_nonce_commitment: bytes
    status: str
    nanosecond_timestamp: int
    key_identifier: bytes
    arbiter_instance_identifier: bytes
    signature: bytes
    extension: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "violation_type": self.violation_type,
            "n_total": self.n_total,
            "n_sampled": self.n_sampled,
            "sampling_rate": self.sampling_rate,
            "n_violations": self.n_violations,
            "observed_rate": self.observed_rate,
            "confidence_level": self.confidence_level,
            "CI_lower": self.ci_lower,
            "CI_upper": self.ci_upper,
            "sampling_threshold": self.sampling_threshold,
            "epoch_nonce_commitment": self.epoch_nonce_commitment.hex(),
            "status": self.status,
            "nanosecond_timestamp": self.nanosecond_timestamp,
            "key_identifier": self.key_identifier.hex(),
            "arbiter_instance_identifier": self.arbiter_instance_identifier.hex(),
            "signature": self.signature.hex(),
            "extension": self.extension,
        }


def make_epoch_nonce_commitment(
    epoch_nonce: bytes,
    *,
    epoch: int,
    operator_key: bytes,
) -> bytes:
    """Commit to a secret epoch nonce per MEA-2.1.

    The nonce stays secret during the epoch (so the operator cannot
    game the sampling decision). The commitment is published at epoch
    start; the nonce is published at epoch close (MEA-2.5). Domain-
    separated HMAC-SHA256 over the (epoch, nonce) pair binds the
    commitment to the epoch counter and prevents cross-epoch replay.
    """
    if not isinstance(epoch_nonce, (bytes, bytearray)):
        raise S3PError("epoch_nonce must be bytes")
    if len(epoch_nonce) < 16:
        raise S3PError("epoch_nonce must be >= 16 bytes")
    if not isinstance(operator_key, (bytes, bytearray)):
        raise S3PError("operator_key must be bytes")
    payload = (
        _S3P_REQUEST_PREFIX
        + epoch.to_bytes(8, "big", signed=False)
        + b"\x00"
        + bytes(epoch_nonce)
    )
    return hmac.new(bytes(operator_key), payload, hashlib.sha256).digest()


def _legacy_raw(pub) -> bytes:
    from cryptography.hazmat.primitives import serialization
    return pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


# ── Clopper-Pearson via regularized incomplete beta ──────────────────────

def _beta_continued_fraction(x: float, a: float, b: float) -> float:
    """Lentz's continued fraction for the regularized incomplete beta.

    Returns the CF used by the standard recursion for I_x(a, b).
    Converges to ~machine precision in ~30 iterations for typical
    (a, b) pairs.
    """
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            return h
    return h


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """I_x(a, b) — the regularized incomplete beta function.

    Defined as B(x; a, b) / B(a, b). Used as the CDF of the Beta
    distribution. Range: [0, 1]. Numerically stable on [0, 1] via the
    standard branch selection at x = (a+1)/(a+b+2).
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    front = math.exp(log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(x, a, b) / a
    return 1.0 - front * _beta_continued_fraction(1.0 - x, b, a) / b


def _beta_quantile(p: float, a: float, b: float) -> float:
    """Inverse of the Beta CDF at probability p via bisection.

    Pure bisection on [0, 1] using forward I_x evaluation. ~100
    iterations converges to ~1e-15 absolute. Sufficient for
    Clopper-Pearson CI construction.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        cdf = regularized_incomplete_beta(mid, a, b)
        if cdf < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return (lo + hi) / 2.0


def clopper_pearson_ci(
    n_violations: int,
    n_sampled: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Exact Clopper-Pearson binomial confidence interval.

    For ``k`` violations in ``n`` trials at confidence ``1 - alpha`` the
    Clopper-Pearson lower bound is the ``alpha/2`` quantile of
    Beta(k, n - k + 1) and the upper bound is the ``1 - alpha/2``
    quantile of Beta(k + 1, n - k). The k = 0 and k = n endpoints are
    handled analytically.
    """
    if n_sampled < 1:
        raise S3PError("n_sampled must be >= 1")
    if not (0 <= n_violations <= n_sampled):
        raise S3PError("n_violations must be in [0, n_sampled]")
    if not (0.0 < confidence_level < 1.0):
        raise S3PError("confidence_level must be in (0, 1)")
    alpha = 1.0 - confidence_level
    k = n_violations
    n = n_sampled
    if k == 0:
        lower = 0.0
    else:
        lower = _beta_quantile(alpha / 2.0, k, n - k + 1)
    if k == n:
        upper = 1.0
    else:
        upper = _beta_quantile(1.0 - alpha / 2.0, k + 1, n - k)
    return (lower, upper)


# ── S3P canonical signing payload ────────────────────────────────────────

def _canonical_s3p_payload(
    *,
    epoch: int,
    violation_type: str,
    n_total: int,
    n_sampled: int,
    sampling_rate: str,
    n_violations: int,
    observed_rate: str,
    confidence_level: str,
    ci_lower: str,
    ci_upper: str,
    sampling_threshold: str,
    epoch_nonce_commitment: bytes,
    status: str,
    nanosecond_timestamp: int,
    key_identifier: bytes,
    arbiter_instance_identifier: bytes,
    extension: dict,
) -> bytes:
    from vaara.attestation.overt import canonical_cbor

    payload = {
        "epoch": int(epoch),
        "violation_type": violation_type,
        "n_total": int(n_total),
        "n_sampled": int(n_sampled),
        "sampling_rate": sampling_rate,
        "n_violations": int(n_violations),
        "observed_rate": observed_rate,
        "confidence_level": confidence_level,
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
        "sampling_threshold": sampling_threshold,
        "epoch_nonce_commitment": bytes(epoch_nonce_commitment),
        "status": status,
        "nanosecond_timestamp": int(nanosecond_timestamp),
        "key_identifier": bytes(key_identifier),
        "arbiter_instance_identifier": bytes(arbiter_instance_identifier),
        "extension": extension,
    }
    return canonical_cbor(payload)


def emit_s3p_attestation(
    *,
    signing_key,
    epoch: int,
    violation_type: str,
    n_total: int,
    n_sampled: int,
    n_violations: int,
    sampling_rate: float | Decimal | str,
    sampling_threshold: float | Decimal | str,
    epoch_nonce_commitment: bytes,
    arbiter_instance_identifier: bytes,
    confidence_level: float = 0.95,
    conformal_extension: Optional[ConformalExtension] = None,
    nanosecond_timestamp: Optional[int] = None,
) -> S3PAttestation:
    """Build, canonical-CBOR-encode, and Ed25519-sign an S3P attestation.

    Computes the Clopper-Pearson CI from (n_violations, n_sampled) and
    assigns status: ``insufficient_sample`` if n_sampled is 0,
    ``threshold_exceeded`` if CI_lower > sampling_threshold,
    ``compliant`` otherwise.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError as exc:
        raise S3PError(
            "cryptography not installed. Install with: pip install "
            "'vaara[attestation]'"
        ) from exc
    if not isinstance(signing_key, Ed25519PrivateKey):
        raise S3PError("signing_key must be an Ed25519PrivateKey")
    if len(arbiter_instance_identifier) != 16:
        raise S3PError("arbiter_instance_identifier must be 16 bytes (UUID)")
    if n_total < 0 or n_sampled < 0 or n_violations < 0:
        raise S3PError("counts must be non-negative")
    if n_sampled > n_total:
        raise S3PError("n_sampled must be <= n_total")
    if n_violations > n_sampled:
        raise S3PError("n_violations must be <= n_sampled")

    sampling_rate_f = float(
        Decimal(sampling_rate) if isinstance(sampling_rate, str)
        else Decimal(repr(float(sampling_rate)))
    )
    sampling_threshold_f = float(
        Decimal(sampling_threshold) if isinstance(sampling_threshold, str)
        else Decimal(repr(float(sampling_threshold)))
    )
    if not (0.0 <= sampling_rate_f <= 1.0):
        raise S3PError("sampling_rate must be in [0, 1]")
    if not (0.0 <= sampling_threshold_f <= 1.0):
        raise S3PError("sampling_threshold must be in [0, 1]")

    if n_sampled == 0:
        observed_rate_f = 0.0
        ci_lower_f, ci_upper_f = 0.0, 1.0
        status = "insufficient_sample"
    else:
        observed_rate_f = n_violations / n_sampled
        ci_lower_f, ci_upper_f = clopper_pearson_ci(
            n_violations, n_sampled, confidence_level
        )
        status = (
            "threshold_exceeded"
            if ci_lower_f > sampling_threshold_f
            else "compliant"
        )

    sampling_rate_str = _decimal_str(
        Decimal(sampling_rate) if isinstance(sampling_rate, str)
        else Decimal(repr(float(sampling_rate)))
    )
    sampling_threshold_str = _decimal_str(
        Decimal(sampling_threshold) if isinstance(sampling_threshold, str)
        else Decimal(repr(float(sampling_threshold)))
    )
    observed_rate_str = _decimal_str(observed_rate_f)
    confidence_level_str = _decimal_str(confidence_level)
    ci_lower_str = _decimal_str(ci_lower_f)
    ci_upper_str = _decimal_str(ci_upper_f)

    if nanosecond_timestamp is None:
        nanosecond_timestamp = time.time_ns()

    pub = signing_key.public_key()
    pub_raw = (
        pub.public_bytes_raw() if hasattr(pub, "public_bytes_raw")
        else _legacy_raw(pub)
    )
    key_identifier = hashlib.sha256(pub_raw).digest()

    extension: dict = {}
    if conformal_extension is not None:
        extension["vaara_conformal_aggregate"] = conformal_extension.to_dict()

    payload = _canonical_s3p_payload(
        epoch=epoch,
        violation_type=violation_type,
        n_total=n_total,
        n_sampled=n_sampled,
        sampling_rate=sampling_rate_str,
        n_violations=n_violations,
        observed_rate=observed_rate_str,
        confidence_level=confidence_level_str,
        ci_lower=ci_lower_str,
        ci_upper=ci_upper_str,
        sampling_threshold=sampling_threshold_str,
        epoch_nonce_commitment=epoch_nonce_commitment,
        status=status,
        nanosecond_timestamp=nanosecond_timestamp,
        key_identifier=key_identifier,
        arbiter_instance_identifier=arbiter_instance_identifier,
        extension=extension,
    )
    signature = signing_key.sign(payload)

    return S3PAttestation(
        epoch=epoch,
        violation_type=violation_type,
        n_total=n_total,
        n_sampled=n_sampled,
        sampling_rate=sampling_rate_str,
        n_violations=n_violations,
        observed_rate=observed_rate_str,
        confidence_level=confidence_level_str,
        ci_lower=ci_lower_str,
        ci_upper=ci_upper_str,
        sampling_threshold=sampling_threshold_str,
        epoch_nonce_commitment=epoch_nonce_commitment,
        status=status,
        nanosecond_timestamp=nanosecond_timestamp,
        key_identifier=key_identifier,
        arbiter_instance_identifier=arbiter_instance_identifier,
        signature=signature,
        extension=extension,
    )


def verify_s3p_attestation(
    attestation: S3PAttestation, public_key_raw: bytes,
) -> bool:
    """Verify the Ed25519 signature on an S3P attestation.

    Recomputes the canonical CBOR encoding of every field except the
    signature and checks that the supplied public key matches the
    attestation's ``key_identifier``.
    """
    try:
        from cryptography.exceptions import (
            InvalidSignature,
            UnsupportedAlgorithm,
        )
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise S3PError(
            "cryptography not installed. Install with: pip install "
            "'vaara[attestation]'"
        ) from exc
    if hashlib.sha256(public_key_raw).digest() != attestation.key_identifier:
        return False
    payload = _canonical_s3p_payload(
        epoch=attestation.epoch,
        violation_type=attestation.violation_type,
        n_total=attestation.n_total,
        n_sampled=attestation.n_sampled,
        sampling_rate=attestation.sampling_rate,
        n_violations=attestation.n_violations,
        observed_rate=attestation.observed_rate,
        confidence_level=attestation.confidence_level,
        ci_lower=attestation.ci_lower,
        ci_upper=attestation.ci_upper,
        sampling_threshold=attestation.sampling_threshold,
        epoch_nonce_commitment=attestation.epoch_nonce_commitment,
        status=attestation.status,
        nanosecond_timestamp=attestation.nanosecond_timestamp,
        key_identifier=attestation.key_identifier,
        arbiter_instance_identifier=attestation.arbiter_instance_identifier,
        extension=attestation.extension,
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key_raw).verify(
            attestation.signature, payload,
        )
        return True
    except (InvalidSignature, ValueError, UnsupportedAlgorithm):
        return False
