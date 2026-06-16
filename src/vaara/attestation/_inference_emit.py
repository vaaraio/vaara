# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Emit and verify for inference-receipt envelopes.

Internal module. Public surface is in ``vaara.attestation.inference``.

Reuses the SEP-2787 canonicalization (RFC 8785 JCS) and signing stack
(HS256 / ES256 / RS256) unchanged. The only new wire shape is the envelope
layout; the cryptographic primitives are shared so a verifier that already
checks Vaara records checks inference receipts with no new crypto code.

Sampling-parameter float discipline: ``canonical_json`` rejects IEEE-754
floats outright (cross-stack float drift is the most common cause of
signature mismatch). Sampling params (``temperature``, ``top_p``, ...) arrive
as floats, so ``normalize_inference_request`` rewrites every float to its
shortest round-trip decimal string before the request is committed. The
committed object is therefore the normalized one; a verifier re-derives the
same digest from the same normalization.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Optional

from vaara.attestation._inference_types import (
    InferenceAttestation,
    InferenceOutcome,
    InferenceReceipt,
    ModelDerived,
    RequestDeclared,
    inference_outcome_to_dict,
    model_derived_to_dict,
    request_declared_to_dict,
)
from vaara.attestation._receipt_types import (
    BackLink,
    ReceiptAsserted,
    back_link_to_dict,
    receipt_asserted_to_dict,
)
from vaara.attestation._sep2787_canonical import (
    canonical_json,
    iso8601_to_epoch,
    make_args_digest,
    new_nonce,
    now_iso8601,
)
from vaara.attestation._sep2787_signing import (
    sign_es256,
    sign_hs256,
    sign_rs256,
    verify_es256,
    verify_hs256,
    verify_rs256,
)
from vaara.attestation._sep2787_types import (
    VALID_ALGS,
    Algorithm,
    ArgsCommitment,
    AttestationError,
    IssuerAsserted,
    issuer_to_dict,
)


def _sign(payload: bytes, *, alg: Algorithm, signing_material: Any) -> str:
    if alg == "HS256":
        if not isinstance(signing_material, (bytes, bytearray)):
            raise AttestationError("HS256 requires bytes shared_secret")
        return sign_hs256(payload, shared_secret=bytes(signing_material))
    if alg == "ES256":
        return sign_es256(payload, private_key=signing_material)
    if alg == "RS256":
        return sign_rs256(payload, private_key=signing_material)
    raise AttestationError(f"unsupported alg: {alg!r}")


def _verify(
    payload: bytes, *, alg: Algorithm, signature_hex: str, verifying_material: Any
) -> bool:
    if alg == "HS256":
        if not isinstance(verifying_material, (bytes, bytearray)):
            return False
        return verify_hs256(
            payload, signature_hex=signature_hex, shared_secret=bytes(verifying_material)
        )
    if alg == "ES256":
        return verify_es256(
            payload, signature_hex=signature_hex, public_key=verifying_material
        )
    if alg == "RS256":
        return verify_rs256(
            payload, signature_hex=signature_hex, public_key=verifying_material
        )
    return False


def _stringify_floats(obj: Any) -> Any:
    """Recursively rewrite floats to their shortest round-trip decimal string.

    ``repr(float)`` is the shortest string that round-trips the value, so the
    rewrite is a pure, deterministic function of the input float. bool is an
    int subclass and is left untouched (JSON booleans canonicalize fine).
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return repr(obj)
    if isinstance(obj, dict):
        return {k: _stringify_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stringify_floats(v) for v in obj]
    return obj


def normalize_inference_request(
    *, messages: Any, sampling: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Canonical request object: messages + float-normalized sampling params."""
    return {
        "messages": _stringify_floats(messages),
        "sampling": _stringify_floats(sampling or {}),
    }


def make_request_commitment(
    *, messages: Any, sampling: Optional[dict[str, Any]] = None
) -> ArgsCommitment:
    """Hash-only commitment over the normalized request object."""
    return make_args_digest(
        normalize_inference_request(messages=messages, sampling=sampling)
    )


def make_output_commitment(output: Any) -> ArgsCommitment:
    """Hash-only commitment over the response payload.

    ``output`` is the assembled response object (e.g. ``{"content": "...",
    "toolCalls": [...]}`` or a bare string). Floats are normalized for the
    same reason as the request.
    """
    return make_args_digest(_stringify_floats(output))


def _attestation_signing_payload(att: InferenceAttestation) -> bytes:
    """JCS-canonical encoding of the attestation blocks, signature excluded."""
    return canonical_json(
        {
            "version": att.version,
            "alg": att.alg,
            "requestDeclared": request_declared_to_dict(att.request_declared),
            "issuerAsserted": issuer_to_dict(att.issuer_asserted),
            "modelDerived": model_derived_to_dict(att.model_derived),
        }
    )


def emit_inference_attestation(
    *,
    request_declared: RequestDeclared,
    model_derived: ModelDerived,
    iss: str,
    sub: str,
    secret_version: str,
    alg: Algorithm,
    signing_material: Any,
    exp_seconds: int = 300,
    nonce: Optional[str] = None,
    iat: Optional[str] = None,
    version: int = 1,
) -> InferenceAttestation:
    """Build, JCS-canonicalize, and sign an InferenceAttestation envelope."""
    if alg not in VALID_ALGS:
        raise AttestationError(f"unsupported alg: {alg!r}")
    if not request_declared.intent.strip():
        raise AttestationError("requestDeclared.intent MUST be non-empty")

    issuer_asserted = IssuerAsserted(
        iss=iss,
        sub=sub,
        iat=iat or now_iso8601(),
        exp_seconds=int(exp_seconds),
        nonce=nonce or new_nonce(),
        secret_version=secret_version,
        alg=alg,
    )
    unsigned = InferenceAttestation(
        version=version,
        alg=alg,
        request_declared=request_declared,
        issuer_asserted=issuer_asserted,
        model_derived=model_derived,
        signature="",
    )
    signature = _sign(
        _attestation_signing_payload(unsigned),
        alg=alg,
        signing_material=signing_material,
    )
    return InferenceAttestation(
        version=version,
        alg=alg,
        request_declared=request_declared,
        issuer_asserted=issuer_asserted,
        model_derived=model_derived,
        signature=signature,
    )


def verify_inference_attestation_detail(
    att: InferenceAttestation,
    *,
    verifying_material: Any,
    now: Optional[float] = None,
    clock_skew_seconds: int = 30,
) -> dict[str, Any]:
    """Decompose attestation verification into signature vs freshness.

    Returns ``{"signatureValid", "fresh", "iatValid", "ageSeconds",
    "expSeconds"}``. ``verify_inference_attestation`` is the strict AND of an
    authentic signature and a live TTL window; this split lets a verifier
    report the two apart so an authentic-but-expired credential is never
    mislabeled as a signature failure. Stored receipts are archival by nature
    and are expected to outlive their live window, so a verifier sweeping a
    directory reports freshness rather than gating on it.

    Freshness mirrors ``verify_attestation``: ``iat - skew <= now <= iat +
    expSeconds + skew`` with a future-dating guard so issuance cannot be
    stamped ahead to extend the live window.
    """
    signature_valid = _verify(
        _attestation_signing_payload(att),
        alg=att.alg,
        signature_hex=att.signature,
        verifying_material=verifying_material,
    )
    iat_epoch = iso8601_to_epoch(att.issuer_asserted.iat)
    current = now if now is not None else time.time()
    if iat_epoch is None:
        return {
            "signatureValid": signature_valid,
            "fresh": False,
            "iatValid": False,
            "ageSeconds": None,
            "expSeconds": att.issuer_asserted.exp_seconds,
        }
    future_dated = iat_epoch > current + clock_skew_seconds
    deadline = iat_epoch + att.issuer_asserted.exp_seconds + clock_skew_seconds
    return {
        "signatureValid": signature_valid,
        "fresh": (not future_dated) and current <= deadline,
        "iatValid": True,
        "ageSeconds": current - iat_epoch,
        "expSeconds": att.issuer_asserted.exp_seconds,
    }


def verify_inference_attestation(
    att: InferenceAttestation,
    *,
    verifying_material: Any,
    now: Optional[float] = None,
    clock_skew_seconds: int = 30,
) -> bool:
    """Verify the attestation signature AND its live TTL window (strict AND).

    For the signature and freshness verdicts reported apart (so an expired but
    authentic credential is not mislabeled as a bad signature), use
    ``verify_inference_attestation_detail``.
    """
    detail = verify_inference_attestation_detail(
        att,
        verifying_material=verifying_material,
        now=now,
        clock_skew_seconds=clock_skew_seconds,
    )
    return bool(detail["signatureValid"] and detail["fresh"])


def inference_attestation_digest(att: InferenceAttestation) -> str:
    """``sha256:<hex>`` over the full attestation wire bytes (signature included)."""
    return "sha256:" + hashlib.sha256(canonical_json(att.to_dict())).hexdigest()


def inference_receipt_digest(receipt: InferenceReceipt) -> str:
    """``sha256:<hex>`` over the full receipt wire bytes (signature included).

    The receipt twin of :func:`inference_attestation_digest`. A session
    manifest pins both digests per inference, so the same JCS-over-wire formula
    is used for both: binding the manifest to a hardware root transitively pins
    every receipt byte-for-byte.
    """
    return "sha256:" + hashlib.sha256(canonical_json(receipt.to_dict())).hexdigest()


def make_inference_back_link(att: InferenceAttestation) -> BackLink:
    """Build the receipt back-link that pins this attestation instance."""
    return BackLink(
        attestation_digest=inference_attestation_digest(att),
        attestation_nonce=att.issuer_asserted.nonce,
    )


def _receipt_signing_payload(receipt: InferenceReceipt) -> bytes:
    """JCS-canonical encoding of the receipt blocks, signature excluded."""
    return canonical_json(
        {
            "version": receipt.version,
            "alg": receipt.alg,
            "backLink": back_link_to_dict(receipt.back_link),
            "outcomeDerived": inference_outcome_to_dict(receipt.outcome_derived),
            "receiptAsserted": receipt_asserted_to_dict(receipt.receipt_asserted),
        }
    )


def emit_inference_receipt(
    *,
    back_link: BackLink,
    outcome_derived: InferenceOutcome,
    iss: str,
    sub: str,
    secret_version: str,
    alg: Algorithm,
    signing_material: Any,
    nonce: Optional[str] = None,
    iat: Optional[str] = None,
    version: int = 1,
) -> InferenceReceipt:
    """Build, JCS-canonicalize, and sign an InferenceReceipt envelope.

    ``back_link`` joins the receipt to the inference attestation it answers
    (build it with ``make_inference_back_link``).
    """
    if alg not in VALID_ALGS:
        raise AttestationError(f"unsupported alg: {alg!r}")
    if not back_link.attestation_digest.startswith("sha256:"):
        raise AttestationError("backLink.attestationDigest MUST be a 'sha256:' digest")
    if not back_link.attestation_nonce:
        raise AttestationError("backLink.attestationNonce MUST be non-empty")

    receipt_asserted = ReceiptAsserted(
        iss=iss,
        sub=sub,
        iat=iat or now_iso8601(),
        nonce=nonce or new_nonce(),
        secret_version=secret_version,
        alg=alg,
    )
    unsigned = InferenceReceipt(
        version=version,
        alg=alg,
        back_link=back_link,
        receipt_asserted=receipt_asserted,
        outcome_derived=outcome_derived,
        signature="",
    )
    signature = _sign(
        _receipt_signing_payload(unsigned), alg=alg, signing_material=signing_material
    )
    return InferenceReceipt(
        version=version,
        alg=alg,
        back_link=back_link,
        receipt_asserted=receipt_asserted,
        outcome_derived=outcome_derived,
        signature=signature,
    )


def verify_inference_receipt_signature(
    receipt: InferenceReceipt, *, verifying_material: Any
) -> bool:
    """Verify the receipt signature only.

    Back-link and output-commitment checks are composed separately by the
    caller (``verify_inference_back_link`` and ``verify_args_commitment``); a
    receipt is a durable record, so there is no TTL to enforce.
    """
    return _verify(
        _receipt_signing_payload(receipt),
        alg=receipt.alg,
        signature_hex=receipt.signature,
        verifying_material=verifying_material,
    )


def verify_inference_back_link(
    receipt: InferenceReceipt, *, attestation: InferenceAttestation
) -> bool:
    """Confirm the receipt's back-link pins ``attestation``.

    Recomputes the attestation digest and constant-time compares it, then
    checks the nonce agrees, so a receipt cannot carry one attestation's
    digest under another's nonce.
    """
    expected_digest = inference_attestation_digest(attestation)
    if not hmac.compare_digest(
        receipt.back_link.attestation_digest, expected_digest
    ):
        return False
    return receipt.back_link.attestation_nonce == attestation.issuer_asserted.nonce
