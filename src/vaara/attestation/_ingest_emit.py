"""Emit and verify-signature for ``vaara.ingest/v0`` envelopes.

Internal module. Public surface is in ``vaara.attestation.receipt``.

This is the universal sink's wire shape. Foreign evidence (a SEP-2643
denial, a SEP-2787 attestation, a did:web ARD, a raw log line) is first
mapped onto the SEP-2828 evidence model by ``vaara...normalize``, then
sealed here into one signed, content-addressed record.

The two existing envelopes do not fit arbitrary foreign evidence:
``emit_receipt`` hard-requires a SEP-2787 back-link, and
``mint_authorization_receipt`` hard-requires a credential and a verdict.
Forcing a heterogeneous source into either would fabricate semantics the
source never carried (a back-link that does not exist, a decision that was
never rendered). The whole value of the sink is honest normalization, so
it gets its own envelope that asserts nothing the source did not establish.

What the envelope binds:

- ``evidenceRef`` content-addresses the normalized evidence object by
  ``sha256:<JCS(normalized)>``. The honest gap report (``missing``), the
  established proof fields (``sep2828``), and the non-proof context
  (``advisory``) all live inside that digested object, so they travel under
  the envelope signature: tamper with the gap report and the digest, and
  therefore the signature, no longer verifies.
- ``completeness`` carries a per-stream ``seq`` and ``runningCount``. A lone
  ingest is ``seq 1`` of a one-record stream; a dropped record inside a
  longer stream is then a provable gap.

Reuses the SEP-2787 canonicalization (RFC 8785 JCS) and signing stack
(HS256 / ES256 / RS256) unchanged. No new crypto. The content-address
recompute is pure JCS + sha256: a third party reproduces it with zero
Vaara import.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._normalize import NormalizedEvidence
from vaara.attestation._sep2787_canonical import (
    canonical_json,
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
    AttestationError,
)

# Pinned in SPEC.md alongside vaara.receipt/v1. ``v0`` because the foreign
# normalization surface is still widening; the crypto stack is v1-stable.
INGEST_SCHEMA = "vaara.ingest/v0"
NORMALIZED_EVIDENCE_SCHEMA = "vaara.normalized-evidence/v0"


def _evidence_digest(evidence: dict[str, Any]) -> str:
    """sha256 over the JCS canonicalization of the normalized evidence.

    Pure JCS + sha256, the content address an independent verifier
    recomputes without importing Vaara.
    """
    return f"sha256:{hashlib.sha256(canonical_json(evidence)).hexdigest()}"


def _single_record_completeness() -> dict[str, int]:
    """Completeness for a lone ingest: seq 1 of a one-record stream."""
    return {"seq": 1, "runningCount": 1}


@dataclass(frozen=True)
class IngestReceipt:
    """A signed ``vaara.ingest/v0`` envelope plus its evidence object.

    ``record`` is the signed envelope. ``evidence`` is the normalized
    evidence object it content-addresses (``NormalizedEvidence.to_dict()``);
    its ``sha256:<JCS>`` digest is pinned by ``record["evidenceRef"]["digest"]``
    under the signature. The two travel together but bind by hash, so the
    evidence bytes may also be carried out of band.
    """

    record: dict[str, Any]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"record": self.record, "evidence": self.evidence}


def _signing_body(
    *,
    version: int,
    alg: Algorithm,
    source_format: str,
    evidence_ref: dict[str, Any],
    ingest_asserted: dict[str, Any],
    completeness: dict[str, Any],
) -> dict[str, Any]:
    """The envelope payload that gets canonicalized and signed.

    Everything substantive about the source lives in the digested evidence
    object; the envelope itself stays thin and carries only what binds and
    routes. ``sourceFormat`` is echoed here purely so a verifier can assert
    it equals the evidence object's own ``sourceFormat`` (a bound cross-check,
    never a second source of truth).
    """
    return {
        "schema": INGEST_SCHEMA,
        "version": version,
        "alg": alg,
        "sourceFormat": source_format,
        "evidenceRef": evidence_ref,
        "ingestAsserted": ingest_asserted,
        "completeness": completeness,
    }


def emit_ingest_receipt(
    *,
    normalized: NormalizedEvidence,
    iss: str,
    sub: str,
    secret_version: str,
    alg: Algorithm,
    signing_material: Any,
    nonce: Optional[str] = None,
    iat: Optional[str] = None,
    completeness: Optional[dict[str, Any]] = None,
    evidence_ref: Optional[str] = None,
    version: int = 0,
    sig_suite: Optional[str] = None,
) -> IngestReceipt:
    """Seal one normalized foreign record into a signed ingest envelope.

    ``normalized`` is the output of ``vaara...normalize`` for one source
    document. It is content-addressed and pinned by the envelope's
    ``evidenceRef.digest``; nothing about it is re-asserted in the envelope,
    so the honest ``missing`` gap report travels under the signature without
    being editable apart from it.

    ``signing_material`` is a bytes shared secret (HS256) or a private-key
    object from ``cryptography.hazmat`` (ES256 / RS256), the same key shapes
    the SEP-2787 stack takes. ``completeness`` defaults to a single-record
    stream; pass ``{"seq": n, "runningCount": m}`` to place this record in a
    longer ingest stream. ``evidence_ref`` is an optional, non-authoritative
    locator (URI or path) for the evidence bytes; the digest is what binds.
    """
    if alg not in VALID_ALGS:
        raise AttestationError(f"unsupported alg: {alg!r}")

    evidence = normalized.to_dict()
    ref: dict[str, Any] = {
        "digest": _evidence_digest(evidence),
        "canonicalization": "JCS",
        "schema": NORMALIZED_EVIDENCE_SCHEMA,
    }
    if evidence_ref is not None:
        if not isinstance(evidence_ref, str) or not evidence_ref:
            raise AttestationError(
                "evidenceRef.ref must be a non-empty string or absent"
            )
        ref["ref"] = evidence_ref

    ingest_asserted: dict[str, Any] = {
        "iss": iss,
        "sub": sub,
        "iat": iat or now_iso8601(),
        "nonce": nonce or new_nonce(),
        "secretVersion": secret_version,
        "alg": alg,
    }
    if sig_suite is not None:
        ingest_asserted["sigSuite"] = sig_suite

    body = _signing_body(
        version=version,
        alg=alg,
        source_format=normalized.source_format,
        evidence_ref=ref,
        ingest_asserted=ingest_asserted,
        completeness=completeness or _single_record_completeness(),
    )

    payload = canonical_json(body)
    if alg == "HS256":
        if not isinstance(signing_material, (bytes, bytearray)):
            raise AttestationError("HS256 requires bytes shared_secret")
        signature_hex = sign_hs256(payload, shared_secret=bytes(signing_material))
    elif alg == "ES256":
        signature_hex = sign_es256(payload, private_key=signing_material)
    elif alg == "RS256":
        signature_hex = sign_rs256(payload, private_key=signing_material)
    else:
        raise AttestationError(f"unreachable alg: {alg!r}")

    record = {**body, "signature": signature_hex}
    return IngestReceipt(record=record, evidence=evidence)


def verify_ingest_signature(
    record: dict[str, Any],
    evidence: dict[str, Any],
    verifying_material: Any,
) -> bool:
    """Verify an ingest envelope against its evidence object.

    Returns True iff (1) the evidence object recomputes to the content
    address pinned in ``record["evidenceRef"]["digest"]``, (2) the envelope's
    ``sourceFormat`` matches the evidence object's, and (3) the signature
    matches the JCS-canonical encoding of the envelope (signature field
    removed) under ``verifying_material``. ``verifying_material`` is a bytes
    shared secret (HS256) or a public-key object (ES256 / RS256).

    Any structural surprise (missing field, wrong type, unknown alg) is a
    verification failure, not an exception.
    """
    if not isinstance(record, dict) or not isinstance(evidence, dict):
        return False
    try:
        ref = record["evidenceRef"]
        pinned_digest = ref["digest"]
        alg = record["alg"]
        signature_hex = record["signature"]
    except (KeyError, TypeError):
        return False
    if not isinstance(signature_hex, str):
        return False

    try:
        recomputed = _evidence_digest(evidence)
    except AttestationError:
        return False
    if recomputed != pinned_digest:
        return False
    if record.get("sourceFormat") != evidence.get("sourceFormat"):
        return False
    if alg not in VALID_ALGS:
        return False

    body = {k: v for k, v in record.items() if k != "signature"}
    try:
        payload = canonical_json(body)
    except AttestationError:
        return False

    if alg == "HS256":
        if not isinstance(verifying_material, (bytes, bytearray)):
            return False
        return verify_hs256(
            payload,
            signature_hex=signature_hex,
            shared_secret=bytes(verifying_material),
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
