# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Emit and signature-verify data-locality records (``vaara.data-locality/v0``).

Internal module. Public surface is in ``vaara.attestation.data_locality``.

A data-locality record binds one agent action's cross-border transfer facts
(data class, endpoint, endpoint region, TLS cert fingerprint, payload digest) to
the enforced allow/block decision, and optionally carries a region attestation
signed by a party distinct from the issuer (a TEE or the receiving provider).

The record is byte-compatible with the ``data_locality_v0`` conformance corpus:
the signature is Ed25519 over the RFC 8785 JCS encoding of the record with the
``signature`` field removed, and the carried attestation signs the JCS of
``{attestedRegion, attester, nonce}``. An outside party grades any record this
module emits with the corpus's dependency-free checker.

This module is deliberately policy-agnostic. It records the decision the
pipeline already made; recomputing whether that decision was correct under a
named policy is the verifier's job (see the corpus checker), not the emitter's.
Turning a *claimed* region into an *attested* one — real TEE/provider
attestation acquisition — is out of scope here by design.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._attest_canonical import canonical_json, new_nonce
from vaara.attestation._attest_types import AttestationError
from vaara.audit.signer import Signer, Verifier

SCHEMA = "vaara.data-locality/v0"


@dataclass(frozen=True)
class TransferFacts:
    """Observable facts about one cross-border transfer.

    ``payload_digest`` and ``tls_cert_sha256`` are ``sha256:<hex>`` strings;
    build the payload digest with :func:`payload_digest`. ``endpoint_region`` is
    the region the endpoint claims — its truth is a Tier-B attestation concern,
    not something this record asserts.
    """

    action_id: str
    data_class: str
    endpoint: str
    endpoint_region: str
    payload_digest: str
    tls_cert_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "actionId": self.action_id,
            "dataClass": self.data_class,
            "endpoint": self.endpoint,
            "endpointRegion": self.endpoint_region,
            "payloadDigest": self.payload_digest,
            "tlsCertSha256": self.tls_cert_sha256,
        }


def payload_digest(payload: Any) -> str:
    """``sha256:<hex>`` over the RFC 8785 JCS bytes of ``payload``."""
    return "sha256:" + hashlib.sha256(canonical_json(payload)).hexdigest()


def region_attestation(
    signer: Signer, *, attester: str, attested_region: str, nonce: Optional[str] = None
) -> dict[str, Any]:
    """Build a carried region attestation, signed by ``signer`` (the attester).

    The attester is a party distinct from the record issuer; a relying party
    verifies this signature against the attester's own public key, never the
    issuer's. In production the attestation comes from a TEE or the provider;
    this helper covers the reference and test paths.
    """
    body = {"attestedRegion": attested_region, "attester": attester,
            "nonce": nonce or new_nonce()}
    sig = signer.sign(canonical_json(body)).hex()
    return {**body, "sig": sig}


def _record_signing_bytes(record: dict[str, Any]) -> bytes:
    """JCS of the record with the ``signature`` field removed."""
    return canonical_json({k: v for k, v in record.items() if k != "signature"})


def emit_data_locality_record(
    *,
    signer: Signer,
    issuer: str,
    transfer: TransferFacts,
    decision: str,
    policy_id: str,
    region_attestation: Optional[dict[str, Any]] = None,
    version: int = 1,
) -> dict[str, Any]:
    """Build, JCS-canonicalize, and sign a data-locality record.

    ``decision`` is the enforced verdict for the transfer (``"allow"`` or
    ``"block"``); ``policy_id`` names the policy that produced it.
    ``region_attestation`` is an optional carried attestation from
    :func:`region_attestation`. Returns the wire-form record dict.
    """
    if decision not in ("allow", "block"):
        raise AttestationError(f"decision must be 'allow' or 'block', got {decision!r}")
    for field in ("payload_digest", "tls_cert_sha256"):
        if not getattr(transfer, field).startswith("sha256:"):
            raise AttestationError(f"transfer.{field} MUST be a 'sha256:' digest")

    record: dict[str, Any] = {
        "alg": signer.algorithm,
        "decision": {"decision": decision, "policyId": policy_id},
        "issuer": issuer,
        "schema": SCHEMA,
        "transfer": transfer.to_dict(),
        "version": version,
    }
    if region_attestation is not None:
        record["regionAttestation"] = region_attestation
    record["signature"] = signer.sign(_record_signing_bytes(record)).hex()
    return record


def verify_record_signature(record: dict[str, Any], *, verifier: Verifier) -> bool:
    """Verify the record's issuer signature only (Tier-A integrity).

    Returns True iff the signature matches the JCS encoding of the record minus
    ``signature`` under ``verifier``. Payload-digest, policy, and carried
    attestation checks are composed separately; the corpus checker recomputes
    all of them from bytes.
    """
    sig_hex = record.get("signature")
    if not isinstance(sig_hex, str):
        return False
    try:
        return verifier.verify(_record_signing_bytes(record), bytes.fromhex(sig_hex))
    except ValueError:
        return False


def emit_from_interception(
    result: Any,
    *,
    signer: Signer,
    issuer: str,
    data_class: str,
    endpoint: str,
    endpoint_region: str,
    payload: Any,
    tls_cert_sha256: str,
    policy_id: str,
    region_attestation: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Emit a locality record from an ``InterceptionResult`` and transfer metadata.

    The seam between the governed decision and a signed locality record:
    ``result.allowed`` maps to ``allow``/``block`` and ``result.action_id``
    identifies the action. The transfer metadata (endpoint, region, data class,
    payload, observed cert fingerprint) comes from the integration that owns the
    outbound call. Acquiring a genuine region attestation for
    ``region_attestation`` is the closed-side concern, left to the caller.
    """
    transfer = TransferFacts(
        action_id=result.action_id,
        data_class=data_class,
        endpoint=endpoint,
        endpoint_region=endpoint_region,
        payload_digest=payload_digest(payload),
        tls_cert_sha256=tls_cert_sha256,
    )
    return emit_data_locality_record(
        signer=signer,
        issuer=issuer,
        transfer=transfer,
        decision="allow" if result.allowed else "block",
        policy_id=policy_id,
        region_attestation=region_attestation,
    )
