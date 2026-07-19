# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""OVERT Phase 3 Independent Attestation Provider (IAP) reference.

A Vaara AAL-3 ``BaseEnvelope`` is a Provisional Receipt signed by the
Arbiter. Reaching AAL-4 requires Phase 3: an independent notary signs
over the envelope and the resulting attestation is anchored in a
transparency log so it cannot be silently retracted.

This module ships a reference IAP that runs end-to-end without a hard
dependency on an external transparency log. The signing surface and the
log adapter shape are kept narrow so a production deployment can swap in
sigstore Rekor (or equivalent) at the same call sites.

Structural independence is enforced: the notary key identifier must
differ from the arbiter key identifier carried inside the AAL-3
envelope. Any attestation where the two collide is rejected at both
emission and verification.

CBOR canonicalisation, domain-separated hashing, and Ed25519 signing
mirror the AAL-3 emitter so an OVERT-aware verifier can reconstruct
every signed payload offline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation.overt import (
    BaseEnvelope,
    EnvelopeError,
    canonical_cbor,
    _canonical_signing_payload,
    _legacy_raw,
    _sha256,
)
from vaara.attestation.transparency_log import (
    InProcessTransparencyLog,
    InclusionProof,
    LogEntry,
    verify_inclusion,
)


class IAPError(RuntimeError):
    """Raised when Phase 3 attestation emission or verification fails."""


_NOTARY_SIGNING_PREFIX = b"vaara/overt-pp1/iap-notary/v1\x00"


@dataclass(frozen=True)
class Phase3Attestation:
    """OVERT 1.0 AAL-4 Phase 3 attestation wrapping an AAL-3 envelope."""

    envelope_cbor: bytes
    notary_signature: bytes
    notary_key_identifier: bytes
    log_index: int
    log_tree_size: int
    log_root_at_append: bytes
    inclusion_proof_siblings: tuple[bytes, ...]
    iap_identifier: str
    attestation_timestamp_ns: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "envelope_cbor": self.envelope_cbor.hex(),
            "notary_signature": self.notary_signature.hex(),
            "notary_key_identifier": self.notary_key_identifier.hex(),
            "log_index": self.log_index,
            "log_tree_size": self.log_tree_size,
            "log_root_at_append": self.log_root_at_append.hex(),
            "inclusion_proof_siblings": [
                s.hex() for s in self.inclusion_proof_siblings
            ],
            "iap_identifier": self.iap_identifier,
            "attestation_timestamp_ns": self.attestation_timestamp_ns,
        }


def envelope_to_canonical_cbor(envelope: BaseEnvelope) -> bytes:
    """Canonical CBOR over all 9 envelope fields, including the signature.

    The Phase 3 notary signs over this byte string so the inner Arbiter
    signature is bound by reference, and the same bytes are appended to
    the transparency log as the leaf.
    """
    full_payload = {
        "blinded_identifier": envelope.blinded_identifier,
        "request_commitment": envelope.request_commitment,
        "encoder_binary_identity": envelope.encoder_binary_identity,
        "non_content_metadata": envelope.non_content_metadata,
        "monotonic_counter": int(envelope.monotonic_counter),
        "nanosecond_timestamp": int(envelope.nanosecond_timestamp),
        "key_identifier": envelope.key_identifier,
        "arbiter_instance_identifier": envelope.arbiter_instance_identifier,
        "signature": envelope.signature,
    }
    return canonical_cbor(full_payload)


def _notary_pubkey_identifier(signing_key) -> bytes:
    pub = signing_key.public_key()
    pub_raw = (
        pub.public_bytes_raw()
        if hasattr(pub, "public_bytes_raw")
        else _legacy_raw(pub)
    )
    return _sha256(pub_raw)


def emit_phase3_attestation(
    *,
    envelope: BaseEnvelope,
    notary_signing_key,
    transparency_log: InProcessTransparencyLog,
    iap_identifier: str,
    attestation_timestamp_ns: Optional[int] = None,
) -> Phase3Attestation:
    """Notary-sign an AAL-3 envelope and anchor it in the transparency log.

    Rejects the call if the notary key identifier matches the arbiter
    key identifier — OVERT Phase 3 requires structural independence.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError as exc:
        raise EnvelopeError(
            "cryptography not installed. Install with: pip install "
            "'vaara[attestation]'"
        ) from exc
    if not isinstance(notary_signing_key, Ed25519PrivateKey):
        raise IAPError("notary_signing_key must be an Ed25519PrivateKey")

    notary_key_id = _notary_pubkey_identifier(notary_signing_key)
    if notary_key_id == envelope.key_identifier:
        raise IAPError(
            "notary key identifier equals arbiter key identifier; OVERT "
            "Phase 3 requires structural independence between the arbiter "
            "and the IAP. Generate the IAP keypair from a separate root."
        )

    envelope_cbor = envelope_to_canonical_cbor(envelope)
    log_entry: LogEntry = transparency_log.append(envelope_cbor)
    proof: InclusionProof = transparency_log.inclusion_proof(log_entry.log_index)

    signature = notary_signing_key.sign(
        _NOTARY_SIGNING_PREFIX + envelope_cbor
    )
    ts = (
        attestation_timestamp_ns
        if attestation_timestamp_ns is not None
        else time.time_ns()
    )

    return Phase3Attestation(
        envelope_cbor=envelope_cbor,
        notary_signature=signature,
        notary_key_identifier=notary_key_id,
        log_index=log_entry.log_index,
        log_tree_size=log_entry.tree_size_at_append,
        log_root_at_append=log_entry.root_hash_at_append,
        inclusion_proof_siblings=proof.siblings,
        iap_identifier=iap_identifier,
        attestation_timestamp_ns=ts,
    )


def verify_phase3_attestation(
    *,
    attestation: Phase3Attestation,
    notary_public_key_raw: bytes,
    expected_log_root: Optional[bytes] = None,
    arbiter_public_key_raw: Optional[bytes] = None,
) -> bool:
    """Verify the notary signature, inclusion proof, and structural independence.

    Args:
        attestation: The Phase 3 attestation to verify.
        notary_public_key_raw: 32-byte raw Ed25519 public key of the
            notary. Must match the attestation's ``notary_key_identifier``.
        expected_log_root: When supplied, the inclusion proof is checked
            against this root. Pass the root the verifier observed at
            audit time, or pin ``attestation.log_root_at_append`` if
            using the per-append root.
        arbiter_public_key_raw: When supplied, the inner envelope's
            Arbiter signature is also verified.
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
        raise EnvelopeError(
            "cryptography not installed. Install with: pip install "
            "'vaara[attestation]'"
        ) from exc

    if _sha256(notary_public_key_raw) != attestation.notary_key_identifier:
        return False

    try:
        Ed25519PublicKey.from_public_bytes(notary_public_key_raw).verify(
            attestation.notary_signature,
            _NOTARY_SIGNING_PREFIX + attestation.envelope_cbor,
        )
    except (InvalidSignature, ValueError, UnsupportedAlgorithm):
        return False

    if expected_log_root is not None:
        proof = InclusionProof(
            log_index=attestation.log_index,
            tree_size=attestation.log_tree_size,
            siblings=attestation.inclusion_proof_siblings,
        )
        if not verify_inclusion(
            leaf_data=attestation.envelope_cbor,
            proof=proof,
            expected_root=expected_log_root,
        ):
            return False

    if arbiter_public_key_raw is not None:
        try:
            import cbor2
        except ImportError as exc:
            raise EnvelopeError(
                "cbor2 not installed. Install with: pip install "
                "'vaara[attestation]'"
            ) from exc

        decoded = cbor2.loads(attestation.envelope_cbor)
        if not isinstance(decoded, dict):
            return False
        if _sha256(arbiter_public_key_raw) != decoded.get("key_identifier"):
            return False
        if decoded.get("key_identifier") == attestation.notary_key_identifier:
            return False

        signing_payload = _canonical_signing_payload(
            blinded_identifier=decoded["blinded_identifier"],
            request_commitment=decoded["request_commitment"],
            encoder_binary_identity=decoded["encoder_binary_identity"],
            non_content_metadata=decoded["non_content_metadata"],
            monotonic_counter=decoded["monotonic_counter"],
            nanosecond_timestamp=decoded["nanosecond_timestamp"],
            key_identifier=decoded["key_identifier"],
            arbiter_instance_identifier=decoded["arbiter_instance_identifier"],
        )
        try:
            Ed25519PublicKey.from_public_bytes(arbiter_public_key_raw).verify(
                decoded["signature"], signing_payload,
            )
        except (InvalidSignature, ValueError, UnsupportedAlgorithm):
            return False

    return True
