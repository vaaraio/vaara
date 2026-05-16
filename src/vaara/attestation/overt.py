"""OVERT 1.0 Protocol Profile 1.0 Base Envelope (Annex B.6).

The Base Envelope is a 9-field closed-schema structure emitted for every
in-scope AI action. Per Protocol Profile 1.0 it is canonically encoded as
deterministic CBOR (RFC 8949 Section 4.2) and signed with Ed25519 (RFC 8032).

Field set is FIXED — no additional fields permitted (closed schema). The
field names below match Annex B.6 of OVERT 1.0. Numeric rules per Protocol
Profile 1.0 Section B.3: IEEE-754 floats are PROHIBITED. Timestamps are
uint64 nanoseconds since the Unix epoch. Rates and probabilities, when
they appear inside `non_content_metadata`, MUST be decimal strings.

This emitter is AAL-3 Phase 2 only (Provisional Receipt with arbiter
signature). AAL-4 promotion requires an external IAP to attach Phase 3
notary attestation and transparency-log inclusion proof; that workflow is
outside Vaara's scope.
"""

from __future__ import annotations

import hashlib
import hmac
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional


class EnvelopeError(RuntimeError):
    """Raised when envelope construction or verification fails."""


@dataclass(frozen=True)
class BaseEnvelope:
    """OVERT 1.0 Protocol Profile 1.0 Base Envelope — Annex B.6.

    Field order matches the spec. The schema is closed: any verifier
    rejecting additional fields will reject envelopes that carry extras.

    Attributes:
        blinded_identifier: Opaque per-action identifier. 32 bytes.
        request_commitment: HMAC-SHA256 keyed commitment over the request
            content digest. The raw content NEVER leaves the operator
            environment; only this commitment crosses the trust boundary.
        encoder_binary_identity: SHA-256 hash identifying the arbiter
            implementation + version + policy hash at the time of
            attestation. At AAL-3 this is operator-derived; at AAL-4 it
            would be notary-derived via a measurement pipeline.
        non_content_metadata: A CBOR-serializable map of structural
            classification fields (action class, severity, decision, etc.)
            that contain no protected content. Operator decides what is
            safe to include.
        monotonic_counter: Strictly increasing per-arbiter-instance
            sequence number. Detects gaps.
        nanosecond_timestamp: uint64 nanoseconds since Unix epoch.
        key_identifier: Identifier (e.g., SHA-256 fingerprint) of the
            Ed25519 public key used to sign this envelope.
        arbiter_instance_identifier: UUID of the specific Vaara instance
            that produced this envelope.
        signature: Ed25519 signature (64 bytes) over the canonical CBOR
            encoding of the OTHER 8 fields in order.
    """

    blinded_identifier: bytes
    request_commitment: bytes
    encoder_binary_identity: bytes
    non_content_metadata: dict
    monotonic_counter: int
    nanosecond_timestamp: int
    key_identifier: bytes
    arbiter_instance_identifier: bytes
    signature: bytes

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for JSON consumers. Bytes are hex-encoded."""
        return {
            "blinded_identifier": self.blinded_identifier.hex(),
            "request_commitment": self.request_commitment.hex(),
            "encoder_binary_identity": self.encoder_binary_identity.hex(),
            "non_content_metadata": self.non_content_metadata,
            "monotonic_counter": self.monotonic_counter,
            "nanosecond_timestamp": self.nanosecond_timestamp,
            "key_identifier": self.key_identifier.hex(),
            "arbiter_instance_identifier": self.arbiter_instance_identifier.hex(),
            "signature": self.signature.hex(),
        }


# Domain separation prefix for request commitments — Protocol Profile 1.0
# uses versioned prefixes on every HMAC operation (Annex B.2). Vaara uses
# this single prefix for v1 commitments. Profile-defined exact bytes live
# in the registered Protocol Profile document; this is Vaara's chosen
# operator-implementation-specific prefix, scoped via the
# `encoder_binary_identity` field.
_REQUEST_COMMITMENT_PREFIX = b"vaara/overt-pp1/request-commitment/v1\x00"
_ENCODER_IDENTITY_PREFIX = b"vaara/overt-pp1/encoder-identity/v1\x00"


def _now_ns() -> int:
    return time.time_ns()


def _hmac_sha256(key: bytes, message: bytes) -> bytes:
    return hmac.new(key, message, hashlib.sha256).digest()


def _sha256(message: bytes) -> bytes:
    return hashlib.sha256(message).digest()


def canonical_cbor(value: Any) -> bytes:
    """Canonical deterministic CBOR encoding per RFC 8949 Section 4.2.

    Uses cbor2 with canonical=True (definite-length, sorted keys, smallest
    int encoding, no floats). IEEE-754 floats are rejected at the boundary
    per Protocol Profile 1.0 numeric rules — callers must pass scaled
    integers or decimal strings for rates and probabilities.
    """
    try:
        import cbor2
    except ImportError as exc:
        raise EnvelopeError(
            "cbor2 not installed. Install with: pip install 'vaara[attestation]'"
        ) from exc

    _reject_floats(value)
    return cbor2.dumps(value, canonical=True)


def _reject_floats(value: Any, path: str = "") -> None:
    """Recursively check that no IEEE-754 floats are present."""
    if isinstance(value, float):
        raise EnvelopeError(
            f"IEEE-754 float at {path or '<root>'} is prohibited by "
            f"Protocol Profile 1.0 numeric rules. Use a scaled integer or "
            f"decimal string instead."
        )
    if isinstance(value, dict):
        for k, v in value.items():
            _reject_floats(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _reject_floats(v, f"{path}[{i}]")


def make_request_commitment(
    request_content: bytes,
    *,
    operator_key: bytes,
) -> bytes:
    """HMAC-SHA256 keyed commitment over a request content digest.

    The operator_key is derived from the operator's root secret via HKDF in
    a Protocol-Profile-conformant deployment. For Vaara's reference
    implementation the operator passes the key directly. Per Annex B.4 the
    raw content digest stays local; only this keyed commitment crosses the
    trust boundary.
    """
    if not isinstance(operator_key, (bytes, bytearray)):
        raise EnvelopeError("operator_key must be bytes")
    if not isinstance(request_content, (bytes, bytearray)):
        raise EnvelopeError("request_content must be bytes")
    digest = _sha256(bytes(request_content))
    return _hmac_sha256(bytes(operator_key), _REQUEST_COMMITMENT_PREFIX + digest)


def encoder_binary_identity(
    *,
    arbiter_version: str,
    policy_hash: bytes,
) -> bytes:
    """SHA-256 over (vaara version + policy hash).

    At AAL-3 this is operator-derived. At AAL-4 a notary measurement
    pipeline would replace this with a hardware-rooted attestation of the
    arbiter binary.
    """
    if not isinstance(policy_hash, (bytes, bytearray)):
        raise EnvelopeError("policy_hash must be bytes")
    payload = (
        _ENCODER_IDENTITY_PREFIX
        + arbiter_version.encode("utf-8")
        + b"\x00"
        + bytes(policy_hash)
    )
    return _sha256(payload)


def _canonical_signing_payload(
    blinded_identifier: bytes,
    request_commitment: bytes,
    encoder_binary_identity: bytes,
    non_content_metadata: dict,
    monotonic_counter: int,
    nanosecond_timestamp: int,
    key_identifier: bytes,
    arbiter_instance_identifier: bytes,
) -> bytes:
    """Canonical CBOR encoding of the 8 signable Base Envelope fields.

    Field order matches Annex B.6. The `signature` field is intentionally
    excluded — it is the result, not an input to itself.
    """
    payload = {
        "blinded_identifier": blinded_identifier,
        "request_commitment": request_commitment,
        "encoder_binary_identity": encoder_binary_identity,
        "non_content_metadata": non_content_metadata,
        "monotonic_counter": int(monotonic_counter),
        "nanosecond_timestamp": int(nanosecond_timestamp),
        "key_identifier": key_identifier,
        "arbiter_instance_identifier": arbiter_instance_identifier,
    }
    return canonical_cbor(payload)


def emit_base_envelope(
    *,
    signing_key,
    request_commitment: bytes,
    encoder_binary_identity: bytes,
    non_content_metadata: dict,
    monotonic_counter: int,
    arbiter_instance_identifier: bytes,
    blinded_identifier: Optional[bytes] = None,
    nanosecond_timestamp: Optional[int] = None,
) -> BaseEnvelope:
    """Build, canonical-CBOR-encode, and Ed25519-sign a Base Envelope.

    The Base Envelope is the AAL-3 Phase 2 Provisional Receipt for a
    single action. An external IAP can co-sign it later to produce the
    AAL-4 Phase 3 attestation; Vaara does not do that work.

    Args:
        signing_key: A cryptography Ed25519PrivateKey instance. Vaara's
            existing key infrastructure (vaara keygen, signing-keys.md)
            applies.
        request_commitment: HMAC-SHA256 keyed commitment (see
            make_request_commitment).
        encoder_binary_identity: SHA-256 identity (see
            encoder_binary_identity).
        non_content_metadata: Structural metadata containing no protected
            content. Floats are rejected.
        monotonic_counter: Strictly increasing per-arbiter sequence.
        arbiter_instance_identifier: 16 bytes (UUID) identifying the
            Vaara arbiter instance.
        blinded_identifier: 32 random bytes if not supplied.
        nanosecond_timestamp: Override timestamp; default uses time.time_ns().
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
    if not isinstance(signing_key, Ed25519PrivateKey):
        raise EnvelopeError("signing_key must be an Ed25519PrivateKey")

    if blinded_identifier is None:
        blinded_identifier = uuid.uuid4().bytes + uuid.uuid4().bytes
    if nanosecond_timestamp is None:
        nanosecond_timestamp = _now_ns()
    if len(arbiter_instance_identifier) != 16:
        raise EnvelopeError(
            "arbiter_instance_identifier must be 16 bytes (UUID)"
        )

    pub = signing_key.public_key()
    pub_raw = pub.public_bytes_raw() if hasattr(pub, "public_bytes_raw") else _legacy_raw(pub)
    key_identifier = _sha256(pub_raw)

    signing_payload = _canonical_signing_payload(
        blinded_identifier=blinded_identifier,
        request_commitment=request_commitment,
        encoder_binary_identity=encoder_binary_identity,
        non_content_metadata=non_content_metadata,
        monotonic_counter=monotonic_counter,
        nanosecond_timestamp=nanosecond_timestamp,
        key_identifier=key_identifier,
        arbiter_instance_identifier=arbiter_instance_identifier,
    )
    signature = signing_key.sign(signing_payload)

    return BaseEnvelope(
        blinded_identifier=blinded_identifier,
        request_commitment=request_commitment,
        encoder_binary_identity=encoder_binary_identity,
        non_content_metadata=non_content_metadata,
        monotonic_counter=monotonic_counter,
        nanosecond_timestamp=nanosecond_timestamp,
        key_identifier=key_identifier,
        arbiter_instance_identifier=arbiter_instance_identifier,
        signature=signature,
    )


def verify_base_envelope(envelope: BaseEnvelope, public_key_raw: bytes) -> bool:
    """Verify an envelope's Ed25519 signature against the raw 32-byte pubkey.

    Returns True iff the canonical CBOR encoding of the 8 signable fields
    matches the signature under the supplied public key, AND the supplied
    public key's SHA-256 matches the envelope's `key_identifier`.
    """
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise EnvelopeError(
            "cryptography not installed. Install with: pip install "
            "'vaara[attestation]'"
        ) from exc

    if _sha256(public_key_raw) != envelope.key_identifier:
        return False
    payload = _canonical_signing_payload(
        blinded_identifier=envelope.blinded_identifier,
        request_commitment=envelope.request_commitment,
        encoder_binary_identity=envelope.encoder_binary_identity,
        non_content_metadata=envelope.non_content_metadata,
        monotonic_counter=envelope.monotonic_counter,
        nanosecond_timestamp=envelope.nanosecond_timestamp,
        key_identifier=envelope.key_identifier,
        arbiter_instance_identifier=envelope.arbiter_instance_identifier,
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key_raw).verify(
            envelope.signature, payload,
        )
        return True
    except InvalidSignature:
        return False


def _legacy_raw(pub) -> bytes:
    from cryptography.hazmat.primitives import serialization
    return pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
