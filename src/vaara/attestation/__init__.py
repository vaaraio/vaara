"""OVERT 1.0 Protocol Profile 1.0 attestation envelope emission.

Vaara's structural position relative to OVERT 1.0 (Glacis Technologies,
overt.is): Vaara is the third-party runtime kernel that intercepts agent
actions, scores risk, and writes the audit trail. In OVERT terms Vaara is
the **Arbiter** at AAL-3 (operator-controlled notary model). Phase 1
(Enforcement) and Phase 2 (Provisional Receipt) are emitted by Vaara
directly. Phase 3 (Full Attestation) requires an external Independent
Attestation Provider (IAP) and is intentionally out of scope.

The `BaseEnvelope` produced here implements Protocol Profile 1.0 Annex B.6
verbatim: a 9-field closed-schema CBOR-encoded structure signed with
Ed25519. Any OVERT-aware verifier (auditor, IAP, relying party) can
recompute the canonical encoding and verify the signature offline.

Install: ``pip install 'vaara[attestation]'``.

See COMPLIANCE.md "Position relative to open runtime-attestation standards"
for the full architectural framing.
"""

from vaara.attestation.overt import (
    BaseEnvelope,
    EnvelopeError,
    canonical_cbor,
    emit_base_envelope,
    verify_base_envelope,
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
from vaara.attestation.iap import (
    IAPError,
    Phase3Attestation,
    emit_phase3_attestation,
    envelope_to_canonical_cbor,
    verify_phase3_attestation,
)
from vaara.attestation.transparency_log import (
    InProcessTransparencyLog,
    InclusionProof,
    LogEntry,
    TransparencyLogError,
    verify_inclusion,
)

__all__ = [
    "BaseEnvelope",
    "ConformalExtension",
    "EnvelopeError",
    "IAPError",
    "InProcessTransparencyLog",
    "InclusionProof",
    "LogEntry",
    "Phase3Attestation",
    "S3PAttestation",
    "S3PError",
    "TransparencyLogError",
    "canonical_cbor",
    "clopper_pearson_ci",
    "emit_base_envelope",
    "emit_phase3_attestation",
    "emit_s3p_attestation",
    "envelope_to_canonical_cbor",
    "make_epoch_nonce_commitment",
    "regularized_incomplete_beta",
    "verify_base_envelope",
    "verify_inclusion",
    "verify_phase3_attestation",
    "verify_s3p_attestation",
]
