"""OVERT 1.0 Protocol Profile 1.0 attestation envelope emission.

Vaara's structural position relative to OVERT 1.0 (Glacis Technologies,
overt.is): Vaara is the third-party runtime kernel that intercepts agent
actions, scores risk, and writes the audit trail. In OVERT terms Vaara is
the **Arbiter** at AAL-3 (operator-controlled notary model). Phase 1
(Enforcement) and Phase 2 (Provisional Receipt) are emitted by Vaara
directly. Phase 3 (Full Attestation) is provided by ``vaara.attestation.iap``
since v0.13.0 as a reference Independent Attestation Provider that
notary-signs the Provisional Receipt and anchors it in a transparency
log. Production deployments can swap in sigstore Rekor or an equivalent
independently-operated log at the same call sites. As of v0.17.0, the
``vaara overt verify`` CLI validates any OVERT 1.0 Base Envelope produced
by a conformant emitter, Vaara or otherwise.

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
from vaara.attestation.tee import (
    MockSEVSNPAttester,
    SEVSNPHostAttester,
    SEVSNPReport,
    TEEAttestationError,
    TEEAttester,
    bind_overt_envelope_to_report_data,
    parse_sev_snp_report,
    verify_envelope_binding,
    verify_sev_snp_report_signature,
)
from vaara.attestation.sep2787 import (
    Algorithm as SEP2787Algorithm,
    ArgsCommitment as SEP2787ArgsCommitment,
    ArgsDigest,
    ArgsProjection,
    ArgsRef,
    Attestation as SEP2787Attestation,
    AttestationError as SEP2787AttestationError,
    IssuerAsserted,
    PlannerDeclared,
    ToolCallBinding,
    canonical_json as sep2787_canonical_json,
    emit_attestation as sep2787_emit_attestation,
    make_args_digest,
    make_args_projection,
    verify_attestation as sep2787_verify_attestation,
)

__all__ = [
    "ArgsDigest",
    "ArgsProjection",
    "ArgsRef",
    "BaseEnvelope",
    "ConformalExtension",
    "EnvelopeError",
    "IAPError",
    "InProcessTransparencyLog",
    "InclusionProof",
    "IssuerAsserted",
    "LogEntry",
    "MockSEVSNPAttester",
    "Phase3Attestation",
    "PlannerDeclared",
    "S3PAttestation",
    "S3PError",
    "SEP2787Algorithm",
    "SEP2787ArgsCommitment",
    "SEP2787Attestation",
    "SEP2787AttestationError",
    "SEVSNPHostAttester",
    "SEVSNPReport",
    "TEEAttestationError",
    "TEEAttester",
    "ToolCallBinding",
    "TransparencyLogError",
    "bind_overt_envelope_to_report_data",
    "canonical_cbor",
    "clopper_pearson_ci",
    "emit_base_envelope",
    "emit_phase3_attestation",
    "emit_s3p_attestation",
    "envelope_to_canonical_cbor",
    "make_args_digest",
    "make_args_projection",
    "make_epoch_nonce_commitment",
    "parse_sev_snp_report",
    "regularized_incomplete_beta",
    "sep2787_canonical_json",
    "sep2787_emit_attestation",
    "sep2787_verify_attestation",
    "verify_base_envelope",
    "verify_envelope_binding",
    "verify_inclusion",
    "verify_phase3_attestation",
    "verify_s3p_attestation",
    "verify_sev_snp_report_signature",
]
