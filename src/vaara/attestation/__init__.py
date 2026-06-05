"""Runtime attestation envelopes: OVERT 1.0 and SEP-2787 v2.

This package ships two coexisting attestation surfaces:

**OVERT 1.0 Protocol Profile 1.0** (``vaara.attestation.overt``).
Per-action CBOR Base Envelope, 9-field closed schema, Ed25519
(optionally ML-DSA-65). Vaara's structural position relative to OVERT
1.0 (Glacis Technologies, overt.is): Vaara is the third-party runtime
kernel that intercepts agent actions, scores risk, and writes the
audit trail. In OVERT terms Vaara is the **Arbiter** at AAL-3
(operator-controlled notary model). Phase 1 (Enforcement) and Phase 2
(Provisional Receipt) are emitted by Vaara directly. Phase 3 (Full
Attestation) is provided by ``vaara.attestation.iap`` since v0.13.0
as a reference Independent Attestation Provider that notary-signs the
Provisional Receipt and anchors it in a transparency log. Production
deployments can swap in sigstore Rekor or an equivalent
independently-operated log at the same call sites. As of v0.17.0,
the ``vaara overt verify`` CLI validates any OVERT 1.0 Base Envelope
produced by a conformant emitter, Vaara or otherwise.

**SEP-2787 v2** (``vaara.attestation.sep2787``). Per-tool-call JSON
envelope carried inside MCP ``_meta``. Three trust-surface blocks
(``plannerDeclared``, ``issuerAsserted``, ``payloadDerived``) plus a
signature computed over the JCS-canonical encoding of those four
blocks. Signing modes: HS256, ES256, RS256. ``parse_attestation``
provides full wire round-trip so third-party consumers of the v0
test vectors can parse JSON bytes, verify the signature, and re-emit
byte-identically. Reference implementation pinned at tag
``sep2787-ref-v2``.

The two envelopes coexist. Field-level mapping lives in
``docs/sep2787-overt-mapping.md``.

Install: ``pip install 'vaara[attestation]'``.

See docs/COMPLIANCE.md "Position relative to open runtime-attestation
standards" for the full architectural framing.
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
    ConsistencyProof,
    InProcessTransparencyLog,
    InclusionProof,
    LogEntry,
    TransparencyLogError,
    verify_consistency,
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
    ArgsProjection,
    ArgsRef,
    Attestation as SEP2787Attestation,
    AttestationError as SEP2787AttestationError,
    IssuerAsserted,
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    canonical_json as sep2787_canonical_json,
    emit_attestation as sep2787_emit_attestation,
    make_args_digest,
    make_args_projection,
    parse_attestation as sep2787_parse_attestation,
    verify_attestation as sep2787_verify_attestation,
)
from vaara.attestation.receipt import (
    BackLink,
    BackLinkResult,
    BundleVerdict,
    EvidenceBundle,
    ExecutionReceipt,
    LensResult,
    LoggedReceiptVerdict,
    OutcomeDerived,
    ReceiptAsserted,
    RevocationEntry,
    RevocationRegistry,
    RevocationStatus,
    attestation_digest,
    build_bundle_document,
    check_receipt_revocation,
    emit_receipt,
    evidence_bundle_from_json,
    load_bundle_pieces_from_dir,
    make_back_link,
    make_result_digest,
    make_result_projection,
    parse_receipt,
    verify_back_link,
    verify_evidence_bundle,
    verify_logged_receipt,
    verify_receipt_signature,
)

__all__ = [
    "ArgsProjection",
    "ArgsRef",
    "BackLink",
    "BackLinkResult",
    "BaseEnvelope",
    "BundleVerdict",
    "ConformalExtension",
    "ConsistencyProof",
    "EnvelopeError",
    "EvidenceBundle",
    "ExecutionReceipt",
    "IAPError",
    "InProcessTransparencyLog",
    "InclusionProof",
    "IssuerAsserted",
    "LensResult",
    "LogEntry",
    "LoggedReceiptVerdict",
    "MockSEVSNPAttester",
    "OutcomeDerived",
    "PayloadDerived",
    "Phase3Attestation",
    "PlannerDeclared",
    "ReceiptAsserted",
    "RevocationEntry",
    "RevocationRegistry",
    "RevocationStatus",
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
    "attestation_digest",
    "bind_overt_envelope_to_report_data",
    "build_bundle_document",
    "canonical_cbor",
    "check_receipt_revocation",
    "clopper_pearson_ci",
    "emit_base_envelope",
    "emit_phase3_attestation",
    "emit_receipt",
    "emit_s3p_attestation",
    "envelope_to_canonical_cbor",
    "evidence_bundle_from_json",
    "load_bundle_pieces_from_dir",
    "make_args_digest",
    "make_args_projection",
    "make_back_link",
    "make_epoch_nonce_commitment",
    "make_result_digest",
    "make_result_projection",
    "parse_receipt",
    "parse_sev_snp_report",
    "regularized_incomplete_beta",
    "sep2787_canonical_json",
    "sep2787_emit_attestation",
    "sep2787_parse_attestation",
    "sep2787_verify_attestation",
    "verify_back_link",
    "verify_base_envelope",
    "verify_consistency",
    "verify_envelope_binding",
    "verify_evidence_bundle",
    "verify_inclusion",
    "verify_logged_receipt",
    "verify_phase3_attestation",
    "verify_receipt_signature",
    "verify_s3p_attestation",
    "verify_sev_snp_report_signature",
]
