"""Execution receipts: the post-execution sibling of SEP-2787.

SEP-2787 attests a ``tools/call`` *request* before it runs: issuer,
subject, target, intent, nonce, time, argument commitment. It defers,
by design, any claim about whether the call executed or what came
back. The execution receipt covers that deferred half. It binds the
outcome of one attested request and links back to the attestation it
answers.

A receipt carries three parts plus a signature:

- ``backLink`` pins the SEP-2787 attestation by nonce and by a digest
  over its full wire bytes.
- ``receiptAsserted`` is the issuer block, set by whoever observed the
  outcome (the executing server, or an intermediary such as a
  governance proxy).
- ``outcomeDerived`` carries the status (``executed`` / ``refused`` /
  ``errored``), the completion time, and an optional commitment over
  the result, reusing the SEP-2787 argument-commitment shapes.

A receipt verifies in three composable checks: the signature
(``verify_receipt_signature``), the back-link to its attestation
(``verify_back_link``), and, when a result commitment is present, the
result binding (``verify_args_commitment`` against the runtime result
object). A receipt is a durable record rather than a time-bounded
capability, so there is no TTL.

Canonicalization (RFC 8785 JCS) and signing (HS256 / ES256 / RS256)
are shared with ``vaara.attestation.tool_call_attestation`` unchanged. A verifier
that already checks SEP-2787 signatures needs no new crypto to check
receipts.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

from vaara.attestation._bundle import (
    BundleVerdict,
    EvidenceBundle,
    LensResult,
    verify_evidence_bundle,
)
from vaara.attestation._bundle_io import (
    build_bundle_document,
    evidence_bundle_from_json,
    load_bundle_pieces_from_dir,
)
from vaara.attestation._bundle_set import (
    BUNDLE_SET_SCHEMA_NAME,
    BundleSetEntry,
    BundleSetReport,
    check_bundle_set,
)
from vaara.attestation._audit_summary import (
    SUMMARY_SCHEMA,
    render_record_set_summary,
)
from vaara.attestation._conformance_statement import (
    STATEMENT_SCHEMA,
    ConformanceCorpusError,
    ConformanceStatement,
    CorpusIntegrity,
    RecordsResult,
    SelfTest,
    SuiteResult,
    build_conformance_statement,
    render_conformance_statement,
    verify_corpus_integrity,
)
from vaara.attestation._decision_conformance import (
    check_decision_conformance,
)
from vaara.attestation._enforcement import (
    ENFORCEMENT_SCHEMA,
    EnforcementVerdict,
    bind_record_to_report_data,
    verify_enforcement,
)
from vaara.attestation._enforcement_set import (
    ENFORCEMENT_SET_SCHEMA_NAME,
    EnforcementSetEntry,
    EnforcementSetReport,
    check_enforcement_set,
)
from vaara.attestation._tpm import (
    MockTPMQuoter,
    TPMAttestationError,
)
from vaara.attestation._tpm_binding import (
    TPM_BINDING_SCHEMA,
    TPMBindingVerdict,
    bind_record_to_chain_extra_data,
    bind_record_to_extra_data,
    verify_tpm_binding,
)
from vaara.attestation._tpm_bundle import (
    TPM_BUNDLE_SCHEMA,
    build_tpm_bundle_document,
    verify_tpm_bundle,
)
from vaara.attestation._tpm_chain import (
    TPM_CHAIN_SCHEMA,
    TPMChainLink,
    TPMChainVerdict,
    verify_tpm_chain,
)
from vaara.attestation._tpm_chain_bundle import (
    build_tpm_chain_document,
    verify_tpm_chain_bundle,
)
from vaara.attestation._attestation_result import (
    SCHEMA as ATTESTATION_RESULT_SCHEMA,
    build_attestation_result,
)
from vaara.attestation._handoff import (
    ComponentDigest,
    HandoffVerdict,
    build_handoff,
    sign_manifest,
    verify_handoff,
)
from vaara.attestation._handoff_set import (
    HANDOFF_SET_SCHEMA_NAME,
    HandoffSetEntry,
    HandoffSetReport,
    check_handoff_set,
)
from vaara.attestation._key_history import (
    KeyHistory,
    KeyValidity,
    KeyValidityStatus,
    within_validity,
)
from vaara.attestation._normalize import (
    NormalizedEvidence,
    detect_format,
    normalize,
)
from vaara.attestation._ingest_emit import (
    INGEST_SCHEMA,
    NORMALIZED_EVIDENCE_SCHEMA,
    IngestReceipt,
    emit_ingest_receipt,
    verify_ingest_signature,
)
from vaara.attestation._receipt_conformance import (
    ConformanceCheck,
    ConformanceReport,
    check_record_conformance,
)
from vaara.attestation._record_set_conformance import (
    RecordSetEntry,
    RecordSetReport,
    SetFinding,
    check_record_set,
    classify_record,
)
from vaara.attestation._receipt_emit import (
    emit_receipt,
    verify_receipt_signature,
)
from vaara.attestation._receipt_identity import (
    IdentityResult,
    did_web_to_url,
    verify_receipt_identity,
)
from vaara.attestation._receipt_identity_live import (
    DidDocumentCache,
    LiveIdentityResult,
    ResolutionMeta,
    https_fetch,
    verify_receipt_identity_live,
)
from vaara.attestation._receipt_pq import (
    PqVerdict,
    attach_pq_signature,
    mldsa65_public_key_from_method,
    mldsa65_sign,
    mldsa65_verify,
    pq_verdict,
    receipt_preimage,
)
from vaara.attestation._receipt_retention import (
    RetentionResult,
    verify_receipt_retained,
)
from vaara.attestation._receipt_cbom import (
    CBOM_ABSENT,
    CBOM_DOWNGRADE,
    CBOM_MISMATCH,
    CBOM_OK,
    crypto_posture_for,
    verify_crypto_posture,
)
from vaara.attestation._receipt_types import (
    BackLink,
    CryptoAlgorithm,
    CryptoPosture,
    ExecutionReceipt,
    ExistenceProof,
    OutcomeDerived,
    PqSignature,
    ReceiptAsserted,
    ReceiptStatus,
    ResultCommitment,
    receipt_from_dict as parse_receipt,
)
from vaara.attestation._receipt_existence import (
    ExistenceResult,
    attach_existence_proof,
    existence_record_digest,
    verify_existence_proof,
)
from vaara.attestation._revocation import (
    LoggedReceiptVerdict,
    RevocationEntry,
    RevocationRegistry,
    RevocationStatus,
    check_receipt_revocation,
    receipt_leaf_bytes,
    revoked_in_time,
    verify_logged_receipt,
)
from vaara.attestation._receipt_verifier import (
    BackLinkResult,
    attestation_digest,
    make_back_link,
    verify_back_link,
)
from vaara.attestation._receipt_vc import (
    VAARA_RECEIPT_CONTEXT_URL,
    load_receipt_context,
    receipt_from_vc,
    receipt_to_vc,
)

# Result commitments reuse the SEP-2787 argument-commitment builders.
# Re-export under result-oriented names so call sites read naturally.
from vaara.attestation._attest_canonical import (
    make_args_digest as make_result_digest,
    make_args_projection as make_result_projection,
)

__all__ = [
    "VAARA_RECEIPT_CONTEXT_URL",
    "BackLink",
    "BackLinkResult",
    "BundleVerdict",
    "ComponentDigest",
    "ConformanceCheck",
    "ConformanceReport",
    "ENFORCEMENT_SCHEMA",
    "EnforcementVerdict",
    "bind_record_to_report_data",
    "verify_enforcement",
    "ENFORCEMENT_SET_SCHEMA_NAME",
    "EnforcementSetEntry",
    "EnforcementSetReport",
    "check_enforcement_set",
    "TPM_BINDING_SCHEMA",
    "TPM_BUNDLE_SCHEMA",
    "TPM_CHAIN_SCHEMA",
    "TPMAttestationError",
    "TPMBindingVerdict",
    "TPMChainLink",
    "TPMChainVerdict",
    "MockTPMQuoter",
    "bind_record_to_chain_extra_data",
    "bind_record_to_extra_data",
    "build_tpm_bundle_document",
    "build_tpm_chain_document",
    "verify_tpm_binding",
    "verify_tpm_bundle",
    "verify_tpm_chain",
    "verify_tpm_chain_bundle",
    "ATTESTATION_RESULT_SCHEMA",
    "build_attestation_result",
    "HandoffVerdict",
    "build_handoff",
    "sign_manifest",
    "verify_handoff",
    "HANDOFF_SET_SCHEMA_NAME",
    "HandoffSetEntry",
    "HandoffSetReport",
    "check_handoff_set",
    "check_record_conformance",
    "check_decision_conformance",
    "ExistenceProof",
    "ExistenceResult",
    "attach_existence_proof",
    "existence_record_digest",
    "verify_existence_proof",
    "RecordSetEntry",
    "RecordSetReport",
    "SetFinding",
    "check_record_set",
    "classify_record",
    "BUNDLE_SET_SCHEMA_NAME",
    "BundleSetEntry",
    "BundleSetReport",
    "check_bundle_set",
    "SUMMARY_SCHEMA",
    "render_record_set_summary",
    "STATEMENT_SCHEMA",
    "ConformanceCorpusError",
    "ConformanceStatement",
    "CorpusIntegrity",
    "RecordsResult",
    "SelfTest",
    "SuiteResult",
    "build_conformance_statement",
    "render_conformance_statement",
    "verify_corpus_integrity",
    "NormalizedEvidence",
    "detect_format",
    "normalize",
    "INGEST_SCHEMA",
    "NORMALIZED_EVIDENCE_SCHEMA",
    "IngestReceipt",
    "emit_ingest_receipt",
    "verify_ingest_signature",
    "CryptoAlgorithm",
    "CryptoPosture",
    "CBOM_OK",
    "CBOM_ABSENT",
    "CBOM_MISMATCH",
    "CBOM_DOWNGRADE",
    "crypto_posture_for",
    "verify_crypto_posture",
    "DidDocumentCache",
    "EvidenceBundle",
    "ExecutionReceipt",
    "IdentityResult",
    "KeyHistory",
    "KeyValidity",
    "KeyValidityStatus",
    "LensResult",
    "LiveIdentityResult",
    "LoggedReceiptVerdict",
    "RetentionResult",
    "OutcomeDerived",
    "PqSignature",
    "PqVerdict",
    "ReceiptAsserted",
    "ReceiptStatus",
    "ResolutionMeta",
    "ResultCommitment",
    "RevocationEntry",
    "RevocationRegistry",
    "RevocationStatus",
    "attestation_digest",
    "attach_pq_signature",
    "check_receipt_revocation",
    "did_web_to_url",
    "emit_receipt",
    "mldsa65_public_key_from_method",
    "mldsa65_sign",
    "mldsa65_verify",
    "pq_verdict",
    "receipt_preimage",
    "build_bundle_document",
    "evidence_bundle_from_json",
    "load_bundle_pieces_from_dir",
    "https_fetch",
    "load_receipt_context",
    "make_back_link",
    "make_result_digest",
    "make_result_projection",
    "parse_receipt",
    "receipt_from_vc",
    "receipt_leaf_bytes",
    "receipt_to_vc",
    "revoked_in_time",
    "verify_back_link",
    "verify_evidence_bundle",
    "verify_logged_receipt",
    "verify_receipt_identity",
    "verify_receipt_identity_live",
    "verify_receipt_retained",
    "verify_receipt_signature",
    "within_validity",
]
