"""Deprecated alias for :mod:`vaara.attestation.tool_call_attestation`.

Vaara's tool-call attestation was formerly exposed here under the SEP-2787
name. The format is Vaara's own: Vaara published the first implementation
(v0.42.0-0.44.0, 29-30 May 2026) and originated the trust-surface grouping
and schema points the community SEP-2787 draft later adopted. This module
re-exports the public surface for backward compatibility and will be removed
in a future release; import from ``vaara.attestation.tool_call_attestation``.
"""

from __future__ import annotations

from vaara.attestation.tool_call_attestation import (  # noqa: F401
    Algorithm,
    ArgsCommitment,
    ArgsCommitmentResult,
    ArgsProjection,
    ArgsRef,
    Attestation,
    AttestationError,
    IssuerAsserted,
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    canonical_json,
    emit_attestation,
    make_args_digest,
    make_args_projection,
    parse_attestation,
    verify_args_commitment,
    verify_attestation,
)

__all__ = [
    "Algorithm",
    "ArgsCommitment",
    "ArgsCommitmentResult",
    "ArgsProjection",
    "ArgsRef",
    "Attestation",
    "AttestationError",
    "IssuerAsserted",
    "PayloadDerived",
    "PlannerDeclared",
    "ToolCallBinding",
    "canonical_json",
    "emit_attestation",
    "make_args_digest",
    "make_args_projection",
    "parse_attestation",
    "verify_args_commitment",
    "verify_attestation",
]
