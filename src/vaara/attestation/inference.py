# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Inference receipts: governing the model call, not just the tool call.

The MCP proxy's SEP-2787 attestation+receipt pair binds a ``tools/call``.
This sibling pair binds the ``chat/completion`` underneath it: which model,
on what silicon-resident weights, given what input, returned what output. It
is the seventh sovereignty lever -- signed evidence that the inference itself,
not only the tooling around it, is accounted for.

Two envelopes mirror the SEP-2787 pair and reuse its canonicalization
(RFC 8785 JCS) and signing stack (HS256 / ES256 / RS256) unchanged:

- ``InferenceAttestation`` is the pre-call request commitment (declared
  intent + request commitment, issuer block with a TTL, and the model facts
  the proxy derived at call time).
- ``InferenceReceipt`` is the post-call outcome, back-linked to the exact
  attestation, carrying status, an output commitment, eval-stat counters, and
  an honest ``tier`` self-label.

Tier A (``integrity``) binds model+input+output with no determinism claim and
ships standalone. Tier B (``replay``) additionally claims byte-reproducibility
and is deferred; see ``research/inference_receipts_design_20260614.md``.

A verifier that already checks Vaara records needs no new crypto: the result
and request commitments reuse the SEP-2787 ``verify_args_commitment`` shapes.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

from vaara.attestation._inference_determinism import (
    DeterminismVerdict,
    determinism_verdict,
    honest_tier,
)
from vaara.attestation._inference_emit import (
    emit_inference_attestation,
    emit_inference_receipt,
    inference_attestation_digest,
    inference_receipt_digest,
    make_inference_back_link,
    make_output_commitment,
    make_request_commitment,
    normalize_inference_request,
    verify_inference_attestation,
    verify_inference_attestation_detail,
    verify_inference_back_link,
    verify_inference_receipt_signature,
)
from vaara.attestation._inference_types import (
    INFER_VALID_STATUSES,
    INFER_VALID_TIERS,
    InferenceAttestation,
    InferenceOutcome,
    InferenceReceipt,
    ModelDerived,
    RequestDeclared,
    inference_attestation_from_dict as parse_inference_attestation,
    inference_receipt_from_dict as parse_inference_receipt,
)

__all__ = [
    "INFER_VALID_STATUSES",
    "INFER_VALID_TIERS",
    "DeterminismVerdict",
    "determinism_verdict",
    "honest_tier",
    "InferenceAttestation",
    "InferenceOutcome",
    "InferenceReceipt",
    "ModelDerived",
    "RequestDeclared",
    "emit_inference_attestation",
    "emit_inference_receipt",
    "inference_attestation_digest",
    "inference_receipt_digest",
    "make_inference_back_link",
    "make_output_commitment",
    "make_request_commitment",
    "normalize_inference_request",
    "parse_inference_attestation",
    "parse_inference_receipt",
    "verify_inference_attestation",
    "verify_inference_attestation_detail",
    "verify_inference_back_link",
    "verify_inference_receipt_signature",
]
