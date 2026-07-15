# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Dataclasses and serialization for inference-receipt envelopes.

Internal module. Public surface is in ``vaara.attestation.inference``.

The inference receipt governs the *model call* underneath a tool call. The
MCP proxy's SEP-2787 attestation+receipt pair binds a ``tools/call``; this
sibling pair binds the ``chat/completion`` that produced it: which model, on
what silicon-resident weights, given what input, returned what output.

Two envelopes mirror the SEP-2787 pair so they share the same
canonicalization (RFC 8785 JCS) and signing stack (HS256 / ES256 / RS256):

1. ``InferenceAttestation`` is the pre-call request commitment. Three blocks
   plus the signature: ``requestDeclared`` (declared intent + a commitment
   over the canonical request, reusing the SEP-2787 argument-commitment
   shapes), ``issuerAsserted`` (the SEP-2787 issuer block reused unchanged,
   ``iss="vaara-infer-proxy"``, carrying an ``expSeconds`` TTL because a
   pre-call attestation is a time-bounded capability), and ``modelDerived``
   (facts the proxy pulled from the inference server at call time: model ref,
   manifest digest, GGUF-metadata hash, optional quant / param-count labels).

2. ``InferenceReceipt`` is the post-call outcome, reusing the execution
   receipt's ``backLink`` join and ``receiptAsserted`` issuer block. Its
   ``outcomeDerived`` block carries the status, completion time, the honest
   ``tier`` self-label (``integrity`` vs ``replay``), an optional output
   commitment, and the inference server's eval-stat counters in the clear.

As with SEP-2787, every signed block is a *closed* schema: parsing rejects
any field outside the modeled set so the model-derived preimage stays
byte-exact to the wire. Extending a block is an explicit version bump.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._receipt_types import (
    BackLink,
    ReceiptAsserted,
    back_link_from_dict,
    back_link_to_dict,
    receipt_asserted_from_dict,
    receipt_asserted_to_dict,
)
from vaara.attestation._attest_types import (
    VALID_ALGS,
    Algorithm,
    ArgsCommitment,
    AttestationError,
    IssuerAsserted,
    args_from_dict,
    args_to_dict,
    issuer_from_dict,
    issuer_to_dict,
)

# An inference outcome is durable: it completed, refused, or errored. Distinct
# vocabulary from the tool-call receipt's "executed" because a model call
# reads naturally as "completed".
INFER_VALID_STATUSES: frozenset[str] = frozenset(
    {"completed", "refused", "errored"}
)
# Tier is the honest self-label, inside the signed preimage so a downgrade is
# detectable: "integrity" binds model+input+output with no determinism claim;
# "replay" additionally claims byte-reproducibility (temp=0 + seed).
INFER_VALID_TIERS: frozenset[str] = frozenset({"integrity", "replay"})


def _reject_unknown_keys(
    d: dict[str, Any], allowed: frozenset[str], where: str
) -> None:
    """Fail closed on any key outside the closed schema for a signed block.

    Same discipline as ``_receipt_types._reject_unknown_keys``: the signature
    covers the JCS encoding of the modeled fields, so a silently-dropped key
    would make a model-derived preimage disagree with a byte-exact verifier.
    """
    extra = set(d) - allowed
    if extra:
        raise AttestationError(
            f"{where} carries unrecognized field(s) {sorted(extra)!r}; "
            "the signed schema is closed"
        )


@dataclass(frozen=True)
class RequestDeclared:
    """What the harness asked the model for.

    ``intent`` is a human-meaningful label (``inference/chat/<model>``).
    ``request_commitment`` is a commitment over the canonical request object
    (messages + normalized sampling params), reusing the SEP-2787
    argument-commitment shapes so a verifier can later re-derive the same
    digest from the runtime request and reject on mismatch.
    """

    intent: str
    request_commitment: ArgsCommitment


@dataclass(frozen=True)
class ModelDerived:
    """Facts the proxy derived about the served model at call time.

    ``manifest_digest`` pins the exact model the inference server resolved the
    request to (from ollama ``/api/show``); ``gguf_metadata_hash`` pins the
    weights' metadata block. Together they answer "was it the model it
    claimed". ``quantization`` and ``param_count`` are optional human labels.
    """

    model_ref: str
    manifest_digest: str
    gguf_metadata_hash: str
    quantization: Optional[str] = None
    param_count: Optional[str] = None


@dataclass(frozen=True)
class InferenceOutcome:
    """Facts about what the model call returned.

    ``status`` is one of ``completed`` / ``refused`` / ``errored``.
    ``output_commitment`` binds the response payload (absent on a refusal,
    which has no output). ``eval_stats`` carries the inference server's
    integer counters (token counts, durations in ns) in the clear; floats are
    rejected so the signed preimage cannot drift across stacks. ``tier`` names
    the strength of the claim being made.
    """

    status: str
    completed_at: str
    tier: str
    output_commitment: Optional[ArgsCommitment] = None
    eval_stats: Optional[dict[str, int]] = None


@dataclass(frozen=True)
class InferenceAttestation:
    """Pre-call inference attestation envelope.

    Three trust-surface blocks plus the signature. The signature is computed
    over the JCS-canonical encoding of ``{version, alg, requestDeclared,
    issuerAsserted, modelDerived}`` and does not cover itself.
    """

    version: int
    alg: Algorithm
    request_declared: RequestDeclared
    issuer_asserted: IssuerAsserted
    model_derived: ModelDerived
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "alg": self.alg,
            "requestDeclared": request_declared_to_dict(self.request_declared),
            "issuerAsserted": issuer_to_dict(self.issuer_asserted),
            "modelDerived": model_derived_to_dict(self.model_derived),
            "signature": self.signature,
        }


@dataclass(frozen=True)
class InferenceReceipt:
    """Post-call inference receipt envelope.

    ``backLink`` (reused from the execution receipt) pins the exact pre-call
    attestation; two more blocks plus the signature. The signature is computed
    over the JCS-canonical encoding of ``{version, alg, backLink,
    outcomeDerived, receiptAsserted}`` and does not cover itself.
    """

    version: int
    alg: Algorithm
    back_link: BackLink
    receipt_asserted: ReceiptAsserted
    outcome_derived: InferenceOutcome
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "alg": self.alg,
            "backLink": back_link_to_dict(self.back_link),
            "outcomeDerived": inference_outcome_to_dict(self.outcome_derived),
            "receiptAsserted": receipt_asserted_to_dict(self.receipt_asserted),
            "signature": self.signature,
        }


# --- block serializers ------------------------------------------------------


def request_declared_to_dict(rd: RequestDeclared) -> dict[str, Any]:
    return {
        "intent": rd.intent,
        "requestCommitment": args_to_dict(rd.request_commitment),
    }


def model_derived_to_dict(md: ModelDerived) -> dict[str, Any]:
    out: dict[str, Any] = {
        "modelRef": md.model_ref,
        "manifestDigest": md.manifest_digest,
        "ggufMetadataHash": md.gguf_metadata_hash,
    }
    if md.quantization is not None:
        out["quantization"] = md.quantization
    if md.param_count is not None:
        out["paramCount"] = md.param_count
    return out


def inference_outcome_to_dict(od: InferenceOutcome) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": od.status,
        "completedAt": od.completed_at,
        "tier": od.tier,
    }
    if od.output_commitment is not None:
        out["outputCommitment"] = args_to_dict(od.output_commitment)
    if od.eval_stats is not None:
        out["evalStats"] = dict(od.eval_stats)
    return out


# --- closed key sets --------------------------------------------------------

_REQUEST_DECLARED_KEYS = frozenset({"intent", "requestCommitment"})
_MODEL_DERIVED_KEYS = frozenset(
    {"modelRef", "manifestDigest", "ggufMetadataHash", "quantization", "paramCount"}
)
_OUTCOME_KEYS = frozenset(
    {"status", "completedAt", "tier", "outputCommitment", "evalStats"}
)
_ATTESTATION_KEYS = frozenset(
    {"version", "alg", "requestDeclared", "issuerAsserted", "modelDerived", "signature"}
)
_RECEIPT_KEYS = frozenset(
    {"version", "alg", "backLink", "outcomeDerived", "receiptAsserted", "signature"}
)


# --- block deserializers ----------------------------------------------------


def request_declared_from_dict(d: dict[str, Any]) -> RequestDeclared:
    _reject_unknown_keys(d, _REQUEST_DECLARED_KEYS, "requestDeclared")
    for required in ("intent", "requestCommitment"):
        if required not in d:
            raise AttestationError(
                f"requestDeclared missing required field {required!r}"
            )
    if not isinstance(d["intent"], str) or not d["intent"].strip():
        raise AttestationError("requestDeclared.intent MUST be a non-empty string")
    return RequestDeclared(
        intent=d["intent"],
        request_commitment=args_from_dict(d["requestCommitment"]),
    )


def model_derived_from_dict(d: dict[str, Any]) -> ModelDerived:
    _reject_unknown_keys(d, _MODEL_DERIVED_KEYS, "modelDerived")
    for required in ("modelRef", "manifestDigest", "ggufMetadataHash"):
        if required not in d:
            raise AttestationError(
                f"modelDerived missing required field {required!r}"
            )
    for digest_field in ("manifestDigest", "ggufMetadataHash"):
        if not str(d[digest_field]).startswith("sha256:"):
            raise AttestationError(
                f"modelDerived.{digest_field} MUST be a 'sha256:' digest"
            )
    return ModelDerived(
        model_ref=d["modelRef"],
        manifest_digest=d["manifestDigest"],
        gguf_metadata_hash=d["ggufMetadataHash"],
        quantization=d.get("quantization"),
        param_count=d.get("paramCount"),
    )


def inference_outcome_from_dict(d: dict[str, Any]) -> InferenceOutcome:
    _reject_unknown_keys(d, _OUTCOME_KEYS, "outcomeDerived")
    for required in ("status", "completedAt", "tier"):
        if required not in d:
            raise AttestationError(
                f"outcomeDerived missing required field {required!r}"
            )
    if d["status"] not in INFER_VALID_STATUSES:
        raise AttestationError(f"invalid status {d['status']!r}")
    if d["tier"] not in INFER_VALID_TIERS:
        raise AttestationError(f"invalid tier {d['tier']!r}")
    commitment = (
        args_from_dict(d["outputCommitment"])
        if "outputCommitment" in d
        else None
    )
    eval_stats = d.get("evalStats")
    if eval_stats is not None:
        if not isinstance(eval_stats, dict):
            raise AttestationError("outcomeDerived.evalStats must be an object")
        for k, v in eval_stats.items():
            # bool is an int subclass; exclude it so True/False can't masquerade
            # as a counter and drift the preimage.
            if not isinstance(v, int) or isinstance(v, bool):
                raise AttestationError(
                    f"outcomeDerived.evalStats[{k!r}] must be an integer; "
                    "floats and bools are rejected to keep the preimage stable"
                )
        eval_stats = dict(eval_stats)
    return InferenceOutcome(
        status=d["status"],
        completed_at=d["completedAt"],
        tier=d["tier"],
        output_commitment=commitment,
        eval_stats=eval_stats,
    )


def inference_attestation_from_dict(d: dict[str, Any]) -> InferenceAttestation:
    """Reconstruct an InferenceAttestation from its wire JSON dict.

    Inverse of ``InferenceAttestation.to_dict()``. Field-presence validation
    only; signature verification still requires the caller's keying material.
    """
    _reject_unknown_keys(d, _ATTESTATION_KEYS, "inferenceAttestation")
    for required in (
        "version", "alg", "requestDeclared", "issuerAsserted",
        "modelDerived", "signature",
    ):
        if required not in d:
            raise AttestationError(
                f"inferenceAttestation missing required field {required!r}"
            )
    if d["alg"] not in VALID_ALGS:
        raise AttestationError(f"unsupported alg {d['alg']!r}")
    return InferenceAttestation(
        version=d["version"],
        alg=d["alg"],
        request_declared=request_declared_from_dict(d["requestDeclared"]),
        issuer_asserted=issuer_from_dict(d["issuerAsserted"]),
        model_derived=model_derived_from_dict(d["modelDerived"]),
        signature=d["signature"],
    )


def inference_receipt_from_dict(d: dict[str, Any]) -> InferenceReceipt:
    """Reconstruct an InferenceReceipt from its wire JSON dict.

    Inverse of ``InferenceReceipt.to_dict()``. Field-presence validation
    only; signature verification still requires the caller's keying material.
    """
    _reject_unknown_keys(d, _RECEIPT_KEYS, "inferenceReceipt")
    for required in (
        "version", "alg", "backLink", "outcomeDerived",
        "receiptAsserted", "signature",
    ):
        if required not in d:
            raise AttestationError(
                f"inferenceReceipt missing required field {required!r}"
            )
    if d["alg"] not in VALID_ALGS:
        raise AttestationError(f"unsupported alg {d['alg']!r}")
    return InferenceReceipt(
        version=d["version"],
        alg=d["alg"],
        back_link=back_link_from_dict(d["backLink"]),
        receipt_asserted=receipt_asserted_from_dict(d["receiptAsserted"]),
        outcome_derived=inference_outcome_from_dict(d["outcomeDerived"]),
        signature=d["signature"],
    )
