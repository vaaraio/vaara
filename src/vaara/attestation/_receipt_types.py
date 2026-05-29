"""Dataclasses and serialization for execution-receipt envelopes.

Internal module. Public surface is in ``vaara.attestation.receipt``.

The execution receipt is the post-execution sibling of the SEP-2787
request attestation. SEP-2787 binds an *observed* ``tools/call``
request before execution: issuer, subject, target, intent, nonce,
time, and an argument commitment. It deliberately says nothing about
whether the call ran or what came back. The receipt covers exactly
that deferred half: it binds the *outcome* of one attested request
and links back to the attestation it answers.

Three blocks plus the signature, mirroring the SEP-2787 trust-surface
layout so the two envelopes verify with the same canonicalization and
signing stack:

1. ``backLink`` joins the receipt to its request attestation. It
   carries the attestation's nonce (fast correlation) and a digest
   over the full SEP-2787 wire envelope including its signature, which
   pins the exact attestation instance the receipt answers.
2. ``receiptAsserted`` is the issuer block, set by whoever observed
   the outcome (the executing server, or an intermediary such as a
   governance proxy). Its signature is the proof the values were bound
   together at receipt time.
3. ``outcomeDerived`` carries the execution status, completion time,
   and an optional commitment over the result. The result commitment
   reuses the SEP-2787 argument-commitment shapes (``ArgsRef`` /
   ``ArgsProjection``) verbatim: structurally they are a commitment
   over a JSON value, which is what a result commitment also is.

A receipt is a durable record, not a time-bounded capability, so it
carries no ``exp`` and the verifier enforces no TTL. This is the same
pre-execution-capability versus post-execution-record distinction the
SEP-2787 thread keeps drawing: the attestation can expire, the record
of what happened does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from vaara.attestation._sep2787_types import (
    VALID_ALGS,
    Algorithm,
    ArgsCommitment,
    AttestationError,
    args_from_dict,
    args_to_dict,
)

# Result commitments are structurally identical to argument commitments:
# a commitment over a JSON-serialisable value. Reuse the shapes rather
# than duplicate them.
ResultCommitment = ArgsCommitment

ReceiptStatus = Literal["executed", "refused", "errored"]
VALID_STATUSES: frozenset[str] = frozenset({"executed", "refused", "errored"})


@dataclass(frozen=True)
class BackLink:
    """Join from a receipt to the SEP-2787 attestation it answers.

    ``attestation_nonce`` echoes the attestation's
    ``issuerAsserted.nonce`` for fast correlation. ``attestation_digest``
    is ``sha256:<hex>`` over the JCS-canonical encoding of the full
    attestation wire envelope (signature included), pinning the exact
    attestation instance.
    """

    attestation_digest: str
    attestation_nonce: str


@dataclass(frozen=True)
class ReceiptAsserted:
    """Issuer block: what the receipt issuer binds at receipt time.

    Set by the party that observed the outcome (executing server or an
    intermediary). The signature over the envelope is the proof these
    values were bound together.
    """

    iss: str
    sub: str
    iat: str
    nonce: str
    secret_version: str
    alg: Algorithm


@dataclass(frozen=True)
class OutcomeDerived:
    """Facts about what happened to the attested request.

    ``status`` is one of ``executed`` / ``refused`` / ``errored``.
    ``result_commitment`` is a commitment over the result payload and
    is optional: a refused call has no result, so the commitment is
    absent. An executed or errored call commits to the result or the
    error object respectively.
    """

    status: ReceiptStatus
    completed_at: str
    result_commitment: Optional[ResultCommitment] = None


@dataclass(frozen=True)
class ExecutionReceipt:
    """Execution-receipt envelope.

    ``backLink`` plus two trust-surface blocks plus the signature. The
    signature is computed over the JCS-canonical encoding of
    ``{version, alg, backLink, outcomeDerived, receiptAsserted}`` and
    does not cover itself.
    """

    version: int
    alg: Algorithm
    back_link: BackLink
    receipt_asserted: ReceiptAsserted
    outcome_derived: OutcomeDerived
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "alg": self.alg,
            "backLink": back_link_to_dict(self.back_link),
            "outcomeDerived": outcome_to_dict(self.outcome_derived),
            "receiptAsserted": receipt_asserted_to_dict(self.receipt_asserted),
            "signature": self.signature,
        }


def back_link_to_dict(bl: BackLink) -> dict[str, Any]:
    return {
        "attestationDigest": bl.attestation_digest,
        "attestationNonce": bl.attestation_nonce,
    }


def receipt_asserted_to_dict(ra: ReceiptAsserted) -> dict[str, Any]:
    return {
        "alg": ra.alg,
        "iat": ra.iat,
        "iss": ra.iss,
        "nonce": ra.nonce,
        "secretVersion": ra.secret_version,
        "sub": ra.sub,
    }


def outcome_to_dict(od: OutcomeDerived) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": od.status,
        "completedAt": od.completed_at,
    }
    if od.result_commitment is not None:
        out["resultCommitment"] = args_to_dict(od.result_commitment)
    return out


def back_link_from_dict(d: dict[str, Any]) -> BackLink:
    for required in ("attestationDigest", "attestationNonce"):
        if required not in d:
            raise AttestationError(f"backLink missing required field {required!r}")
    return BackLink(
        attestation_digest=d["attestationDigest"],
        attestation_nonce=d["attestationNonce"],
    )


def receipt_asserted_from_dict(d: dict[str, Any]) -> ReceiptAsserted:
    for required in ("alg", "iat", "iss", "nonce", "secretVersion", "sub"):
        if required not in d:
            raise AttestationError(
                f"receiptAsserted missing required field {required!r}"
            )
    if d["alg"] not in VALID_ALGS:
        raise AttestationError(f"unsupported alg {d['alg']!r}")
    return ReceiptAsserted(
        alg=d["alg"],
        iat=d["iat"],
        iss=d["iss"],
        nonce=d["nonce"],
        secret_version=d["secretVersion"],
        sub=d["sub"],
    )


def outcome_from_dict(d: dict[str, Any]) -> OutcomeDerived:
    for required in ("status", "completedAt"):
        if required not in d:
            raise AttestationError(
                f"outcomeDerived missing required field {required!r}"
            )
    if d["status"] not in VALID_STATUSES:
        raise AttestationError(f"invalid status {d['status']!r}")
    commitment = (
        args_from_dict(d["resultCommitment"])
        if "resultCommitment" in d
        else None
    )
    return OutcomeDerived(
        status=d["status"],
        completed_at=d["completedAt"],
        result_commitment=commitment,
    )


def receipt_from_dict(d: dict[str, Any]) -> ExecutionReceipt:
    """Reconstruct an ExecutionReceipt from its wire JSON dict.

    Inverse of ``ExecutionReceipt.to_dict()``. Field-presence
    validation only; signature verification still requires the
    caller's keying material.
    """
    for required in (
        "version", "alg", "backLink", "outcomeDerived",
        "receiptAsserted", "signature",
    ):
        if required not in d:
            raise AttestationError(f"receipt missing required field {required!r}")
    if d["alg"] not in VALID_ALGS:
        raise AttestationError(f"unsupported alg {d['alg']!r}")
    return ExecutionReceipt(
        version=d["version"],
        alg=d["alg"],
        back_link=back_link_from_dict(d["backLink"]),
        receipt_asserted=receipt_asserted_from_dict(d["receiptAsserted"]),
        outcome_derived=outcome_from_dict(d["outcomeDerived"]),
        signature=d["signature"],
    )
