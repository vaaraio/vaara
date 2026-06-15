"""Dataclasses and serialization for decision-record envelopes.

Internal module. Public surface is in ``vaara.attestation.decision``.

The decision record is the pre-execution sibling of the SEP-2787 request
attestation and the post-execution sibling is the execution receipt. The
attestation binds an *observed* ``tools/call`` request before execution.
The receipt binds the *outcome* after execution. The decision record
covers the half between them: the governing server's policy verdict and
its risk basis, signed and committed *before* the side effect runs, so a
verifier can prove the verdict was fixed before the action.

Three blocks plus the signature, mirroring the SEP-2787 and receipt
trust-surface layout so all three envelopes verify with the same
canonicalization and signing stack:

1. ``backLink`` joins the decision to the SEP-2787 attestation it
   governs. Same shape as the receipt back-link: the attestation's
   nonce plus a digest over the full attestation wire bytes (signature
   included), which pins the exact attestation instance.
2. ``issuerAsserted`` is the governing server's issuer block. It carries
   the same fields as the receipt's ``receiptAsserted`` block; the wire
   key differs because the decision and the outcome are distinct records.
3. ``decisionDerived`` carries the verdict (``allow`` / ``block`` /
   ``escalate``), the risk basis that drove it, and the decision time.

A decision record is a durable record, not a time-bounded capability, so
it carries no ``exp`` and the verifier enforces no TTL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from vaara.attestation._receipt_types import (
    BackLink,
    ReceiptAsserted,
    _reject_unknown_keys,
    back_link_from_dict,
    back_link_to_dict,
    receipt_asserted_from_dict,
    receipt_asserted_to_dict,
)
from vaara.attestation._sep2787_types import VALID_ALGS, Algorithm, AttestationError

# The governing server's issuer block is identical in shape to the
# receipt's ``receiptAsserted`` (iss, sub, iat, nonce, secretVersion,
# alg). Reuse the dataclass; only the wire key differs (``issuerAsserted``).
IssuerAsserted = ReceiptAsserted

DecisionVerdict = Literal["allow", "block", "escalate"]
VALID_VERDICTS: frozenset[str] = frozenset({"allow", "block", "escalate"})


@dataclass(frozen=True)
class EvidenceRef:
    """A content-addressed reference from the decision basis to external evidence.

    The decision record carries the verdict and its risk basis but does
    not fix what *produced* that basis. ``EvidenceRef`` is the slot that
    names the external evidence by content address rather than re-describing
    it: a detector (e.g. a runtime tool-drift gateway) emits its own record,
    that record gets a content address, and the decision's basis cites the
    address under a ``policyId``. The two records stay independent, bound by
    hash, and a third party who trusts neither side can recompute the address
    from the referenced bytes and confirm the decision rested on that exact
    evidence.

    ``digest`` is ``sha256:<hex>`` over the canonical bytes of the referenced
    evidence object, matching the digest convention used by ``backLink`` and
    ``outcomeDerived.decisionDigest``. ``canonicalization`` names how those
    bytes were canonicalized before hashing (e.g. ``"JCS"``), so an
    independent implementation reproduces the same address. ``schema`` names
    the referenced object's shape and version (e.g.
    ``"interlock.drift-record/v0"``), so the verifier knows how to interpret
    it. ``ref`` is an optional, non-authoritative locator (URI or path) for
    fetching the bytes; the ``digest`` is what binds, so the bytes may also
    travel out of band. Sitting inside the signed ``decisionDerived`` block,
    the reference is covered by the decision signature: a swapped or stripped
    citation breaks verification.
    """

    digest: str
    canonicalization: str
    schema: str
    ref: Optional[str] = None


@dataclass(frozen=True)
class DecisionDerived:
    """The governing server's verdict and the basis for it.

    ``decision`` is one of ``allow`` / ``block`` / ``escalate``.
    ``decided_at`` is the ISO 8601 UTC decision time. The risk-basis
    fields are optional and, when present, are decimal strings: floats
    are banned on the wire (the JCS boundary rejects them) because
    cross-stack float behaviour is the most common source of signature
    drift. ``client_turn_id``, when present, records that the client
    *claimed* a turn id (SEP-2817 correlation), not that the server
    vouches for it. ``evidence_ref``, when present, is a content-addressed
    pointer to the external evidence the basis rests on (see ``EvidenceRef``).
    """

    decision: DecisionVerdict
    decided_at: str
    reason: Optional[str] = None
    risk_score: Optional[str] = None
    threshold_allow: Optional[str] = None
    threshold_block: Optional[str] = None
    policy_id: Optional[str] = None
    client_turn_id: Optional[str] = None
    evidence_ref: Optional[EvidenceRef] = None


@dataclass(frozen=True)
class DecisionRecord:
    """Decision-record envelope.

    ``backLink`` plus two trust-surface blocks plus the signature. The
    signature is computed over the JCS-canonical encoding of
    ``{version, alg, backLink, decisionDerived, issuerAsserted}`` and
    does not cover itself.
    """

    version: int
    alg: Algorithm
    back_link: BackLink
    decision_derived: DecisionDerived
    issuer_asserted: IssuerAsserted
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "alg": self.alg,
            "backLink": back_link_to_dict(self.back_link),
            "decisionDerived": decision_to_dict(self.decision_derived),
            "issuerAsserted": receipt_asserted_to_dict(self.issuer_asserted),
            "signature": self.signature,
        }


_EVIDENCE_REF_KEYS = frozenset({"digest", "canonicalization", "schema", "ref"})


def evidence_ref_to_dict(er: EvidenceRef) -> dict[str, Any]:
    out: dict[str, Any] = {
        "digest": er.digest,
        "canonicalization": er.canonicalization,
        "schema": er.schema,
    }
    if er.ref is not None:
        out["ref"] = er.ref
    return out


def evidence_ref_from_dict(d: dict[str, Any]) -> EvidenceRef:
    _reject_unknown_keys(d, _EVIDENCE_REF_KEYS, "evidenceRef")
    for required in ("digest", "canonicalization", "schema"):
        value = d.get(required)
        if not isinstance(value, str) or not value:
            raise AttestationError(
                f"evidenceRef.{required} must be a non-empty string"
            )
    if not d["digest"].startswith("sha256:"):
        raise AttestationError("evidenceRef.digest MUST be a 'sha256:' digest")
    ref = d.get("ref")
    if ref is not None and (not isinstance(ref, str) or not ref):
        raise AttestationError("evidenceRef.ref must be a non-empty string or absent")
    return EvidenceRef(
        digest=d["digest"],
        canonicalization=d["canonicalization"],
        schema=d["schema"],
        ref=ref,
    )


def decision_to_dict(dd: DecisionDerived) -> dict[str, Any]:
    out: dict[str, Any] = {
        "decision": dd.decision,
        "decidedAt": dd.decided_at,
    }
    if dd.reason is not None:
        out["reason"] = dd.reason
    if dd.risk_score is not None:
        out["riskScore"] = dd.risk_score
    if dd.threshold_allow is not None:
        out["thresholdAllow"] = dd.threshold_allow
    if dd.threshold_block is not None:
        out["thresholdBlock"] = dd.threshold_block
    if dd.policy_id is not None:
        out["policyId"] = dd.policy_id
    if dd.client_turn_id is not None:
        out["clientTurnId"] = dd.client_turn_id
    if dd.evidence_ref is not None:
        out["evidenceRef"] = evidence_ref_to_dict(dd.evidence_ref)
    return out


def decision_from_dict(d: dict[str, Any]) -> DecisionDerived:
    for required in ("decision", "decidedAt"):
        if required not in d:
            raise AttestationError(
                f"decisionDerived missing required field {required!r}"
            )
    if d["decision"] not in VALID_VERDICTS:
        raise AttestationError(f"invalid decision verdict {d['decision']!r}")
    evidence_ref = (
        evidence_ref_from_dict(d["evidenceRef"]) if "evidenceRef" in d else None
    )
    return DecisionDerived(
        decision=d["decision"],
        decided_at=d["decidedAt"],
        reason=d.get("reason"),
        risk_score=d.get("riskScore"),
        threshold_allow=d.get("thresholdAllow"),
        threshold_block=d.get("thresholdBlock"),
        policy_id=d.get("policyId"),
        client_turn_id=d.get("clientTurnId"),
        evidence_ref=evidence_ref,
    )


def decision_record_from_dict(d: dict[str, Any]) -> DecisionRecord:
    """Reconstruct a DecisionRecord from its wire JSON dict.

    Inverse of ``DecisionRecord.to_dict()``. Field-presence validation
    only; signature verification still requires the caller's keying
    material.
    """
    for required in (
        "version", "alg", "backLink", "decisionDerived",
        "issuerAsserted", "signature",
    ):
        if required not in d:
            raise AttestationError(f"decision record missing required field {required!r}")
    if d["alg"] not in VALID_ALGS:
        raise AttestationError(f"unsupported alg {d['alg']!r}")
    return DecisionRecord(
        version=d["version"],
        alg=d["alg"],
        back_link=back_link_from_dict(d["backLink"]),
        decision_derived=decision_from_dict(d["decisionDerived"]),
        issuer_asserted=receipt_asserted_from_dict(d["issuerAsserted"]),
        signature=d["signature"],
    )
