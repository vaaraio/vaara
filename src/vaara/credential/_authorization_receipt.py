# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mint a signed, recomputable authorization receipt from a gateway verdict.

Internal module. Public surface is in ``vaara.credential``.

The gateway answers "may this call run?" with a :class:`GrantVerdict`. On its
own that verdict is ephemeral: it lives and dies inside the gateway process,
the same ceiling a kernel egress filter hits when it drops a packet and moves
on. This module turns the verdict into a durable, content-addressed,
independently recomputable artifact.

A refusal mints a receipt too, and that is the point. A denied call leaves a
portable proof of the non-action: a third party recomputes the verdict from the
grant and the runtime arguments and confirms the refusal, trusting only the
issuer's public key, not the producer. No kernel filter can hand an auditor
that.

The receipt is the canonical ``vaara.receipt/v1`` envelope (SPEC.md), assembled
and signed by :func:`emit_decision_record` unchanged, so it carries no new
crypto and verifies under the same stack as every other Vaara receipt. The
authorization verdict maps onto the envelope's verdict vocabulary
(``allow`` / ``block`` / ``escalate``): an allowed call is ``allow``; a refused
call is ``block``, carrying the :data:`GrantReason` literal
(``missing_credential``, ``capability_exceeded``, ``binding_unknown``, ...) as
the decision ``reason`` so the precise machine cause survives into the receipt.

The evidence behind the receipt is a ``vaara.authorization/v0`` record binding
the tool, the tenant, the exact signed grant (by content address), the runtime
argument commitment, and the verdict. The raw arguments never enter the record;
only their commitment does, so the receipt is publishable while the arguments
stay private. An auditor who holds the arguments out of band recomputes the
commitment and re-runs the verdict against the grant's capabilities.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal, Optional, cast

from vaara.attestation._attest_canonical import canonical_json, now_iso8601
from vaara.attestation.decision import (
    BackLink,
    DecisionDerived,
    DecisionRecord,
    EvidenceRef,
    emit_decision_record,
)
from vaara.credential._grant_capability import capability_to_dict
from vaara.credential._grant_types import BrokeredCredential
from vaara.credential._grant_verify import GrantVerdict

# Evidence-record schema id, registered as a downstream profile of
# vaara.receipt/v1 in SPEC.md Section 5.1.
AUTHORIZATION_SCHEMA = "vaara.authorization/v0"

# Default policy id stamped on the decision when the caller supplies none.
_DEFAULT_POLICY_ID = "policy:vaara-credential-broker/1"


def _digest(obj: Any) -> str:
    """Content address ``obj`` as ``sha256:`` over its JCS-canonical bytes."""
    return "sha256:" + hashlib.sha256(canonical_json(obj)).hexdigest()


def grant_fingerprint(credential: BrokeredCredential) -> str:
    """Content address of the exact signed grant (signature included).

    Pins the precise credential the decision rested on: a re-minted or
    tampered grant produces a different fingerprint, so the receipt cannot be
    replayed against a different authorization.
    """
    return _digest(credential.to_dict())


def args_commitment(runtime_args: Any) -> str:
    """Content address of the runtime arguments. Only the commitment travels.

    The raw arguments stay out of the receipt; this commitment is what binds.
    An auditor holding the arguments recomputes this and confirms the match.
    """
    return _digest(runtime_args)


def build_authorization_evidence(
    *,
    credential: BrokeredCredential,
    runtime_args: Any,
    verdict: GrantVerdict,
    coverage: Optional[dict[str, Any]] = None,
    completeness: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the ``vaara.authorization/v0`` evidence record for one decision.

    The record is self-describing and content-addressable: its JCS digest is
    what the receipt's ``evidenceRef.digest`` pins. ``verdict``/``reason`` carry
    the gateway's exact answer (``allow`` + ``ok`` on success, otherwise ``deny``
    plus the failing :data:`GrantReason`). The grant's capabilities are echoed
    so the record names the bounds the verdict was computed against without a
    reader having to dereference the grant.

    ``coverage``, when supplied, states the observation boundary the decision
    was made under: which capability surface was in scope, named in the trace
    itself. It is what lets a reader tell absence-as-fact from absence-as-silence,
    so a missing deny for an in-scope call reads as "not refused within a stated
    boundary" rather than "not observed". A recomputing auditor reads the verdict
    against the same declared scope. Omitted (the default) when no boundary is
    asserted, keeping the record byte-identical to a coverage-free decision.

    ``completeness``, when supplied, carries a monotonic per-boundary ``seq`` and
    the ``runningCount`` of receipts issued under that boundary up to and
    including this one. Both ride inside the content-addressed evidence, so they
    are pinned by ``evidenceRef.digest`` and travel under the receipt's
    signature. Because the issuer assigns ``seq`` gap-free by construction, a
    missing sequence number in an exported set is a provable gap, and the latest
    ``runningCount`` makes a short set self-evidently incomplete, with no
    external witness required. Omitted (the default) when no boundary asserts a
    sequence, keeping the record byte-identical to a completeness-free decision.
    """
    record: dict[str, Any] = {
        "schema": AUTHORIZATION_SCHEMA,
        "toolName": credential.scope.tool_name,
        "tenantId": credential.scope.tenant_id,
        "grantFingerprint": grant_fingerprint(credential),
        "argsCommitment": args_commitment(runtime_args),
        "verdict": "allow" if verdict.ok else "deny",
        "reason": verdict.reason,
    }
    if credential.capabilities:
        record["capabilities"] = [
            capability_to_dict(c) for c in credential.capabilities
        ]
    if coverage:
        record["coverage"] = coverage
    if completeness:
        record["completeness"] = completeness
    return record


@dataclass(frozen=True)
class ReceiptSigner:
    """The issuer identity and key a gateway mints authorization receipts under.

    Bundles the signing material with the asserted issuer claims so a gateway
    holds one opt-in object rather than five loose parameters. ``signing_material``
    is a bytes secret for HS256 or a private-key object for ES256 / RS256, the
    same key the broker signs grants and execution receipts with.
    """

    signing_material: Any
    iss: str
    sub: str
    secret_version: str
    alg: str = "ES256"


@dataclass(frozen=True)
class AuthorizationReceipt:
    """A minted authorization decision and the evidence it is addressed to.

    ``record`` is the signed ``vaara.receipt/v1`` envelope; ``evidence`` is the
    ``vaara.authorization/v0`` record whose JCS digest the envelope's
    ``evidenceRef.digest`` pins. Both must be persisted: the receipt is the
    signed verdict, the evidence is the bytes that make the verdict recomputable.
    """

    record: DecisionRecord
    evidence: dict[str, Any]


def mint_authorization_receipt(
    *,
    credential: BrokeredCredential,
    runtime_args: Any,
    verdict: GrantVerdict,
    iss: str,
    sub: str,
    secret_version: str,
    alg: str,
    signing_material: Any,
    decided_at: Optional[str] = None,
    nonce: Optional[str] = None,
    policy_id: Optional[str] = None,
    ref: Optional[str] = None,
    coverage: Optional[dict[str, Any]] = None,
    completeness: Optional[dict[str, Any]] = None,
) -> AuthorizationReceipt:
    """Mint a signed authorization receipt for one gateway verdict.

    Allowed and refused calls both mint. The receipt back-links to the same
    attestation the grant is bound to (``binding.attestationDigest``), so an
    auditor joins grant -> attestation -> authorization receipt -> execution
    receipt offline. ``signing_material`` is the issuer's signing key under
    ``alg`` (a bytes secret for HS256, a private-key object for ES256 / RS256),
    the same key used for the grant and the execution receipt.

    ``coverage`` states the observation boundary the decision was made under and
    is folded into the evidence record so the boundary is named in the trace and
    travels under the same signature. ``completeness`` carries the per-boundary
    ``seq`` and ``runningCount`` for the same purpose, making a dropped receipt
    inside the boundary a provable gap.
    """
    evidence = build_authorization_evidence(
        credential=credential,
        runtime_args=runtime_args,
        verdict=verdict,
        coverage=coverage,
        completeness=completeness,
    )
    evidence_ref = EvidenceRef(
        digest=_digest(evidence),
        canonicalization="JCS",
        schema=AUTHORIZATION_SCHEMA,
        ref=ref or f"vaara:authorization/{credential.scope.tool_name}",
    )
    decision_derived = DecisionDerived(
        decision="allow" if verdict.ok else "block",
        decided_at=decided_at or now_iso8601(),
        reason=verdict.reason,
        policy_id=policy_id or _DEFAULT_POLICY_ID,
        evidence_ref=evidence_ref,
    )
    back_link = BackLink(
        attestation_digest=credential.binding.attestation_digest,
        attestation_nonce=credential.binding.attestation_nonce,
    )
    record = emit_decision_record(
        back_link=back_link,
        decision_derived=decision_derived,
        iss=iss,
        sub=sub,
        secret_version=secret_version,
        alg=cast(Literal["HS256", "ES256", "RS256"], alg),
        signing_material=signing_material,
        nonce=nonce,
    )
    return AuthorizationReceipt(record=record, evidence=evidence)


def mint_for_signer(
    signer: ReceiptSigner,
    *,
    credential: BrokeredCredential,
    runtime_args: Any,
    verdict: GrantVerdict,
    decided_at: Optional[str] = None,
    nonce: Optional[str] = None,
    policy_id: Optional[str] = None,
    ref: Optional[str] = None,
    coverage: Optional[dict[str, Any]] = None,
    completeness: Optional[dict[str, Any]] = None,
) -> AuthorizationReceipt:
    """Mint an authorization receipt using a :class:`ReceiptSigner` bundle."""
    return mint_authorization_receipt(
        credential=credential,
        runtime_args=runtime_args,
        verdict=verdict,
        iss=signer.iss,
        sub=signer.sub,
        secret_version=signer.secret_version,
        alg=signer.alg,
        signing_material=signer.signing_material,
        decided_at=decided_at,
        nonce=nonce,
        policy_id=policy_id,
        ref=ref,
        coverage=coverage,
        completeness=completeness,
    )
