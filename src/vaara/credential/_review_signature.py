# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reviewer-signed review resolutions.

Internal module. Public surface is in ``vaara.credential``.

An ``ESCALATION_RESOLVED`` row names its reviewer as free text and is signed
only by the trail key, so it says the operator recorded that someone approved.
This module lets the reviewer sign the resolution themselves, and lets a third
party check which person it was without trusting the operator.

The reviewer signs the JCS encoding of::

    {"type": "vaara.review-resolution/v1", "actionId": "...",
     "decisionRecordHash": "...", "resolution": "allow", "reviewer": "did:...",
     "justification": "..."}

``decisionRecordHash`` is the ``record_hash`` of the last ``DECISION_MADE``
record for the action, so the signature covers the exact decision it
resolves. The row carries a closed block::

    "reviewer_signature": {"type": "vaara.review-resolution/v1", "alg": "ES256",
                           "decisionRecordHash": "...", "signature": "..."}

The block never carries a key. The verifier takes the key from the
``reviewers`` the principal pinned in a signed ``vaara.permission/v1``.

Each resolution of a call under the permission lands in one state:

* ``signed``: the signature verifies under the pinned key of the named reviewer
  and binds this action's decision record.
* ``unsigned``: no signature. A finding only when the permission names reviewers.
* ``unpinned``: signed, but the permission pins no reviewer keys, so nobody can
  say whose key it was. Not a finding, and not ``signed``.
* ``unknown_reviewer``: signed by a reviewer the permission does not name.
* ``bad_signature``: the signature does not verify under the pinned key.
* ``decision_mismatch``: the signature names another decision record, or the
  trail holds no decision for the action.
* ``malformed``: the block is not the closed shape above.

``override`` is true when the reviewer allowed what the decision denied or
escalated. It is scoped to that one action and never counted as a policy allow.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from vaara.attestation._attest_canonical import canonical_json
from vaara.attestation._attest_signing import (
    sign_es256,
    sign_rs256,
    verify_es256,
    verify_rs256,
)
from vaara.attestation._attest_types import AttestationError

REVIEW_RESOLUTION_TYPE = "vaara.review-resolution/v1"
SIGNATURE_KEYS = frozenset({"type", "alg", "decisionRecordHash", "signature"})

#: States that keep ``ok`` true whether or not the permission names reviewers.
_CLEAN_STATES = frozenset({"signed", "unpinned"})


def resolution_payload(
    *,
    action_id: str,
    decision_record_hash: str,
    resolution: str,
    reviewer: str,
    justification: str,
) -> bytes:
    """The exact bytes a reviewer's signature covers."""
    return canonical_json({
        "type": REVIEW_RESOLUTION_TYPE,
        "actionId": action_id,
        "decisionRecordHash": decision_record_hash,
        "resolution": resolution,
        "reviewer": reviewer,
        "justification": justification,
    })


def decision_record_for(records: list) -> Optional[Any]:
    """The last ``DECISION_MADE`` record in ``records``, or None."""
    from vaara.audit.trail import EventType

    found = None
    for rec in records:
        if rec.event_type == EventType.DECISION_MADE:
            found = rec
    return found


def sign_resolution(
    *,
    action_id: str,
    decision_record_hash: str,
    resolution: str,
    reviewer: str,
    justification: str,
    private_key: Any,
) -> dict[str, str]:
    """Sign a resolution with the reviewer's own key (EC P-256 or RSA).

    Returns the block the ``ESCALATION_RESOLVED`` row carries.
    """
    from cryptography.hazmat.primitives.asymmetric import ec, rsa

    payload = resolution_payload(
        action_id=action_id, decision_record_hash=decision_record_hash,
        resolution=resolution, reviewer=reviewer, justification=justification,
    )
    if isinstance(private_key, ec.EllipticCurvePrivateKey):
        alg, sig = "ES256", sign_es256(payload, private_key=private_key)
    elif isinstance(private_key, rsa.RSAPrivateKey):
        alg, sig = "RS256", sign_rs256(payload, private_key=private_key)
    else:
        raise AttestationError("a reviewer signs with an EC P-256 or RSA private key")
    return {
        "type": REVIEW_RESOLUTION_TYPE,
        "alg": alg,
        "decisionRecordHash": decision_record_hash,
        "signature": sig,
    }


@dataclass(frozen=True)
class ReviewedResolution:
    """One resolution of a call under the permission, and where it landed."""

    action_id: str
    reviewer: str
    resolution: str
    decision: str
    state: str
    override: bool


@dataclass(frozen=True)
class ReviewReport:
    """The verdict over every resolution in scope. ``ok`` is false on any finding."""

    reviewers_named: bool
    reviews: tuple[ReviewedResolution, ...] = ()
    chain_problem: str = ""
    grant_problem: str = ""
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        if self.chain_problem or self.grant_problem:
            return False
        clean = _CLEAN_STATES if self.reviewers_named else _CLEAN_STATES | {"unsigned"}
        return all(r.state in clean for r in self.reviews)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reviewersNamed": self.reviewers_named,
            "chainProblem": self.chain_problem,
            "grantProblem": self.grant_problem,
            "counts": dict(self.counts),
            "reviews": [
                {
                    "actionId": r.action_id,
                    "reviewer": r.reviewer,
                    "resolution": r.resolution,
                    "decision": r.decision,
                    "state": r.state,
                    "override": r.override,
                }
                for r in self.reviews
            ],
        }


def _judge(row: Any, decision_rec: Optional[Any], pinned: dict[str, Any]) -> str:
    data = row.data if isinstance(row.data, dict) else {}
    block = data.get("reviewer_signature")
    if block is None:
        return "unsigned"
    if (not isinstance(block, dict) or set(block) != SIGNATURE_KEYS
            or block.get("type") != REVIEW_RESOLUTION_TYPE
            or not all(isinstance(block[k], str) for k in SIGNATURE_KEYS)):
        return "malformed"
    if not pinned:
        return "unpinned"
    reviewer = data.get("reviewer")
    entry = pinned.get(reviewer) if isinstance(reviewer, str) else None
    if entry is None:
        return "unknown_reviewer"
    if decision_rec is None or block["decisionRecordHash"] != decision_rec.record_hash:
        return "decision_mismatch"
    if block["alg"] != entry.alg:
        return "bad_signature"
    try:
        key = entry.load_public_key()
    except AttestationError:
        return "bad_signature"
    resolution, justification = data.get("resolution"), data.get("justification", "")
    if not isinstance(resolution, str) or not isinstance(justification, str):
        return "malformed"
    payload = resolution_payload(
        action_id=row.action_id, decision_record_hash=block["decisionRecordHash"],
        resolution=resolution, reviewer=reviewer, justification=justification,
    )
    verify = verify_es256 if entry.alg == "ES256" else verify_rs256
    if not verify(payload, signature_hex=block["signature"], public_key=key):
        return "bad_signature"
    return "signed"


def check_review_resolutions(
    trail: Any,
    p: Any,
    *,
    verifying_material: Any,
    verify_chain: bool = True,
) -> ReviewReport:
    """Check every resolution of a call under the permission, offline.

    In scope: ``ESCALATION_RESOLVED`` rows whose action was requested by the
    permission's agent, for its tool and tenant. The chain and the principal's
    signature are checked first; if either fails, nothing is judged, because
    the pinned reviewer keys come from that signature.
    """
    from vaara.audit.trail import EventType
    from vaara.credential._permission import verify_permission_signature

    named = bool(p.reviewers)
    if verify_chain:
        broken = trail.verify_chain()
        if broken:
            return ReviewReport(named, chain_problem=str(broken))
    if not verify_permission_signature(p, verifying_material=verifying_material):
        return ReviewReport(named, grant_problem="bad_signature")

    pinned = {r.reviewer: r for r in p.reviewers}
    reviews: list[ReviewedResolution] = []
    for rec in list(trail._records):
        if rec.event_type != EventType.ACTION_REQUESTED:
            continue
        if (rec.agent_id != p.agent or rec.tool_name != p.tool_name
                or (rec.tenant_id or "") != p.tenant_id):
            continue
        records = trail.get_action_trail(rec.action_id)
        for i, row in enumerate(records):
            if row.event_type != EventType.ESCALATION_RESOLVED:
                continue
            decision_rec = decision_record_for(records[:i])
            decision = ""
            if decision_rec is not None and isinstance(decision_rec.data, dict):
                decision = str(decision_rec.data.get("decision") or "")
            data = row.data if isinstance(row.data, dict) else {}
            resolution = str(data.get("resolution") or "")
            reviews.append(ReviewedResolution(
                action_id=row.action_id,
                reviewer=str(data.get("reviewer") or ""),
                resolution=resolution,
                decision=decision,
                state=_judge(row, decision_rec, pinned),
                override=resolution == "allow" and decision in ("deny", "escalate"),
            ))
    return ReviewReport(named, reviews=tuple(reviews),
                        counts=dict(Counter(r.state for r in reviews)))
