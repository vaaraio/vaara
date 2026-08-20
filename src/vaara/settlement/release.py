# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Release conditions: the receipt gates the payment.

A release condition (``vaara.release-condition/v0``) is a signed,
content-addressed statement made by whoever holds money: here is what is held,
here is exactly what must be proved before it moves, and here is when the offer
closes. The proof it names is an ordinary Vaara receipt. Nothing in this module
holds a key belonging to a payer, signs a transaction, talks to a chain, or
moves a cent. It answers one question about bytes, and the answer is what a
settlement agent acts on.

That placement is the point: it puts a verifier in the settlement path while
holding nobody's funds and nobody's keys.

## Four states, and why they are four

:func:`evaluate` returns one of ``RELEASED``, ``HELD``, ``EXPIRED`` or
``REFUSED``, each carrying a reason drawn from a closed set. The reason space is
partitioned by :data:`REASON_STATE`, so a reason belongs to exactly one state and
the two negatives can never collapse into each other.

They must not collapse, because a verifier that proved nothing must never read as
green, and must never read as the same false as a genuine failure. ``HELD``
because no receipt has arrived and ``REFUSED`` because a receipt was tampered
with are different facts about the world. Answering both with one boolean throws
away the difference between "not yet" and "no".

The partition runs on one axis, asked in order:

- Is the artifact *sound*? A broken condition signature, a receipt signed under a
  key the condition does not pin, a broken receipt signature, or evidence that
  does not resolve to the digest the receipt signed: the presented evidence fails
  as evidence. ``REFUSED``.
- Has the window closed? ``EXPIRED``. It names the clock and accuses nobody.
- Is the sound evidence *sufficient*? A missing receipt, a receipt for another
  action, another authorization, another issuer, or one that proves a refusal
  rather than an allow: the evidence is fine and does not satisfy the condition.
  ``HELD``, and the money stays where it is.
- Everything holds: ``RELEASED``.

Soundness is asked before the clock so an expired window cannot swallow a
tampering finding, and the clock is asked before sufficiency so a closed window
is reported as the reason the money is not moving.

## What it costs a reader

Nothing beyond what the rest of the tree already costs. Canonicalization is RFC
8785 JCS, the condition is signed through the ``vaara.audit.signer`` ``Signer``
protocol (Ed25519 by default), and the receipt is verified with the same ES256
stack every other Vaara receipt uses. There is no new cryptography here and no
new dependency. The ``tests/vectors/release_condition_v0/`` corpus recomputes
every verdict from the case bytes with no Vaara import.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

from vaara.attestation._attest_canonical import canonical_json, iso8601_to_epoch
from vaara.attestation._attest_types import AttestationError
from vaara.attestation._decision_types import decision_record_from_dict
from vaara.attestation._decision_emit import verify_decision_signature
from vaara.audit.signer import Signer, Verifier

SCHEMA = "vaara.release-condition/v0"

#: The evidence schema a v0 condition requires by default: an authorization
#: decision (SPEC.md Section 5.3), which is the record that says an agent action
#: was authorised and by which grant.
DEFAULT_EVIDENCE_SCHEMA = "vaara.authorization/v0"

#: The receipt verdict a v0 condition requires by default. A receipt carrying
#: ``block`` is sound evidence of a refusal and never releases money.
DEFAULT_DECISION = "allow"

#: Receipt signing algorithm a v0 condition pins. The condition pins one key by
#: fingerprint, so it also pins the algorithm that key belongs to.
RECEIPT_ALG = "ES256"


class ReleaseState(str, Enum):
    """What the settlement agent should do with the held value."""

    RELEASED = "released"
    HELD = "held"
    EXPIRED = "expired"
    REFUSED = "refused"


class ReleaseReason(str, Enum):
    """Why. Closed set; every member is mapped by :data:`REASON_STATE`."""

    # released
    RECEIPT_MATCHES = "receipt_matches"
    # held: sound evidence, insufficient (or none yet)
    RECEIPT_ABSENT = "receipt_absent"
    EVIDENCE_SCHEMA_MISMATCH = "evidence_schema_mismatch"
    ISSUER_NOT_ACCEPTED = "issuer_not_accepted"
    DECISION_NOT_ACCEPTED = "decision_not_accepted"
    AUTHORIZATION_MISMATCH = "authorization_mismatch"
    ACTION_DIGEST_MISMATCH = "action_digest_mismatch"
    # expired: the window closed
    CONDITION_EXPIRED = "condition_expired"
    # refused: the artifact fails as evidence
    CONDITION_MALFORMED = "condition_malformed"
    CONDITION_KEY_ABSENT = "condition_key_absent"
    CONDITION_SIGNATURE_INVALID = "condition_signature_invalid"
    RECEIPT_MALFORMED = "receipt_malformed"
    RECEIPT_KEY_ABSENT = "receipt_key_absent"
    RECEIPT_KEY_UNTRUSTED = "receipt_key_untrusted"
    RECEIPT_SIGNATURE_INVALID = "receipt_signature_invalid"
    EVIDENCE_DIGEST_MISMATCH = "evidence_digest_mismatch"


#: The partition, as data rather than as control flow. A reason has exactly one
#: state, so no code path can quietly file a refusal under a hold or answer an
#: absence with the same value as a forgery. The corpus and the unit tests both
#: assert this mapping is total over :class:`ReleaseReason` and onto
#: :class:`ReleaseState`.
REASON_STATE: Mapping[ReleaseReason, ReleaseState] = {
    ReleaseReason.RECEIPT_MATCHES: ReleaseState.RELEASED,
    ReleaseReason.RECEIPT_ABSENT: ReleaseState.HELD,
    ReleaseReason.EVIDENCE_SCHEMA_MISMATCH: ReleaseState.HELD,
    ReleaseReason.ISSUER_NOT_ACCEPTED: ReleaseState.HELD,
    ReleaseReason.DECISION_NOT_ACCEPTED: ReleaseState.HELD,
    ReleaseReason.AUTHORIZATION_MISMATCH: ReleaseState.HELD,
    ReleaseReason.ACTION_DIGEST_MISMATCH: ReleaseState.HELD,
    ReleaseReason.CONDITION_EXPIRED: ReleaseState.EXPIRED,
    ReleaseReason.CONDITION_MALFORMED: ReleaseState.REFUSED,
    ReleaseReason.CONDITION_KEY_ABSENT: ReleaseState.REFUSED,
    ReleaseReason.CONDITION_SIGNATURE_INVALID: ReleaseState.REFUSED,
    ReleaseReason.RECEIPT_MALFORMED: ReleaseState.REFUSED,
    ReleaseReason.RECEIPT_KEY_ABSENT: ReleaseState.REFUSED,
    ReleaseReason.RECEIPT_KEY_UNTRUSTED: ReleaseState.REFUSED,
    ReleaseReason.RECEIPT_SIGNATURE_INVALID: ReleaseState.REFUSED,
    ReleaseReason.EVIDENCE_DIGEST_MISMATCH: ReleaseState.REFUSED,
}


def _digest(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj)).hexdigest()


def _require_decimal_string(name: str, value: Any) -> str:
    """Amounts are decimal strings on the wire, never floats.

    Cross-stack float behaviour is the most common source of signature drift, so
    the JCS boundary rejects floats everywhere in this tree. Catching it in the
    constructor gives the caller the field name instead of a canonicalization
    error three frames later.
    """
    if not isinstance(value, str) or not value:
        raise AttestationError(
            f"{name} must be a non-empty decimal string, not {type(value).__name__}"
        )
    return value


@dataclass(frozen=True)
class HeldValue:
    """What the condition governs, stated so a reader knows what is at stake.

    Descriptive and signed, never acted on by this module: nothing here reaches a
    wallet, a chain, or a custodian. ``amount`` is a decimal string in the
    asset's own units.
    """

    amount: str
    asset: str
    network: str
    payee: str

    def __post_init__(self) -> None:
        _require_decimal_string("held.amount", self.amount)
        for field in ("asset", "network", "payee"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise AttestationError(f"held.{field} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "amount": self.amount,
            "asset": self.asset,
            "network": self.network,
            "payee": self.payee,
        }


@dataclass(frozen=True)
class ReleaseRequirements:
    """Exactly what must be proved. Every field is a match, not a hint.

    ``action_digest`` is the runtime-argument commitment of the authorised call
    (``argsCommitment`` in the authorization record); ``grant_fingerprint`` is
    the content address of the grant that governed it. Together they say "this
    action, under this authorization", so a receipt for a neighbouring call or
    for the same call under a different grant does not release.

    ``receipt_key_fingerprint`` pins the one key whose receipts count, computed
    by :func:`receipt_key_fingerprint`. A receipt from anyone else is not weaker
    evidence, it is evidence addressed to a different condition.
    """

    action_digest: str
    grant_fingerprint: str
    receipt_issuer: str
    receipt_key_fingerprint: str
    decision: str = DEFAULT_DECISION
    evidence_schema: str = DEFAULT_EVIDENCE_SCHEMA

    def __post_init__(self) -> None:
        for field in ("action_digest", "grant_fingerprint", "receipt_key_fingerprint"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.startswith("sha256:"):
                raise AttestationError(f"requires.{field} MUST be a 'sha256:' digest")
        for field in ("receipt_issuer", "decision", "evidence_schema"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise AttestationError(f"requires.{field} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "actionDigest": self.action_digest,
            "decision": self.decision,
            "evidenceSchema": self.evidence_schema,
            "grantFingerprint": self.grant_fingerprint,
            "receiptIssuer": self.receipt_issuer,
            "receiptKeyFingerprint": self.receipt_key_fingerprint,
        }


def receipt_key_fingerprint(public_key_pem: bytes) -> str:
    """``sha256:`` over the SubjectPublicKeyInfo DER of an ES256 public key.

    Taken over the DER rather than the PEM so a re-wrapped or re-wrapped-at-a-
    different-line-width copy of the same key fingerprints identically. A key is
    pinned by what it is, not by how it was typed out.
    """
    from cryptography.hazmat.primitives import serialization

    key = serialization.load_pem_public_key(public_key_pem)
    der = key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return "sha256:" + hashlib.sha256(der).hexdigest()


def _condition_signing_bytes(condition: Mapping[str, Any]) -> bytes:
    """JCS of the condition with ``signature`` removed."""
    return canonical_json({k: v for k, v in condition.items() if k != "signature"})


def emit_release_condition(
    *,
    signer: Signer,
    issuer: str,
    condition_id: str,
    held: HeldValue,
    requires: ReleaseRequirements,
    not_after: str,
    version: int = 1,
) -> dict[str, Any]:
    """Build, JCS-canonicalize, and sign a release condition.

    ``not_after`` is an ISO 8601 UTC instant and is inclusive: a receipt
    presented at exactly that instant still releases. ``signer`` is the party
    holding the value; a relying party verifies against that party's public key,
    which it holds out of band.
    """
    if iso8601_to_epoch(not_after) is None:
        raise AttestationError("notAfter must be an ISO 8601 instant")
    if not condition_id:
        raise AttestationError("conditionId must be non-empty")
    if not issuer:
        raise AttestationError("issuer must be non-empty")

    condition: dict[str, Any] = {
        "alg": signer.algorithm,
        "conditionId": condition_id,
        "holds": held.to_dict(),
        "issuer": issuer,
        "notAfter": not_after,
        "requires": requires.to_dict(),
        "schema": SCHEMA,
        "version": version,
    }
    condition["signature"] = signer.sign(_condition_signing_bytes(condition)).hex()
    return condition


def condition_digest(condition: Mapping[str, Any]) -> str:
    """Content address of the exact signed condition, signature included.

    This is what a decision names, so a decision is bound to one condition's
    bytes and cannot be replayed against a re-issued one.
    """
    return _digest(dict(condition))


def verify_condition_signature(
    condition: Mapping[str, Any], *, verifier: Verifier
) -> bool:
    """Verify the issuer signature over the condition. Nothing else."""
    signature = condition.get("signature")
    if not isinstance(signature, str):
        return False
    try:
        return verifier.verify(
            _condition_signing_bytes(condition), bytes.fromhex(signature)
        )
    except ValueError:
        return False


@dataclass(frozen=True)
class ReleaseBundle:
    """Everything presented alongside the held value at evaluation time.

    ``now`` is the evaluating party's clock as an ISO 8601 UTC instant.
    ``receipt`` is a ``vaara.receipt/v1`` envelope and ``evidence`` the record
    its ``evidenceRef.digest`` pins; both absent is the ordinary case before
    anything has been proved. ``condition_verifier`` holds the condition
    issuer's key, and ``receipt_public_key_pem`` the receipt signer's, both
    obtained out of band. Keys travel here rather than inside the condition
    because a document cannot vouch for the key that signed it.
    """

    now: str
    receipt: Optional[dict[str, Any]] = None
    evidence: Optional[dict[str, Any]] = None
    condition_verifier: Optional[Verifier] = None
    receipt_public_key_pem: Optional[bytes] = None


@dataclass(frozen=True)
class ReleaseDecision:
    """The answer, and what it was computed over."""

    state: ReleaseState
    reason: ReleaseReason
    condition_digest: str
    evaluated_at: str
    receipt_digest: Optional[str] = None

    @property
    def released(self) -> bool:
        return self.state is ReleaseState.RELEASED

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema": "vaara.release-decision/v0",
            "state": self.state.value,
            "reason": self.reason.value,
            "conditionDigest": self.condition_digest,
            "evaluatedAt": self.evaluated_at,
        }
        if self.receipt_digest is not None:
            out["receiptDigest"] = self.receipt_digest
        return out


_REQUIRED_CONDITION_KEYS = (
    "alg", "conditionId", "holds", "issuer", "notAfter", "requires", "schema",
    "version", "signature",
)
_REQUIRED_HELD_KEYS = ("amount", "asset", "network", "payee")
_REQUIRED_REQUIRES_KEYS = (
    "actionDigest", "decision", "evidenceSchema", "grantFingerprint",
    "receiptIssuer", "receiptKeyFingerprint",
)
_DIGEST_REQUIRES_KEYS = ("actionDigest", "grantFingerprint", "receiptKeyFingerprint")


def _condition_well_formed(condition: Any) -> bool:
    if not isinstance(condition, dict):
        return False
    if condition.get("schema") != SCHEMA:
        return False
    if any(k not in condition for k in _REQUIRED_CONDITION_KEYS):
        return False
    if not isinstance(condition["version"], int) or isinstance(condition["version"], bool):
        return False
    for key in ("alg", "conditionId", "issuer", "signature"):
        if not isinstance(condition[key], str) or not condition[key]:
            return False
    held, requires = condition["holds"], condition["requires"]
    if not isinstance(held, dict) or not isinstance(requires, dict):
        return False
    if any(not isinstance(held.get(k), str) or not held.get(k) for k in _REQUIRED_HELD_KEYS):
        return False
    if any(
        not isinstance(requires.get(k), str) or not requires.get(k)
        for k in _REQUIRED_REQUIRES_KEYS
    ):
        return False
    if any(not requires[k].startswith("sha256:") for k in _DIGEST_REQUIRES_KEYS):
        return False
    return iso8601_to_epoch(condition["notAfter"]) is not None


def _decide(
    reason: ReleaseReason,
    *,
    digest: str,
    now: str,
    receipt_digest: Optional[str] = None,
) -> ReleaseDecision:
    return ReleaseDecision(
        state=REASON_STATE[reason],
        reason=reason,
        condition_digest=digest,
        evaluated_at=now,
        receipt_digest=receipt_digest,
    )


def evaluate(condition: Mapping[str, Any], bundle: ReleaseBundle) -> ReleaseDecision:
    """Decide whether the held value releases, and say exactly why.

    Soundness first, then the clock, then sufficiency. See the module docstring
    for why that order is the whole design and not an implementation detail.
    """
    now_epoch = iso8601_to_epoch(bundle.now)
    if now_epoch is None:
        raise AttestationError("bundle.now must be an ISO 8601 instant")

    # --- the condition itself must be sound ---------------------------------
    if not _condition_well_formed(condition):
        # No trustworthy content address exists for a document that is not a
        # release condition, so the decision names the bytes it was handed.
        return _decide(
            ReleaseReason.CONDITION_MALFORMED,
            digest=_digest(dict(condition)) if isinstance(condition, dict) else "",
            now=bundle.now,
        )
    digest = condition_digest(condition)
    if bundle.condition_verifier is None:
        return _decide(ReleaseReason.CONDITION_KEY_ABSENT, digest=digest, now=bundle.now)
    if not verify_condition_signature(condition, verifier=bundle.condition_verifier):
        return _decide(
            ReleaseReason.CONDITION_SIGNATURE_INVALID, digest=digest, now=bundle.now
        )

    requires = condition["requires"]
    receipt = bundle.receipt
    receipt_digest = _digest(receipt) if isinstance(receipt, dict) else None

    def decided(reason: ReleaseReason) -> ReleaseDecision:
        return _decide(
            reason, digest=digest, now=bundle.now, receipt_digest=receipt_digest
        )

    # --- a presented receipt must be sound, before the clock is consulted ----
    #
    # Ordering matters here. If expiry were checked first, a tampered receipt
    # presented one second after the window closed would be reported as a closed
    # window, and the forgery would leave no trace in the answer.
    record = None
    if receipt is not None:
        try:
            record = decision_record_from_dict(receipt)
        except (AttestationError, AttributeError, TypeError, KeyError):
            return decided(ReleaseReason.RECEIPT_MALFORMED)
        if bundle.receipt_public_key_pem is None:
            return decided(ReleaseReason.RECEIPT_KEY_ABSENT)
        if record.alg != RECEIPT_ALG:
            # v0 pins one key by fingerprint, so it pins that key's algorithm.
            # A receipt under another algorithm is not signed by the pinned key.
            return decided(ReleaseReason.RECEIPT_KEY_UNTRUSTED)
        try:
            presented = receipt_key_fingerprint(bundle.receipt_public_key_pem)
        except (ValueError, TypeError):
            return decided(ReleaseReason.RECEIPT_KEY_ABSENT)
        if presented != requires["receiptKeyFingerprint"]:
            return decided(ReleaseReason.RECEIPT_KEY_UNTRUSTED)

        from cryptography.hazmat.primitives import serialization

        public_key = serialization.load_pem_public_key(bundle.receipt_public_key_pem)
        if not verify_decision_signature(record, verifying_material=public_key):
            return decided(ReleaseReason.RECEIPT_SIGNATURE_INVALID)

        # The evidence must be the bytes the receipt signed over. Covers a
        # receipt that binds no evidence, evidence that was not presented, and
        # evidence that was swapped after signing: in each case the receipt does
        # not vouch for what is in hand.
        evidence_ref = record.decision_derived.evidence_ref
        if evidence_ref is None or not isinstance(bundle.evidence, dict):
            return decided(ReleaseReason.EVIDENCE_DIGEST_MISMATCH)
        if _digest(bundle.evidence) != evidence_ref.digest:
            return decided(ReleaseReason.EVIDENCE_DIGEST_MISMATCH)

    # --- the clock ----------------------------------------------------------
    not_after_epoch = iso8601_to_epoch(condition["notAfter"])
    assert not_after_epoch is not None  # guarded by _condition_well_formed
    if now_epoch > not_after_epoch:
        return decided(ReleaseReason.CONDITION_EXPIRED)

    # --- sound evidence, but is it sufficient? ------------------------------
    if receipt is None or record is None:
        return decided(ReleaseReason.RECEIPT_ABSENT)
    evidence = bundle.evidence or {}
    evidence_ref = record.decision_derived.evidence_ref
    assert evidence_ref is not None  # guarded above

    required_schema = requires["evidenceSchema"]
    if evidence_ref.schema != required_schema or evidence.get("schema") != required_schema:
        return decided(ReleaseReason.EVIDENCE_SCHEMA_MISMATCH)
    if record.issuer_asserted.iss != requires["receiptIssuer"]:
        return decided(ReleaseReason.ISSUER_NOT_ACCEPTED)
    if record.decision_derived.decision != requires["decision"]:
        return decided(ReleaseReason.DECISION_NOT_ACCEPTED)
    if evidence.get("grantFingerprint") != requires["grantFingerprint"]:
        return decided(ReleaseReason.AUTHORIZATION_MISMATCH)
    if evidence.get("argsCommitment") != requires["actionDigest"]:
        return decided(ReleaseReason.ACTION_DIGEST_MISMATCH)

    return decided(ReleaseReason.RECEIPT_MATCHES)


__all__ = [
    "DEFAULT_DECISION",
    "DEFAULT_EVIDENCE_SCHEMA",
    "RECEIPT_ALG",
    "REASON_STATE",
    "SCHEMA",
    "HeldValue",
    "ReleaseBundle",
    "ReleaseDecision",
    "ReleaseReason",
    "ReleaseRequirements",
    "ReleaseState",
    "condition_digest",
    "emit_release_condition",
    "evaluate",
    "receipt_key_fingerprint",
    "verify_condition_signature",
]
