# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Attribute attestations: a signed statement that names its own source strength.

Internal module. Public surface is :mod:`vaara.attestation.attribute`.

An attribute attestation binds a subject to attribute values, states where each
value came from, and says how long it holds. It is the shape a qualified
attestation of attributes has once the supervisory wrapper is removed, and it is
deliberately *not* that: nothing here is qualified, nothing here is listed on a
trusted list, and no term reserved by eIDAS is used to describe it. What it is
instead is self-hostable. The issuer holds its own key, the subject never leaves
the issuer's premises, and a relying party checks the result from the bytes.

## The field that carries the whole design

Any signed record can assert an attribute. The question a relying party actually
has is whether the assertion is worth anything, and that depends entirely on
where the value came from. So every attribute names its own source, drawn from a
closed, ordered set:

    undeclared        nothing is claimed about where this came from
    operator_declared the party being judged typed it in
    measured          the issuer observed it directly
    protocol_defined  the value is fixed by a specification and cannot differ

A relying party states the floor it needs. An attestation carrying a value below
that floor does not fail as evidence; it is sound and insufficient, and it says
so. That distinction is the same one :mod:`vaara.settlement.release` makes
between a held and a refused release, and it is made the same way here, by
partitioning the reason space rather than by adding a boolean.

The ordering is a total order and the choice is deliberate. ``protocol_defined``
outranks ``measured`` because a value fixed by a specification cannot be wrong,
while a measurement can come from a broken sensor. Anything a relying party
cannot place on this ladder is ``undeclared``, which is the floor and never
converts upward.

## Why the source field matters more than the signature

An alarm that is classified as an animal by the supplier being paid to reduce
alarms is ``operator_declared``. The same classification produced by a model
whose identity and version ride inside the attestation is ``measured``. Both are
signed, both verify, and only one of them is evidence of anything. A record
format that cannot express that difference lets a supplier reclassify its way to
a target while every signature still checks out.

## What this is not

It is not a qualified electronic attestation of attributes and MUST NOT be
described as one. Those terms are tied to a supervised, audited, trusted-list
entry, and no amount of correct cryptography substitutes for the listing. An
attestation issued and signed by the party it describes proves integrity, never
independence, and :func:`evaluate` reports that standing rather than hiding it.

Canonicalization is RFC 8785 JCS and signing runs through the
``vaara.audit.signer`` protocol, so this adds no cryptography and no dependency.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from vaara.attestation._attest_canonical import canonical_json, iso8601_to_epoch
from vaara.attestation._attest_types import AttestationError
from vaara.audit.signer import Signer, Verifier

SCHEMA = "vaara.attribute-attestation/v0"


class SourceStanding(str, Enum):
    """Where an attribute value came from. Ordered, closed, and never inferred."""

    UNDECLARED = "undeclared"
    OPERATOR_DECLARED = "operator_declared"
    MEASURED = "measured"
    PROTOCOL_DEFINED = "protocol_defined"


#: The total order. A relying party names a floor and this decides whether a
#: value clears it. Kept as data so the ladder is one authority rather than a
#: comparison written out at each call site.
STANDING_RANK: Mapping[SourceStanding, int] = {
    SourceStanding.UNDECLARED: 0,
    SourceStanding.OPERATOR_DECLARED: 1,
    SourceStanding.MEASURED: 2,
    SourceStanding.PROTOCOL_DEFINED: 3,
}


class AttributeState(str, Enum):
    """What a relying party should do with the attestation."""

    ACCEPTED = "accepted"
    WITHHELD = "withheld"
    EXPIRED = "expired"
    REFUSED = "refused"


class AttributeReason(str, Enum):
    """Why. Closed set; every member is mapped by :data:`REASON_STATE`."""

    # accepted
    ATTRIBUTE_ATTESTED = "attribute_attested"
    # withheld: sound evidence, insufficient for what was asked
    ATTRIBUTE_ABSENT = "attribute_absent"
    SUBJECT_MISMATCH = "subject_mismatch"
    ISSUER_NOT_ACCEPTED = "issuer_not_accepted"
    SOURCE_BELOW_FLOOR = "source_below_floor"
    # expired: outside the stated window
    OUTSIDE_VALIDITY_WINDOW = "outside_validity_window"
    # refused: fails as evidence
    ATTESTATION_MALFORMED = "attestation_malformed"
    KEY_ABSENT = "key_absent"
    SIGNATURE_INVALID = "signature_invalid"


#: The partition, as data. A reason belongs to exactly one state, so no code
#: path can file a forged attestation under the same answer as an honest one
#: that merely fell short of the floor.
REASON_STATE: Mapping[AttributeReason, AttributeState] = {
    AttributeReason.ATTRIBUTE_ATTESTED: AttributeState.ACCEPTED,
    AttributeReason.ATTRIBUTE_ABSENT: AttributeState.WITHHELD,
    AttributeReason.SUBJECT_MISMATCH: AttributeState.WITHHELD,
    AttributeReason.ISSUER_NOT_ACCEPTED: AttributeState.WITHHELD,
    AttributeReason.SOURCE_BELOW_FLOOR: AttributeState.WITHHELD,
    AttributeReason.OUTSIDE_VALIDITY_WINDOW: AttributeState.EXPIRED,
    AttributeReason.ATTESTATION_MALFORMED: AttributeState.REFUSED,
    AttributeReason.KEY_ABSENT: AttributeState.REFUSED,
    AttributeReason.SIGNATURE_INVALID: AttributeState.REFUSED,
}


def _digest(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj)).hexdigest()


@dataclass(frozen=True)
class Attribute:
    """One attested value, and the standing of where it came from.

    ``value`` is a string on the wire. Numbers travel as decimal strings for the
    same reason they do everywhere else in this tree: a float is the most common
    source of cross-stack signature drift and the JCS boundary rejects it.
    ``source_detail`` is optional and names the thing that produced the value, a
    sensor id, a model and version, a specification section. It carries no
    authority on its own; the standing is what a verifier grades.
    """

    name: str
    value: str
    source: SourceStanding
    source_detail: Optional[str] = None

    def __post_init__(self) -> None:
        for field in ("name", "value"):
            v = getattr(self, field)
            if not isinstance(v, str) or not v:
                raise AttestationError(f"attribute.{field} must be a non-empty string")
        if not isinstance(self.source, SourceStanding):
            raise AttestationError(
                f"attribute.source must be a SourceStanding, got {self.source!r}"
            )
        if self.source_detail is not None and (
            not isinstance(self.source_detail, str) or not self.source_detail
        ):
            raise AttestationError(
                "attribute.source_detail must be a non-empty string or absent"
            )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "source": self.source.value,
            "value": self.value,
        }
        if self.source_detail is not None:
            out["sourceDetail"] = self.source_detail
        return out


@dataclass(frozen=True)
class Subject:
    """What the attributes are about.

    ``kind`` names the class of thing (``detection-event``, ``device``,
    ``agent``, ``document``) and ``id`` identifies one of them inside the
    issuer's own namespace. Nothing here is a natural person identifier and
    nothing in this module resolves one; an issuer that wants to attest about a
    person carries that in its own attribute values and answers for it.
    """

    kind: str
    id: str

    def __post_init__(self) -> None:
        for field in ("kind", "id"):
            v = getattr(self, field)
            if not isinstance(v, str) or not v:
                raise AttestationError(f"subject.{field} must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind}


def _signing_bytes(attestation: Mapping[str, Any]) -> bytes:
    """JCS of the attestation with ``signature`` removed.

    The same rule the release condition and the data-locality record use, so a
    verifier that checks one checks all three with no new code.
    """
    return canonical_json({k: v for k, v in attestation.items() if k != "signature"})


def emit_attribute_attestation(
    *,
    signer: Signer,
    issuer: str,
    attestation_id: str,
    subject: Subject,
    attributes: Sequence[Attribute],
    not_before: str,
    not_after: str,
    version: int = 1,
) -> dict[str, Any]:
    """Build, JCS-canonicalize, and sign an attribute attestation.

    ``not_before`` and ``not_after`` are ISO 8601 UTC instants and both are
    inclusive. Attributes are emitted sorted by name so two issuers building the
    same statement produce the same bytes.
    """
    if not attributes:
        raise AttestationError("an attestation with no attributes attests nothing")
    for field, value in (("issuer", issuer), ("attestation_id", attestation_id)):
        if not isinstance(value, str) or not value:
            raise AttestationError(f"{field} must be a non-empty string")
    for field, value in (("not_before", not_before), ("not_after", not_after)):
        if iso8601_to_epoch(value) is None:
            raise AttestationError(f"{field} must be an ISO 8601 instant")
    if iso8601_to_epoch(not_after) < iso8601_to_epoch(not_before):  # type: ignore[operator]
        raise AttestationError("notAfter precedes notBefore")
    names = [a.name for a in attributes]
    if len(set(names)) != len(names):
        raise AttestationError("duplicate attribute name in one attestation")

    attestation: dict[str, Any] = {
        "alg": signer.algorithm,
        "attestationId": attestation_id,
        "attributes": [a.to_dict() for a in sorted(attributes, key=lambda a: a.name)],
        "issuer": issuer,
        "notAfter": not_after,
        "notBefore": not_before,
        "schema": SCHEMA,
        "subject": subject.to_dict(),
        "version": version,
    }
    attestation["signature"] = signer.sign(_signing_bytes(attestation)).hex()
    return attestation


def attestation_digest(attestation: Mapping[str, Any]) -> str:
    """Content address of the exact signed attestation, signature included."""
    return _digest(dict(attestation))


def verify_attestation_signature(
    attestation: Mapping[str, Any], *, verifier: Verifier
) -> bool:
    """Verify the issuer signature over the attestation. Nothing else."""
    signature = attestation.get("signature")
    if not isinstance(signature, str):
        return False
    try:
        return verifier.verify(_signing_bytes(attestation), bytes.fromhex(signature))
    except ValueError:
        return False


@dataclass(frozen=True)
class AttributeQuery:
    """What a relying party is asking for, and the floor it will accept.

    ``minimum_source`` is the whole point of asking through this object rather
    than reading the attestation directly: it forces the caller to state, up
    front, how strong the provenance has to be. A caller that genuinely accepts
    anything says so by naming ``UNDECLARED``, which is a decision on the record
    rather than an omission.
    """

    name: str
    minimum_source: SourceStanding
    subject_id: Optional[str] = None
    accepted_issuers: Optional[frozenset[str]] = None


@dataclass(frozen=True)
class AttributeDecision:
    """The answer, the value when there is one, and what it was computed over."""

    state: AttributeState
    reason: AttributeReason
    attestation_digest: str
    evaluated_at: str
    value: Optional[str] = None
    source: Optional[SourceStanding] = None

    @property
    def accepted(self) -> bool:
        return self.state is AttributeState.ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema": "vaara.attribute-decision/v0",
            "state": self.state.value,
            "reason": self.reason.value,
            "attestationDigest": self.attestation_digest,
            "evaluatedAt": self.evaluated_at,
        }
        if self.value is not None:
            out["value"] = self.value
        if self.source is not None:
            out["source"] = self.source.value
        return out


_REQUIRED_KEYS = (
    "alg", "attestationId", "attributes", "issuer", "notAfter", "notBefore",
    "schema", "signature", "subject", "version",
)


def _well_formed(attestation: Any) -> bool:
    if not isinstance(attestation, dict) or attestation.get("schema") != SCHEMA:
        return False
    if any(k not in attestation for k in _REQUIRED_KEYS):
        return False
    if not isinstance(attestation["version"], int) or isinstance(
        attestation["version"], bool
    ):
        return False
    for key in ("alg", "attestationId", "issuer", "signature"):
        if not isinstance(attestation[key], str) or not attestation[key]:
            return False
    subject = attestation["subject"]
    if not isinstance(subject, dict):
        return False
    if any(not isinstance(subject.get(k), str) or not subject.get(k)
           for k in ("id", "kind")):
        return False
    attributes = attestation["attributes"]
    if not isinstance(attributes, list) or not attributes:
        return False
    seen = set()
    for a in attributes:
        if not isinstance(a, dict):
            return False
        if any(not isinstance(a.get(k), str) or not a.get(k)
               for k in ("name", "source", "value")):
            return False
        # An unrecognised standing is malformed rather than silently floored:
        # a verifier that quietly downgrades a value it does not understand
        # gives a forger a way to introduce one.
        if a["source"] not in {s.value for s in SourceStanding}:
            return False
        if a["name"] in seen:
            return False
        seen.add(a["name"])
    for field in ("notBefore", "notAfter"):
        if iso8601_to_epoch(attestation[field]) is None:
            return False
    return True


def _decide(
    reason: AttributeReason,
    *,
    digest: str,
    now: str,
    value: Optional[str] = None,
    source: Optional[SourceStanding] = None,
) -> AttributeDecision:
    return AttributeDecision(
        state=REASON_STATE[reason],
        reason=reason,
        attestation_digest=digest,
        evaluated_at=now,
        value=value,
        source=source,
    )


def evaluate(
    attestation: Mapping[str, Any],
    query: AttributeQuery,
    *,
    now: str,
    verifier: Optional[Verifier] = None,
) -> AttributeDecision:
    """Decide whether one attribute may be relied on, and say exactly why.

    Soundness first, then the clock, then sufficiency. Soundness runs before the
    clock so an expired window cannot swallow a bad signature, and the clock runs
    before sufficiency so a closed window is reported as the reason rather than a
    missing attribute.
    """
    now_epoch = iso8601_to_epoch(now)
    if now_epoch is None:
        raise AttestationError("now must be an ISO 8601 instant")

    if not _well_formed(attestation):
        return _decide(
            AttributeReason.ATTESTATION_MALFORMED,
            digest=_digest(dict(attestation)) if isinstance(attestation, dict) else "",
            now=now,
        )
    digest = attestation_digest(attestation)
    if verifier is None:
        return _decide(AttributeReason.KEY_ABSENT, digest=digest, now=now)
    if not verify_attestation_signature(attestation, verifier=verifier):
        return _decide(AttributeReason.SIGNATURE_INVALID, digest=digest, now=now)

    if not (
        iso8601_to_epoch(attestation["notBefore"])  # type: ignore[operator]
        <= now_epoch
        <= iso8601_to_epoch(attestation["notAfter"])  # type: ignore[operator]
    ):
        return _decide(
            AttributeReason.OUTSIDE_VALIDITY_WINDOW, digest=digest, now=now
        )

    if query.subject_id is not None and attestation["subject"]["id"] != query.subject_id:
        return _decide(AttributeReason.SUBJECT_MISMATCH, digest=digest, now=now)
    if query.accepted_issuers is not None and (
        attestation["issuer"] not in query.accepted_issuers
    ):
        return _decide(AttributeReason.ISSUER_NOT_ACCEPTED, digest=digest, now=now)

    match = next(
        (a for a in attestation["attributes"] if a["name"] == query.name), None
    )
    if match is None:
        return _decide(AttributeReason.ATTRIBUTE_ABSENT, digest=digest, now=now)

    source = SourceStanding(match["source"])
    if STANDING_RANK[source] < STANDING_RANK[query.minimum_source]:
        return _decide(
            AttributeReason.SOURCE_BELOW_FLOOR,
            digest=digest, now=now, value=match["value"], source=source,
        )

    return _decide(
        AttributeReason.ATTRIBUTE_ATTESTED,
        digest=digest, now=now, value=match["value"], source=source,
    )


__all__ = [
    "REASON_STATE",
    "SCHEMA",
    "STANDING_RANK",
    "Attribute",
    "AttributeDecision",
    "AttributeQuery",
    "AttributeReason",
    "AttributeState",
    "SourceStanding",
    "Subject",
    "attestation_digest",
    "emit_attribute_attestation",
    "evaluate",
    "verify_attestation_signature",
]
