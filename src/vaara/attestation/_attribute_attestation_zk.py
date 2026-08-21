# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Attribute attestations whose values are committed, not carried.

Internal module. Public surface is :mod:`vaara.attestation.attribute_zk`.

Section 5.8 of the specification asks what a value is worth, and answers it by
making every attribute name its own source. This module asks the question that
follows: what does the issuer have to *keep* in order to say it.

## The asset that stops existing

An attestation provider that vouches for an attribute has to hold the attribute.
Date of birth, mother's maiden name, the images of an identity document, the
video of the customer proving they are themselves. Anything held can be sold,
subpoenaed, breached or repurposed, and no policy statement changes what the
holder is capable of.

Commit to the value at issuance and hand the opening to the holder, and the
issuer has nothing left to sell. Not a promise not to sell it: the asset stops
existing at the moment of issuance. What survives is a Pedersen commitment, which
is perfectly hiding, and a signature over it.

A relying party still gets what it actually needed, which was never the value:

- that a **predicate** holds over the hidden value (``age >= 18``), proved
- **how strongly the value was sourced**, in the clear, from the same closed
  ordered ladder as Section 5.8

That split is the design. The standing stays readable and gradeable precisely
because it is not the secret; what a relying party must be able to judge is the
strength of the evidence, and what it has no business learning is the value.

## What changes from :mod:`vaara.attestation.attribute`

One field::

    v0:   Attribute(name, value,      source, source_detail)
    this: Attribute(name, commitment, source, source_detail)

Everything else is deliberately identical: JCS canonicalization, a signature over
the document with its own ``signature`` member removed, four states with the
reason space partitioned as data, and checks ordered soundness, clock,
sufficiency.

## Predicates need no new cryptography

The range proof already in :mod:`vaara.attestation.zk` proves that a commitment
opens to a value in ``[0, 2**RANGE_BITS)``. Shifting the commitment turns that
into a comparison, because the commitment is additively homomorphic::

    value >= t    prove w = value - t          verifier target  C - t*G
    value <= t    prove w = t - value          verifier target  t*G - C
    a <= v <= b   both of the above

The blind follows the same shift: it stays as issued for the ``>=`` direction and
negates for the ``<=`` direction, since ``t*G - C`` opens to ``t - value`` under
the blind ``-blind``.

Each proof's Fiat-Shamir transcript is seeded with the attestation digest, the
attribute name and the predicate, so a proof does not move to another document,
another attribute, or a weaker threshold.

## What this is not

It is not selective disclosure. One signature covers every commitment in the
document, so a holder cannot present three attributes out of ten from a single
signed credential; that needs a signature scheme built for it (BBS+) and is not
here. It is not a qualified electronic attestation of attributes under Regulation
(EU) 910/2014 and MUST NOT be described as one. And it does not make the issuer
honest: zero knowledge hides the value from the relying party and from whoever
the issuer might later sell to, and says nothing about whether the issuer
committed to the truth in the first place.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

from vaara.attestation._attest_canonical import canonical_json, iso8601_to_epoch
from vaara.attestation._attest_types import AttestationError
from vaara.attestation._attribute_attestation import (
    STANDING_RANK,
    AttributeState,
    SourceStanding,
    Subject,
)
from vaara.attestation.zk._commit import commit, random_scalar
from vaara.attestation.zk._group import G, N, Point, scalar_mul
from vaara.attestation.zk._params import PROOF_SYSTEM, RANGE_BITS, params_digest
from vaara.attestation.zk._prove import _neg, _range_prove
from vaara.attestation.zk._verify import _RANGE_LEN, _range_verify
from vaara.audit.signer import Signer, Verifier

SCHEMA = "vaara.attribute-attestation-zk/v0"
PROOF_SCHEMA = "vaara.attribute-predicate/v0"

#: Every committed value and every predicate bound lives in ``[0, MAX_VALUE)``.
#: The range proof is what makes the comparison sound, and it proves membership
#: of exactly this interval, so the interval is part of the format rather than an
#: implementation detail. A value outside it is refused at issuance.
MAX_VALUE = 1 << RANGE_BITS

#: Length of the hex commitment on the wire: SEC1 compressed P-256 point.
_COMMITMENT_HEX_LEN = 66


class PredicateKind(str, Enum):
    """The shape of the question asked of a hidden value."""

    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    IN_RANGE = "in_range"


class PredicateReason(str, Enum):
    """Why. Closed set; every member is mapped by :data:`REASON_STATE`."""

    # accepted
    PREDICATE_PROVEN = "predicate_proven"
    # withheld: sound evidence, insufficient for what was asked
    ATTRIBUTE_ABSENT = "attribute_absent"
    SUBJECT_MISMATCH = "subject_mismatch"
    ISSUER_NOT_ACCEPTED = "issuer_not_accepted"
    SOURCE_BELOW_FLOOR = "source_below_floor"
    PROOF_ABSENT = "proof_absent"
    # expired: outside the stated window
    OUTSIDE_VALIDITY_WINDOW = "outside_validity_window"
    # refused: fails as evidence
    ATTESTATION_MALFORMED = "attestation_malformed"
    KEY_ABSENT = "key_absent"
    SIGNATURE_INVALID = "signature_invalid"
    PROOF_MALFORMED = "proof_malformed"
    PROOF_NOT_BOUND = "proof_not_bound"
    PROOF_INVALID = "proof_invalid"


#: The partition, as data. ``PROOF_ABSENT`` withholds and ``PROOF_INVALID``
#: refuses, and the gap between them is the point: nothing was proved is not the
#: same fact as something was forged, and one boolean for both throws away the
#: difference between "not yet" and "no".
REASON_STATE: Mapping[PredicateReason, AttributeState] = {
    PredicateReason.PREDICATE_PROVEN: AttributeState.ACCEPTED,
    PredicateReason.ATTRIBUTE_ABSENT: AttributeState.WITHHELD,
    PredicateReason.SUBJECT_MISMATCH: AttributeState.WITHHELD,
    PredicateReason.ISSUER_NOT_ACCEPTED: AttributeState.WITHHELD,
    PredicateReason.SOURCE_BELOW_FLOOR: AttributeState.WITHHELD,
    PredicateReason.PROOF_ABSENT: AttributeState.WITHHELD,
    PredicateReason.OUTSIDE_VALIDITY_WINDOW: AttributeState.EXPIRED,
    PredicateReason.ATTESTATION_MALFORMED: AttributeState.REFUSED,
    PredicateReason.KEY_ABSENT: AttributeState.REFUSED,
    PredicateReason.SIGNATURE_INVALID: AttributeState.REFUSED,
    PredicateReason.PROOF_MALFORMED: AttributeState.REFUSED,
    PredicateReason.PROOF_NOT_BOUND: AttributeState.REFUSED,
    PredicateReason.PROOF_INVALID: AttributeState.REFUSED,
}


def _digest(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj)).hexdigest()


def _in_range(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < MAX_VALUE


# --- the predicate ----------------------------------------------------------


@dataclass(frozen=True)
class Predicate:
    """A comparison against a hidden value, and the bounds it is taken over.

    Exactly the bounds the kind uses are present, and each is an integer in
    ``[0, MAX_VALUE)``. Only two directions exist because only two are needed:
    everything else is one of them, or both of them.
    """

    kind: PredicateKind
    lower: Optional[int] = None
    upper: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PredicateKind):
            raise AttestationError(f"predicate.kind must be a PredicateKind, got {self.kind!r}")
        needs = {
            PredicateKind.AT_LEAST: ("lower",),
            PredicateKind.AT_MOST: ("upper",),
            PredicateKind.IN_RANGE: ("lower", "upper"),
        }[self.kind]
        for field in ("lower", "upper"):
            bound = getattr(self, field)
            if field in needs:
                if not _in_range(bound):
                    raise AttestationError(
                        f"predicate.{field} must be an integer in [0, {MAX_VALUE})"
                    )
            elif bound is not None:
                raise AttestationError(
                    f"predicate.{field} has no meaning for {self.kind.value}"
                )
        if self.kind is PredicateKind.IN_RANGE and self.lower > self.upper:  # type: ignore[operator]
            raise AttestationError("predicate.lower exceeds predicate.upper")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind.value}
        if self.lower is not None:
            out["lower"] = self.lower
        if self.upper is not None:
            out["upper"] = self.upper
        return out

    @classmethod
    def from_dict(cls, obj: Any) -> "Predicate":
        if not isinstance(obj, dict) or set(obj) - {"kind", "lower", "upper"}:
            raise AttestationError("predicate is not a predicate object")
        try:
            kind = PredicateKind(obj.get("kind"))
        except ValueError as exc:
            raise AttestationError(f"unknown predicate kind {obj.get('kind')!r}") from exc
        return cls(kind=kind, lower=obj.get("lower"), upper=obj.get("upper"))

    @property
    def directions(self) -> tuple[str, ...]:
        """The range statements this predicate compiles to, in transcript order."""
        if self.kind is PredicateKind.AT_LEAST:
            return ("ge",)
        if self.kind is PredicateKind.AT_MOST:
            return ("le",)
        return ("ge", "le")

    def bound_for(self, direction: str) -> int:
        return self.lower if direction == "ge" else self.upper  # type: ignore[return-value]

    def holds(self, value: int) -> bool:
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value > self.upper:
            return False
        return True


# --- the committed document -------------------------------------------------


@dataclass(frozen=True)
class AttributeValue:
    """A plaintext attribute on its way in. It never reaches the wire.

    Values are integers because the range proof is an argument about integers.
    A string attribute would need a membership proof against a committed set,
    which this version does not carry.
    """

    name: str
    value: int
    source: SourceStanding
    source_detail: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise AttestationError("attribute.name must be a non-empty string")
        if not _in_range(self.value):
            raise AttestationError(
                f"attribute.value must be an integer in [0, {MAX_VALUE}), "
                f"got {self.value!r}"
            )
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


@dataclass(frozen=True)
class CommittedAttribute:
    """One committed value, and the standing of where it came from."""

    name: str
    commitment: str
    source: SourceStanding
    source_detail: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise AttestationError("attribute.name must be a non-empty string")
        if not _point_from_hex(self.commitment):
            raise AttestationError("attribute.commitment is not a P-256 point")
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
            "commitment": self.commitment,
            "name": self.name,
            "source": self.source.value,
        }
        if self.source_detail is not None:
            out["sourceDetail"] = self.source_detail
        return out


@dataclass(frozen=True)
class Opening:
    """What the holder keeps and the issuer does not: the value and its blind.

    Losing this makes the attestation unusable and reveals nothing. Publishing it
    reveals the value to whoever reads it, which is the holder's decision to make
    and nobody else's.
    """

    name: str
    value: int
    blind: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise AttestationError("opening.name must be a non-empty string")
        if not _in_range(self.value):
            raise AttestationError(f"opening.value must be in [0, {MAX_VALUE})")
        if not isinstance(self.blind, int) or isinstance(self.blind, bool):
            raise AttestationError("opening.blind must be an integer")
        if not 0 <= self.blind < N:
            raise AttestationError("opening.blind must be a P-256 scalar")

    def to_dict(self) -> dict[str, Any]:
        return {"blind": str(self.blind), "name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls, obj: Any) -> "Opening":
        if not isinstance(obj, dict):
            raise AttestationError("opening is not an object")
        try:
            blind = int(obj["blind"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AttestationError("opening.blind is not an integer") from exc
        return cls(name=obj.get("name"), value=obj.get("value"), blind=blind)


class IssuedAttestation:
    """The two halves of an issuance, kept apart on purpose.

    The attestation is the issuer's to publish. The openings are the holder's,
    and :meth:`release_to_holder` is the moment they stop being the issuer's:
    after it returns, this object cannot produce them again, so an issuer that
    followed the ritual cannot later be compelled to hand over what it no longer
    has a reference to.

    This is a structural guarantee about the object, not a claim about process
    memory. Zeroing RAM is not something a garbage-collected runtime offers, and
    the format does not pretend otherwise. What it removes is the *reason* to
    retain the value, which is the part that a subpoena, a breach or a change of
    ownership actually reaches.
    """

    __slots__ = ("_attestation", "_openings")

    def __init__(self, attestation: dict[str, Any], openings: Sequence[Opening]):
        self._attestation = attestation
        self._openings: Optional[tuple[Opening, ...]] = tuple(openings)

    @property
    def attestation(self) -> dict[str, Any]:
        return self._attestation

    @property
    def retains_openings(self) -> bool:
        return self._openings is not None

    def release_to_holder(self) -> tuple[Opening, ...]:
        """Hand the openings over and drop them here. Callable exactly once."""
        if self._openings is None:
            raise AttestationError(
                "the openings were already released to the holder and are not "
                "retained by the issuer"
            )
        openings, self._openings = self._openings, None
        return openings


def _point_from_hex(value: Any) -> Optional[Point]:
    """Decode a wire commitment, or None if it is not one. Never raises."""
    if not isinstance(value, str) or len(value) != _COMMITMENT_HEX_LEN:
        return None
    try:
        point = Point.from_bytes(bytes.fromhex(value))
    except ValueError:
        return None
    return None if point.is_infinity() else point


def commit_attribute(
    value: AttributeValue, *, blind: Optional[int] = None
) -> tuple[CommittedAttribute, Opening]:
    """Commit one plaintext attribute, returning the public half and the opening.

    ``blind`` exists so a conformance generator can produce byte-identical
    vectors. Leave it unset in production: a reused blind across two issuances of
    the same value makes the two commitments equal, which is exactly the linkage
    the commitment is there to prevent.
    """
    gamma = random_scalar() if blind is None else blind % N
    published = CommittedAttribute(
        name=value.name,
        commitment=commit(value.value, gamma).to_bytes().hex(),
        source=value.source,
        source_detail=value.source_detail,
    )
    return published, Opening(name=value.name, value=value.value, blind=gamma)


def _signing_bytes(attestation: Mapping[str, Any]) -> bytes:
    """JCS of the attestation with ``signature`` removed. Section 5.8's rule."""
    return canonical_json({k: v for k, v in attestation.items() if k != "signature"})


def emit_attribute_attestation_zk(
    *,
    signer: Signer,
    issuer: str,
    attestation_id: str,
    subject: Subject,
    attributes: Sequence[CommittedAttribute],
    not_before: str,
    not_after: str,
    version: int = 1,
) -> dict[str, Any]:
    """Build, JCS-canonicalize, and sign a document over already-committed values.

    Use :func:`issue` unless the commitments were produced elsewhere; it is the
    same thing with the ritual attached.
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
        "proofSystem": PROOF_SYSTEM,
        "schema": SCHEMA,
        "subject": subject.to_dict(),
        "version": version,
    }
    attestation["signature"] = signer.sign(_signing_bytes(attestation)).hex()
    return attestation


def issue(
    *,
    signer: Signer,
    issuer: str,
    attestation_id: str,
    subject: Subject,
    values: Sequence[AttributeValue],
    not_before: str,
    not_after: str,
    version: int = 1,
    blinds: Optional[Mapping[str, int]] = None,
) -> IssuedAttestation:
    """Commit every value, sign the commitments, and separate the two halves.

    This is the ritual the format exists for, in the order that makes the claim
    true: commit, sign, hand the openings to the holder, hold nothing. The
    returned object is the only place the openings live, and
    :meth:`IssuedAttestation.release_to_holder` empties it.
    """
    published: list[CommittedAttribute] = []
    openings: list[Opening] = []
    for value in values:
        blind = None if blinds is None else blinds.get(value.name)
        attribute, opening = commit_attribute(value, blind=blind)
        published.append(attribute)
        openings.append(opening)

    attestation = emit_attribute_attestation_zk(
        signer=signer,
        issuer=issuer,
        attestation_id=attestation_id,
        subject=subject,
        attributes=published,
        not_before=not_before,
        not_after=not_after,
        version=version,
    )
    return IssuedAttestation(attestation, openings)


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


# --- proving and verifying a predicate --------------------------------------


def _transcript(digest: str, name: str, predicate: Predicate, direction: str) -> bytes:
    """The Fiat-Shamir prefix, which is what stops a proof from travelling.

    It names the exact signed document, the attribute inside it, the predicate
    asked, and which of the predicate's two directions this proof is. Change any
    of the four and the challenges change, so the proof no longer verifies.
    """
    return b"/".join([
        b"vaara/attribute-zk/v0",
        digest.encode("utf-8"),
        name.encode("utf-8"),
        canonical_json(predicate.to_dict()),
        direction.encode("ascii"),
    ])


def _target(commitment: Point, predicate: Predicate, direction: str) -> Point:
    """The shifted commitment whose opening the range proof is taken over."""
    bound = scalar_mul(predicate.bound_for(direction), G)
    return commitment + _neg(bound) if direction == "ge" else bound + _neg(commitment)


def _find(attestation: Mapping[str, Any], name: str) -> Optional[Mapping[str, Any]]:
    attributes = attestation.get("attributes")
    if not isinstance(attributes, list):
        return None
    return next(
        (a for a in attributes if isinstance(a, dict) and a.get("name") == name), None
    )


def open_predicate(
    attestation: Mapping[str, Any], opening: Opening, predicate: Predicate
) -> dict[str, Any]:
    """Prove that the predicate holds over the committed value. Holder side.

    Raises rather than emitting a proof for a statement that is false. A prover
    that hands back something unverifiable pushes the discovery to the relying
    party and reads there as a forgery, which is not what happened; a lie has no
    witness and the honest answer is to say so here.
    """
    if not isinstance(opening, Opening):
        raise AttestationError("opening must be an Opening")
    published = _find(attestation, opening.name)
    if published is None:
        raise AttestationError(f"attestation carries no attribute {opening.name!r}")
    commitment = _point_from_hex(published.get("commitment"))
    if commitment is None:
        raise AttestationError("attribute.commitment is not a P-256 point")
    if commit(opening.value, opening.blind) != commitment:
        raise AttestationError("the opening does not open this commitment")
    if not predicate.holds(opening.value):
        raise AttestationError(
            "the predicate does not hold over the committed value, so it has no proof"
        )

    digest = attestation_digest(attestation)
    blob = bytearray()
    for direction in predicate.directions:
        bound = predicate.bound_for(direction)
        if direction == "ge":
            witness, gamma = opening.value - bound, opening.blind
        else:
            witness, gamma = bound - opening.value, (N - opening.blind) % N
        blob += _range_prove(
            witness, gamma, _transcript(digest, opening.name, predicate, direction)
        )

    return {
        "attestationDigest": digest,
        "name": opening.name,
        "predicate": predicate.to_dict(),
        "proof": bytes(blob).hex(),
        "proofSystem": PROOF_SYSTEM,
        "schema": PROOF_SCHEMA,
        "verifierParamsDigest": params_digest(),
    }


_PROOF_KEYS = (
    "attestationDigest", "name", "predicate", "proof", "proofSystem", "schema",
    "verifierParamsDigest",
)


def _parse_proof(envelope: Any) -> Optional[tuple[str, str, Predicate, bytes]]:
    """Structural read of a proof envelope, or None if it is not one."""
    if not isinstance(envelope, dict) or any(k not in envelope for k in _PROOF_KEYS):
        return None
    if envelope["schema"] != PROOF_SCHEMA or envelope["proofSystem"] != PROOF_SYSTEM:
        return None
    if envelope["verifierParamsDigest"] != params_digest():
        return None
    for key in ("attestationDigest", "name", "proof"):
        if not isinstance(envelope[key], str) or not envelope[key]:
            return None
    try:
        predicate = Predicate.from_dict(envelope["predicate"])
    except AttestationError:
        return None
    try:
        blob = bytes.fromhex(envelope["proof"])
    except ValueError:
        return None
    if len(blob) != len(predicate.directions) * _RANGE_LEN:
        return None
    return envelope["attestationDigest"], envelope["name"], predicate, blob


def verify_predicate(
    attestation: Mapping[str, Any], proof: Any
) -> bool:
    """Check a predicate proof against the attestation it names. Never raises.

    This answers one question, whether the proof is sound over this document. It
    says nothing about the issuer signature, the clock, or whether the standing
    clears anyone's floor; :func:`evaluate` is what puts those in order.
    """
    parsed = _parse_proof(proof)
    if parsed is None:
        return False
    digest, name, predicate, blob = parsed
    if digest != attestation_digest(attestation):
        return False
    published = _find(attestation, name)
    if published is None:
        return False
    commitment = _point_from_hex(published.get("commitment"))
    if commitment is None:
        return False
    try:
        for index, direction in enumerate(predicate.directions):
            chunk = blob[index * _RANGE_LEN : (index + 1) * _RANGE_LEN]
            if not _range_verify(
                _target(commitment, predicate, direction),
                chunk,
                _transcript(digest, name, predicate, direction),
            ):
                return False
    except (ValueError, IndexError):
        return False
    return True


# --- evaluation -------------------------------------------------------------


@dataclass(frozen=True)
class PredicateQuery:
    """What a relying party is asking, and the floor it will accept.

    There is no way to ask for a value, which is the point. A caller states a
    predicate and a minimum standing, and gets back whether the predicate was
    proved over a value sourced at least that strongly.
    """

    name: str
    predicate: Predicate
    minimum_source: SourceStanding
    subject_id: Optional[str] = None
    accepted_issuers: Optional[frozenset[str]] = None


@dataclass(frozen=True)
class PredicateDecision:
    """The answer, the standing behind it, and what it was computed over.

    It carries no value and there is none to carry.
    """

    state: AttributeState
    reason: PredicateReason
    attestation_digest: str
    evaluated_at: str
    source: Optional[SourceStanding] = None
    predicate: Optional[Mapping[str, Any]] = None

    @property
    def accepted(self) -> bool:
        return self.state is AttributeState.ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema": "vaara.attribute-predicate-decision/v0",
            "state": self.state.value,
            "reason": self.reason.value,
            "attestationDigest": self.attestation_digest,
            "evaluatedAt": self.evaluated_at,
        }
        if self.source is not None:
            out["source"] = self.source.value
        if self.predicate is not None:
            out["predicate"] = dict(self.predicate)
        return out


_REQUIRED_KEYS = (
    "alg", "attestationId", "attributes", "issuer", "notAfter", "notBefore",
    "proofSystem", "schema", "signature", "subject", "version",
)


def _well_formed(attestation: Any) -> bool:
    if not isinstance(attestation, dict) or attestation.get("schema") != SCHEMA:
        return False
    if any(k not in attestation for k in _REQUIRED_KEYS):
        return False
    if attestation["proofSystem"] != PROOF_SYSTEM:
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
               for k in ("name", "source", "commitment")):
            return False
        # An unrecognised standing is malformed rather than silently floored: a
        # verifier that quietly downgrades what it does not understand gives a
        # forger a way to introduce a standing of their own.
        if a["source"] not in {s.value for s in SourceStanding}:
            return False
        # A commitment that is not a point on the curve is not a commitment. It
        # is checked here, once, so no later step has to guess what it holds.
        if _point_from_hex(a["commitment"]) is None:
            return False
        if a["name"] in seen:
            return False
        seen.add(a["name"])
    for field in ("notBefore", "notAfter"):
        if iso8601_to_epoch(attestation[field]) is None:
            return False
    return True


def _decide(
    reason: PredicateReason,
    *,
    digest: str,
    now: str,
    source: Optional[SourceStanding] = None,
    predicate: Optional[Mapping[str, Any]] = None,
) -> PredicateDecision:
    return PredicateDecision(
        state=REASON_STATE[reason],
        reason=reason,
        attestation_digest=digest,
        evaluated_at=now,
        source=source,
        predicate=predicate,
    )


def evaluate(
    attestation: Mapping[str, Any],
    query: PredicateQuery,
    *,
    proof: Any = None,
    now: str,
    verifier: Optional[Verifier] = None,
) -> PredicateDecision:
    """Decide whether a predicate over a hidden value may be relied on, and say why.

    Soundness first, then the clock, then sufficiency, the same order as Section
    5.8. Inside soundness the presented proof is judged before the standing floor,
    so a forged proof is reported as forged rather than leaving the room as merely
    weaker than asked for.
    """
    now_epoch = iso8601_to_epoch(now)
    if now_epoch is None:
        raise AttestationError("now must be an ISO 8601 instant")
    if not isinstance(query, PredicateQuery):
        raise AttestationError("query must be a PredicateQuery")

    if not _well_formed(attestation):
        return _decide(
            PredicateReason.ATTESTATION_MALFORMED,
            digest=_digest(dict(attestation)) if isinstance(attestation, dict) else "",
            now=now,
        )
    digest = attestation_digest(attestation)
    if verifier is None:
        return _decide(PredicateReason.KEY_ABSENT, digest=digest, now=now)
    if not verify_attestation_signature(attestation, verifier=verifier):
        return _decide(PredicateReason.SIGNATURE_INVALID, digest=digest, now=now)

    if not (
        iso8601_to_epoch(attestation["notBefore"])  # type: ignore[operator]
        <= now_epoch
        <= iso8601_to_epoch(attestation["notAfter"])  # type: ignore[operator]
    ):
        return _decide(PredicateReason.OUTSIDE_VALIDITY_WINDOW, digest=digest, now=now)

    if query.subject_id is not None and attestation["subject"]["id"] != query.subject_id:
        return _decide(PredicateReason.SUBJECT_MISMATCH, digest=digest, now=now)
    if query.accepted_issuers is not None and (
        attestation["issuer"] not in query.accepted_issuers
    ):
        return _decide(PredicateReason.ISSUER_NOT_ACCEPTED, digest=digest, now=now)

    match = _find(attestation, query.name)
    if match is None:
        return _decide(PredicateReason.ATTRIBUTE_ABSENT, digest=digest, now=now)
    source = SourceStanding(match["source"])
    asked = query.predicate.to_dict()

    if proof is None:
        return _decide(PredicateReason.PROOF_ABSENT, digest=digest, now=now,
                       source=source, predicate=asked)
    parsed = _parse_proof(proof)
    if parsed is None:
        return _decide(PredicateReason.PROOF_MALFORMED, digest=digest, now=now,
                       source=source, predicate=asked)
    # A proof answers exactly one question about exactly one document. One that
    # answers a different one is not weak evidence, it is the wrong artifact.
    proof_digest, proof_name, proof_predicate, _ = parsed
    if (proof_digest, proof_name, proof_predicate.to_dict()) != (
        digest, query.name, asked
    ):
        return _decide(PredicateReason.PROOF_NOT_BOUND, digest=digest, now=now,
                       source=source, predicate=asked)
    if not verify_predicate(attestation, proof):
        return _decide(PredicateReason.PROOF_INVALID, digest=digest, now=now,
                       source=source, predicate=asked)

    if STANDING_RANK[source] < STANDING_RANK[query.minimum_source]:
        return _decide(PredicateReason.SOURCE_BELOW_FLOOR, digest=digest, now=now,
                       source=source, predicate=asked)

    return _decide(PredicateReason.PREDICATE_PROVEN, digest=digest, now=now,
                   source=source, predicate=asked)


__all__ = [
    "MAX_VALUE",
    "PROOF_SCHEMA",
    "PROOF_SYSTEM",
    "REASON_STATE",
    "SCHEMA",
    "STANDING_RANK",
    "AttributeState",
    "AttributeValue",
    "CommittedAttribute",
    "IssuedAttestation",
    "Opening",
    "Predicate",
    "PredicateDecision",
    "PredicateKind",
    "PredicateQuery",
    "PredicateReason",
    "SourceStanding",
    "Subject",
    "attestation_digest",
    "commit_attribute",
    "emit_attribute_attestation_zk",
    "evaluate",
    "issue",
    "open_predicate",
    "verify_attestation_signature",
    "verify_predicate",
]
