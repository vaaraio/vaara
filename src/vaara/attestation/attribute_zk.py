# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Attribute attestations that commit to the value instead of carrying it.

An attestation provider that vouches for an attribute has to hold the attribute,
and anything held can be sold, breached, subpoenaed or repurposed. This format
commits to the value at issuance, hands the opening to the holder, and leaves the
issuer with nothing to sell. Not a promise not to sell it: the asset stops
existing at the moment of issuance.

A relying party still learns the two things it actually needed. That a predicate
holds over the hidden value, proved rather than asserted, and how strongly the
value was sourced, in the clear, from the same closed ordered ladder
:mod:`vaara.attestation.attribute` uses. The standing stays readable because
judging the strength of evidence is the relying party's job; the value is not.

    >>> issued = issue(signer=..., values=[AttributeValue("age", 37,
    ...                SourceStanding.PROTOCOL_DEFINED, "passport MRZ")], ...)
    >>> openings = issued.release_to_holder()   # the issuer now holds nothing
    >>> proof = open_predicate(issued.attestation, openings[0],
    ...                        Predicate(PredicateKind.AT_LEAST, lower=18))
    >>> evaluate(issued.attestation, query, proof=proof, now=..., verifier=...)

:func:`evaluate` answers ``accepted`` / ``withheld`` / ``expired`` / ``refused``,
each with a reason from a closed set, partitioned so that nothing proved can
never be reported the same way as something forged.

The proof system is the transparent P-256 commit-and-prove engine already in the
tree: Pedersen commitments, perfectly hiding and computationally binding, with a
second generator derived by hash-to-curve so there is no trusted setup and anyone
can recompute it. Comparisons come from shifting the commitment, so this adds no
cryptography and no dependency.

**Not selective disclosure.** One signature covers every commitment in the
document, so a holder cannot present three attributes out of ten from a single
signed credential. That needs a signature scheme built for it and is not here.

**Not qualified.** This is not a qualified electronic attestation of attributes
under Regulation (EU) 910/2014 and must not be described as one. Those terms are
tied to a supervised trusted-list entry that no cryptographic property
substitutes for. Zero knowledge hides the value from the relying party and from
whoever the issuer might later sell to; it does not make the issuer honest about
what it committed to.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

from vaara.attestation._attribute_attestation_zk import (
    MAX_VALUE,
    PROOF_SCHEMA,
    PROOF_SYSTEM,
    REASON_STATE,
    SCHEMA,
    STANDING_RANK,
    AttributeState,
    AttributeValue,
    CommittedAttribute,
    IssuedAttestation,
    Opening,
    Predicate,
    PredicateDecision,
    PredicateKind,
    PredicateQuery,
    PredicateReason,
    SourceStanding,
    Subject,
    attestation_digest,
    commit_attribute,
    emit_attribute_attestation_zk,
    evaluate,
    issue,
    open_predicate,
    verify_attestation_signature,
    verify_predicate,
)

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
