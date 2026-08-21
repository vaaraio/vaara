# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Attribute attestations: signed statements that name their own source strength.

An attribute attestation binds a subject to attribute values, states where each
value came from, and says how long it holds. Every value carries a standing from
a closed, ordered set (``undeclared``, ``operator_declared``, ``measured``,
``protocol_defined``), and a relying party names the floor it will accept.

The distinction that field makes is the reason the format exists. An alarm
classified as an animal by the supplier being paid to reduce alarms is
``operator_declared``. The same classification produced by a model whose identity
and version ride inside the attestation is ``measured``. Both are signed, both
verify, and only one is evidence. A format that cannot express that lets a
supplier reclassify its way to a target with every signature intact.

:func:`evaluate` answers one of ``accepted`` / ``withheld`` / ``expired`` /
``refused``, each carrying a reason from a closed set, partitioned so that sound
but insufficient evidence can never be reported the same way as a forgery.

This is not a qualified electronic attestation of attributes and must not be
described as one: those terms are tied to a supervised trusted-list entry that no
amount of correct cryptography substitutes for. What this is instead is
self-hostable. The issuer holds its own key and the subject data never leaves the
issuer's premises.

Canonicalization is RFC 8785 JCS; signing runs through the ``vaara.audit.signer``
protocol (Ed25519 by default, ML-DSA-65 optional). No new cryptography and no new
dependency.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

from vaara.attestation._attribute_attestation import (
    REASON_STATE,
    SCHEMA,
    STANDING_RANK,
    Attribute,
    AttributeDecision,
    AttributeQuery,
    AttributeReason,
    AttributeState,
    SourceStanding,
    Subject,
    attestation_digest,
    emit_attribute_attestation,
    evaluate,
    verify_attestation_signature,
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
