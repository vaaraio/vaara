"""Handoff verification over a whole directory of cross-org packages.

``verify_handoff`` checks one self-contained handoff package: one org's record,
handed to another org's regulator. A regulator who receives a pile of packages,
from one provider or several, asks the batch question: across the whole set, how
many records verify offline under their rotated-out keys, and how many are
anchor-corroborated rather than resting on the signature alone?
``check_handoff_set`` is the receiving side of that, the batch twin of
:func:`verify_handoff` and the handoff analogue of :func:`check_bundle_set`.

Each package is run through :func:`verify_handoff`. The set then rolls up:

* **Pass count.** How many packages are ``ok`` for the chosen mode. A document
  that does not parse into a package is a failing entry that gates the set,
  never a silent drop.
* **Tier tally.** How many records reached ``verifiable`` (signature binds to a
  listed key, inside its window, not revoked) and how many reached
  ``corroborated`` (an enclosed eIDAS anchor predates retirement and
  revocation). The gap between the two is the part of the set whose record
  authenticity rests on the signature alone.
* **Pinning coverage.** How many packages had their producer identity pinned
  against a trusted DID document. The coverage note (advisory) fires when no
  package in the set was pinned: every record's authenticity rests on a
  self-asserted identity, the handoff analogue of the lens coverage gap.

The set is ``ok`` iff every document loaded and every loaded package verified
for the chosen mode; the coverage note is advisory and does not gate. Like the
single verb, this needs the crypto the record lens uses, so it runs only with
the attestation extra installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from vaara.attestation._handoff import verify_handoff

HANDOFF_SET_SCHEMA_NAME = "sep2828-handoff-set"
HANDOFF_SET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HandoffSetEntry:
    """Per-package verdict inside a set.

    ``loaded`` is False when the document did not parse into a handoff package;
    then ``error`` names why and the tier booleans are False. Otherwise ``ok``,
    ``verifiable``, ``corroborated``, and ``producer_identity_basis`` come from
    the handoff verdict.
    """

    name: str
    loaded: bool
    ok: bool
    verifiable: bool
    corroborated: bool
    producer_identity_basis: str
    producer: Optional[str]
    error: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "loaded": self.loaded,
            "ok": self.ok,
            "verifiable": self.verifiable,
            "corroborated": self.corroborated,
            "producerIdentityBasis": self.producer_identity_basis,
            "producer": self.producer,
            "error": self.error,
        }


@dataclass(frozen=True)
class HandoffSetReport:
    """Outcome of verifying a set of cross-org handoff packages."""

    ok: bool
    strict: bool
    total: int
    loaded: int
    passed: int
    verifiable: int
    corroborated: int
    pinned: int
    entries: tuple[HandoffSetEntry, ...]
    pinning_gap: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HANDOFF_SET_SCHEMA_NAME,
            "schemaVersion": HANDOFF_SET_SCHEMA_VERSION,
            "ok": self.ok,
            "strict": self.strict,
            "total": self.total,
            "loaded": self.loaded,
            "passed": self.passed,
            "verifiable": self.verifiable,
            "corroborated": self.corroborated,
            "pinned": self.pinned,
            "pinningGap": self.pinning_gap,
            "entries": [e.to_dict() for e in self.entries],
        }


def check_handoff_set(
    packages: Sequence[tuple[str, Any, Optional[str]]],
    *,
    trusted_did_document: Optional[dict[str, Any]] = None,
    strict: bool = False,
) -> HandoffSetReport:
    """Verify a set of ``(name, document, anchor_attested_time)`` packages.

    Each document is run through :func:`verify_handoff` with the per-package
    anchor time the caller resolved (None when the package carries no verified
    anchor). The set rolls up the pass count, the verifiable / corroborated
    tiers, and the producer-identity pinning. Never raises on a malformed
    document: it becomes a failing entry that gates the set. The set is ``ok``
    iff every document loaded and every loaded package verified for the chosen
    mode. The pinning gap (no package pinned its producer) is advisory.
    """
    entries: list[HandoffSetEntry] = []
    loaded = 0
    passed = 0
    verifiable = 0
    corroborated = 0
    pinned = 0

    for name, doc, anchor_time in packages:
        try:
            verdict = verify_handoff(
                doc, anchor_attested_time=anchor_time,
                trusted_did_document=trusted_did_document, strict=strict,
            )
        except ValueError as exc:
            entries.append(
                HandoffSetEntry(
                    name, False, False, False, False,
                    "self_asserted_unpinned", None, str(exc)
                )
            )
            continue

        loaded += 1
        if verdict.verifiable:
            verifiable += 1
        if verdict.corroborated:
            corroborated += 1
        if verdict.producer_identity_basis == "pinned":
            pinned += 1
        if verdict.ok:
            passed += 1
        entries.append(
            HandoffSetEntry(
                name=name,
                loaded=True,
                ok=verdict.ok,
                verifiable=verdict.verifiable,
                corroborated=verdict.corroborated,
                producer_identity_basis=verdict.producer_identity_basis,
                producer=verdict.producer,
                error=None,
            )
        )

    all_loaded = all(e.loaded for e in entries)
    all_ok = all(e.ok for e in entries if e.loaded)
    pinning_gap = loaded > 0 and pinned == 0

    return HandoffSetReport(
        ok=all_loaded and all_ok,
        strict=strict,
        total=len(entries),
        loaded=loaded,
        passed=passed,
        verifiable=verifiable,
        corroborated=corroborated,
        pinned=pinned,
        entries=tuple(entries),
        pinning_gap=pinning_gap,
    )
