"""Enforcement verification over a whole directory of (record, report, VCEK) triples.

``verify_enforcement`` binds one signed SEP-2828 record to one SEV-SNP report.
An auditor who receives a pile of enforced records, one report and VCEK per
record, asks the batch question: across the whole set, how many bind to a
confidential VM, and at what tier? ``check_enforcement_set`` is the receiving
side of that, the batch twin of :func:`verify_enforcement` and the enforcement
analogue of :func:`check_bundle_set`.

Each triple is run through :func:`verify_enforcement`. The set then rolls up:

* **Pass count.** How many triples are ``ok`` for the chosen mode. A triple
  whose report does not parse, whose signature fails, or whose ``REPORT_DATA``
  does not bind is a failing entry that gates the set, never a silent drop.
* **Tier tally.** How many landed at each tier (``unverified`` / ``bound`` /
  ``measurement_pinned``). The tier ``attested`` is reserved for the future
  KDS-chained release and is never emitted in v0, so it never appears here.
* **Pinning coverage.** How many records pinned a launch measurement. The
  coverage note (advisory) fires when no record in the set pinned an image:
  the whole set bound to *a* CVM but never to a *vetted* one, the enforcement
  analogue of the lens coverage gap.

The set is ``ok`` iff every triple loaded and every loaded triple verified for
the chosen mode; the pinning coverage note is advisory and does not gate. Like
the single verb, this needs the crypto the binding uses, so it runs only with
the attestation extra installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from vaara.attestation._enforcement import verify_enforcement
from vaara.attestation.tee import TEEAttestationError

ENFORCEMENT_SET_SCHEMA_NAME = "sep2828-enforcement-set"
ENFORCEMENT_SET_SCHEMA_VERSION = 1

# The tiers a v0 verdict can carry, in ascending order. ``attested`` is
# deliberately absent: it is reserved for the chain-rooted future tier and is
# never emitted, so it is not a key the tally pre-seeds.
TIER_NAMES = ("unverified", "bound", "measurement_pinned")


@dataclass(frozen=True)
class EnforcementSetEntry:
    """Per-triple verdict inside a set.

    ``loaded`` is False when the triple could not be evaluated at all (an
    unreadable report, a malformed VCEK, a record that will not canonicalise);
    then ``error`` names why and ``tier`` is ``unverified``. Otherwise ``ok``,
    ``tier``, and ``bound`` come from the enforcement verdict.
    """

    name: str
    loaded: bool
    ok: bool
    tier: str
    bound: bool
    measurement_basis: str
    error: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "loaded": self.loaded,
            "ok": self.ok,
            "tier": self.tier,
            "bound": self.bound,
            "measurementBasis": self.measurement_basis,
            "error": self.error,
        }


@dataclass(frozen=True)
class EnforcementSetReport:
    """Outcome of binding a whole set of records to SEV-SNP reports."""

    ok: bool
    strict: bool
    total: int
    loaded: int
    passed: int
    bound: int
    measurement_pinned: int
    tier_counts: dict[str, int]
    entries: tuple[EnforcementSetEntry, ...]
    pinning_gap: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ENFORCEMENT_SET_SCHEMA_NAME,
            "schemaVersion": ENFORCEMENT_SET_SCHEMA_VERSION,
            "ok": self.ok,
            "strict": self.strict,
            "total": self.total,
            "loaded": self.loaded,
            "passed": self.passed,
            "bound": self.bound,
            "measurementPinned": self.measurement_pinned,
            "tierCounts": dict(self.tier_counts),
            "pinningGap": self.pinning_gap,
            "entries": [e.to_dict() for e in self.entries],
        }


def check_enforcement_set(
    triples: Sequence[tuple[str, Any, bytes, bytes]],
    *,
    expected_measurement: Optional[str] = None,
    strict: bool = False,
) -> EnforcementSetReport:
    """Bind a set of ``(name, record, report_bytes, vcek_pem)`` triples.

    Each triple is run through :func:`verify_enforcement`. The set rolls up the
    pass count, the per-tier tally, and the pinning coverage. Never raises on a
    bad triple: a report that will not parse, a malformed VCEK, or a record that
    will not canonicalise becomes a failing entry that gates the set. The set is
    ``ok`` iff every triple loaded and every loaded triple verified for the
    chosen mode. The pinning gap (no record pinned a measurement) is advisory.
    """
    entries: list[EnforcementSetEntry] = []
    tier_counts: dict[str, int] = {name: 0 for name in TIER_NAMES}
    loaded = 0
    passed = 0
    bound = 0
    measurement_pinned = 0

    for name, record, report_bytes, vcek_pem in triples:
        try:
            verdict = verify_enforcement(
                record, report_bytes, vcek_pem,
                expected_measurement=expected_measurement, strict=strict,
            )
        except (TEEAttestationError, ValueError, KeyError, TypeError) as exc:
            entries.append(
                EnforcementSetEntry(
                    name, False, False, "unverified", False, "unpinned", str(exc)
                )
            )
            continue

        loaded += 1
        # A future tier outside the v0 set would still be counted, never dropped.
        tier_counts[verdict.tier] = tier_counts.get(verdict.tier, 0) + 1
        if verdict.bound:
            bound += 1
        if verdict.measurement_basis == "pinned":
            measurement_pinned += 1
        if verdict.ok:
            passed += 1
        entries.append(
            EnforcementSetEntry(
                name=name,
                loaded=True,
                ok=verdict.ok,
                tier=verdict.tier,
                bound=verdict.bound,
                measurement_basis=verdict.measurement_basis,
                error=None,
            )
        )

    all_loaded = all(e.loaded for e in entries)
    all_ok = all(e.ok for e in entries if e.loaded)
    pinning_gap = loaded > 0 and measurement_pinned == 0

    return EnforcementSetReport(
        ok=all_loaded and all_ok,
        strict=strict,
        total=len(entries),
        loaded=loaded,
        passed=passed,
        bound=bound,
        measurement_pinned=measurement_pinned,
        tier_counts=tier_counts,
        entries=tuple(entries),
        pinning_gap=pinning_gap,
    )
