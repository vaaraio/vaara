# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full-lens verification over a whole directory of evidence bundles.

``verify_evidence_bundle`` runs every lens over one bundle. An auditor who
receives a pile of bundles, possibly from more than one issuer, asks the
batch question: over the whole set, how many verify, and what does the
evidence cover? ``check_bundle_set`` is the receiving side of that, the
full-lens twin of :func:`check_record_set` (which answers the keyless
conformance question; this one runs the signed, crypto-backed lenses).

Each bundle document is loaded with :func:`evidence_bundle_from_json` and
run through :func:`verify_evidence_bundle`. The set then rolls up:

* **Pass count.** How many bundles are ``ok``: signature established and
  every applicable lens passed. A bundle that fails to load (a malformed
  document) is a failing entry that gates the set, never a silent drop.
* **Authentication count.** How many had their signature established at
  all. A set whose bundles verify their logs but never their signatures is
  a different posture than one that authenticates every issuer.
* **Lens coverage.** For each lens, how many bundles carried the evidence
  it needs and how many passed. The coverage gap (advisory) names the
  lenses no bundle in the set exercised: evidence the whole set never
  carried, the batch analogue of the executed-coverage hole.

The set is ``ok`` iff every bundle loaded and every loaded bundle verified;
the coverage gaps are reported but do not gate. Unlike ``check_record_set``,
this needs the crypto the lenses use, so it runs only with the attestation
extra installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from vaara.attestation._bundle import LENS_NAMES, verify_evidence_bundle
from vaara.attestation._bundle_io import evidence_bundle_from_json

BUNDLE_SET_SCHEMA_NAME = "sep2828-evidence-bundle-set"
BUNDLE_SET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BundleSetEntry:
    """Per-bundle verdict inside a set.

    ``loaded`` is False when the document did not parse into a bundle; then
    ``error`` names why and ``lens_states`` is empty. Otherwise ``ok`` and
    ``authenticity_established`` come from the bundle verdict and
    ``lens_states`` maps each lens to ``pass`` / ``fail`` / ``n/a``.
    """

    name: str
    loaded: bool
    ok: bool
    authenticity_established: bool
    keyid: Optional[str]
    lens_states: dict[str, str]
    error: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "loaded": self.loaded,
            "ok": self.ok,
            "authenticityEstablished": self.authenticity_established,
            "keyid": self.keyid,
            "lensStates": dict(self.lens_states),
            "error": self.error,
        }


@dataclass(frozen=True)
class BundleSetReport:
    """Outcome of running the full lens stack over a set of bundles."""

    ok: bool
    total: int
    loaded: int
    passed: int
    authenticated: int
    entries: tuple[BundleSetEntry, ...]
    lens_applicable: dict[str, int]
    lens_passed: dict[str, int]
    lens_gaps: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BUNDLE_SET_SCHEMA_NAME,
            "schemaVersion": BUNDLE_SET_SCHEMA_VERSION,
            "ok": self.ok,
            "total": self.total,
            "loaded": self.loaded,
            "passed": self.passed,
            "authenticated": self.authenticated,
            "lensApplicable": dict(self.lens_applicable),
            "lensPassed": dict(self.lens_passed),
            "lensGaps": list(self.lens_gaps),
            "entries": [e.to_dict() for e in self.entries],
        }


def check_bundle_set(bundles: Sequence[tuple[str, Any]]) -> BundleSetReport:
    """Run the full lens stack over a set of ``(name, parsed_json)`` bundles.

    Each document is loaded into an :class:`EvidenceBundle` and verified;
    the set rolls up the pass and authentication counts and the per-lens
    coverage. Never raises on a malformed document: it becomes a failing
    entry that gates the set. The set is ``ok`` iff every document loaded
    and every loaded bundle verified. The coverage gaps (lenses no bundle
    exercised) are advisory and do not gate.
    """
    entries: list[BundleSetEntry] = []
    lens_applicable: dict[str, int] = {name: 0 for name in LENS_NAMES}
    lens_passed: dict[str, int] = {name: 0 for name in LENS_NAMES}
    loaded = 0
    passed = 0
    authenticated = 0

    for name, doc in bundles:
        try:
            bundle = evidence_bundle_from_json(doc)
        except ValueError as exc:
            entries.append(
                BundleSetEntry(name, False, False, False, None, {}, str(exc))
            )
            continue

        loaded += 1
        verdict = verify_evidence_bundle(bundle)
        states: dict[str, str] = {}
        for result in verdict.lenses:
            if result.applicable:
                lens_applicable[result.lens] += 1
                if result.ok:
                    lens_passed[result.lens] += 1
                states[result.lens] = "pass" if result.ok else "fail"
            else:
                states[result.lens] = "n/a"
        if verdict.authenticity_established:
            authenticated += 1
        if verdict.ok:
            passed += 1
        entries.append(
            BundleSetEntry(
                name=name,
                loaded=True,
                ok=verdict.ok,
                authenticity_established=verdict.authenticity_established,
                keyid=verdict.keyid,
                lens_states=states,
                error=None,
            )
        )

    lens_gaps = tuple(name for name in LENS_NAMES if lens_applicable[name] == 0)
    all_loaded = all(e.loaded for e in entries)
    all_ok = all(e.ok for e in entries if e.loaded)

    return BundleSetReport(
        ok=all_loaded and all_ok,
        total=len(entries),
        loaded=loaded,
        passed=passed,
        authenticated=authenticated,
        entries=tuple(entries),
        lens_applicable=lens_applicable,
        lens_passed=lens_passed,
        lens_gaps=lens_gaps,
    )
