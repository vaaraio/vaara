#!/usr/bin/env python3
"""Generate the revocation_freshness_v0 conformance vectors.

draft-sirkkavaara-vaara-receipt-08 Section 10 states the limit these vectors
hold a verifier to: offline verification is a computation over the parameters
the consumer holds, revocation is a property of the present, and a signature
that verifies is not evidence that the signing key is still valid. Where a
decision depends on revocation state, the freshness a deployment accepts is
an operational parameter that deployment must state.

So the suite is about the QUALIFIER on a clean answer, not about the answer.
Seven cases pin the two rules that follow:

1. A not-revoked verdict may be read as a statement about now only when the
   registry carries an observation instant AND the caller states a staleness
   bound AND the registry falls inside it. Missing either input yields
   ``unknown``, never a quiet promotion to current.
2. A revocation the verifier can see is binding however stale the registry
   is, because a revocation fact does not expire. Only the negative verdict
   weakens. ``revoked_stale`` is the case that pins this and it is the one
   most likely to regress.

Deterministic: no keys, no signatures, no clock. Every instant is a literal,
so re-running produces byte-identical cases. Run from the repo root:
``python tests/vectors/revocation_freshness_v0/_generate.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from vaara.attestation.receipt import RevocationEntry, RevocationRegistry

HERE = Path(__file__).resolve().parent

ISS = "did:web:agent.example"
KEYID = "key-1"
IAT = "2026-09-01T00:00:00Z"
NOW = "2026-09-02T12:00:00Z"
HOUR = 3600.0

# Observed two hours before NOW: inside a one-day bound, outside a one-hour one.
RECENT = "2026-09-02T10:00:00Z"
OLD = "2026-08-01T00:00:00Z"
FUTURE = "2026-12-01T00:00:00Z"

REVOKED_BEFORE_IAT = "2026-08-15T00:00:00Z"


def main() -> None:
    cases: list[dict] = []
    expected: dict[str, dict] = {}

    def case(name, registry, *, now, bound, note):
        st = registry.status(
            ISS, IAT, keyid=KEYID, now=now, max_staleness_seconds=bound
        )
        cases.append(
            {
                "name": name,
                "note": note,
                "iss": ISS,
                "keyid": KEYID,
                "issued_at": IAT,
                "now": now,
                "max_staleness_seconds": bound,
                "registry": registry.to_dict(),
            }
        )
        expected[name] = {
            "revoked": st.revoked,
            "freshness": st.freshness,
            "establishes_current": st.establishes_current,
            "registry_digest": registry.digest(),
        }

    case(
        "fresh_clean",
        RevocationRegistry([], as_of=RECENT),
        now=NOW,
        bound=24 * HOUR,
        note="observed two hours ago against a one day bound; the only case "
        "whose clean answer speaks to the present",
    )
    case(
        "stale_clean",
        RevocationRegistry([], as_of=RECENT),
        now=NOW,
        bound=HOUR,
        note="same registry, one hour bound; two hours old is outside it, so "
        "the clean answer says nothing about revocations since",
    )
    case(
        "undated_clean",
        RevocationRegistry([]),
        now=NOW,
        bound=24 * HOUR,
        note="no observation instant, so there is no basis on which to call "
        "the clean answer current however generous the bound",
    )
    case(
        "unbounded_clean",
        RevocationRegistry([], as_of=RECENT),
        now=NOW,
        bound=None,
        note="registry is dated but the deployment stated no bound; Section 10 "
        "puts that parameter on the deployment, so absent it the answer is unknown",
    )
    case(
        "revoked_fresh",
        RevocationRegistry(
            [RevocationEntry("key", KEYID, REVOKED_BEFORE_IAT)], as_of=RECENT
        ),
        now=NOW,
        bound=24 * HOUR,
        note="revoked in time and the registry is fresh; revoked, and a "
        "revoked verdict never establishes current validity either",
    )
    case(
        "revoked_stale",
        RevocationRegistry(
            [RevocationEntry("key", KEYID, REVOKED_BEFORE_IAT)], as_of=OLD
        ),
        now=NOW,
        bound=HOUR,
        note="THE ASYMMETRY. The registry is a month stale and the receipt is "
        "still revoked, because a revocation fact does not expire. Staleness "
        "weakens only the negative answer",
    )
    case(
        "future_as_of",
        RevocationRegistry([], as_of=FUTURE),
        now=NOW,
        bound=24 * HOUR,
        note="observed after now, so the two clocks disagree and the bound "
        "cannot be evaluated honestly; unknown rather than trivially fresh",
    )

    (HERE / "cases.json").write_text(
        json.dumps({"version": 1, "cases": cases}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(cases)} cases")


if __name__ == "__main__":
    main()
