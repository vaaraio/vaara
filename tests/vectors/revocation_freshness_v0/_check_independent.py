#!/usr/bin/env python3
"""Independent conformance checker for the revocation_freshness_v0 vectors.

Imports only the standard library plus ``rfc8785``. It does not import Vaara.
A second implementation can run this file to confirm the freshness rules are
consumable from the committed bytes alone, which is what
draft-sirkkavaara-vaara-receipt-08 Section 10 asks of a consumer and what the
European Commission's Article 50 transparency guidelines describe at
paragraph 76 as detection "ideally locally executable on the digital device".

Two rules are reproduced from scratch:

1. **Revocation in time.** A registry entry binds when its ``revoked_at`` is
   at or before the receipt's ``issued_at``. Identity-scope entries match on
   ``iss``, key-scope entries on the bound keyid. Unparseable instants fail
   closed, so an unreadable date revokes rather than being skipped.
2. **Freshness.** ``fresh`` requires an ``as_of`` on the registry, a stated
   ``max_staleness_seconds``, both instants parseable, and an age between
   zero and the bound inclusive. Anything else is ``unknown``, except an age
   beyond the bound, which is ``stale``. An ``as_of`` in the future is
   ``unknown``, not fresh, because the clocks disagree.

``establishes_current`` is then true for exactly one combination: not revoked
AND fresh. A revoked verdict never establishes current validity, and neither
does a clean verdict the checker cannot date.

Run: ``python tests/vectors/revocation_freshness_v0/_check_independent.py``.
Exit code 0 means every case matched its expected verdict.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import rfc8785

HERE = Path(__file__).resolve().parent


def _parse_iso(value):
    """ISO 8601 to an aware datetime, or None when it will not parse."""
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _revoked(registry, iss, issued_at, keyid):
    """The revocation-in-time predicate, rebuilt from the text."""
    issued = _parse_iso(issued_at)
    for entry in registry.get("entries", []):
        scope = entry.get("scope")
        subject = entry.get("subject")
        if scope == "key":
            if keyid is None or subject != keyid:
                continue
        elif scope == "identity":
            if subject != iss:
                continue
        else:
            continue
        revoked_at = _parse_iso(entry.get("revoked_at"))
        if revoked_at is None or issued is None:
            return True  # fail closed on an instant that will not parse
        if revoked_at <= issued:
            return True
    return False


def _freshness(registry, now, bound):
    if bound is None:
        return "unknown"
    as_of = registry.get("as_of")
    if as_of is None:
        return "unknown"
    observed = _parse_iso(as_of)
    current = _parse_iso(now)
    if observed is None or current is None:
        return "unknown"
    age = (current - observed).total_seconds()
    if age < 0:
        return "unknown"
    return "fresh" if age <= bound else "stale"


def _digest(registry):
    return "sha256:" + hashlib.sha256(rfc8785.dumps(registry)).hexdigest()


def _evaluate(case):
    registry = case["registry"]
    revoked = _revoked(registry, case["iss"], case["issued_at"], case.get("keyid"))
    freshness = _freshness(registry, case.get("now"), case.get("max_staleness_seconds"))
    return {
        "revoked": revoked,
        "freshness": freshness,
        "establishes_current": (not revoked) and freshness == "fresh",
        "registry_digest": _digest(registry),
    }


def main() -> int:
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))

    failures = 0
    for case in cases:
        name = case["name"]
        got = _evaluate(case)
        want = expected.get(name)
        if want is None:
            print(f"FAIL {name}: no expected verdict committed")
            failures += 1
            continue
        if got != want:
            print(f"FAIL {name}")
            for key in sorted(set(got) | set(want)):
                if got.get(key) != want.get(key):
                    print(f"       {key}: got {got.get(key)!r} want {want.get(key)!r}")
            failures += 1
        else:
            print(
                f"ok   {name}: revoked={got['revoked']} "
                f"freshness={got['freshness']} "
                f"establishes_current={got['establishes_current']}"
            )

    print(f"\n{len(cases) - failures}/{len(cases)} cases matched")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
