# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""VATE status freshness through the proposed credential-TTL bridge.

The bridge is Takao Sato's, unmodified, from discussion #502:

    source_issued_at  -> asserted.iat
    max_age_seconds   -> expSeconds
    checked_at        -> now
    clock_skew_seconds is a parameter of verify_grant

This calls ``vaara.credential.verify_grant`` directly. ``CredentialGateway``
takes no ``now``, so a fixed-clock run cannot go through it, and the whole
point of the case is a boundary one second wide.

The result to read is not the first line. It is the second: the shipped
default skew of 30 seconds covers a one-second margin, so the denial the case
expects appears only under a skew of zero.

Usage:  python scripts/vate-al2/ttl_bridge.py <path-to-context.json>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vaara.credential import verify_grant  # noqa: E402

from _common import (  # noqa: E402
    BINDING_DIGEST,
    SECRET,
    epoch,
    header,
    load_fixture,
    mint_grant,
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    fixture_path = Path(argv[1])
    ctx = load_fixture(fixture_path)
    header("VATE AL2: deny-status-stale-just-over-boundary, TTL bridge", fixture_path)

    issued_at = ctx["source_issued_at"]
    checked_at = ctx["checked_at"]
    max_age = int(ctx["max_age_seconds"])
    elapsed = epoch(checked_at) - epoch(issued_at)

    print("## Bridge, as proposed in the ask")
    print(f"  source_issued_at {issued_at}  -> asserted.iat      epoch {epoch(issued_at):.0f}")
    print(f"  checked_at       {checked_at}  -> now               epoch {epoch(checked_at):.0f}")
    print(f"  max_age_seconds  {max_age}                  -> expSeconds")
    print(f"  elapsed          {elapsed:.0f}s against a {max_age}s bound, over by {elapsed - max_age:.0f}s")
    print()

    grant = mint_grant(iat=issued_at, exp_seconds=max_age)
    known = frozenset({BINDING_DIGEST})

    def run(*, now: float, skew: int):
        return verify_grant(
            grant,
            verifying_material=SECRET,
            runtime_tool_name=grant.scope.tool_name,
            runtime_args={"path": "/tmp/x"},
            runtime_tenant_id=grant.scope.tenant_id,
            revocation=None,
            known_attestation_digests=known,
            now=now,
            clock_skew_seconds=skew,
        )

    at_checked = epoch(checked_at)
    rows = [
        ("clock_skew_seconds=0", run(now=at_checked, skew=0)),
        ("clock_skew_seconds=30 (shipped default)", run(now=at_checked, skew=30)),
        ("control: checked_at minus 1s, skew=0", run(now=at_checked - 1, skew=0)),
    ]

    print("## Raw output")
    for label, verdict in rows:
        print(f"  {label:<42} -> GrantVerdict(ok={verdict.ok}, reason={verdict.reason!r})")
    print()

    denied_at_zero = rows[0][1]
    admitted_at_default = rows[1][1]

    print("## Reading")
    print("  The predicted verdict reproduces: at zero skew the grant is expired,")
    print("  and the boundary sits exactly where the source reading put it.")
    print()
    print("  The line worth keeping is the second one. clock_skew_seconds defaults")
    print("  to 30 in verify_grant and in CredentialGateway, and an operator can")
    print("  set it, including to zero, through --attest-clock-skew-seconds. A")
    print("  one-second margin sits inside the default tolerance, so the shipped")
    print("  default admits. A case built to catch a verifier with no freshness")
    print("  check cannot distinguish it from one that has a check plus any")
    print("  ordinary clock tolerance, unless zero skew is part of what the case")
    print("  fixes or the margin clears the tolerances implementations ship.")
    print()
    print("## Assertions")
    ok = True
    for label, want_ok, want_reason, verdict in [
        ("skew=0 denies as expired", False, "expired", denied_at_zero),
        ("skew=30 admits", True, "ok", admitted_at_default),
        ("control admits", True, "ok", rows[2][1]),
    ]:
        good = verdict.ok is want_ok and verdict.reason == want_reason
        ok = ok and good
        print(f"  [{'pass' if good else 'FAIL'}] {label}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
