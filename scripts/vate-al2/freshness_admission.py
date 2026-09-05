# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""VATE status freshness against the Vaara surface that is the same object.

The registry verdict comes from `RevocationRegistry.status()`, mapped as:

    source_issued_at -> RevocationRegistry(as_of=...)
    checked_at       -> now
    max_age_seconds  -> max_staleness_seconds

No clock skew participates in this path. That much was run on 2 September and
is reproduced here unchanged.

What follows it is the step from a freshness observation to an admission
decision, which the 2 September run did NOT evaluate. `verify_grant` consumes
`establishes_current` and refuses with `revocation_stale`, but only when the
deployment states a bound: with `max_staleness_seconds=None` the branch is
unreachable and a clean answer of any age stands. `CredentialGateway` forwards
the bound, so the verdict is reachable through the shipped enforcement path,
but it exposes no injectable clock, so the one-second boundary cannot be
pinned there. Each layer prints its own run, so a reader can see where they
diverge.

Usage:  python scripts/vate-al2/freshness_admission.py <path-to-context.json>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vaara.attestation._revocation import RevocationRegistry  # noqa: E402
from vaara.credential import verify_grant  # noqa: E402

from _common import (  # noqa: E402
    BINDING_DIGEST,
    ISS,
    TEST_KEY,
    epoch,
    gateway_params,
    header,
    load_fixture,
    mint_grant,
    mint_through_emitter,
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    fixture_path = Path(argv[1])
    ctx = load_fixture(fixture_path)
    header(
        "VATE AL2: deny-status-stale-just-over-boundary, freshness and admission",
        fixture_path,
    )

    issued_at = ctx["source_issued_at"]
    checked_at = ctx["checked_at"]
    max_age = float(ctx["max_age_seconds"])
    at_bound = _shift(issued_at, max_age)

    registry = RevocationRegistry((), as_of=issued_at)

    print("## The registry verdict at a fixed clock")
    print(f"  RevocationRegistry(entries=(), as_of={issued_at!r})")
    print()

    status = registry.status(
        ISS, issued_at, now=checked_at, max_staleness_seconds=max_age
    )
    print(f"  .status(now={checked_at!r}, max_staleness_seconds={max_age:.0f})")
    print(f"    revoked             {status.revoked}")
    print(f"    freshness           {status.freshness!r}")
    print(f"    registry_as_of      {status.registry_as_of!r}")
    print(f"    establishes_current {status.establishes_current}")
    print()

    controls = [
        (
            f"{max_age:.0f}s, exactly at the bound",
            registry.status(ISS, issued_at, now=at_bound, max_staleness_seconds=max_age),
        ),
        (
            f"{max_age + 1:.0f}s, one over (the case)",
            registry.status(ISS, issued_at, now=checked_at, max_staleness_seconds=max_age),
        ),
        (
            "no as_of at all",
            RevocationRegistry((), as_of=None).status(
                ISS, issued_at, now=checked_at, max_staleness_seconds=max_age
            ),
        ),
    ]
    print("  boundary controls:")
    for label, st in controls:
        print(f"    {label:<28} -> {st.freshness!r}")
    print()

    print("## Freshness carried into an admission decision")
    print("  This is the step the 2 September run did not evaluate. The deny and")
    print("  should_execute rows reported that day were a proposed mapping from")
    print("  establishes_current, not the output of a decision step. Here it is")
    print("  actually evaluated.")
    print()

    grant = mint_grant(iat=issued_at, exp_seconds=int(max_age))
    known = frozenset({BINDING_DIGEST})

    def admit(*, revocation_now: str, bound: float | None):
        return verify_grant(
            grant,
            verifying_material=TEST_KEY,
            runtime_tool_name=grant.scope.tool_name,
            runtime_args={"path": "/tmp/x"},
            runtime_tenant_id=grant.scope.tenant_id,
            revocation=registry,
            known_attestation_digests=known,
            # Pinned so the credential TTL cannot fail first and mask the
            # revocation branch. The TTL is the other script's subject.
            now=epoch(issued_at),
            clock_skew_seconds=0,
            revocation_now=revocation_now,
            max_staleness_seconds=bound,
        )

    admission = [
        (
            f"bound {max_age:.0f}s, registry {max_age + 1:.0f}s old (the case)",
            admit(revocation_now=checked_at, bound=max_age),
        ),
        (
            f"bound {max_age:.0f}s, registry exactly at the bound",
            admit(revocation_now=at_bound, bound=max_age),
        ),
        (
            "no bound stated (the shipped default)",
            admit(revocation_now=checked_at, bound=None),
        ),
    ]
    print("  verify_grant, fixed clock:")
    for label, verdict in admission:
        print(f"    {label:<48} -> ok={verdict.ok}, reason={verdict.reason!r}")
    print()

    gateway_rows = _run_gateway_cases()
    print("## The same verdicts through the shipped enforcement path")
    print("  CredentialGateway, with a real attestation and grant written to disk")
    print("  by AttestPairEmitter, as the proxy writes them. The gateway takes no")
    print("  now, so these run on the real clock and the one-second boundary above")
    print("  cannot be pinned here. What they show is that the bound and the skew")
    print("  reach verify_grant through the class itself.")
    print()
    for label, verdict in gateway_rows:
        print(f"    {label:<48} -> ok={verdict.ok}, reason={verdict.reason!r}")
    print()

    print("## Reading, in the terms the ask used")
    print("  admission_decision deny   evaluated, verify_grant reason")
    print("                            'revocation_stale'")
    print("  should_execute     false  evaluated, GrantVerdict.ok is False")
    print("  reason_codes              ordered by the verifier, not by me:")
    print("                            revoked is answered before staleness, so a")
    print("                            visible revocation is never downgraded to")
    print("                            revocation_stale by a stale registry")
    print()
    print("  Unchanged from 2 September. These stay written out as mismatches,")
    print("  because a table cell would overstate each one:")
    print("    required     unmapped. Vaara has no per-call notion of status")
    print("                 evidence being required. Absent registry is")
    print("                 revocation=None, which skips the check silently.")
    print("    availability analogue only, through freshness 'unknown'.")
    print("    status       'active' maps to revoked False only if credential")
    print("                 revocation and status-source liveness are one")
    print("                 predicate. I do not think they are.")
    print()

    print("## Assertions")
    ok = True
    checks = [
        ("case is stale", status.freshness == "stale"),
        ("case does not establish current", status.establishes_current is False),
        ("case is not revoked", status.revoked is False),
        ("at the bound is fresh", controls[0][1].freshness == "fresh"),
        ("one over is stale", controls[1][1].freshness == "stale"),
        ("no as_of is unknown", controls[2][1].freshness == "unknown"),
        (
            "admission denies with revocation_stale",
            admission[0][1].ok is False and admission[0][1].reason == "revocation_stale",
        ),
        ("admission at the bound admits", admission[1][1].ok is True),
        ("no bound stated admits", admission[2][1].ok is True),
        ("gateway without a bound admits an old registry", gateway_rows[0][1].ok is True),
        (
            "gateway with a bound refuses it as revocation_stale",
            gateway_rows[1][1].ok is False
            and gateway_rows[1][1].reason == "revocation_stale",
        ),
        (
            "gateway forwards skew 0, expired grant refused",
            gateway_rows[2][1].ok is False and gateway_rows[2][1].reason == "expired",
        ),
        ("gateway forwards skew 30, the same grant admitted", gateway_rows[3][1].ok is True),
    ]
    for label, good in checks:
        ok = ok and good
        print(f"  [{'pass' if good else 'FAIL'}] {label}")
    return 0 if ok else 1


def _run_gateway_cases() -> list:
    """Execute the forwarding claims instead of asserting them in prose.

    Two parameters are under test. ``max_staleness_seconds`` needs only a
    registry too old to speak to the present. ``clock_skew_seconds`` needs a
    grant that has actually expired, and the gateway accepts no injected
    clock, so the grant is minted with a one second lifetime and the script
    waits it out. Two seconds of real time, and it puts the shipped class under
    the same assertions as everything else here.
    """
    import tempfile
    import time

    from vaara.attestation._revocation import RevocationRegistry
    from vaara.credential import CredentialGateway

    old = "2020-01-01T00:00:00Z"
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        receipts = Path(tmp) / "receipts"
        receipts.mkdir()

        cred = mint_through_emitter(receipts)
        params = gateway_params(cred)

        def gw(**kwargs):
            return CredentialGateway(
                verifying_material=TEST_KEY,
                receipts_dir=receipts,
                expected_tenant=cred["scope"]["tenantId"],
                **kwargs,
            ).authorize(
                params,
                tool_name=cred["scope"]["toolName"],
                arguments={"path": "/tmp/x"},
            )

        rows.append(
            (
                "registry from 2020, no bound stated",
                gw(revocation=RevocationRegistry((), as_of=old)),
            )
        )
        rows.append(
            (
                "registry from 2020, bound 300s",
                gw(
                    revocation=RevocationRegistry((), as_of=old),
                    max_staleness_seconds=300,
                ),
            )
        )

        short = mint_through_emitter(receipts, grant_exp_seconds=1)
        short_params = gateway_params(short)

        def gw_short(skew: int):
            return CredentialGateway(
                verifying_material=TEST_KEY,
                receipts_dir=receipts,
                expected_tenant=short["scope"]["tenantId"],
                clock_skew_seconds=skew,
            ).authorize(
                short_params,
                tool_name=short["scope"]["toolName"],
                arguments={"path": "/tmp/x"},
            )

        time.sleep(2)
        rows.append(("grant 1s old past a 1s lifetime, skew 0", gw_short(0)))
        rows.append(("the same grant, skew 30 (the default)", gw_short(30)))
    return rows


def _shift(ts: str, seconds: float) -> str:
    """ISO 8601 timestamp moved forward by ``seconds``, in the fixture's form."""
    from datetime import datetime, timedelta, timezone

    base = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
    moved = base + timedelta(seconds=seconds)
    return moved.strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
