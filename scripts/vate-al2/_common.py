# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared scaffolding for the two VATE AL2 reproducers.

Everything here is non-VATE. It exists so that in each run exactly one thing
can fail: the check under test. Kept in one place so both scripts hold the
same constants and a reader has one file to audit rather than two copies.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from vaara.attestation._attest_canonical import iso8601_to_epoch, make_args_digest
from vaara.credential import GrantBinding, GrantScope, emit_grant

#: The pinned VATE fixture this work is scoped to.
FIXTURE_SHA256 = "5045e9bcf3711e2de9431d50befb03662c0c868988dcccd616d2392375a545b2"
FIXTURE_NAME = "status-stale-just-over-boundary-context.json"
FIXTURE_URL = (
    "https://raw.githubusercontent.com/Poke-nushi/Verifiable-Agent-Trust-Envelope"
    "/v0.4.0/conformance/al2-vate-v0.3/fixtures/" + FIXTURE_NAME
)

# Test scaffolding. The secret protects nothing and is published on purpose so
# the runs are reproducible byte for byte.
SECRET = b"x" * 32
TOOL = "read_file"
ARGS = {"path": "/tmp/x"}
TENANT = "t1"
ISS = "vaara-proxy"
SUB = "agent-1"
SECRET_VERSION = "v1"

# A binding digest is required to be a sha256: string and to be in the set of
# digests the verifier knows. Both scripts supply this one as known, so the
# binding check can never be the reason a run fails.
BINDING_DIGEST = "sha256:" + "0" * 64
BINDING_NONCE = "n-vate-al2"


def load_fixture(path: Path) -> dict[str, Any]:
    """Read the pinned fixture, refusing anything whose digest does not match."""
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != FIXTURE_SHA256:
        raise SystemExit(
            f"fixture digest mismatch\n"
            f"  expected {FIXTURE_SHA256}\n"
            f"  got      {got}\n"
            f"  fetch the pinned bytes from {FIXTURE_URL}"
        )
    return json.loads(raw.decode("utf-8"))


def mint_grant(*, iat: str, exp_seconds: int) -> Any:
    """A grant whose only reachable failure is the timing one under test."""
    return emit_grant(
        scope=GrantScope(
            tool_name=TOOL,
            args_commitment=make_args_digest(ARGS).projection_digest,
            tenant_id=TENANT,
        ),
        binding=GrantBinding(
            attestation_digest=BINDING_DIGEST,
            attestation_nonce=BINDING_NONCE,
        ),
        iss=ISS,
        sub=SUB,
        secret_version=SECRET_VERSION,
        alg="HS256",
        signing_material=SECRET,
        exp_seconds=exp_seconds,
        iat=iat,
        nonce="n-grant-vate-al2",
    )


def epoch(ts: str) -> float:
    """ISO 8601 to epoch seconds, refusing anything the parser cannot read."""
    value = iso8601_to_epoch(ts)
    if value is None:
        raise SystemExit(f"unparseable timestamp in fixture: {ts!r}")
    return value


def header(title: str, fixture_path: Path) -> None:
    """Print the provenance every external-SUT record needs to carry."""
    import vaara

    print(f"# {title}")
    print()
    print("## Consumed input")
    print(f"{FIXTURE_NAME}")
    print(f"  path   {fixture_path}")
    print(f"  sha256 {FIXTURE_SHA256} (verified)")
    print()
    print("## Environment")
    print(f"  vaara     {getattr(vaara, '__version__', 'unknown')}")
    print(f"  python    {platform.python_version()} ({sys.implementation.name})")
    print(f"  platform  {platform.system()} {platform.machine()}")
    print()
    print("## Non-VATE scaffolding held constant")
    print("  alg HS256 with a published test secret")
    print("  argsCommitment recomputed from the runtime arguments")
    print(f"  tool {TOOL!r} and tenant {TENANT!r} matching the grant scope")
    print("  binding digest supplied to the verifier as known")
    print("  no capabilities block, so the exact-args commitment path is taken")
    print()
