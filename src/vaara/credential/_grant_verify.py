# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify a brokered credential against the runtime call it claims to authorize.

Internal module. Public surface is in ``vaara.credential``.

``verify_grant`` is the standalone, composable check a gateway runs before
letting a tool execute. It gates on five facts, reported as the first failing
reason so an expired-but-authentic grant is never mislabeled as a bad
signature:

1. ``bad_signature``  - signature does not match the grant blocks.
2. ``expired``        - now is past ``iat + expSeconds`` (or the grant is
   future-dated), with the same clock-skew window the SEP-2787/inference
   verifiers use.
3. ``scope_mismatch`` - the runtime tool / tenant / args do not match the
   committed scope (args re-derived with ``make_args_digest``).
4. ``revoked``        - the issuer or its bound key was revoked at or before
   issuance (``RevocationRegistry``).
5. ``binding_unknown``- the bound attestation digest is not in the set of
   known mediation digests the verifier was given (fail-closed when no set is
   supplied).

This is detection of a defeated broker, not a mathematical-completeness claim.
Completeness holds only for tools placed behind a gateway that runs this; see
``gateway.py`` and the reconciliation fallback in the design doc.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._attest_canonical import iso8601_to_epoch, make_args_digest
from vaara.credential._grant_capability import evaluate
from vaara.credential._grant_emit import verify_grant_signature
from vaara.credential._grant_types import BrokeredCredential

GrantReason = str  # one of the literals documented above; "ok" on success


@dataclass(frozen=True)
class GrantVerdict:
    """Result of ``verify_grant``: ``ok`` plus the first failing ``reason``."""

    ok: bool
    reason: GrantReason


def verify_grant(
    credential: BrokeredCredential,
    *,
    verifying_material: Any,
    runtime_tool_name: str,
    runtime_args: Any,
    runtime_tenant_id: str,
    revocation: Any = None,
    known_attestation_digests: Optional[frozenset[str]] = None,
    now: Optional[float] = None,
    clock_skew_seconds: int = 30,
) -> GrantVerdict:
    """Check a credential against the runtime call. See module docstring."""
    if not verify_grant_signature(credential, verifying_material=verifying_material):
        return GrantVerdict(False, "bad_signature")

    asserted = credential.asserted
    iat_epoch = iso8601_to_epoch(asserted.iat)
    current = now if now is not None else time.time()
    if iat_epoch is None:
        return GrantVerdict(False, "expired")
    future_dated = iat_epoch > current + clock_skew_seconds
    deadline = iat_epoch + asserted.exp_seconds + clock_skew_seconds
    if future_dated or current > deadline:
        return GrantVerdict(False, "expired")

    scope = credential.scope
    if scope.tool_name != runtime_tool_name:
        return GrantVerdict(False, "scope_mismatch")
    if scope.tenant_id != runtime_tenant_id:
        return GrantVerdict(False, "scope_mismatch")
    if credential.capabilities:
        # Capability mode: enforce typed constraints (closed coverage) instead
        # of the exact args commitment, which becomes a mint-time anchor only.
        ok, reason = evaluate(credential.capabilities, runtime_args)
        if not ok:
            return GrantVerdict(False, reason)
    else:
        runtime_commitment = make_args_digest(runtime_args).projection_digest
        if scope.args_commitment != runtime_commitment:
            return GrantVerdict(False, "scope_mismatch")

    if revocation is not None:
        status = revocation.status(
            asserted.iss, asserted.iat, keyid=asserted.secret_version
        )
        if status.revoked:
            return GrantVerdict(False, "revoked")

    known = known_attestation_digests or frozenset()
    if credential.binding.attestation_digest not in known:
        return GrantVerdict(False, "binding_unknown")

    return GrantVerdict(True, "ok")
