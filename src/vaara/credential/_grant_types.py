# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dataclasses and serialization for brokered-credential (grant) envelopes.

Internal module. Public surface is in ``vaara.credential``.

A brokered credential is a standalone signed envelope the proxy mints between
the *allow* decision and the upstream *forward*. It is the authority half of
Vaara's "separate intelligence from authority" move: the model proposes a tool
call, but only a valid, unexpired, non-revoked, attestation-bound grant lets a
gateway-protected tool actually run.

The credential is NOT embedded in the attestation or the receipt. The
attestation is emitted *before* the forward and the receipt *after* the
outcome, so a credential that must travel *with* the request cannot live in a
receipt that does not exist yet. The grant instead pins the attestation digest
(``binding.attestationDigest``), which the receipt also back-links, so an
auditor can join grant -> attestation -> receipt offline.

The signing stack is the SEP-2787 one (HS256 / ES256 / RS256 over RFC 8785
JCS) reused unchanged, so the grant key matches the attestation/receipt key.
Each signed block is a closed schema (parsers in ``_grant_parse``): an
unrecognized wire key is a hard reject, keeping the modeled preimage
byte-exact to the wire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vaara.attestation._attest_types import Algorithm
from vaara.credential._grant_capability import Capability, capability_to_dict

GrantAlgorithm = Algorithm


@dataclass(frozen=True)
class GrantScope:
    """What the credential authorizes: one tool, one args commitment, one tenant.

    ``args_commitment`` is the ``ArgsProjection.projection_digest`` of the
    attested arguments. The gateway re-derives it from the runtime arguments
    and rejects on mismatch, so a mutated argument after minting fails scope.
    """

    tool_name: str
    args_commitment: str
    tenant_id: str


@dataclass(frozen=True)
class GrantBinding:
    """The attestation instance this grant is bound to.

    ``attestation_digest`` is ``attestation_digest(att)``; ``attestation_nonce``
    is the attestation's ``issuerAsserted.nonce``.
    """

    attestation_digest: str
    attestation_nonce: str


@dataclass(frozen=True)
class GrantAsserted:
    """Issuer-asserted grant facts: who, for whom, when, how long, which key."""

    iss: str
    sub: str
    iat: str
    exp_seconds: int
    nonce: str
    secret_version: str


@dataclass(frozen=True)
class BrokeredCredential:
    """A signed, short-lived, scoped, attestation-bound credential envelope.

    The signature is over the JCS encoding of
    ``{version, alg, scope, binding, asserted}`` and does not cover itself.
    """

    version: int
    alg: GrantAlgorithm
    scope: GrantScope
    binding: GrantBinding
    asserted: GrantAsserted
    signature: str
    capabilities: tuple[Capability, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "alg": self.alg,
            "scope": scope_to_dict(self.scope),
            "binding": binding_to_dict(self.binding),
            "asserted": asserted_to_dict(self.asserted),
            "signature": self.signature,
        }
        if self.capabilities:
            d["capabilities"] = [capability_to_dict(c) for c in self.capabilities]
        return d


def scope_to_dict(s: GrantScope) -> dict[str, Any]:
    return {
        "argsCommitment": s.args_commitment,
        "tenantId": s.tenant_id,
        "toolName": s.tool_name,
    }


def binding_to_dict(b: GrantBinding) -> dict[str, Any]:
    return {
        "attestationDigest": b.attestation_digest,
        "attestationNonce": b.attestation_nonce,
    }


def asserted_to_dict(a: GrantAsserted) -> dict[str, Any]:
    return {
        "expSeconds": a.exp_seconds,
        "iat": a.iat,
        "iss": a.iss,
        "nonce": a.nonce,
        "secretVersion": a.secret_version,
        "sub": a.sub,
    }


SCOPE_KEYS = frozenset({"argsCommitment", "tenantId", "toolName"})
BINDING_KEYS = frozenset({"attestationDigest", "attestationNonce"})
ASSERTED_KEYS = frozenset(
    {"expSeconds", "iat", "iss", "nonce", "secretVersion", "sub"}
)
GRANT_KEYS = frozenset(
    {"version", "alg", "scope", "binding", "asserted", "signature", "capabilities"}
)
