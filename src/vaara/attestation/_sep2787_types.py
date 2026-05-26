"""Dataclasses and serialization helpers for SEP-2787 envelopes.

Internal module. Public surface is in ``vaara.attestation.sep2787``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Union

Algorithm = Literal["HS256", "ES256", "RS256"]
VALID_ALGS: frozenset[str] = frozenset({"HS256", "ES256", "RS256"})


class AttestationError(RuntimeError):
    """Raised when SEP-2787 envelope construction or verification fails."""


@dataclass(frozen=True)
class ArgsDigest:
    """Commitment-only args binding. Payload never crosses the verifier.

    Privacy-friendly default. The audit invariant is "this call was
    bound to this exact commitment." The verifier sees ``digest`` but
    not the underlying arguments.
    """

    digest: str
    canonicalization: Literal["jcs", "cbor"] = "jcs"
    kind: Literal["digest"] = field(default="digest", init=False)


@dataclass(frozen=True)
class ArgsRef:
    """Content-addressed reference. Verifier may fetch and check digest."""

    ref: str
    digest: str
    canonicalization: Literal["jcs", "cbor"] = "jcs"
    kind: Literal["ref"] = field(default="ref", init=False)


@dataclass(frozen=True)
class ArgsProjection:
    """Redacted or transformed projection of the args, with its own digest.

    The projection is what the issuer reviewed and bound; the original
    arguments are NOT recoverable from the projection. Useful when the
    arguments contain personal data, secrets, or trade-secret payloads
    but the audit needs a reviewable summary.
    """

    projection: dict[str, Any]
    projection_digest: str
    kind: Literal["projection"] = field(default="projection", init=False)


ArgsCommitment = Union[ArgsDigest, ArgsRef, ArgsProjection]


@dataclass(frozen=True)
class ToolCallBinding:
    """One tool-call entry. The planner declares ``name`` and
    ``server_fingerprint``; ``args`` is the trust-surface-separated
    args binding for this call."""

    name: str
    server_fingerprint: str
    args: ArgsCommitment


@dataclass(frozen=True)
class PlannerDeclared:
    """Trust surface 1: what the client / agent planner claims.

    Set by the planner upstream of the issuer. The issuer binds these
    under its signing key but does not assert their truth, only that
    the planner claimed them.
    """

    intent: str
    tool_calls: tuple[ToolCallBinding, ...]
    requested_capability: Optional[str] = None


@dataclass(frozen=True)
class IssuerAsserted:
    """Trust surface 2: what the attestation issuer binds at signing time.

    Set by the issuer (often a separate identity from the planner: a
    compliance gateway or notary). The issuer's signature is the proof
    that these values were bound together at the issuance instant.
    """

    iss: str
    sub: str
    iat: str
    exp_seconds: int
    nonce: str
    secret_version: str
    alg: Algorithm


@dataclass(frozen=True)
class Attestation:
    """SEP-2787 tool call attestation envelope, proposed shape.

    Composed of three trust-surface blocks plus the signature. The
    signature is computed over the JCS-canonical encoding of
    ``{version, alg, planner_declared, issuer_asserted,
    payload_derived}`` and does not cover itself.

    The ``payload_derived`` tuple parallels
    ``planner_declared.tool_calls`` in order: index N here is the args
    commitment for the N-th tool call.
    """

    version: int
    alg: Algorithm
    planner_declared: PlannerDeclared
    issuer_asserted: IssuerAsserted
    payload_derived: tuple[ArgsCommitment, ...]
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "alg": self.alg,
            "planner_declared": planner_to_dict(self.planner_declared),
            "issuer_asserted": asdict(self.issuer_asserted),
            "payload_derived": [args_to_dict(a) for a in self.payload_derived],
            "signature": self.signature,
        }


def args_to_dict(args: ArgsCommitment) -> dict[str, Any]:
    if isinstance(args, ArgsDigest):
        return {
            "kind": "digest",
            "digest": args.digest,
            "canonicalization": args.canonicalization,
        }
    if isinstance(args, ArgsRef):
        return {
            "kind": "ref",
            "ref": args.ref,
            "digest": args.digest,
            "canonicalization": args.canonicalization,
        }
    if isinstance(args, ArgsProjection):
        return {
            "kind": "projection",
            "projection": args.projection,
            "projection_digest": args.projection_digest,
        }
    raise AttestationError(f"unknown args commitment kind: {type(args)!r}")


def planner_to_dict(planner: PlannerDeclared) -> dict[str, Any]:
    out: dict[str, Any] = {
        "intent": planner.intent,
        "tool_calls": [
            {
                "name": tc.name,
                "server_fingerprint": tc.server_fingerprint,
                "args": args_to_dict(tc.args),
            }
            for tc in planner.tool_calls
        ],
    }
    if planner.requested_capability is not None:
        out["requested_capability"] = planner.requested_capability
    return out
