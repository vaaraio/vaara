"""Dataclasses and serialization helpers for SEP-2787 envelopes.

Internal module. Public surface is in ``vaara.attestation.sep2787``.

v2 envelope shape lands the four changes Vaara committed to in
``modelcontextprotocol/modelcontextprotocol#2787`` after the trust-surface
grouping was incorporated into the SEP draft on soup-oss commit
``dd030d5b``:

1. ``toolCalls`` lives under ``payloadDerived``, not ``plannerDeclared``.
   Tool bindings are facts derived from the request payload, not planner
   declarations.
2. ``argsProjection`` serialises with a JSON-stringified ``projection``
   field carrying the JCS-canonical encoding of the projection object.
   The digest is taken over those bytes.
3. The v1 ``kind``-discriminated union is dropped; the two commitment
   shapes (``ArgsRef``, ``ArgsProjection``) self-discriminate by which
   fields are present.
4. Commitment-only audit composes on ``ArgsProjection`` as a
   hash-only-identity projection of the form ``{"digest": "sha256:..."}``;
   no separate ``ArgsDigest`` type ships in the spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

Algorithm = Literal["HS256", "ES256", "RS256"]
VALID_ALGS: frozenset[str] = frozenset({"HS256", "ES256", "RS256"})


class AttestationError(RuntimeError):
    """Raised when SEP-2787 envelope construction or verification fails."""


@dataclass(frozen=True)
class ArgsRef:
    """Content-addressed reference. Verifier may fetch and check digest."""

    ref: str
    digest: str
    canonicalization: Literal["jcs"] = "jcs"


@dataclass(frozen=True)
class ArgsProjection:
    """Reviewed projection of the args, with its own digest.

    ``projection`` is the JCS-canonical JSON encoding of the projection
    object as a UTF-8 string. ``projection_digest`` is ``sha256:<hex>``
    over the UTF-8 bytes of ``projection``.

    Commitment-only audit (payload stays local) is expressed as a
    hash-only-identity projection: ``projection`` carries the
    JCS-canonical encoding of ``{"digest": "sha256:..."}`` and the
    embedded digest binds the underlying arguments. The verifier
    recomputes the same digest from the runtime arguments and rejects
    on mismatch. See ``make_args_digest``.
    """

    projection: str
    projection_digest: str


ArgsCommitment = Union[ArgsRef, ArgsProjection]


@dataclass(frozen=True)
class ToolCallBinding:
    """One tool-call entry: name + server fingerprint + args commitment.

    Lives under ``payloadDerived.toolCalls`` in the v2 envelope. Each
    binding is a fact derived from the request payload, not a planner
    declaration.
    """

    name: str
    server_fingerprint: str
    args: ArgsCommitment


@dataclass(frozen=True)
class PlannerDeclared:
    """Trust surface 1: what the client / agent planner claims.

    Holds intent and an optional requested-capability claim. Tool-call
    bindings moved to ``payloadDerived.toolCalls`` in the v2 envelope.
    The issuer binds these fields under its signing key but does not
    assert their truth, only that the planner claimed them.
    """

    intent: str
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
class PayloadDerived:
    """Trust surface 3: facts derived from the request payload.

    Holds the tool-call bindings (name, server fingerprint, args
    commitment). The args commitment is computed deterministically from
    the request payload, not declared by the planner.
    """

    tool_calls: tuple[ToolCallBinding, ...]


@dataclass(frozen=True)
class Attestation:
    """SEP-2787 tool call attestation envelope, v2 shape.

    Three trust-surface blocks plus the signature. The signature is
    computed over the JCS-canonical encoding of ``{version, alg,
    plannerDeclared, issuerAsserted, payloadDerived}`` and does not
    cover itself.
    """

    version: int
    alg: Algorithm
    planner_declared: PlannerDeclared
    issuer_asserted: IssuerAsserted
    payload_derived: PayloadDerived
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "alg": self.alg,
            "plannerDeclared": planner_to_dict(self.planner_declared),
            "issuerAsserted": issuer_to_dict(self.issuer_asserted),
            "payloadDerived": payload_to_dict(self.payload_derived),
            "signature": self.signature,
        }


def args_to_dict(args: ArgsCommitment) -> dict[str, Any]:
    if isinstance(args, ArgsRef):
        return {
            "ref": args.ref,
            "digest": args.digest,
            "canonicalization": args.canonicalization,
        }
    if isinstance(args, ArgsProjection):
        return {
            "projection": args.projection,
            "projectionDigest": args.projection_digest,
        }
    raise AttestationError(f"unknown args commitment kind: {type(args)!r}")


def tool_call_to_dict(tc: ToolCallBinding) -> dict[str, Any]:
    return {
        "name": tc.name,
        "serverFingerprint": tc.server_fingerprint,
        "args": args_to_dict(tc.args),
    }


def planner_to_dict(planner: PlannerDeclared) -> dict[str, Any]:
    out: dict[str, Any] = {"intent": planner.intent}
    if planner.requested_capability is not None:
        out["requestedCapability"] = planner.requested_capability
    return out


def issuer_to_dict(issuer: IssuerAsserted) -> dict[str, Any]:
    return {
        "alg": issuer.alg,
        "expSeconds": issuer.exp_seconds,
        "iat": issuer.iat,
        "iss": issuer.iss,
        "nonce": issuer.nonce,
        "secretVersion": issuer.secret_version,
        "sub": issuer.sub,
    }


def payload_to_dict(payload: PayloadDerived) -> dict[str, Any]:
    return {
        "toolCalls": [tool_call_to_dict(tc) for tc in payload.tool_calls],
    }


def args_from_dict(d: dict[str, Any]) -> ArgsCommitment:
    if "ref" in d:
        if "digest" not in d:
            raise AttestationError("ArgsRef missing 'digest'")
        return ArgsRef(
            ref=d["ref"],
            digest=d["digest"],
            canonicalization=d.get("canonicalization", "jcs"),
        )
    if "projection" in d:
        if "projectionDigest" not in d:
            raise AttestationError("ArgsProjection missing 'projectionDigest'")
        return ArgsProjection(
            projection=d["projection"],
            projection_digest=d["projectionDigest"],
        )
    raise AttestationError(
        "args commitment missing both 'ref' and 'projection'; cannot discriminate"
    )


def tool_call_from_dict(d: dict[str, Any]) -> ToolCallBinding:
    for required in ("name", "serverFingerprint", "args"):
        if required not in d:
            raise AttestationError(f"toolCall missing required field {required!r}")
    return ToolCallBinding(
        name=d["name"],
        server_fingerprint=d["serverFingerprint"],
        args=args_from_dict(d["args"]),
    )


def planner_from_dict(d: dict[str, Any]) -> PlannerDeclared:
    if "intent" not in d:
        raise AttestationError("plannerDeclared missing required field 'intent'")
    return PlannerDeclared(
        intent=d["intent"],
        requested_capability=d.get("requestedCapability"),
    )


def issuer_from_dict(d: dict[str, Any]) -> IssuerAsserted:
    for required in (
        "alg", "expSeconds", "iat", "iss", "nonce", "secretVersion", "sub"
    ):
        if required not in d:
            raise AttestationError(f"issuerAsserted missing required field {required!r}")
    if d["alg"] not in VALID_ALGS:
        raise AttestationError(f"unsupported alg {d['alg']!r}")
    return IssuerAsserted(
        alg=d["alg"],
        exp_seconds=d["expSeconds"],
        iat=d["iat"],
        iss=d["iss"],
        nonce=d["nonce"],
        secret_version=d["secretVersion"],
        sub=d["sub"],
    )


def payload_from_dict(d: dict[str, Any]) -> PayloadDerived:
    if "toolCalls" not in d:
        raise AttestationError("payloadDerived missing required field 'toolCalls'")
    calls = d["toolCalls"]
    if not isinstance(calls, list):
        raise AttestationError("payloadDerived.toolCalls must be an array")
    return PayloadDerived(
        tool_calls=tuple(tool_call_from_dict(c) for c in calls),
    )


def attestation_from_dict(d: dict[str, Any]) -> Attestation:
    """Reconstruct an Attestation from its wire JSON dict.

    Inverse of ``Attestation.to_dict()``. Accepts a parsed JSON object
    (camelCase keys on the boundary), reconstructs the Python dataclass
    tree, and returns an Attestation ready for ``verify_attestation``.
    Field-presence validation only; signature verification still
    requires the caller's keying material.
    """
    for required in (
        "version", "alg", "plannerDeclared", "issuerAsserted",
        "payloadDerived", "signature",
    ):
        if required not in d:
            raise AttestationError(f"attestation missing required field {required!r}")
    if d["alg"] not in VALID_ALGS:
        raise AttestationError(f"unsupported alg {d['alg']!r}")
    return Attestation(
        version=d["version"],
        alg=d["alg"],
        planner_declared=planner_from_dict(d["plannerDeclared"]),
        issuer_asserted=issuer_from_dict(d["issuerAsserted"]),
        payload_derived=payload_from_dict(d["payloadDerived"]),
        signature=d["signature"],
    )
