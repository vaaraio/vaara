# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The standing permission: what a person allowed an agent, signed by that person.

Internal module. Public surface is in ``vaara.credential``.

A grant authorizes one call and is minted by the proxy. A permission is the
person's own statement, made once and valid for days or weeks: this agent may
call this tool, with arguments inside these bounds, and no more than this much
in total over this period. The person signs it with their own key, so a
verifier holding the trail and the permission can tell whether the agent
stayed within what the person allowed, which is a different question from
whether it stayed within what the operator's policy allowed.

Wire shape (``vaara.permission/v1``), a closed schema like every other signed
Vaara block::

    {
      "type": "vaara.permission/v1", "version": 1, "alg": "ES256",
      "principal": "...", "keyId": "...", "agent": "...",
      "toolName": "...", "tenantId": "",
      "notBefore": "2026-10-13T00:00:00Z", "notAfter": "2026-11-13T00:00:00Z",
      "nonce": "...",
      "capabilities": [{"arg": "amount", "op": "le", "value": "500"}],
      "limits": [
        {"kind": "sum", "arg": "amount", "max": "2000", "windowSeconds": 604800},
        {"kind": "count", "max": "20", "windowSeconds": 86400}
      ],
      "reviewers": [{"reviewer": "did:...", "alg": "ES256", "publicKey": "-----BEGIN..."}],
      "signature": "..."
    }

``capabilities`` bound each call, with the same operators and closed coverage
as a capability grant. ``limits`` bound calls taken together: ``sum`` adds one
numeric argument across the calls, ``count`` counts them. A limit with
``windowSeconds`` covers a rolling window ending at the call; without it, the
whole validity period. Amounts are decimal strings, never floats.

``reviewers`` is optional and omitted when empty, so a permission without it
signs and digests exactly as before the field existed. Each entry names a
person the principal trusts to resolve escalations under this permission and
pins that person's public key (PEM, ES256 or RS256). A resolution in the trail
is checked against these keys, never against a key the row carries; see
``_review_signature``.

The signature is over the JCS encoding of every field except ``signature``,
with the SEP-2787 signing stack the grants use. ``permission_digest`` is the
SHA-256 of the JCS encoding of the whole signed object, signature included,
and is what a grant or a record names the permission by.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Optional, Sequence

from vaara.attestation._attest_canonical import (
    canonical_json,
    iso8601_to_epoch,
    new_nonce,
)
from vaara.attestation._attest_signing import (
    sign_es256,
    sign_hs256,
    sign_rs256,
    verify_es256,
    verify_hs256,
    verify_rs256,
)
from vaara.attestation._attest_types import VALID_ALGS, Algorithm, AttestationError
from vaara.credential._grant_capability import (
    Capability,
    capability_from_dict,
    capability_to_dict,
    evaluate,
)

PERMISSION_TYPE = "vaara.permission/v1"
LIMIT_KINDS = frozenset({"sum", "count"})

PERMISSION_KEYS = frozenset({
    "type", "version", "alg", "principal", "keyId", "agent", "toolName",
    "tenantId", "notBefore", "notAfter", "nonce", "capabilities", "limits",
    "reviewers", "signature",
})
LIMIT_KEYS = frozenset({"kind", "arg", "max", "windowSeconds"})
REVIEWER_KEYS = frozenset({"reviewer", "alg", "publicKey"})
#: A shared secret cannot say which person signed, so reviewers sign with a key pair.
REVIEWER_ALGS = frozenset({"ES256", "RS256"})


@dataclass(frozen=True)
class Limit:
    """A bound over calls taken together."""

    kind: str
    max: str
    arg: Optional[str] = None
    window_seconds: Optional[int] = None


@dataclass(frozen=True)
class Reviewer:
    """A person the principal trusts to resolve escalations, with their pinned key."""

    reviewer: str
    alg: str
    public_key: str

    def to_dict(self) -> dict[str, Any]:
        return {"reviewer": self.reviewer, "alg": self.alg, "publicKey": self.public_key}

    def load_public_key(self) -> Any:
        """The pinned key as a key object; raises AttestationError if unusable."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec, rsa

        try:
            key = serialization.load_pem_public_key(self.public_key.encode())
        except (ValueError, TypeError) as exc:
            raise AttestationError(f"reviewer.publicKey is not a PEM public key: {exc}") from exc
        want = ec.EllipticCurvePublicKey if self.alg == "ES256" else rsa.RSAPublicKey
        if not isinstance(key, want):
            raise AttestationError(f"reviewer.publicKey does not match alg {self.alg}")
        return key


@dataclass(frozen=True)
class Permission:
    """A signed standing permission from a principal to one agent for one tool."""

    alg: Algorithm
    principal: str
    key_id: str
    agent: str
    tool_name: str
    tenant_id: str
    not_before: str
    not_after: str
    nonce: str
    capabilities: tuple[Capability, ...] = ()
    limits: tuple[Limit, ...] = ()
    signature: str = ""
    version: int = 1
    reviewers: tuple[Reviewer, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = _unsigned_dict(self)
        d["signature"] = self.signature
        return d


def _limit_to_dict(lim: Limit) -> dict[str, Any]:
    d: dict[str, Any] = {"kind": lim.kind, "max": lim.max}
    if lim.arg is not None:
        d["arg"] = lim.arg
    if lim.window_seconds is not None:
        d["windowSeconds"] = lim.window_seconds
    return d


def _unsigned_dict(p: Permission) -> dict[str, Any]:
    d = {
        "type": PERMISSION_TYPE,
        "version": p.version,
        "alg": p.alg,
        "principal": p.principal,
        "keyId": p.key_id,
        "agent": p.agent,
        "toolName": p.tool_name,
        "tenantId": p.tenant_id,
        "notBefore": p.not_before,
        "notAfter": p.not_after,
        "nonce": p.nonce,
        "capabilities": [capability_to_dict(c) for c in p.capabilities],
        "limits": [_limit_to_dict(lim) for lim in p.limits],
    }
    if p.reviewers:
        d["reviewers"] = [r.to_dict() for r in p.reviewers]
    return d


def _decimal(value: Any) -> Optional[Decimal]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        d = Decimal(str(value))
    except InvalidOperation:
        return None
    return d if d.is_finite() else None


def _check_limit(lim: Limit) -> None:
    if lim.kind not in LIMIT_KINDS:
        raise AttestationError(f"limit.kind must be one of {sorted(LIMIT_KINDS)!r}")
    bound = _decimal(lim.max) if isinstance(lim.max, str) else None
    if bound is None or bound < 0:
        raise AttestationError("limit.max must be a non-negative decimal string")
    if lim.kind == "sum" and not lim.arg:
        raise AttestationError("a sum limit names the argument it adds up")
    if lim.kind == "count" and lim.arg is not None:
        raise AttestationError("a count limit names no argument")
    if lim.window_seconds is not None and (
            isinstance(lim.window_seconds, bool) or not isinstance(lim.window_seconds, int)
            or lim.window_seconds <= 0):
        raise AttestationError("limit.windowSeconds must be a positive integer")


def _check_fields(p: Permission) -> None:
    if p.alg not in VALID_ALGS:
        raise AttestationError(f"unsupported alg: {p.alg!r}")
    for name in ("principal", "agent", "tool_name", "not_before", "not_after", "nonce"):
        if not isinstance(getattr(p, name), str) or not getattr(p, name):
            raise AttestationError(f"permission.{name} must be a non-empty string")
    start, end = iso8601_to_epoch(p.not_before), iso8601_to_epoch(p.not_after)
    if start is None or end is None or end <= start:
        raise AttestationError("notBefore and notAfter must be times with notBefore first")
    for lim in p.limits:
        _check_limit(lim)
    for r in p.reviewers:
        _check_reviewer(r)


def _check_reviewer(r: Reviewer) -> None:
    if not isinstance(r.reviewer, str) or not r.reviewer:
        raise AttestationError("reviewer.reviewer must be a non-empty string")
    if r.alg not in REVIEWER_ALGS:
        raise AttestationError(f"reviewer.alg must be one of {sorted(REVIEWER_ALGS)!r}")
    if not isinstance(r.public_key, str):
        raise AttestationError("reviewer.publicKey must be a PEM string")
    r.load_public_key()


def signing_payload(p: Permission) -> bytes:
    """The exact bytes the permission's signature covers."""
    return canonical_json(_unsigned_dict(p))


def sign_permission(
    *,
    principal: str,
    key_id: str,
    agent: str,
    tool_name: str,
    not_before: str,
    not_after: str,
    alg: Algorithm,
    signing_material: Any,
    tenant_id: str = "",
    capabilities: Sequence[Capability] = (),
    limits: Sequence[Limit] = (),
    nonce: Optional[str] = None,
    reviewers: Sequence[Reviewer] = (),
) -> Permission:
    """Build and sign a permission. ``signing_material`` is the principal's key:
    bytes for HS256, a private-key object for ES256 / RS256."""
    unsigned = Permission(
        alg=alg, principal=principal, key_id=key_id, agent=agent,
        tool_name=tool_name, tenant_id=tenant_id, not_before=not_before,
        not_after=not_after, nonce=nonce or new_nonce(),
        capabilities=tuple(capabilities), limits=tuple(limits),
        reviewers=tuple(reviewers),
    )
    _check_fields(unsigned)
    payload = signing_payload(unsigned)
    if alg == "HS256":
        if not isinstance(signing_material, (bytes, bytearray)):
            raise AttestationError("HS256 requires a bytes shared secret")
        sig = sign_hs256(payload, shared_secret=bytes(signing_material))
    elif alg == "ES256":
        sig = sign_es256(payload, private_key=signing_material)
    else:
        sig = sign_rs256(payload, private_key=signing_material)
    return Permission(**{**unsigned.__dict__, "signature": sig})


def verify_permission_signature(p: Permission, *, verifying_material: Any) -> bool:
    """True iff the signature matches under the principal's verifying key."""
    payload = signing_payload(p)
    if p.alg == "HS256":
        if not isinstance(verifying_material, (bytes, bytearray)):
            return False
        return verify_hs256(payload, signature_hex=p.signature,
                            shared_secret=bytes(verifying_material))
    if p.alg == "ES256":
        return verify_es256(payload, signature_hex=p.signature, public_key=verifying_material)
    if p.alg == "RS256":
        return verify_rs256(payload, signature_hex=p.signature, public_key=verifying_material)
    return False


def permission_digest(p: Permission) -> str:
    """``sha256:`` digest of the whole signed object, signature included."""
    return "sha256:" + hashlib.sha256(canonical_json(p.to_dict())).hexdigest()


def _field(d: dict[str, Any], key: str, default: Any = None) -> Any:
    """A raw wire value; ``_check_fields`` validates it after construction."""
    return d.get(key, default)


def permission_from_dict(d: Any) -> Permission:
    """Parse a permission. Unknown fields, a wrong type tag or a malformed limit
    are refused, so the parsed object is byte-exact to what was signed."""
    if not isinstance(d, dict):
        raise AttestationError("permission must be an object")
    extra = set(d) - PERMISSION_KEYS
    if extra:
        raise AttestationError(
            f"permission carries unrecognized field(s) {sorted(extra)!r}; "
            "the signed schema is closed"
        )
    if _field(d, "type") != PERMISSION_TYPE:
        raise AttestationError(f"permission.type must be {PERMISSION_TYPE!r}")
    if _field(d, "version") != 1:
        raise AttestationError("permission.version must be 1")
    caps = _field(d, "capabilities", [])
    lims = _field(d, "limits", [])
    if not isinstance(caps, list) or not isinstance(lims, list):
        raise AttestationError("capabilities and limits must be lists")
    limits = []
    for item in lims:
        if not isinstance(item, dict):
            raise AttestationError("limit must be an object")
        extra = set(item) - LIMIT_KEYS
        if extra:
            raise AttestationError(f"limit carries unrecognized field(s) {sorted(extra)!r}")
        limits.append(Limit(kind=_field(item, "kind"), max=_field(item, "max"),
                            arg=_field(item, "arg"), window_seconds=_field(item, "windowSeconds")))
    revs = _field(d, "reviewers", [])
    if not isinstance(revs, list) or ("reviewers" in d and not revs):
        # An empty list is refused rather than dropped: the field is omitted
        # when empty, so a parsed permission stays byte-exact to what was signed.
        raise AttestationError("reviewers must be a non-empty list when present")
    reviewers = []
    for item in revs:
        if not isinstance(item, dict):
            raise AttestationError("reviewer must be an object")
        extra = set(item) - REVIEWER_KEYS
        if extra:
            raise AttestationError(f"reviewer carries unrecognized field(s) {sorted(extra)!r}")
        reviewers.append(Reviewer(reviewer=_field(item, "reviewer"), alg=_field(item, "alg"),
                                  public_key=_field(item, "publicKey")))
    p = Permission(
        alg=_field(d, "alg"), principal=_field(d, "principal"), key_id=_field(d, "keyId", ""),
        agent=_field(d, "agent"), tool_name=_field(d, "toolName"),
        tenant_id=_field(d, "tenantId", ""), not_before=_field(d, "notBefore"),
        not_after=_field(d, "notAfter"), nonce=_field(d, "nonce"),
        capabilities=tuple(capability_from_dict(c) for c in caps),
        limits=tuple(limits), signature=_field(d, "signature", ""),
        reviewers=tuple(reviewers),
    )
    if not isinstance(p.key_id, str) or not isinstance(p.tenant_id, str) \
            or not isinstance(p.signature, str):
        raise AttestationError("keyId, tenantId and signature must be strings")
    _check_fields(p)
    return p


def evaluate_call(
    p: Permission,
    args: Any,
    at: float,
    prior: Iterable[tuple[float, Any]] = (),
) -> tuple[bool, str]:
    """Would a call with ``args`` at time ``at`` stay within the permission?

    ``prior`` is every earlier call made under it that was allowed, as
    ``(timestamp, args)``. Returns ``(True, "ok")`` or ``(False, reason)`` with
    ``outside_validity``, ``capability_exceeded``, ``capability_uncovered``,
    ``limit_argument_unreadable`` or ``limit_exceeded: <detail>``.
    """
    start, end = iso8601_to_epoch(p.not_before), iso8601_to_epoch(p.not_after)
    if start is None or end is None or not start <= at <= end:
        return (False, "outside_validity")
    if p.capabilities:
        ok, reason = evaluate(p.capabilities, args)
        if not ok:
            return (False, reason)
    earlier = [(t, a) for t, a in prior if t <= at]
    for lim in p.limits:
        floor = at - lim.window_seconds if lim.window_seconds is not None else start
        in_window = [a for t, a in earlier if t > floor]
        bound = Decimal(lim.max)
        span = f"{lim.window_seconds} s" if lim.window_seconds is not None else "validity"
        if lim.kind == "count":
            total = Decimal(len(in_window) + 1)
            if total > bound:
                return (False, f"limit_exceeded: count {total} > {bound} in {span}")
            continue
        this = _decimal(args.get(lim.arg)) if isinstance(args, dict) else None
        if this is None:
            return (False, "limit_argument_unreadable")
        total = this
        for a in in_window:
            value = _decimal(a.get(lim.arg)) if isinstance(a, dict) else None
            total += value if value is not None else Decimal(0)
        if total > bound:
            return (False, f"limit_exceeded: sum {lim.arg} {total} > {bound} in {span}")
    return (True, "ok")


def check_trail_against_permission(
    trail: Any,
    p: Permission,
    *,
    verifying_material: Any,
    verify_chain: bool = True,
) -> Any:
    """Replay a trail against a permission, offline.

    Every recorded request from the permission's agent to its tool and tenant,
    inside the validity period, is judged with :func:`evaluate_call` against
    the calls before it that the trail shows as allowed. Allowed calls count
    toward the totals whether or not they were inside the permission, because
    they happened. States and ``ok`` are those of ``check_trail_against_grant``:
    an allowed call outside the permission is ``violated``, a refused one
    ``refused``, an escalated one ``held``. Calls outside the validity period
    are ``outside_window``. The chain and the principal's signature are checked
    first; if either fails, no call is judged.
    """
    from collections import Counter

    from vaara.audit.trail import EventType
    from vaara.credential._trail_grant import GrantedCall, TrailGrantReport, _last_decision

    def report(**kw: Any) -> TrailGrantReport:
        return TrailGrantReport(p.tool_name, p.tenant_id, **kw)

    if verify_chain:
        broken = trail.verify_chain()
        if broken:
            return report(chain_problem=str(broken))
    if not verify_permission_signature(p, verifying_material=verifying_material):
        return report(grant_problem="bad_signature")

    allowed: list[tuple[float, Any]] = []
    calls: list[GrantedCall] = []
    for rec in list(trail._records):
        if rec.event_type != EventType.ACTION_REQUESTED:
            continue
        if (rec.agent_id != p.agent or rec.tool_name != p.tool_name
                or (rec.tenant_id or "") != p.tenant_id):
            continue
        decision = _last_decision(trail.get_action_trail(rec.action_id)[1:])
        params = rec.data.get("parameters") if isinstance(rec.data, dict) else None
        if isinstance(params, dict) and params.get("_truncated"):
            state, reason = "unverifiable", "parameters_truncated"
        else:
            args = params if params is not None else {}
            fits, reason = evaluate_call(p, args, rec.timestamp, allowed)
            if reason == "outside_validity":
                state, reason = "outside_window", ""
            elif fits:
                state = "within" if decision else "undecided"
            elif decision == "deny":
                state = "refused"
            elif decision == "escalate":
                state = "held"
            elif decision:
                state = "violated"
            else:
                state = "undecided"
            if decision == "allow":
                allowed.append((rec.timestamp, args))
        calls.append(GrantedCall(
            action_id=rec.action_id, timestamp=rec.timestamp, agent_id=rec.agent_id,
            state=state, decision=decision, reason="" if reason == "ok" else reason,
        ))
    return report(calls=tuple(calls), counts=dict(Counter(c.state for c in calls)))
