"""Typed capability constraints for capability-mode grants (Phase C).

Internal module. Public surface is in ``vaara.credential``.

A Phase A grant pins exact arguments via ``scope.argsCommitment`` and the
gateway enforces byte-equality. A capability grant instead carries a list of
typed constraints over named arguments, and the gateway enforces each at call
time, so one credential authorizes a *bounded class* of calls (the
authority-layer move: control what is allowed, not just prove what happened).

Coverage is CLOSED: in capability mode the set of runtime argument keys must
equal the set of constrained argument names. An unnamed runtime arg fails
``capability_uncovered``; pin an argument that must not vary with an ``eq``
constraint.

Each capability is a signed, closed-schema block (camelCase-free: the wire
keys are ``arg``/``op``/``value``). Numeric bounds are decimal-as-string;
floats and bools in the signed value are rejected, consistent with the JCS
reject-floats guard in ``_sep2787_canonical``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from vaara.attestation._attest_types import AttestationError

CAPABILITY_KEYS = frozenset({"arg", "op", "value"})
NUMERIC_OPS = frozenset({"le", "ge"})
VALID_OPS = NUMERIC_OPS | frozenset({"eq", "in"})


@dataclass(frozen=True)
class Capability:
    """One typed constraint over a named tool argument.

    ``op`` is ``le``/``ge`` (numeric, Decimal compare), ``eq`` (pin to a fixed
    value), or ``in`` (membership). ``value`` is a decimal string for
    ``le``/``ge``, an arbitrary string for ``eq``, or a tuple of strings for
    ``in``.
    """

    arg: str
    op: str
    value: Any


def capability_to_dict(c: Capability) -> dict[str, Any]:
    value: Any = list(c.value) if c.op == "in" else c.value
    return {"arg": c.arg, "op": c.op, "value": value}


def capability_from_dict(d: dict[str, Any]) -> Capability:
    """Parse one capability wire object under its closed schema."""
    if not isinstance(d, dict):
        raise AttestationError("capability must be an object")
    extra = set(d) - CAPABILITY_KEYS
    if extra:
        raise AttestationError(
            f"capability carries unrecognized field(s) {sorted(extra)!r}; "
            "the signed schema is closed"
        )
    arg = d.get("arg")
    op = d.get("op")
    if not isinstance(arg, str) or not arg:
        raise AttestationError("capability.arg must be a non-empty string")
    if op not in VALID_OPS:
        raise AttestationError(f"capability.op must be one of {sorted(VALID_OPS)!r}")
    value = d.get("value")
    if op == "in":
        if not isinstance(value, list) or not value:
            raise AttestationError("capability.value for 'in' must be a non-empty list")
        if not all(isinstance(item, str) for item in value):
            raise AttestationError("capability.value 'in' items must be strings")
        return Capability(arg=arg, op=op, value=tuple(value))
    if op in NUMERIC_OPS:
        if not isinstance(value, str) or not value:
            raise AttestationError(f"capability.value for {op!r} must be a decimal string")
        try:
            Decimal(value)
        except InvalidOperation as e:
            raise AttestationError(f"capability.value {value!r} is not a decimal") from e
        return Capability(arg=arg, op=op, value=value)
    # eq
    if not isinstance(value, str):
        raise AttestationError("capability.value for 'eq' must be a string")
    return Capability(arg=arg, op=op, value=value)


def _as_decimal(v: Any) -> Decimal | None:
    """Coerce a runtime arg value to Decimal, or None if not a real number."""
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        return None
    try:
        return Decimal(str(v))
    except InvalidOperation:
        return None


def _check(cap: Capability, actual: Any) -> bool:
    if cap.op == "eq":
        return not isinstance(actual, bool) and str(actual) == cap.value
    if cap.op == "in":
        return not isinstance(actual, bool) and str(actual) in cap.value
    a = _as_decimal(actual)
    bound = _as_decimal(cap.value)
    if a is None or bound is None:
        return False  # fail closed on a malformed value or bound
    return a <= bound if cap.op == "le" else a >= bound


def evaluate(capabilities: tuple[Capability, ...], runtime_args: Any) -> tuple[bool, str]:
    """Enforce capability constraints against runtime args (closed coverage).

    Returns ``(ok, reason)``: ``(True, "ok")`` when every runtime arg is named
    by a capability and every constraint holds; otherwise ``(False, reason)``
    with ``capability_uncovered`` (a runtime arg no capability names) or
    ``capability_exceeded`` (a constraint failed or a named arg is missing).
    """
    if not isinstance(runtime_args, dict):
        return (False, "capability_exceeded")
    named = {c.arg for c in capabilities}
    for key in runtime_args:
        if key not in named:
            return (False, "capability_uncovered")
    for cap in capabilities:
        if cap.arg not in runtime_args or not _check(cap, runtime_args[cap.arg]):
            return (False, "capability_exceeded")
    return (True, "ok")


def _demo() -> None:
    """Self-check: bound holds, over-bound / unnamed / pinned-mismatch fail."""
    caps = (
        Capability("amount", "le", "500"),
        Capability("vendor", "in", ("acme", "globex")),
        Capability("destination", "eq", "0xABC"),
    )
    ok = {"amount": 400, "vendor": "acme", "destination": "0xABC"}
    assert evaluate(caps, ok) == (True, "ok")
    assert evaluate(caps, {**ok, "amount": 600}) == (False, "capability_exceeded")
    assert evaluate(caps, {**ok, "vendor": "evilcorp"}) == (False, "capability_exceeded")
    assert evaluate(caps, {**ok, "destination": "0xDEAD"}) == (False, "capability_exceeded")
    assert evaluate(caps, {**ok, "memo": "hi"}) == (False, "capability_uncovered")
    assert evaluate(caps, {"amount": 400, "vendor": "acme"}) == (False, "capability_exceeded")
    # round-trip + bool rejected as a numeric
    assert capability_from_dict(capability_to_dict(caps[0])) == caps[0]
    assert evaluate((Capability("n", "le", "5"),), {"n": True}) == (False, "capability_exceeded")
    # malformed bound fails closed rather than raising
    assert evaluate((Capability("n", "le", "abc"),), {"n": 1}) == (False, "capability_exceeded")
    print("ok")


if __name__ == "__main__":
    _demo()
