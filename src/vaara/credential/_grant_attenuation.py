# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Delegated-privilege attenuation over capability grants.

Internal module. Public surface is re-exported from ``vaara.credential``.

A delegation chain is safe only if authority never *grows* as it is handed
down: agent B, acting on A's behalf, must not be able to do anything A could
not, and C (delegated by B) must not exceed B. This module decides that
relation over the typed ``Capability`` grants in ``_grant_capability``.

The core is ``capability_subsumes(parent, child)``: does the child grant allow
a *subset* of the calls the parent grant allows? Equivalently, is every runtime
-args dict the child would accept also accepted by the parent? If yes, the
child is an attenuation (narrowing) of the parent and the hand-off is safe.

The check is deliberately **sound and conservative**: it returns ``True`` only
when subsumption is provable, and ``False`` (fail closed) on anything it cannot
prove — mixed numeric/string domains, unbounded child ranges, differing arg
sets. It will never approve a broadening; at worst it conservatively rejects a
narrowing it cannot verify. That is the correct bias for an authority control.

This is an evidence/verification primitive, not a credential issuer: it decides
whether one grant attenuates another. Minting and rotating the grants stays in
the deployer's identity layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Sequence

from vaara.credential._grant_capability import Capability, _as_decimal


def _names(caps: Sequence[Capability]) -> set[str]:
    return {c.arg for c in caps}


def _child_finite_set(caps: Sequence[Capability]) -> Optional[frozenset[str]]:
    """The finite set of values the child permits for one arg, or None.

    Returns None when the child places no ``eq``/``in`` constraint (so the
    permitted set is not finite — it is a numeric interval or unconstrained).
    Any ``le``/``ge`` bounds present are applied to filter the finite set;
    non-numeric members are dropped against a numeric bound (they cannot
    satisfy it), which can yield the empty set — a child that permits nothing.
    """
    universe: Optional[set[str]] = None
    for c in caps:
        if c.op == "eq":
            s = {c.value}
        elif c.op == "in":
            s = set(c.value)
        else:
            continue
        universe = s if universe is None else (universe & s)
    if universe is None:
        return None
    for c in caps:
        if c.op in ("le", "ge"):
            bound = _as_decimal(c.value)
            if bound is None:
                return frozenset()  # malformed bound: child permits nothing
            kept = set()
            for m in universe:
                md = _as_decimal(m)
                if md is None:
                    continue
                if (c.op == "le" and md <= bound) or (c.op == "ge" and md >= bound):
                    kept.add(m)
            universe = kept
    return frozenset(universe)


def _child_bounds(caps: Sequence[Capability]) -> tuple[Optional[Decimal], Optional[Decimal]]:
    """Tightest (lower, upper) numeric bounds the child imposes via le/ge."""
    lo: Optional[Decimal] = None
    hi: Optional[Decimal] = None
    for c in caps:
        b = _as_decimal(c.value)
        if b is None:
            continue
        if c.op == "le":
            hi = b if hi is None else min(hi, b)
        elif c.op == "ge":
            lo = b if lo is None else max(lo, b)
    return lo, hi


def _child_guarantees(child_caps: Sequence[Capability], parent_cap: Capability) -> bool:
    """Does every value the child permits (for this arg) satisfy parent_cap?"""
    finite = _child_finite_set(child_caps)
    if finite is not None and len(finite) == 0:
        return True  # child permits nothing here -> vacuously within parent

    if parent_cap.op == "eq":
        return finite is not None and finite <= {parent_cap.value}
    if parent_cap.op == "in":
        return finite is not None and finite <= set(parent_cap.value)

    # Numeric parent bound (le / ge).
    bound = _as_decimal(parent_cap.value)
    if bound is None:
        return False
    if finite is not None:
        decs = [_as_decimal(m) for m in finite]
        if any(d is None for d in decs):
            return False  # a non-numeric permitted value cannot meet a numeric bound
        if parent_cap.op == "le":
            return all(d <= bound for d in decs)  # type: ignore[operator]
        return all(d >= bound for d in decs)  # type: ignore[operator]
    # Child is an interval: it guarantees the bound only if its own bound is
    # at least as tight. An unbounded side (None) cannot guarantee anything.
    lo, hi = _child_bounds(child_caps)
    if parent_cap.op == "le":
        return hi is not None and hi <= bound
    return lo is not None and lo >= bound


def capability_subsumes(
    parent: Sequence[Capability], child: Sequence[Capability]
) -> tuple[bool, str]:
    """Is ``child`` no broader than ``parent`` (child-allowed ⊆ parent-allowed)?

    Returns ``(True, "ok")`` when the child grant attenuates the parent, else
    ``(False, reason)`` — ``arg_set_mismatch`` when the two grants constrain
    different argument sets (not a clean narrowing, rejected fail-closed), or
    ``broadens:<arg>`` when the child would permit a value the parent forbids.
    """
    np_, nc_ = _names(parent), _names(child)
    if np_ != nc_:
        return (False, "arg_set_mismatch")
    for arg in np_:
        parent_caps = [c for c in parent if c.arg == arg]
        child_caps = [c for c in child if c.arg == arg]
        for pc in parent_caps:
            if not _child_guarantees(child_caps, pc):
                return (False, f"broadens:{arg}")
    return (True, "ok")


@dataclass(frozen=True)
class AttenuationReport:
    """Result of checking privilege attenuation down a delegation chain.

    ``ok`` is True when every hop attenuates (or holds) the prior grant.
    ``first_broadening_index`` is the index of the first child grant that
    broadens its parent (>=1), or -1 when the chain is clean. ``reason``
    carries the subsumption reason at that hop.
    """

    ok: bool
    first_broadening_index: int
    reason: str


def chain_is_attenuating(
    grants: Sequence[Sequence[Capability]],
) -> AttenuationReport:
    """Verify authority never grows along an ordered chain of grants.

    ``grants`` is ordered root -> leaf (e.g. the capability sets aligned to
    ``DelegationGraph.chain_for``). Each grant must subsume the next. A chain
    of length 0 or 1 is trivially attenuating.
    """
    for i in range(1, len(grants)):
        ok, reason = capability_subsumes(grants[i - 1], grants[i])
        if not ok:
            return AttenuationReport(False, i, reason)
    return AttenuationReport(True, -1, "ok")
