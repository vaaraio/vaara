"""The constrained decision predicate, compiled to a set of range statements.

The runtime verdict is a threshold branch over a risk score. Proving membership
of a small set of non-negative differences in [0, 2**RANGE_BITS) is exactly the
statement "this verdict is the correct branch". A false verdict forces at least
one negative difference, which has no valid range proof, so the prover cannot
even construct a proof for a wrong verdict.

Thresholds follow the usual ordering escalate <= deny (lower risk escalates,
higher risk blocks). The record vocabulary is allow / block / escalate.
"""

from __future__ import annotations

from ._params import RANGE_BITS, SCALE

_MAX = 1 << RANGE_BITS


def to_fixed(x: float) -> int:
    """Scale a [0, 1] float to a fixed-point integer in [0, 2**RANGE_BITS)."""
    v = round(x * SCALE)
    if not (0 <= v < _MAX):
        raise ValueError(f"value {x} out of fixed-point range")
    return v


def decide(score_fp: int, deny_fp: int, escalate_fp: int) -> str:
    if score_fp >= deny_fp:
        return "block"
    if score_fp >= escalate_fp:
        return "escalate"
    return "allow"


def range_witnesses(
    verdict: str, score_fp: int, deny_fp: int, escalate_fp: int
) -> list[int]:
    """Non-negative differences that are all in range iff `verdict` is the true
    output of `decide`. Raises ValueError if the verdict is inconsistent with the
    values (the soundness anchor: a lie has no witness)."""
    if verdict == "block":
        ws = [score_fp - deny_fp]
    elif verdict == "escalate":
        ws = [score_fp - escalate_fp, deny_fp - score_fp - 1]
    elif verdict == "allow":
        ws = [escalate_fp - score_fp - 1, deny_fp - score_fp - 1]
    else:
        raise ValueError(f"unknown verdict {verdict!r}")
    for w in ws:
        if not (0 <= w < _MAX):
            raise ValueError("verdict inconsistent with committed values")
    return ws
