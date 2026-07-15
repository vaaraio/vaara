"""Pedersen commitments and generator vectors over P-256.

A Pedersen commitment `commit(value, blind) = value*G + blind*H` is perfectly
hiding and computationally binding, given an `H` whose discrete log relative to
`G` is unknown. `H` and the inner-product generator vectors are derived by
hash-to-curve from fixed labels, so no trusted setup is needed and anyone can
recompute them.
"""

from __future__ import annotations

import secrets

from ._group import G, N, Point, hash_to_point, scalar_mul

# Second Pedersen generator. Derived by hash-to-curve, so its discrete log to G
# is unknown by construction (nothing-up-my-sleeve).
H = hash_to_point(b"vaara/zk/H/v0")


def commit(value: int, blind: int) -> Point:
    """Pedersen commitment value*G + blind*H."""
    return scalar_mul(value, G) + scalar_mul(blind, H)


def gens(count: int) -> tuple[list[Point], list[Point]]:
    """Deterministic generator vectors (G_i, H_i) for the inner-product argument.

    Each generator is an independent hash-to-curve point, so the whole set is
    recomputable by a verifier and has no known discrete-log relations.
    """
    gv = [hash_to_point(b"vaara/zk/Gv/v0/" + i.to_bytes(4, "big")) for i in range(count)]
    hv = [hash_to_point(b"vaara/zk/Hv/v0/" + i.to_bytes(4, "big")) for i in range(count)]
    return gv, hv


def random_scalar() -> int:
    """A uniform scalar in [0, N), from the system CSPRNG."""
    return secrets.randbelow(N)
