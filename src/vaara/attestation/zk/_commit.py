# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pedersen commitments and generator vectors over P-256.

A Pedersen commitment `commit(value, blind) = value*G + blind*H` is perfectly
hiding and computationally binding, given an `H` whose discrete log relative to
`G` is unknown. `H` and the auxiliary generator vectors are derived by
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



def random_scalar() -> int:
    """A uniform scalar in [0, N), from the system CSPRNG."""
    return secrets.randbelow(N)
