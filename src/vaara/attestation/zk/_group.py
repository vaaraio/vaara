"""P-256 (secp256r1) group and scalar field, implemented from the public NIST
parameters using only the standard library.

The `cryptography` library deliberately does not expose arbitrary elliptic-curve
point arithmetic (add and scalar-multiply on chosen points), which the Pedersen
commitments and the range-proof argument require. This module provides that
arithmetic directly. It is used only by the zero-knowledge decisionProof engine,
never by the signing or anchoring paths, which continue to use `cryptography`.
"""

from __future__ import annotations

import hashlib

# NIST P-256 / secp256r1 domain parameters.
P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
A = P - 3
B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
GX = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
GY = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551

# P-256 has p == 3 (mod 4), so a square root is r^((p+1)/4) when one exists.
_SQRT_EXP = (P + 1) // 4


def _mod_sqrt(v: int) -> int | None:
    """Return a square root of v mod P, or None if v is not a quadratic residue."""
    v %= P
    r = pow(v, _SQRT_EXP, P)
    if (r * r) % P == v:
        return r
    return None


class Point:
    """An affine point on P-256, or the point at infinity (x is None)."""

    __slots__ = ("x", "y")

    def __init__(self, x: int | None, y: int | None):
        self.x = x
        self.y = y

    def is_infinity(self) -> bool:
        return self.x is None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def double(self) -> "Point":
        if self.is_infinity() or self.y == 0:
            return INF
        assert self.x is not None and self.y is not None
        s = ((3 * self.x * self.x + A) * pow(2 * self.y, -1, P)) % P
        x3 = (s * s - 2 * self.x) % P
        y3 = (s * (self.x - x3) - self.y) % P
        return Point(x3, y3)

    def __add__(self, other: "Point") -> "Point":
        if self.is_infinity():
            return other
        if other.is_infinity():
            return self
        assert self.x is not None and self.y is not None
        assert other.x is not None and other.y is not None
        if self.x == other.x:
            if (self.y + other.y) % P == 0:
                return INF
            return self.double()
        s = ((other.y - self.y) * pow(other.x - self.x, -1, P)) % P
        x3 = (s * s - self.x - other.x) % P
        y3 = (s * (self.x - x3) - self.y) % P
        return Point(x3, y3)

    def mul(self, k: int) -> "Point":
        k %= N
        result = INF
        addend = self
        while k:
            if k & 1:
                result = result + addend
            addend = addend.double()
            k >>= 1
        return result

    def to_bytes(self) -> bytes:
        """SEC1 compressed encoding (0x00 for the point at infinity)."""
        if self.is_infinity():
            return b"\x00"
        assert self.x is not None and self.y is not None
        prefix = 0x02 | (self.y & 1)
        return bytes([prefix]) + self.x.to_bytes(32, "big")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Point":
        if data == b"\x00":
            return INF
        if len(data) != 33 or data[0] not in (0x02, 0x03):
            raise ValueError("invalid compressed point encoding")
        x = int.from_bytes(data[1:33], "big")
        if x >= P:
            raise ValueError("non-canonical point: x >= field prime")
        rhs = (x * x * x + A * x + B) % P
        y = _mod_sqrt(rhs)
        if y is None:
            raise ValueError("x is not on the curve")
        if (y & 1) != (data[0] & 1):
            y = P - y
        return Point(x, y)


INF = Point(None, None)
G = Point(GX, GY)


def scalar_mul(k: int, pt: Point) -> Point:
    return pt.mul(k)


def hash_to_scalar(*chunks: bytes) -> int:
    h = hashlib.sha256(b"".join(chunks)).digest()
    return int.from_bytes(h, "big") % N


def hash_to_point(label: bytes) -> Point:
    """Deterministic try-and-increment hash-to-curve. Returns a point whose y is
    even, so the result is independent of the sign convention."""
    for ctr in range(256):
        x = int.from_bytes(
            hashlib.sha256(label + ctr.to_bytes(4, "big")).digest(), "big"
        ) % P
        rhs = (x * x * x + A * x + B) % P
        y = _mod_sqrt(rhs)
        if y is not None:
            if y & 1:
                y = P - y
            return Point(x, y)
    raise ValueError("hash_to_point failed to find a point")
