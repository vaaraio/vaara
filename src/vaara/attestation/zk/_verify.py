"""Verifier for the decisionProof.

Rebuilds the published commitments and the verdict's homomorphic difference
targets, then checks each bit-decomposition range proof: that the bit commitments
sum (weighted by 2^i) to the target, and that every bit carries a valid Schnorr
OR-proof. Returns False on any structural or algebraic failure; never raises on
adversarial input.
"""

from __future__ import annotations

from ._commit import H
from ._group import G, N, Point, scalar_mul
from ._params import RANGE_BITS
from ._prove import (
    POINT_LEN,
    SCALAR_LEN,
    _neg,
    _or_challenge,
    _seed,
    _targets,
)

_OR_LEN = 2 * POINT_LEN + 3 * SCALAR_LEN
_BIT_LEN = POINT_LEN + _OR_LEN
_RANGE_LEN = RANGE_BITS * _BIT_LEN


def _witness_count(verdict: str) -> int:
    if verdict == "block":
        return 1
    if verdict in ("escalate", "allow"):
        return 2
    raise ValueError(f"unknown verdict {verdict!r}")


def _read_scalar(b: bytes) -> int:
    return int.from_bytes(b, "big")


def _or_verify(c: Point, blob: bytes, prefix: bytes) -> bool:
    a0 = Point.from_bytes(blob[0:POINT_LEN])
    a1 = Point.from_bytes(blob[POINT_LEN : 2 * POINT_LEN])
    off = 2 * POINT_LEN
    e0 = _read_scalar(blob[off : off + SCALAR_LEN])
    z0 = _read_scalar(blob[off + SCALAR_LEN : off + 2 * SCALAR_LEN])
    z1 = _read_scalar(blob[off + 2 * SCALAR_LEN : off + 3 * SCALAR_LEN])
    y0 = c
    y1 = c + _neg(G)
    e = _or_challenge(prefix, c, a0, a1)
    e1 = (e - e0) % N
    if scalar_mul(z0, H) != a0 + scalar_mul(e0, y0):
        return False
    if scalar_mul(z1, H) != a1 + scalar_mul(e1, y1):
        return False
    return True


def _range_verify(target: Point, blob: bytes, prefix: bytes) -> bool:
    acc = Point(None, None)  # identity
    for i in range(RANGE_BITS):
        base = i * _BIT_LEN
        ci = Point.from_bytes(blob[base : base + POINT_LEN])
        orblob = blob[base + POINT_LEN : base + _BIT_LEN]
        if not _or_verify(ci, orblob, prefix + b"/" + i.to_bytes(2, "big")):
            return False
        acc = acc + scalar_mul(1 << i, ci)
    return acc == target


def verify(proof: bytes, verdict: str, binding_digest_hex: str) -> bool:
    try:
        n_wit = _witness_count(verdict)
        expected = 3 * POINT_LEN + n_wit * _RANGE_LEN
        if len(proof) != expected:
            return False
        vs = Point.from_bytes(proof[0:POINT_LEN])
        vd = Point.from_bytes(proof[POINT_LEN : 2 * POINT_LEN])
        ve = Point.from_bytes(proof[2 * POINT_LEN : 3 * POINT_LEN])
        targets = _targets(verdict, vs, vd, ve)
        seed = _seed(binding_digest_hex, verdict, vs, vd, ve)
        off = 3 * POINT_LEN
        for j, target in enumerate(targets):
            blob = proof[off + j * _RANGE_LEN : off + (j + 1) * _RANGE_LEN]
            if not _range_verify(target, blob, seed + b"/w" + j.to_bytes(2, "big")):
                return False
        return True
    except (ValueError, IndexError):
        return False
