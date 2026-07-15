"""Prover for the decisionProof.

The statement: the committed score and thresholds (published as Pedersen
commitments V_s, V_d, V_e) yield the claimed verdict under the threshold rule,
without revealing them. The verdict determines a small set of non-negative
differences; each difference is shown to lie in [0, 2**RANGE_BITS) with a
bit-decomposition range proof, and each bit is shown to be 0 or 1 with a Schnorr
OR-proof. The two differences of a verdict are homomorphic combinations of the
same V_s, V_d, V_e, so they refer to one consistent set of hidden values.

Everything is transparent (generators are hash-to-curve, no trusted setup) and
bound by Fiat-Shamir to the params digest, the binding digest, the verdict, and
the published commitments, so a proof does not transfer to another record.
"""

from __future__ import annotations

from ._circuit import range_witnesses
from ._commit import H, commit, random_scalar
from ._group import G, N, Point, hash_to_scalar, scalar_mul
from ._params import RANGE_BITS, params_digest

POINT_LEN = 33
SCALAR_LEN = 32


def _neg(pt: Point) -> Point:
    return scalar_mul(N - 1, pt)


def _scalar_bytes(s: int) -> bytes:
    return (s % N).to_bytes(SCALAR_LEN, "big")


def _seed(binding_digest_hex: str, verdict: str, vs: Point, vd: Point, ve: Point) -> bytes:
    return b"|".join(
        [
            params_digest().encode(),
            binding_digest_hex.encode(),
            verdict.encode(),
            vs.to_bytes(),
            vd.to_bytes(),
            ve.to_bytes(),
        ]
    )


def _targets(verdict: str, vs: Point, vd: Point, ve: Point) -> list[Point]:
    """Homomorphic commitments to the verdict's non-negative differences. Order
    matches range_witnesses. A constant 1 is committed as G (blind 0)."""
    ng = _neg(G)
    if verdict == "block":
        return [vs + _neg(vd)]
    if verdict == "escalate":
        return [vs + _neg(ve), vd + _neg(vs) + ng]
    if verdict == "allow":
        return [ve + _neg(vs) + ng, vd + _neg(vs) + ng]
    raise ValueError(f"unknown verdict {verdict!r}")


def _or_challenge(prefix: bytes, c: Point, a0: Point, a1: Point) -> int:
    return hash_to_scalar(b"vaara/zk/or", prefix, c.to_bytes(), a0.to_bytes(), a1.to_bytes())


def _or_prove(c: Point, bit: int, r: int, prefix: bytes) -> bytes:
    """Schnorr OR-proof that c commits to 0 or 1: c = r*H (bit 0) or c - G = r*H
    (bit 1), in zero knowledge of which."""
    y = [c, c + _neg(G)]  # Y0, Y1
    t = bit
    f = 1 - bit
    # Simulate the false branch.
    e_f = random_scalar()
    z_f = random_scalar()
    a_f = scalar_mul(z_f, H) + scalar_mul((N - e_f) % N, y[f])
    # Commit honestly on the true branch.
    k = random_scalar()
    a_t = scalar_mul(k, H)
    a_list: list[Point | None] = [None, None]
    a_list[t] = a_t
    a_list[f] = a_f
    a0, a1 = a_list
    assert a0 is not None and a1 is not None
    e = _or_challenge(prefix, c, a0, a1)
    e_t = (e - e_f) % N
    z_t = (k + e_t * r) % N
    e0 = e_t if t == 0 else e_f
    z_list: list[int | None] = [None, None]
    z_list[t] = z_t
    z_list[f] = z_f
    z0, z1 = z_list
    assert z0 is not None and z1 is not None
    return (
        a0.to_bytes()
        + a1.to_bytes()
        + _scalar_bytes(e0)
        + _scalar_bytes(z0)
        + _scalar_bytes(z1)
    )


def _range_prove(w: int, gamma: int, prefix: bytes) -> bytes:
    """Bit-decomposition range proof that commit(w, gamma) is in [0, 2**RANGE_BITS).
    The per-bit blinds are chosen so sum 2^i * C_i equals the target commitment."""
    bits = [(w >> i) & 1 for i in range(RANGE_BITS)]
    r = [random_scalar() for _ in range(RANGE_BITS - 1)]
    partial = sum((1 << i) * r[i] for i in range(RANGE_BITS - 1)) % N
    inv_top = pow(1 << (RANGE_BITS - 1), -1, N)
    r.append(((gamma - partial) * inv_top) % N)
    out = bytearray()
    for i in range(RANGE_BITS):
        ci = commit(bits[i], r[i])
        out += ci.to_bytes()
        out += _or_prove(ci, bits[i], r[i], prefix + b"/" + i.to_bytes(2, "big"))
    return bytes(out)


def prove(
    verdict: str,
    score_fp: int,
    deny_fp: int,
    escalate_fp: int,
    binding_digest_hex: str,
) -> bytes:
    # Validate the verdict against the values; raises if inconsistent.
    ws = range_witnesses(verdict, score_fp, deny_fp, escalate_fp)

    gs, gd, ge = random_scalar(), random_scalar(), random_scalar()
    vs = commit(score_fp, gs)
    vd = commit(deny_fp, gd)
    ve = commit(escalate_fp, ge)

    if verdict == "block":
        blinds = [(gs - gd) % N]
    elif verdict == "escalate":
        blinds = [(gs - ge) % N, (gd - gs) % N]
    else:  # allow
        blinds = [(ge - gs) % N, (gd - gs) % N]

    seed = _seed(binding_digest_hex, verdict, vs, vd, ve)
    out = bytearray()
    out += vs.to_bytes() + vd.to_bytes() + ve.to_bytes()
    for j, (w, gamma) in enumerate(zip(ws, blinds)):
        out += _range_prove(w, gamma, seed + b"/w" + j.to_bytes(2, "big"))
    return bytes(out)


def build_proof_envelope(
    verdict: str,
    score_fp: int,
    deny_fp: int,
    escalate_fp: int,
    binding_digest_hex: str,
) -> dict:
    """Assemble the SEP-2828 ``decisionProof`` envelope around a fresh proof."""
    from ._params import PROOF_SYSTEM

    raw = prove(verdict, score_fp, deny_fp, escalate_fp, binding_digest_hex)
    return {
        "proofSystem": PROOF_SYSTEM,
        "publicInputs": {"bindingDigest": binding_digest_hex, "verdict": verdict},
        "proof": raw.hex(),
        "verifierParamsDigest": params_digest(),
    }
