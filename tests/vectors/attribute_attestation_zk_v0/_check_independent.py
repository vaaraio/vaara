#!/usr/bin/env python3
"""Independent checker for attribute_attestation_zk_v0: the value is not here.

Imports only the standard library plus ``rfc8785`` and ``cryptography``. It does
not import Vaara. The elliptic-curve arithmetic, the Pedersen commitments, the
bit-decomposition range argument and the Schnorr OR-proofs below are rebuilt from
the published parameters, and every verdict is recomputed from the bytes of the
case file and the committed public key.

## What the format claims

An attestation binds a subject to *commitments*, not to values. Each attribute
carries the standing of its source in the clear, drawn from the same closed and
totally ordered set the plaintext profile uses:

    undeclared  <  operator_declared  <  measured  <  protocol_defined

A relying party asks whether a predicate holds over the hidden value, and names
the floor it will accept for the standing. The issuer, having handed the openings
to the holder, cannot answer either question and is not asked to.

## The cryptography, stated so it can be rebuilt

The curve is NIST P-256. A commitment is ``C = v*G + r*H`` where ``G`` is the
standard base point and ``H`` is derived by try-and-increment hash-to-curve from
the label ``vaara/zk/H/v0``, normalised to even ``y``. Nobody knows the discrete
log of ``H`` to ``G``, and there is no trusted setup: recompute ``H`` and check.
The commitment is perfectly hiding and computationally binding.

A range proof shows that a commitment opens to a value in ``[0, 2**32)``. It
publishes one commitment ``C_i`` per bit, proves each opens to 0 or 1 with a
Schnorr OR-proof over base ``H``, and the verifier checks that
``sum(2**i * C_i)`` equals the target commitment. The blinds are chosen by the
prover so that identity holds for an honest witness.

Comparisons are the same argument over a shifted commitment, because Pedersen
commitments add:

    value >= t   target  C - t*G     opens to (value - t) under blind r
    value <= t   target  t*G - C     opens to (t - value) under blind -r

A witness outside ``[0, 2**32)`` has no valid bit decomposition, so a predicate
that does not hold has no proof.

Each proof's Fiat-Shamir transcript is seeded with the attestation digest, the
attribute name, the JCS of the predicate and the direction, so a proof does not
move to another document, another attribute or another threshold.

## Four states, and the reason space is partitioned

  accepted  the predicate was proved over a value sourced at or above the floor
  withheld  the document is sound and does not answer what was asked: no proof
            was presented, the attribute is absent, the subject or issuer is not
            the one asked about, or the standing is below the floor
  expired   now is outside notBefore..notAfter
  refused   something fails as evidence: the document is malformed or unsigned by
            the pinned key, or the proof is malformed, bound to something else,
            or does not verify

``proof_absent`` withholds and ``proof_invalid`` refuses, and the gap between
them is the point. Nothing proved is not the same fact as something forged, and
one boolean for both discards the difference between "not yet" and "no".

Order is soundness, then the clock, then sufficiency. Soundness runs first so an
expired window cannot swallow a broken signature, and within sufficiency the
presented proof is judged before the standing floor, so a forged proof is
reported as forged rather than as merely weaker than asked for.

The issuer signature is Ed25519 over the JCS encoding of the attestation with its
own ``signature`` member removed, the same rule the plaintext profile, the
release condition and the data-locality record use.

Run: python3 tests/vectors/attribute_attestation_zk_v0/_check_independent.py
Exit 0 means every case produced the state and reason recorded in expected.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization

HERE = Path(__file__).resolve().parent
SCHEMA = "vaara.attribute-attestation-zk/v0"
PROOF_SCHEMA = "vaara.attribute-predicate/v0"
PROOF_SYSTEM = "vaara-p256-cap-v0"

# ---------------------------------------------------------------------------
# NIST P-256 (secp256r1), from the published domain parameters.
# ---------------------------------------------------------------------------

P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
A = P - 3
B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
GX = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
GY = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551

RANGE_BITS = 32
SCALE = 10**6
MAX_VALUE = 1 << RANGE_BITS

POINT_LEN = 33
SCALAR_LEN = 32
OR_LEN = 2 * POINT_LEN + 3 * SCALAR_LEN
BIT_LEN = POINT_LEN + OR_LEN
RANGE_LEN = RANGE_BITS * BIT_LEN


def _sqrt(v):
    """P has p == 3 (mod 4), so a root is v**((p+1)/4) when one exists."""
    v %= P
    r = pow(v, (P + 1) // 4, P)
    return r if (r * r) % P == v else None


class Pt:
    """An affine point, or the point at infinity when x is None."""

    __slots__ = ("x", "y")

    def __init__(self, x, y):
        self.x, self.y = x, y

    def inf(self):
        return self.x is None

    def __eq__(self, o):
        return isinstance(o, Pt) and self.x == o.x and self.y == o.y

    def dbl(self):
        if self.inf() or self.y == 0:
            return INF
        s = ((3 * self.x * self.x + A) * pow(2 * self.y, -1, P)) % P
        x = (s * s - 2 * self.x) % P
        return Pt(x, (s * (self.x - x) - self.y) % P)

    def __add__(self, o):
        if self.inf():
            return o
        if o.inf():
            return self
        if self.x == o.x:
            if (self.y + o.y) % P == 0:
                return INF
            return self.dbl()
        s = ((o.y - self.y) * pow(o.x - self.x, -1, P)) % P
        x = (s * s - self.x - o.x) % P
        return Pt(x, (s * (self.x - x) - self.y) % P)

    def __mul__(self, k):
        k %= N
        out, add = INF, self
        while k:
            if k & 1:
                out = out + add
            add = add.dbl()
            k >>= 1
        return out

    def neg(self):
        return self if self.inf() else Pt(self.x, (P - self.y) % P)

    def sec1(self):
        if self.inf():
            return b"\x00"
        return bytes([0x02 | (self.y & 1)]) + self.x.to_bytes(32, "big")

    @staticmethod
    def parse(data):
        """SEC1 compressed decoding. Raises ValueError on anything else."""
        if len(data) != POINT_LEN or data[0] not in (0x02, 0x03):
            raise ValueError("not a compressed point")
        x = int.from_bytes(data[1:], "big")
        if x >= P:
            raise ValueError("x is not a field element")
        y = _sqrt((x * x * x + A * x + B) % P)
        if y is None:
            raise ValueError("x is not on the curve")
        if (y & 1) != (data[0] & 1):
            y = P - y
        return Pt(x, y)


INF = Pt(None, None)
G = Pt(GX, GY)


def hash_to_point(label: bytes) -> Pt:
    """Try-and-increment, normalised to even y so the sign convention is fixed."""
    for ctr in range(256):
        x = int.from_bytes(
            hashlib.sha256(label + ctr.to_bytes(4, "big")).digest(), "big"
        ) % P
        y = _sqrt((x * x * x + A * x + B) % P)
        if y is not None:
            return Pt(x, P - y if y & 1 else y)
    raise ValueError("no point found")


#: The second generator. Nothing up anyone's sleeve: it is the hash of a label.
H = hash_to_point(b"vaara/zk/H/v0")


def params_digest() -> str:
    obj = {
        "system": PROOF_SYSTEM,
        "curve": "P-256",
        "H": H.sec1().hex(),
        "rangeBits": RANGE_BITS,
        "scale": SCALE,
    }
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def commit(value: int, blind: int) -> Pt:
    return G * value + H * blind


# ---------------------------------------------------------------------------
# The range argument.
# ---------------------------------------------------------------------------


def _scalar(chunks) -> int:
    return int.from_bytes(hashlib.sha256(b"".join(chunks)).digest(), "big") % N


def or_verify(c: Pt, blob: bytes, prefix: bytes) -> bool:
    """A Schnorr OR-proof that c opens to 0 or 1 under base H.

    Either ``c = r*H`` (the bit is 0) or ``c - G = r*H`` (the bit is 1), and the
    proof does not say which. The two challenges are forced to sum to the
    Fiat-Shamir challenge, so a prover who knows neither opening cannot answer
    both branches.
    """
    a0 = Pt.parse(blob[0:POINT_LEN])
    a1 = Pt.parse(blob[POINT_LEN:2 * POINT_LEN])
    off = 2 * POINT_LEN
    e0 = int.from_bytes(blob[off:off + SCALAR_LEN], "big")
    z0 = int.from_bytes(blob[off + SCALAR_LEN:off + 2 * SCALAR_LEN], "big")
    z1 = int.from_bytes(blob[off + 2 * SCALAR_LEN:off + 3 * SCALAR_LEN], "big")
    # Non-canonical scalars are rejected so a proof has exactly one encoding.
    if e0 >= N or z0 >= N or z1 >= N:
        return False
    e = _scalar([b"vaara/zk/or", prefix, c.sec1(), a0.sec1(), a1.sec1()])
    e1 = (e - e0) % N
    if H * z0 != a0 + c * e0:
        return False
    return H * z1 == a1 + (c + G.neg()) * e1


def range_verify(target: Pt, blob: bytes, prefix: bytes) -> bool:
    """Every bit is 0 or 1, and the bits weigh out to the target commitment."""
    if len(blob) != RANGE_LEN:
        return False
    acc = INF
    for i in range(RANGE_BITS):
        base = i * BIT_LEN
        ci = Pt.parse(blob[base:base + POINT_LEN])
        if not or_verify(ci, blob[base + POINT_LEN:base + BIT_LEN],
                         prefix + b"/" + i.to_bytes(2, "big")):
            return False
        acc = acc + ci * (1 << i)
    return acc == target


# ---------------------------------------------------------------------------
# The document and the predicate.
# ---------------------------------------------------------------------------

RANK = {
    "undeclared": 0,
    "operator_declared": 1,
    "measured": 2,
    "protocol_defined": 3,
}

REASON_STATE = {
    "predicate_proven": "accepted",
    "attribute_absent": "withheld",
    "subject_mismatch": "withheld",
    "issuer_not_accepted": "withheld",
    "source_below_floor": "withheld",
    "proof_absent": "withheld",
    "outside_validity_window": "expired",
    "attestation_malformed": "refused",
    "key_absent": "refused",
    "signature_invalid": "refused",
    "proof_malformed": "refused",
    "proof_not_bound": "refused",
    "proof_invalid": "refused",
}

DIRECTIONS = {
    "at_least": ("ge",),
    "at_most": ("le",),
    "in_range": ("ge", "le"),
}

BOUNDS = {"at_least": ("lower",), "at_most": ("upper",), "in_range": ("lower", "upper")}

_REQUIRED = ("alg", "attestationId", "attributes", "issuer", "notAfter",
             "notBefore", "proofSystem", "schema", "signature", "subject",
             "version")

_PROOF_REQUIRED = ("attestationDigest", "name", "predicate", "proof",
                   "proofSystem", "schema", "verifierParamsDigest")


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _digest(obj) -> str:
    return "sha256:" + hashlib.sha256(_jcs(obj)).hexdigest()


def _epoch(iso):
    try:
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        return datetime.fromisoformat(iso).timestamp()
    except (ValueError, TypeError, AttributeError):
        return None


def _int_in_range(v):
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < MAX_VALUE


def _point(commitment):
    """Decode a wire commitment, or None. The point at infinity is not one."""
    if not isinstance(commitment, str) or len(commitment) != 2 * POINT_LEN:
        return None
    try:
        pt = Pt.parse(bytes.fromhex(commitment))
    except ValueError:
        return None
    return None if pt.inf() else pt


def predicate_ok(pred):
    """A predicate carries exactly the bounds its kind uses, each in range."""
    if not isinstance(pred, dict) or set(pred) - {"kind", "lower", "upper"}:
        return False
    kind = pred.get("kind")
    if kind not in BOUNDS:
        return False
    for field in ("lower", "upper"):
        if field in BOUNDS[kind]:
            if not _int_in_range(pred.get(field)):
                return False
        elif pred.get(field) is not None:
            return False
    if kind == "in_range" and pred["lower"] > pred["upper"]:
        return False
    return True


def well_formed(a) -> bool:
    if not isinstance(a, dict) or a.get("schema") != SCHEMA:
        return False
    if any(k not in a for k in _REQUIRED):
        return False
    if a["proofSystem"] != PROOF_SYSTEM:
        return False
    if not isinstance(a["version"], int) or isinstance(a["version"], bool):
        return False
    for k in ("alg", "attestationId", "issuer", "signature"):
        if not isinstance(a[k], str) or not a[k]:
            return False
    subj = a["subject"]
    if not isinstance(subj, dict):
        return False
    if any(not isinstance(subj.get(k), str) or not subj.get(k) for k in ("id", "kind")):
        return False
    attrs = a["attributes"]
    if not isinstance(attrs, list) or not attrs:
        return False
    seen = set()
    for at in attrs:
        if not isinstance(at, dict):
            return False
        if any(not isinstance(at.get(k), str) or not at.get(k)
               for k in ("name", "source", "commitment")):
            return False
        # An unrecognised standing is malformed, never silently floored.
        if at["source"] not in RANK:
            return False
        # A commitment that is not a point on the curve is not a commitment.
        if _point(at["commitment"]) is None:
            return False
        if at["name"] in seen:
            return False
        seen.add(at["name"])
    return all(_epoch(a[f]) is not None for f in ("notBefore", "notAfter"))


def signature_ok(a: dict, public_key) -> bool:
    body = _jcs({k: v for k, v in a.items() if k != "signature"})
    try:
        sig = bytes.fromhex(a["signature"])
    except (ValueError, TypeError):
        return False
    try:
        public_key.verify(sig, body)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def transcript(digest: str, name: str, predicate: dict, direction: str) -> bytes:
    return b"/".join([
        b"vaara/attribute-zk/v0",
        digest.encode("utf-8"),
        name.encode("utf-8"),
        _jcs(predicate),
        direction.encode("ascii"),
    ])


def proof_well_formed(envelope) -> bool:
    if not isinstance(envelope, dict) or any(
        k not in envelope for k in _PROOF_REQUIRED
    ):
        return False
    if envelope["schema"] != PROOF_SCHEMA or envelope["proofSystem"] != PROOF_SYSTEM:
        return False
    if envelope["verifierParamsDigest"] != params_digest():
        return False
    for k in ("attestationDigest", "name", "proof"):
        if not isinstance(envelope[k], str) or not envelope[k]:
            return False
    if not predicate_ok(envelope["predicate"]):
        return False
    try:
        blob = bytes.fromhex(envelope["proof"])
    except ValueError:
        return False
    return len(blob) == len(DIRECTIONS[envelope["predicate"]["kind"]]) * RANGE_LEN


def proof_verifies(attestation: dict, envelope: dict) -> bool:
    """Recompute the shifted targets and check every range proof against them."""
    predicate = envelope["predicate"]
    name = envelope["name"]
    published = next(
        (x for x in attestation["attributes"] if x["name"] == name), None
    )
    if published is None:
        return False
    commitment = _point(published["commitment"])
    if commitment is None:
        return False
    blob = bytes.fromhex(envelope["proof"])
    digest = _digest(attestation)
    for index, direction in enumerate(DIRECTIONS[predicate["kind"]]):
        bound = G * (predicate["lower"] if direction == "ge" else predicate["upper"])
        target = (
            commitment + bound.neg() if direction == "ge"
            else bound + commitment.neg()
        )
        chunk = blob[index * RANGE_LEN:(index + 1) * RANGE_LEN]
        try:
            ok = range_verify(target, chunk,
                              transcript(digest, name, predicate, direction))
        except ValueError:
            return False
        if not ok:
            return False
    return True


def evaluate(case: dict) -> str:
    """Recompute the reason for one case. Returns a REASON_STATE key."""
    a = case.get("attestation")
    q = case["query"]
    proof = case.get("proof")
    now = _epoch(case["now"])
    if now is None:
        raise ValueError("case 'now' is not an ISO 8601 instant")
    if not predicate_ok(q.get("predicate")):
        raise ValueError("case query carries no usable predicate")

    # 1. sound as an artifact
    if not well_formed(a):
        return "attestation_malformed"
    key_path = case.get("issuer_key")
    if not key_path:
        return "key_absent"
    pub = serialization.load_pem_public_key((HERE / key_path).read_bytes())
    if not signature_ok(a, pub):
        return "signature_invalid"

    # 2. the clock
    if not (_epoch(a["notBefore"]) <= now <= _epoch(a["notAfter"])):
        return "outside_validity_window"

    # 3. does it answer what was asked
    if q.get("subject_id") is not None and a["subject"]["id"] != q["subject_id"]:
        return "subject_mismatch"
    accepted = q.get("accepted_issuers")
    if accepted is not None and a["issuer"] not in accepted:
        return "issuer_not_accepted"
    match = next((x for x in a["attributes"] if x["name"] == q["name"]), None)
    if match is None:
        return "attribute_absent"

    if proof is None:
        return "proof_absent"
    if not proof_well_formed(proof):
        return "proof_malformed"
    if (proof["attestationDigest"], proof["name"], proof["predicate"]) != (
        _digest(a), q["name"], q["predicate"]
    ):
        return "proof_not_bound"
    if not proof_verifies(a, proof):
        return "proof_invalid"

    if RANK[match["source"]] < RANK[q["minimum_source"]]:
        return "source_below_floor"
    return "predicate_proven"


# ---------------------------------------------------------------------------
# Structural properties, asserted before any case is graded.
# ---------------------------------------------------------------------------


def _partition_is_total() -> bool:
    return (set(REASON_STATE.values()) == {"accepted", "withheld", "expired", "refused"}
            and len(REASON_STATE) == len(set(REASON_STATE)))


def _ladder_is_total_order() -> bool:
    vals = list(RANK.values())
    return len(set(vals)) == len(vals) and RANK["undeclared"] == min(vals)


def _nothing_proved_is_not_forged() -> bool:
    """The distinction the whole format turns on, asserted rather than assumed."""
    return (REASON_STATE["proof_absent"] == "withheld"
            and REASON_STATE["proof_invalid"] == "refused")


def _generator_is_recomputable() -> bool:
    """H is a curve point derived from a public label, with no known dlog to G."""
    return (not H.inf()
            and (H.y * H.y - (H.x ** 3 + A * H.x + B)) % P == 0
            and H.y % 2 == 0
            and H == hash_to_point(b"vaara/zk/H/v0")
            and H != G)


def _commitment_is_homomorphic() -> bool:
    """The property every predicate here rests on: shifting the commitment by
    t*G shifts what it opens to by t, so a comparison is a range statement."""
    v, r, t = 37, 0x5EED, 18
    return commit(v, r) + (G * t).neg() == commit(v - t, r)


def _commitment_hides() -> bool:
    """Two commitments to the same value under different blinds are unequal, so
    a relying party holding both learns nothing by comparing them."""
    return commit(37, 11) != commit(37, 12)


def main() -> int:
    expected_path = HERE / "expected.json"
    if not expected_path.exists():
        print("expected.json not found, run _generate.py first", file=sys.stderr)
        return 1
    expected = json.loads(expected_path.read_text(encoding="utf-8"))["cases"]
    cases_dir = HERE / "cases"
    failures = []

    for label, ok in (
        ("reason space partitions over the four states", _partition_is_total()),
        ("standing ladder is a total order", _ladder_is_total_order()),
        ("nothing proved and something forged are different states",
         _nothing_proved_is_not_forged()),
        ("H recomputes from its label and is on the curve",
         _generator_is_recomputable()),
        ("commitments are additively homomorphic", _commitment_is_homomorphic()),
        ("the same value under two blinds gives two commitments",
         _commitment_hides()),
    ):
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            failures.append(label)

    for path in sorted(cases_dir.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        reason = evaluate(case)
        state = REASON_STATE[reason]
        ok = reason == case["expected_reason"] and state == case["expected_state"]
        print(f"  {'PASS' if ok else 'FAIL'}  {path.stem:<28}"
              f"  computed={state}/{reason}"
              f"  expected={case['expected_state']}/{case['expected_reason']}")
        if not ok:
            failures.append(path.stem)

    for name, meta in expected.items():
        path = cases_dir / f"{name}.json"
        if not path.exists():
            print(f"  MISSING  {name}", file=sys.stderr)
            failures.append(name)
            continue
        case = json.loads(path.read_text(encoding="utf-8"))
        reason = evaluate(case)
        if (reason != meta["expected_reason"]
                or REASON_STATE[reason] != meta["expected_state"]):
            failures.append(f"{name}(expected.json cross-check)")

    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    print(f"\nall {len(list(cases_dir.glob('*.json')))} cases pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
