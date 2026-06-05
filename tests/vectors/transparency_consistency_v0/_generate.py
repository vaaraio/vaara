#!/usr/bin/env python3
"""Generate the transparency_consistency_v0 conformance vectors.

These vectors capture RFC 9162 (RFC 6962-bis) consistency proofs over an
append-only Merkle transparency log. A consistency proof shows that the log
at one size is a verifiable prefix of the log at a later size: nothing
earlier was rewritten, the log only ever grew. That is the append-only
guarantee a transparency log exists to provide.

The committed log is a fixed sequence of leaves. Each case names a
``first_size`` and ``second_size`` and carries the proof hashes plus the two
roots an independent verifier would hold (the signed tree heads at those two
points). Positive cases expect ``consistent: true``. Negative cases keep a
genuine proof but corrupt an input (a flipped proof hash, a root from an
unrelated tree); they expect ``consistent: false``, so a checker that always
returned true would be caught.

Hashing is RFC 6962: ``SHA-256(0x00 || leaf)`` for leaves,
``SHA-256(0x01 || left || right)`` for internal nodes. No signatures, so the
vectors need only the standard library to verify. The committed JSON is the
vector; ``_check_independent.py`` verifies whatever is committed. Run from
the repo root: ``python tests/vectors/transparency_consistency_v0/_generate.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from vaara.attestation.transparency_log import InProcessTransparencyLog

HERE = Path(__file__).resolve().parent

# A fixed, deterministic log. Twelve leaves exercise non-power-of-two tree
# sizes, the case the consistency algorithm is most likely to get wrong.
LEAVES = [f"entry-{i:02d}".encode() for i in range(12)]


def _hex(b: bytes) -> str:
    return b.hex()


def main() -> None:
    log = InProcessTransparencyLog()
    for leaf in LEAVES:
        log.append(leaf)

    # A second, unrelated log: same length, different leaves. Its root stands
    # in for a forked history in the negative case.
    forked = InProcessTransparencyLog()
    for i in range(12):
        forked.append(f"forked-{i:02d}".encode())

    positive_pairs = [
        (0, 12),   # empty prefix: trivially consistent, empty proof
        (1, 12),   # single-leaf (power-of-two) prefix
        (3, 12),   # non-power-of-two first size
        (7, 12),   # odd, deep first size
        (8, 12),   # power-of-two first size, partial second
        (12, 12),  # identical trees: empty proof
        (5, 9),    # both non-power-of-two
    ]

    cases = []
    for first_size, second_size in positive_pairs:
        proof = log.consistency_proof(first_size, second_size)
        cases.append({
            "name": f"consistent_{first_size}_to_{second_size}",
            "first_size": first_size,
            "second_size": second_size,
            "first_root": _hex(log.root_at(first_size)),
            "second_root": _hex(log.root_at(second_size)),
            "proof": [_hex(h) for h in proof.hashes],
        })

    # Negative 1: a genuine 3->12 proof with one sibling hash flipped.
    p = log.consistency_proof(3, 12)
    tampered = [_hex(h) for h in p.hashes]
    first = bytes.fromhex(tampered[0])
    tampered[0] = bytes([first[0] ^ 0x01, *first[1:]]).hex()
    cases.append({
        "name": "tampered_proof_hash_3_to_12",
        "first_size": 3,
        "second_size": 12,
        "first_root": _hex(log.root_at(3)),
        "second_root": _hex(log.root_at(12)),
        "proof": tampered,
    })

    # Negative 2: a genuine 3->12 proof checked against a second root taken
    # from a forked log. The first three leaves never produced that root.
    p = log.consistency_proof(3, 12)
    cases.append({
        "name": "forked_second_root_3_to_12",
        "first_size": 3,
        "second_size": 12,
        "first_root": _hex(log.root_at(3)),
        "second_root": _hex(forked.root_at(12)),
        "proof": [_hex(h) for h in p.hashes],
    })

    expected = {
        "consistent_0_to_12": {"consistent": True},
        "consistent_1_to_12": {"consistent": True},
        "consistent_3_to_12": {"consistent": True},
        "consistent_7_to_12": {"consistent": True},
        "consistent_8_to_12": {"consistent": True},
        "consistent_12_to_12": {"consistent": True},
        "consistent_5_to_9": {"consistent": True},
        "tampered_proof_hash_3_to_12": {"consistent": False},
        "forked_second_root_3_to_12": {"consistent": False},
    }

    log_doc = {
        "leaf_hashing": "SHA-256(0x00 || leaf)",
        "node_hashing": "SHA-256(0x01 || left || right)",
        "leaves": [leaf.decode() for leaf in LEAVES],
    }

    (HERE / "log.json").write_text(json.dumps(log_doc, indent=2) + "\n")
    (HERE / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (HERE / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    print(f"wrote {len(cases)} cases over a {len(LEAVES)}-leaf log")


if __name__ == "__main__":
    main()
