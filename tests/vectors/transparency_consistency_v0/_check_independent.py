#!/usr/bin/env python3
"""Independent conformance checker for the transparency_consistency_v0 vectors.

Imports only the Python standard library. It does not import Vaara. For each
committed case it reproduces RFC 9162 (RFC 6962-bis) consistency verification:
given two tree sizes, the two roots a verifier holds, and the proof hashes,
recompute both roots and confirm the smaller tree is a verifiable prefix of
the larger one. Verdicts are compared against ``expected.json``.

The verdict is three-valued: ``consistent``, ``inconsistent``, or
``could_not_compare``. RFC 9162 section 2.1.4.2 bounds a consistency proof at
``0 < m < n``, and a checker asked about sizes outside that range has no
comparison to report. Cases assert the verdict identity, not its truthiness,
so folding ``could_not_compare`` into either boolean fails the case.

As a second, stronger check it recomputes ``first_root`` and ``second_root``
directly from the committed log leaves and confirms each positive case's roots
are the genuine Merkle roots over those leaves, so the vectors cannot pass by
asserting roots that no honest log would produce.

A second implementation that can run this file (or reproduce its logic) shows
the append-only guarantee is consumable without depending on Vaara. Run:
``python tests/vectors/transparency_consistency_v0/_check_independent.py``.
Exit code 0 means every case matched its expected verdict.
"""

from __future__ import annotations

import enum
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _hash_leaf(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _hash_node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _root_from_leaves(leaf_hashes: list[bytes]) -> bytes:
    if not leaf_hashes:
        return hashlib.sha256(b"").digest()
    nodes = list(leaf_hashes)
    while len(nodes) > 1:
        nxt: list[bytes] = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                nxt.append(_hash_node(nodes[i], nodes[i + 1]))
            else:
                nxt.append(nodes[i])
        nodes = nxt
    return nodes[0]


class Verdict(enum.Enum):
    """Three-valued consistency verdict.

    ``COULD_NOT_COMPARE`` is the value that makes this vector set a
    constraint rather than a documented disagreement: a checker that answers
    either ``CONSISTENT`` or ``INCONSISTENT`` on an input RFC 9162 does not
    define has answered a question it never asked, and fails the case.

    Only ``CONSISTENT`` is truthy, so a checker written against a boolean
    return fails closed on an input it could not compare.
    """

    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    COULD_NOT_COMPARE = "could_not_compare"

    def __bool__(self) -> bool:
        return self is Verdict.CONSISTENT


def verify_consistency(
    first_size: int,
    first_root: bytes,
    second_size: int,
    second_root: bytes,
    proof: list[bytes],
) -> Verdict:
    """RFC 9162 section 2.1.4.2 consistency-proof verification.

    Section 2.1.4.2 bounds the proof at ``0 < m < n``. Sizes outside that
    range are decided first and return ``COULD_NOT_COMPARE``, so an input
    the document does not define can never be reported as a verdict about
    the log.
    """
    if first_size <= 0:
        return Verdict.COULD_NOT_COMPARE
    if second_size < first_size:
        return Verdict.COULD_NOT_COMPARE

    if first_size == second_size:
        if not proof and first_root == second_root:
            return Verdict.CONSISTENT
        return Verdict.INCONSISTENT

    path = list(proof)
    if first_size & (first_size - 1) == 0:
        path = [first_root, *path]
    if not path:
        return Verdict.INCONSISTENT

    fn = first_size - 1
    sn = second_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1

    nodes = iter(path)
    fr = sr = next(nodes)
    for sibling in nodes:
        if sn == 0:
            return Verdict.INCONSISTENT
        if fn & 1 or fn == sn:
            fr = _hash_node(sibling, fr)
            sr = _hash_node(sibling, sr)
            while fn != 0 and not (fn & 1):
                fn >>= 1
                sn >>= 1
        else:
            sr = _hash_node(sr, sibling)
        fn >>= 1
        sn >>= 1

    if sn == 0 and fr == first_root and sr == second_root:
        return Verdict.CONSISTENT
    return Verdict.INCONSISTENT


def main() -> int:
    log_doc = json.loads((HERE / "log.json").read_text())
    cases = json.loads((HERE / "cases.json").read_text())
    expected = json.loads((HERE / "expected.json").read_text())

    leaf_hashes = [_hash_leaf(s.encode("utf-8")) for s in log_doc["leaves"]]

    failures = 0
    for case in cases:
        name = case["name"]
        first_root = bytes.fromhex(case["first_root"])
        second_root = bytes.fromhex(case["second_root"])
        proof = [bytes.fromhex(h) for h in case["proof"]]

        got = verify_consistency(
            case["first_size"], first_root,
            case["second_size"], second_root, proof,
        )
        # The case asserts the verdict identity, not its truthiness. A checker
        # that folds could-not-compare into either boolean answer fails here,
        # which is the whole point of the third value.
        want = expected[name]["verdict"]
        if got.value != want:
            print(f"FAIL {name}: verdict={got.value}, expected {want}")
            failures += 1
            continue

        # For positive cases, the committed roots must be the genuine Merkle
        # roots over the log prefixes. (Negative cases intentionally carry a
        # corrupted root, so this stronger check applies only when consistent.)
        if got is Verdict.CONSISTENT:
            real_first = _root_from_leaves(leaf_hashes[: case["first_size"]])
            real_second = _root_from_leaves(leaf_hashes[: case["second_size"]])
            if real_first != first_root or real_second != second_root:
                print(f"FAIL {name}: committed roots are not the genuine log roots")
                failures += 1
                continue

        print(f"ok   {name}: verdict={got.value}")

    if failures:
        print(f"\n{failures} case(s) failed")
        return 1
    print(f"\nall {len(cases)} cases matched expected verdicts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
