#!/usr/bin/env python3
"""Independent conformance checker for the transparency_consistency_v0 vectors.

Imports only the Python standard library. It does not import Vaara. For each
committed case it reproduces RFC 9162 (RFC 6962-bis) consistency verification:
given two tree sizes, the two roots a verifier holds, and the proof hashes,
recompute both roots and confirm the smaller tree is a verifiable prefix of
the larger one. Verdicts are compared against ``expected.json``.

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


def verify_consistency(
    first_size: int,
    first_root: bytes,
    second_size: int,
    second_root: bytes,
    proof: list[bytes],
) -> bool:
    """RFC 9162 section 2.1.4.2 consistency-proof verification."""
    if first_size > second_size:
        return False
    if first_size == second_size:
        return not proof and first_root == second_root
    if first_size == 0:
        return not proof

    path = list(proof)
    if first_size & (first_size - 1) == 0:
        path = [first_root, *path]
    if not path:
        return False

    fn = first_size - 1
    sn = second_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1

    nodes = iter(path)
    fr = sr = next(nodes)
    for sibling in nodes:
        if sn == 0:
            return False
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

    return sn == 0 and fr == first_root and sr == second_root


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
        want = expected[name]["consistent"]
        if got != want:
            print(f"FAIL {name}: consistency={got}, expected {want}")
            failures += 1
            continue

        # For positive cases, the committed roots must be the genuine Merkle
        # roots over the log prefixes. (Negative cases intentionally carry a
        # corrupted root, so this stronger check applies only when consistent.)
        if want:
            real_first = _root_from_leaves(leaf_hashes[: case["first_size"]])
            real_second = _root_from_leaves(leaf_hashes[: case["second_size"]])
            if real_first != first_root or real_second != second_root:
                print(f"FAIL {name}: committed roots are not the genuine log roots")
                failures += 1
                continue

        print(f"ok   {name}: consistent={got}")

    if failures:
        print(f"\n{failures} case(s) failed")
        return 1
    print(f"\nall {len(cases)} cases matched expected verdicts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
