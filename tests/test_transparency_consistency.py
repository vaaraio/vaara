"""Tests for RFC 9162 transparency-log consistency proofs (v0.54).

Consistency proofs are the append-only guarantee: they show the log at an
earlier size is a verifiable prefix of the log at a later size, so a fork or
a rewrite of earlier history is detectable even when every inclusion proof
still verifies.

The verdict is three-valued. RFC 9162 section 2.1.4.2 bounds a consistency
proof at 0 < m < n, and for sizes outside that range there is no comparison
to report: `could_not_compare` is not a claim about the log, it is the
absence of one. It is decided before any comparison is attempted, and it is
falsy, so a caller written against the older bool return fails closed.

Coverage:
1. Genuine prover/verifier agreement across every (first, second) size pair
   in a range, including the non-power-of-two cases.
2. Rejection of a tampered proof, a wrong root, and a forked second tree.
3. Out-of-range inputs (empty prefix, first larger than second, a proof for
   another pair of heads) report could_not_compare, are decided before
   invalid, and stay falsy.
4. The committed transparency_consistency_v0 vectors verify two ways: via
   Vaara, and via the standalone stdlib-only checker.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.attestation.transparency_log import (
    ConsistencyProof,
    ConsistencyVerdict,
    InProcessTransparencyLog,
    TransparencyLogError,
    verify_consistency,
)

VECTORS = Path(__file__).resolve().parent / "vectors" / "transparency_consistency_v0"


def _log(n: int) -> InProcessTransparencyLog:
    log = InProcessTransparencyLog()
    for i in range(n):
        log.append(f"leaf-{i}".encode())
    return log


# ── Genuine prover/verifier agreement ───────────────────────────────────────

@pytest.mark.parametrize("second", range(0, 18))
def test_genuine_proof_verifies_for_every_prefix(second: int) -> None:
    log = _log(second)
    # first == 0 is outside RFC 9162's 0 < m and is covered by
    # test_empty_prefix_cannot_be_compared, not here.
    for first in range(1, second + 1):
        proof = log.consistency_proof(first, second)
        assert verify_consistency(
            first_size=first,
            first_root=log.root_at(first),
            second_size=second,
            second_root=log.root_at(second),
            proof=proof,
        ) is ConsistencyVerdict.CONSISTENT, f"genuine proof {first} to {second} failed"


def test_root_at_matches_root_hash_at_full_size() -> None:
    log = _log(11)
    assert log.root_at(11) == log.root_hash
    assert log.tree_size == 11


def test_root_at_matches_append_time_root() -> None:
    log = InProcessTransparencyLog()
    roots = [log.root_at(0)]
    for i in range(9):
        entry = log.append(f"x{i}".encode())
        assert entry.root_hash_at_append == log.root_at(i + 1)
        roots.append(entry.root_hash_at_append)
    # Each historical root is reproducible after later appends.
    for size, root in enumerate(roots):
        assert log.root_at(size) == root


# ── Negative: tamper, wrong root, forked history ────────────────────────────

def test_tampered_proof_hash_is_rejected() -> None:
    log = _log(12)
    proof = log.consistency_proof(3, 12)
    flipped = list(proof.hashes)
    flipped[0] = bytes([flipped[0][0] ^ 0x01, *flipped[0][1:]])
    bad = dataclasses.replace(proof, hashes=tuple(flipped))
    assert not verify_consistency(
        first_size=3, first_root=log.root_at(3),
        second_size=12, second_root=log.root_at(12), proof=bad,
    )


def test_wrong_second_root_is_rejected() -> None:
    log = _log(12)
    proof = log.consistency_proof(5, 12)
    assert not verify_consistency(
        first_size=5, first_root=log.root_at(5),
        second_size=12, second_root=b"\x00" * 32, proof=proof,
    )


def test_forked_history_is_rejected() -> None:
    """A second tree that rewrote earlier leaves has no valid proof."""
    honest = _log(12)
    forked = InProcessTransparencyLog()
    for i in range(12):
        forked.append(f"FORKED-{i}".encode())
    # Honest 4-leaf prefix root against the forked tree's later root: no proof
    # from either tree should reconcile them.
    for proof in (honest.consistency_proof(4, 12), forked.consistency_proof(4, 12)):
        assert not verify_consistency(
            first_size=4, first_root=honest.root_at(4),
            second_size=12, second_root=forked.root_at(12), proof=proof,
        )


def test_extra_proof_hash_is_rejected() -> None:
    log = _log(12)
    proof = log.consistency_proof(3, 12)
    padded = dataclasses.replace(proof, hashes=(*proof.hashes, b"\x11" * 32))
    assert not verify_consistency(
        first_size=3, first_root=log.root_at(3),
        second_size=12, second_root=log.root_at(12), proof=padded,
    )


# ── Edge cases and input guards ─────────────────────────────────────────────

def test_empty_prefix_cannot_be_compared() -> None:
    """An empty first tree is outside RFC 9162's 0 < m, so nothing is compared.

    The old answer here was ``True``, which asserted the log was append-only
    on the strength of a check that never looked at either root. It would have
    held for any pair of roots at all, as long as the proof list was empty.
    """
    log = _log(7)
    proof = log.consistency_proof(0, 7)
    assert proof.hashes == ()
    assert verify_consistency(
        first_size=0, first_root=log.root_at(0),
        second_size=7, second_root=log.root_at(7), proof=proof,
    ) is ConsistencyVerdict.COULD_NOT_COMPARE


def test_empty_prefix_verdict_is_falsy_so_old_callers_fail_closed() -> None:
    """A caller written against the old bool must refuse, not pass."""
    log = _log(7)
    proof = log.consistency_proof(0, 7)
    verdict = verify_consistency(
        first_size=0, first_root=log.root_at(0),
        second_size=7, second_root=log.root_at(7), proof=proof,
    )
    assert not verdict
    assert not bool(verdict)


def test_empty_prefix_does_not_launder_an_unrelated_root() -> None:
    """The exact case Blake Morrison found: any second_root used to pass."""
    log = _log(7)
    proof = log.consistency_proof(0, 7)
    assert verify_consistency(
        first_size=0, first_root=b"\x00" * 32,
        second_size=7, second_root=b"\xff" * 32, proof=proof,
    ) is ConsistencyVerdict.COULD_NOT_COMPARE


def test_identical_trees_need_empty_proof_and_equal_roots() -> None:
    log = _log(6)
    proof = log.consistency_proof(6, 6)
    assert proof.hashes == ()
    root = log.root_at(6)
    assert verify_consistency(
        first_size=6, first_root=root,
        second_size=6, second_root=root, proof=proof,
    )
    assert not verify_consistency(
        first_size=6, first_root=root,
        second_size=6, second_root=b"\x00" * 32, proof=proof,
    )
    assert not verify_consistency(
        first_size=6, first_root=root, second_size=6, second_root=root,
        proof=ConsistencyProof(6, 6, (b"\x22" * 32,)),
    )


def test_first_larger_than_second_cannot_be_compared() -> None:
    """Outside RFC 9162's m < n: there is no larger tree to be a prefix of.

    ``inconsistent`` would be a claim about the log. Nothing was compared, so
    the honest answer is that the question could not be asked.
    """
    log = _log(8)
    assert verify_consistency(
        first_size=8, first_root=log.root_at(8),
        second_size=4, second_root=log.root_at(4),
        proof=ConsistencyProof(8, 4, ()),
    ) is ConsistencyVerdict.COULD_NOT_COMPARE


def test_proof_for_a_different_pair_of_heads_cannot_be_compared() -> None:
    """A proof describing other sizes says nothing about the pair asked about."""
    log = _log(10)
    proof = log.consistency_proof(3, 10)
    mismatched = dataclasses.replace(proof, second_size=9)
    assert verify_consistency(
        first_size=3, first_root=log.root_at(3),
        second_size=10, second_root=log.root_at(10), proof=mismatched,
    ) is ConsistencyVerdict.COULD_NOT_COMPARE


def test_could_not_compare_is_decided_before_invalid() -> None:
    """Ordering is the other half of the rule.

    An out-of-range pair carrying a proof that would also fail verification
    must report that it could not be compared. If invalid were decided first
    the third value would exist in the type and never reach a caller.
    """
    log = _log(8)
    junk = ConsistencyProof(8, 4, (b"\x99" * 32, b"\x88" * 32))
    assert verify_consistency(
        first_size=8, first_root=log.root_at(8),
        second_size=4, second_root=b"\x00" * 32,
        proof=junk,
    ) is ConsistencyVerdict.COULD_NOT_COMPARE


def test_consistency_proof_out_of_range_raises() -> None:
    log = _log(5)
    with pytest.raises(TransparencyLogError):
        log.consistency_proof(3, 9)  # second_size > tree_size
    with pytest.raises(TransparencyLogError):
        log.consistency_proof(4, 2)  # first_size > second_size
    with pytest.raises(TransparencyLogError):
        log.consistency_proof(-1, 5)  # negative first_size


def test_root_at_out_of_range_raises() -> None:
    log = _log(5)
    with pytest.raises(TransparencyLogError):
        log.root_at(6)
    with pytest.raises(TransparencyLogError):
        log.root_at(-1)


# ── Committed conformance vectors ───────────────────────────────────────────

def test_vaara_reproduces_committed_vectors() -> None:
    cases = json.loads((VECTORS / "cases.json").read_text())
    expected = json.loads((VECTORS / "expected.json").read_text())
    for case in cases:
        proof = ConsistencyProof(
            first_size=case["first_size"],
            second_size=case["second_size"],
            hashes=tuple(bytes.fromhex(h) for h in case["proof"]),
        )
        got = verify_consistency(
            first_size=case["first_size"],
            first_root=bytes.fromhex(case["first_root"]),
            second_size=case["second_size"],
            second_root=bytes.fromhex(case["second_root"]),
            proof=proof,
        )
        # Identity, not truthiness: folding could_not_compare into a boolean
        # would leave the wrong answer green.
        assert got.value == expected[case["name"]]["verdict"], case["name"]


def test_independent_checker_passes() -> None:
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
