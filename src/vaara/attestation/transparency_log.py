# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-process Merkle transparency log for OVERT Phase 3 inclusion proofs.

Reference log compatible with RFC 6962-style binary Merkle trees. The IAP
appends a leaf for each Phase 3 attestation it issues, then hands the
inclusion proof to the verifier alongside the signed envelope. The
verifier reconstructs the root from (leaf, log_index, proof_siblings)
and checks it against a published root hash.

The log is in-process so the protocol stays demonstrable without a hard
dependency on an external transparency log such as sigstore Rekor. The
public surface (append to entry, inclusion_proof, root_hash) is shaped to
match what a Rekor-backed adapter would expose, so a future
``RekorTransparencyLog`` can drop into the same call site.

Domain separators (`b"\\x00"` for leaves, `b"\\x01"` for internal nodes)
match RFC 6962 to prevent second-preimage attacks across the leaf /
internal boundary.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass


class TransparencyLogError(RuntimeError):
    """Raised when log operations fail (e.g. malformed entry, bad index)."""


@dataclass(frozen=True)
class LogEntry:
    """One appended leaf in the transparency log.

    Attributes:
        log_index: 0-based position of this leaf.
        leaf_hash: SHA-256(0x00 || leaf_bytes).
        root_hash_at_append: Merkle root hash after this append.
        tree_size_at_append: Total leaves in the tree after this append.
    """

    log_index: int
    leaf_hash: bytes
    root_hash_at_append: bytes
    tree_size_at_append: int


@dataclass(frozen=True)
class ConsistencyProof:
    """RFC 9162 (RFC 6962-bis) consistency proof between two tree sizes.

    Proves that the log of ``first_size`` leaves whose root was
    ``first_root`` is a prefix of the log of ``second_size`` leaves whose
    root is ``second_root``: every leaf in the smaller tree is present, in
    the same position, in the larger one, and nothing earlier was rewritten.
    This is the append-only guarantee. An inclusion proof shows an entry is
    *in* the log; a consistency proof shows the log only ever *grew*.

    ``first_size <= second_size``. The proof verifies against two roots the
    verifier obtained independently (e.g. signed tree heads published by the
    log operator at two points in time); the hashes alone do not bind to a
    specific pair of roots.
    """

    first_size: int
    second_size: int
    hashes: tuple[bytes, ...]


@dataclass(frozen=True)
class InclusionProof:
    """RFC 6962-style inclusion proof.

    The verifier recomputes the root by hashing ``leaf_hash`` with the
    sibling hashes along the Merkle path. ``log_index`` and ``tree_size``
    determine the direction (left or right) at each level.

    The proof is verified against ``root_hash`` published independently
    by the log operator. If the IAP is the log operator (the reference
    case), the verifier must obtain the root hash from a separate
    audit-time fetch or pinned configuration; the proof alone does not
    bind to a specific root.
    """

    log_index: int
    tree_size: int
    siblings: tuple[bytes, ...]


def _hash_leaf(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _hash_node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _root_from_leaves(leaves: list[bytes]) -> bytes:
    """Compute the RFC 6962 root from a list of leaf hashes."""
    if not leaves:
        return hashlib.sha256(b"").digest()
    nodes = list(leaves)
    while len(nodes) > 1:
        next_level: list[bytes] = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                next_level.append(_hash_node(nodes[i], nodes[i + 1]))
            else:
                next_level.append(nodes[i])
        nodes = next_level
    return nodes[0]


def _proof_for(leaves: list[bytes], index: int) -> tuple[bytes, ...]:
    """Sibling hashes for ``leaves[index]`` in the current tree."""
    if not (0 <= index < len(leaves)):
        raise TransparencyLogError(
            f"index {index} out of range for tree size {len(leaves)}"
        )
    proof: list[bytes] = []
    nodes = list(leaves)
    idx = index
    while len(nodes) > 1:
        sibling = idx ^ 1
        if sibling < len(nodes):
            proof.append(nodes[sibling])
        # Else this node has no sibling at this level; promoted unchanged.
        next_level: list[bytes] = []
        for i in range(0, len(nodes), 2):
            if i + 1 < len(nodes):
                next_level.append(_hash_node(nodes[i], nodes[i + 1]))
            else:
                next_level.append(nodes[i])
        nodes = next_level
        idx //= 2
    return tuple(proof)


def _largest_power_of_two_lt(n: int) -> int:
    """Largest power of two strictly less than ``n`` (requires ``n >= 2``)."""
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def _subproof(m: int, leaves: list[bytes], on_path: bool) -> list[bytes]:
    """RFC 9162 SUBPROOF over ``leaves`` for prefix size ``m``.

    ``on_path`` is the spec's ``b`` flag: when the recursion has narrowed to
    the subtree that exactly equals the ``m``-leaf prefix and that subtree is
    still on the path from the root, its hash is implied (the verifier
    recomputes it) and omitted; otherwise the subtree hash is included.
    """
    n = len(leaves)
    if m == n:
        return [] if on_path else [_root_from_leaves(leaves)]
    k = _largest_power_of_two_lt(n)
    if m <= k:
        return _subproof(m, leaves[:k], on_path) + [_root_from_leaves(leaves[k:])]
    return _subproof(m - k, leaves[k:], False) + [_root_from_leaves(leaves[:k])]


def verify_consistency(
    *,
    first_size: int,
    first_root: bytes,
    second_size: int,
    second_root: bytes,
    proof: ConsistencyProof,
) -> bool:
    """Verify an RFC 9162 consistency proof between two tree heads.

    Recomputes both ``first_root`` and ``second_root`` from the proof and
    returns whether both match. A ``True`` result means the ``first_size``
    tree is a verifiable prefix of the ``second_size`` tree: the log is
    append-only across the two snapshots. Implements RFC 9162 section 2.1.4.2.
    """
    if proof.first_size != first_size or proof.second_size != second_size:
        return False
    if first_size > second_size:
        return False
    if first_size == second_size:
        # Same tree: nothing to prove beyond identical roots.
        return not proof.hashes and first_root == second_root
    if first_size == 0:
        # The empty tree is a prefix of every tree; no path hashes needed.
        return not proof.hashes

    # 0 < first_size < second_size. For a power-of-two first tree the spec
    # omits first_root from the path because the verifier already holds it;
    # prepend it so the seed logic below is uniform.
    path = list(proof.hashes)
    if first_size & (first_size - 1) == 0:
        path = [first_root, *path]
    if not path:
        return False

    fn = first_size - 1
    sn = second_size - 1
    # Strip the shared low run: these bits are interior to the first subtree
    # and contribute no path node.
    while fn & 1:
        fn >>= 1
        sn >>= 1

    nodes = iter(path)
    fr = sr = next(nodes)
    for sibling in nodes:
        if sn == 0:
            # More path nodes than the second tree can account for.
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


def verify_inclusion(
    *,
    leaf_data: bytes,
    proof: InclusionProof,
    expected_root: bytes,
) -> bool:
    """Recompute the root from leaf_data + proof and compare."""
    if not (0 <= proof.log_index < proof.tree_size):
        return False
    node = _hash_leaf(leaf_data)
    idx = proof.log_index
    size = proof.tree_size
    sib_iter = iter(proof.siblings)
    while size > 1:
        last_in_level = size - 1
        if idx == last_in_level and idx % 2 == 0:
            # Unpaired right edge: promoted without consuming a sibling.
            pass
        else:
            try:
                sibling = next(sib_iter)
            except StopIteration:
                return False
            if idx % 2 == 0:
                node = _hash_node(node, sibling)
            else:
                node = _hash_node(sibling, node)
        idx //= 2
        size = (size + 1) // 2
    # All proof entries must be consumed.
    if next(sib_iter, None) is not None:
        return False
    return node == expected_root


class InProcessTransparencyLog:
    """Append-only RFC 6962-style Merkle log held in memory.

    Thread-safe via an internal lock. The log is the reference operator
    for Vaara's IAP demonstrations; a production deployment would back
    the same interface with sigstore Rekor or an equivalent
    independently-operated log.
    """

    def __init__(self) -> None:
        self._leaves: list[bytes] = []
        self._lock = threading.Lock()

    def append(self, leaf_data: bytes) -> LogEntry:
        if not isinstance(leaf_data, (bytes, bytearray)):
            raise TransparencyLogError("leaf_data must be bytes")
        leaf_hash = _hash_leaf(bytes(leaf_data))
        with self._lock:
            self._leaves.append(leaf_hash)
            idx = len(self._leaves) - 1
            root = _root_from_leaves(self._leaves)
            return LogEntry(
                log_index=idx,
                leaf_hash=leaf_hash,
                root_hash_at_append=root,
                tree_size_at_append=len(self._leaves),
            )

    def inclusion_proof(self, log_index: int) -> InclusionProof:
        with self._lock:
            size = len(self._leaves)
            siblings = _proof_for(list(self._leaves), log_index)
            return InclusionProof(
                log_index=log_index, tree_size=size, siblings=siblings,
            )

    def root_at(self, tree_size: int) -> bytes:
        """Merkle root over the first ``tree_size`` leaves.

        ``root_at(0)`` is the empty-tree hash; ``root_at(tree_size)`` equals
        the ``root_hash`` the log had after its first ``tree_size`` appends.
        Useful for pinning a historical signed tree head before requesting a
        consistency proof against a later one.
        """
        with self._lock:
            if not (0 <= tree_size <= len(self._leaves)):
                raise TransparencyLogError(
                    f"tree_size {tree_size} out of range for log of "
                    f"{len(self._leaves)} leaves"
                )
            return _root_from_leaves(self._leaves[:tree_size])

    def consistency_proof(
        self, first_size: int, second_size: int
    ) -> ConsistencyProof:
        """Prove the ``first_size`` tree is a prefix of the ``second_size`` one.

        ``0 <= first_size <= second_size <= tree_size``. The proof is empty
        for the trivial cases (``first_size`` is 0 or equals ``second_size``).
        Pair the result with ``root_at(first_size)`` and ``root_at(second_size)``,
        or with two independently published roots, and check it via
        ``verify_consistency``.
        """
        with self._lock:
            size = len(self._leaves)
            if not (0 <= first_size <= second_size <= size):
                raise TransparencyLogError(
                    f"consistency proof requires 0 <= first_size "
                    f"({first_size}) <= second_size ({second_size}) <= "
                    f"tree_size ({size})"
                )
            if first_size == 0 or first_size == second_size:
                hashes: tuple[bytes, ...] = ()
            else:
                hashes = tuple(
                    _subproof(first_size, self._leaves[:second_size], True)
                )
            return ConsistencyProof(
                first_size=first_size,
                second_size=second_size,
                hashes=hashes,
            )

    @property
    def root_hash(self) -> bytes:
        with self._lock:
            return _root_from_leaves(self._leaves)

    @property
    def tree_size(self) -> int:
        with self._lock:
            return len(self._leaves)
