"""In-process Merkle transparency log for OVERT Phase 3 inclusion proofs.

Reference log compatible with RFC 6962-style binary Merkle trees. The IAP
appends a leaf for each Phase 3 attestation it issues, then hands the
inclusion proof to the verifier alongside the signed envelope. The
verifier reconstructs the root from (leaf, log_index, proof_siblings)
and checks it against a published root hash.

The log is in-process so the protocol stays demonstrable without a hard
dependency on an external transparency log such as sigstore Rekor. The
public surface (append → entry, inclusion_proof, root_hash) is shaped to
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

    @property
    def root_hash(self) -> bytes:
        with self._lock:
            return _root_from_leaves(self._leaves)

    @property
    def tree_size(self) -> int:
        with self._lock:
            return len(self._leaves)
