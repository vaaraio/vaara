#!/usr/bin/env python3
"""Independent checker for the hcs27_checkpoint_v0 conformance vector.

Imports only the standard library plus ``rfc8785`` (JCS). It imports **no
Vaara code**: producer and auditor share no implementation, so a passing check
means the vector stands on the bytes alone.

Independence here is not just "a different import list". Every primitive is
built the *other* way round from the way Vaara builds it:

  Merkle root      Vaara folds bottom-up, promoting an unpaired node. This
                   checker recurses top-down, splitting at the largest power
                   of two below n, which is how RFC 9162 and the HCS-27 SDK
                   define it. The two constructions are equivalent; agreement
                   is evidence rather than tautology.
  inclusion proof  verified with the RFC 9162 fn/sn bit-walk, not Vaara's
                   index-and-size descent.
  canonical JSON   rfc8785, a third-party JCS implementation, not Vaara's
                   stdlib ``json.dumps`` spelling.
  record hash      re-implemented here from the documented field list rather
                   than called from ``AuditRecord.compute_hash``.

What it checks:

  chainRecomputes           each record's stored hash is the hash of its own
                            content, and each links to the one before it.
  entriesBindToRecords      every checkpoint entry carries the record hash of
                            the record at the same position.
  rootRecomputes            the Merkle root over the entries equals the root
                            published in the checkpoint.
  prevRootRecomputes        the root over the first four entries equals the
                            checkpoint's prev linkage and the earlier
                            checkpoint's own root.
  allInclusionProofsVerify  each of the seven proofs recomputes the root.
  consistencyProofVerifies  the four-record tree is a prefix of the seven.
  checkpointShapeConforms   the published messages satisfy HCS-27's structural
                            rules: fixed literals, canonical decimal tree
                            sizes, unpadded base64url roots, 1 KB message cap,
                            299-character memo cap, and prev <= root.

Run: ``python tests/vectors/hcs27_checkpoint_v0/_check_independent.py``.
Exit 0 means every verdict matched ``expected.json``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import rfc8785

HERE = Path(__file__).resolve().parent

MAX_MESSAGE_BYTES = 1024
MAX_MEMO_CHARS = 299
CANONICAL_UINT = re.compile(r"^(0|[1-9]\d*)$")
STRICT_B64URL = re.compile(r"^[A-Za-z0-9_-]+$")


# --- RFC 9162, built top-down (the opposite of Vaara's fold) -----------------

def _hash_leaf(data: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + data).digest()


def _hash_node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _largest_power_of_two_lt(n: int) -> int:
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def merkle_root(canonical_entries: list[bytes]) -> bytes:
    """RFC 9162 MTH, by recursive split."""
    if not canonical_entries:
        return hashlib.sha256(b"").digest()
    if len(canonical_entries) == 1:
        return _hash_leaf(canonical_entries[0])
    k = _largest_power_of_two_lt(len(canonical_entries))
    return _hash_node(
        merkle_root(canonical_entries[:k]), merkle_root(canonical_entries[k:])
    )


def verify_inclusion(proof: dict[str, Any]) -> bool:
    """RFC 9162 section 2.1.3.2 verification, by the fn/sn bit-walk."""
    leaf_index = int(proof["leafIndex"])
    tree_size = int(proof["treeSize"])
    if not 0 <= leaf_index < tree_size:
        return False
    node = bytes.fromhex(proof["leafHash"])
    fn, sn = leaf_index, tree_size - 1
    for sibling_b64 in proof["path"]:
        if sn == 0:
            return False
        sibling = base64.b64decode(sibling_b64)
        if fn & 1 or fn == sn:
            node = _hash_node(sibling, node)
            while fn != 0 and not fn & 1:
                fn >>= 1
                sn >>= 1
        else:
            node = _hash_node(node, sibling)
        fn >>= 1
        sn >>= 1
    return sn == 0 and base64.b64encode(node).decode() == proof["rootHash"]


def verify_consistency(proof: dict[str, Any]) -> bool:
    """RFC 9162 section 2.1.4.2 verification."""
    first_size = int(proof["oldTreeSize"])
    second_size = int(proof["newTreeSize"])
    old_root = proof["oldRootHash"]
    new_root = proof["newRootHash"]
    path = [base64.b64decode(h) for h in proof["consistencyPath"]]

    if first_size == 0:
        return True
    if first_size == second_size:
        return old_root == new_root and not path
    if first_size > second_size or not path:
        return False

    if first_size & (first_size - 1) == 0:
        path = [base64.b64decode(old_root), *path]

    fn, sn = first_size - 1, second_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1

    first_root = second_root = path[0]
    for node in path[1:]:
        if sn == 0:
            return False
        if fn & 1 or fn == sn:
            first_root = _hash_node(node, first_root)
            second_root = _hash_node(node, second_root)
            while fn != 0 and not fn & 1:
                fn >>= 1
                sn >>= 1
        else:
            second_root = _hash_node(second_root, node)
        fn >>= 1
        sn >>= 1

    return (
        sn == 0
        and base64.b64encode(first_root).decode() == old_root
        and base64.b64encode(second_root).decode() == new_root
    )


# --- Vaara's record hash, re-implemented from its documented field list ------

def record_hash(record: dict[str, Any]) -> str:
    """Recompute ``AuditRecord.compute_hash`` without calling it.

    Note the canonicalisation is NOT JCS: the chain digest uses Python's
    default ``ensure_ascii=True``, so non-ASCII escapes to ``\\uXXXX``. That
    is deliberate on Vaara's side, because every trail already on disk was
    written that way. The HCS-27 leaf below uses real JCS instead. Two
    different functions, and this checker exercises both.
    """
    content = {
        "record_id": record["record_id"],
        "action_id": record["action_id"],
        "event_type": record["event_type"],
        "timestamp": record["timestamp"],
        "agent_id": record["agent_id"],
        "tool_name": record["tool_name"],
        "data": record["data"],
        "regulatory_articles": record["regulatory_articles"],
        "previous_hash": record["previous_hash"],
    }
    if record["chain_version"] >= 2:
        content["tenant_id"] = record["tenant_id"]
        content["chain_version"] = record["chain_version"]
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


# --- checkpoint structural conformance --------------------------------------

def checkpoint_conforms(message: dict[str, Any], problems: list[str]) -> bool:
    ok = True

    def bad(reason: str) -> None:
        nonlocal ok
        problems.append(reason)
        ok = False

    if message.get("p") != "hcs-27":
        bad(f"p is {message.get('p')!r}, must be 'hcs-27'")
    if message.get("op") != "register":
        bad(f"op is {message.get('op')!r}, must be 'register'")

    metadata = message.get("metadata")
    if not isinstance(metadata, dict):
        bad("metadata is not an inline object")
        return ok

    if metadata.get("type") != "ans-checkpoint-v1":
        bad(f"metadata.type is {metadata.get('type')!r}; the upstream schema "
            f"types this as a literal and accepts nothing else")
    log = metadata.get("log", {})
    if log.get("alg") != "sha-256":
        bad(f"log.alg is {log.get('alg')!r}, must be 'sha-256'")
    if log.get("merkle") != "rfc9162":
        bad(f"log.merkle is {log.get('merkle')!r}, must be 'rfc9162'")
    if not log.get("leaf"):
        bad("log.leaf must be a non-empty declared leaf formula")

    stream = metadata.get("stream", {})
    if not stream.get("registry") or not stream.get("log_id"):
        bad("stream.registry and stream.log_id must both be non-empty")

    for field in ("root", "prev"):
        commitment = metadata.get(field)
        if commitment is None:
            continue
        if not CANONICAL_UINT.match(str(commitment.get("treeSize", ""))):
            bad(f"{field}.treeSize {commitment.get('treeSize')!r} is not a canonical uint")
        root_b64u = str(commitment.get("rootHashB64u", ""))
        if not STRICT_B64URL.match(root_b64u):
            bad(f"{field}.rootHashB64u is not unpadded base64url: {root_b64u!r}")

    if "prev" in metadata:
        if int(metadata["prev"]["treeSize"]) > int(metadata["root"]["treeSize"]):
            bad("prev.treeSize exceeds root.treeSize: a log cannot shrink")

    memo = message.get("m")
    if memo is not None and len(memo) > MAX_MEMO_CHARS:
        bad(f"m is {len(memo)} chars, over the {MAX_MEMO_CHARS} cap")

    size = len(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if size > MAX_MESSAGE_BYTES:
        bad(f"message is {size} bytes, over Hedera's {MAX_MESSAGE_BYTES}-byte cap")

    return ok


def main() -> int:
    records = json.loads((HERE / "records.json").read_text(encoding="utf-8"))
    entries = json.loads((HERE / "entries.json").read_text(encoding="utf-8"))
    checkpoints = json.loads((HERE / "checkpoints.json").read_text(encoding="utf-8"))
    proofs = json.loads((HERE / "proofs.json").read_text(encoding="utf-8"))
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))

    problems: list[str] = []

    previous = ""
    chain_ok = True
    for i, record in enumerate(records):
        if record["previous_hash"] != previous:
            problems.append(f"record {i} links to {record['previous_hash']!r}, expected {previous!r}")
            chain_ok = False
        recomputed = record_hash(record)
        if recomputed != record["record_hash"]:
            problems.append(f"record {i} hash mismatch: stored {record['record_hash']}, recomputed {recomputed}")
            chain_ok = False
        previous = record["record_hash"]
    if previous != expected["chainHead"]:
        problems.append(f"chain head {previous} != expected {expected['chainHead']}")
        chain_ok = False

    bind_ok = len(entries) == len(records)
    if not bind_ok:
        problems.append(f"{len(entries)} entries for {len(records)} records")
    else:
        for i, (entry, record) in enumerate(zip(entries, records)):
            if entry["recordHash"] != record["record_hash"]:
                problems.append(f"entry {i} recordHash does not match record {i}")
                bind_ok = False

    canonical = [rfc8785.dumps(e) for e in entries]
    root = merkle_root(canonical)
    prev_root = merkle_root(canonical[: expected["prevTreeSize"]])

    def b64u(raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    current_meta = checkpoints["current"]["metadata"]
    root_ok = (
        base64.b64encode(root).decode() == expected["rootHashB64"]
        and b64u(root) == current_meta["root"]["rootHashB64u"]
        and current_meta["root"]["treeSize"] == str(expected["treeSize"])
    )
    if not root_ok:
        problems.append("merkle root does not match the published checkpoint")

    prev_ok = (
        base64.b64encode(prev_root).decode() == expected["prevRootHashB64"]
        and b64u(prev_root) == current_meta["prev"]["rootHashB64u"]
        and b64u(prev_root) == checkpoints["previous"]["metadata"]["root"]["rootHashB64u"]
    )
    if not prev_ok:
        problems.append("prev root does not match the prev linkage or the earlier checkpoint")

    inclusion_ok = len(proofs["inclusion"]) == len(entries)
    if not inclusion_ok:
        problems.append(f"{len(proofs['inclusion'])} inclusion proofs for {len(entries)} entries")
    for i, proof in enumerate(proofs["inclusion"]):
        expected_leaf = _hash_leaf(canonical[i]).hex()
        if proof["leafHash"] != expected_leaf:
            problems.append(f"inclusion proof {i} leafHash is not the leaf of entry {i}")
            inclusion_ok = False
        if not verify_inclusion(proof):
            problems.append(f"inclusion proof {i} does not recompute the root")
            inclusion_ok = False

    consistency_ok = verify_consistency(proofs["consistency"])
    if not consistency_ok:
        problems.append("consistency proof does not verify")

    shape_ok = checkpoint_conforms(checkpoints["previous"], problems) and checkpoint_conforms(
        checkpoints["current"], problems
    )

    verdicts = {
        "chainRecomputes": chain_ok,
        "entriesBindToRecords": bind_ok,
        "rootRecomputes": root_ok,
        "prevRootRecomputes": prev_ok,
        "allInclusionProofsVerify": inclusion_ok,
        "consistencyProofVerifies": consistency_ok,
        "checkpointShapeConforms": shape_ok,
    }

    width = max(len(k) for k in verdicts)
    for name, value in verdicts.items():
        want = expected["verdicts"][name]
        mark = "ok  " if value == want else "FAIL"
        print(f"  {mark} {name:<{width}}  got {str(value):<5} want {want}")

    for problem in problems:
        print(f"       - {problem}")

    if verdicts != expected["verdicts"]:
        print("\nverdicts do not match expected.json")
        return 1
    print(f"\nall {len(verdicts)} verdicts match expected.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
