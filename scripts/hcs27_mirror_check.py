#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read a Vaara checkpoint back off a Hedera topic and check it, importing nothing.

This is the half that has to convince someone who trusts neither Vaara nor the
party that published the checkpoint. It imports **no Vaara code and no Hedera
SDK**. Standard library only, over the public mirror-node REST API, which needs
no account, no key and no install.

    scripts/hcs27_mirror_check.py --topic-id 0.0.NNNNN

What it proves, stated narrowly
-------------------------------
1. A message exists on that topic, at a consensus timestamp the network agreed.
2. It parses as an HCS-27 `op=register` checkpoint.
3. The root it publishes recomputes from the entries held locally, by a Merkle
   construction built the opposite way round from the one that produced it:
   recursive top-down split, where Vaara folds bottom up.
4. Successive checkpoints on the topic chain: each `prev` equals the previous
   message's `(treeSize, rootHashB64u)`, and the tree never shrinks.

What it does not prove
----------------------
That the entries describe anything true. A checkpoint commits to a log. It says
nothing about whether any record inside it describes a permitted act, and
HCS-27 does not define log entry schemas. Checking the records themselves is
`tests/vectors/hcs27_checkpoint_v0/_check_independent.py`, which recomputes the
hash chain.

**And it does not prove append-only growth between checkpoints.** `prev`
chaining only asserts that a checkpoint names its predecessor correctly; a
publisher who discards a log and starts a new one still produces a chain that
links. Two cases are caught here, a shrinking tree and a tree that stayed the
same size while its root moved, because both are visible from the checkpoints
alone. The general case is not: proving that the earlier tree is a prefix of
the later one needs an RFC 9162 consistency proof, which HCS-27 serves
off-ledger rather than in the checkpoint. The vector carries one and
`_check_independent.py` verifies it. A verifier who needs that guarantee
between two live checkpoints has to ask the log operator for the proof.

Canonicalisation is done here with `json.dumps(sort_keys=True,
ensure_ascii=False, separators=(",", ":"))`, which reproduces JCS exactly for
the fixed ASCII-keyed entry shape and needs no dependency. The independence
claim of this script is about the network read-back. The independent JCS
implementation is exercised by the vector checker, which uses `rfc8785`.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ENTRIES = REPO / "tests" / "vectors" / "hcs27_checkpoint_v0" / "entries.json"

MIRROR = {
    "testnet": "https://testnet.mirrornode.hedera.com",
    "previewnet": "https://previewnet.mirrornode.hedera.com",
    "mainnet": "https://mainnet.mirrornode.hedera.com",
}


# --- RFC 9162, built top-down, the opposite of the way Vaara builds it -------

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
    if not canonical_entries:
        return hashlib.sha256(b"").digest()
    if len(canonical_entries) == 1:
        return _hash_leaf(canonical_entries[0])
    k = _largest_power_of_two_lt(len(canonical_entries))
    return _hash_node(
        merkle_root(canonical_entries[:k]), merkle_root(canonical_entries[k:])
    )


def jcs(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


# --- mirror node -------------------------------------------------------------

def fetch_messages(network: str, topic_id: str, limit: int) -> list[dict[str, Any]]:
    base = MIRROR[network]
    url = f"{base}/api/v1/topics/{topic_id}/messages?limit={limit}&order=asc"
    out: list[dict[str, Any]] = []
    while url:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                page = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print(
                    f"  topic {topic_id} has no messages on {network}, "
                    f"or does not exist",
                    file=sys.stderr,
                )
                return []
            raise
        out.extend(page.get("messages", []))
        nxt = (page.get("links") or {}).get("next")
        url = f"{base}{nxt}" if nxt else None
        if len(out) >= limit:
            break
    return out


def decode_message(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Decode one mirror-node message into a checkpoint, or None if it is not one."""
    chunk = raw.get("chunk_info")
    if chunk and chunk.get("total", 1) > 1:
        # A Vaara checkpoint is a few hundred bytes and never chunks. A chunked
        # message on this topic came from something else, and silently
        # reassembling it would be guessing at another producer's framing.
        return None
    try:
        payload = base64.b64decode(raw["message"])
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return None


# --- checks ------------------------------------------------------------------

def check_topic(
    network: str, topic_id: str, entries: list[dict[str, Any]], limit: int
) -> int:
    print(f"topic     {topic_id} on {network}")
    print(f"mirror    {MIRROR[network]}")
    print()

    messages = fetch_messages(network, topic_id, limit)
    if not messages:
        print("FAIL: no messages on the topic")
        return 1

    local_root = b64u(merkle_root([jcs(e) for e in entries]))
    print(f"local entries      {len(entries)}")
    print(f"local root         {local_root}")
    print()

    checkpoints: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for raw in messages:
        decoded = decode_message(raw)
        if not decoded:
            continue
        if decoded.get("p") != "hcs-27" or decoded.get("op") != "register":
            continue
        checkpoints.append((raw, decoded))

    if not checkpoints:
        print(f"FAIL: {len(messages)} message(s) on the topic, none an hcs-27 register")
        return 1

    failures = 0
    matched_root = False

    # One topic may legitimately carry several independent logs. HCS-27 gives
    # each a `stream.log_id`, so `prev` chains within a log and says nothing
    # across logs. Chaining per topic would report a false break the moment a
    # second stream shares the topic, which is exactly what happened the first
    # time this checker ran against a real one.
    prev_by_log: dict[tuple[str, str], dict[str, Any]] = {}

    for raw, message in checkpoints:
        metadata = message.get("metadata") or {}
        root = metadata.get("root") or {}
        stream = metadata.get("stream") or {}
        seq = raw.get("sequence_number")
        ts = raw.get("consensus_timestamp")
        tree_size = root.get("treeSize")
        root_hash = root.get("rootHashB64u")
        log_key = (str(stream.get("registry")), str(stream.get("log_id")))
        prev_seen = prev_by_log.get(log_key)

        print(f"seq {seq}  consensus {ts}")
        print(f"  log        {log_key[0]} / {log_key[1]}")
        print(f"  registry   {(metadata.get('stream') or {}).get('registry')}")
        print(f"  profile    {metadata.get('vaaraProfile', '(none)')}")
        print(f"  leaf       {(metadata.get('log') or {}).get('leaf')}")
        print(f"  treeSize   {tree_size}")
        print(f"  root       {root_hash}")

        # Structural rules HCS-27 states, checked rather than assumed.
        if not isinstance(tree_size, str) or not tree_size.isdigit():
            print("  FAIL: treeSize is not a canonical decimal string")
            failures += 1
        if not isinstance(root_hash, str) or "=" in root_hash or "+" in root_hash:
            print("  FAIL: rootHashB64u is not unpadded base64url")
            failures += 1

        # The chain between successive checkpoints on this topic.
        prev = metadata.get("prev")
        if prev_seen is not None:
            if not prev:
                print("  FAIL: non-genesis checkpoint carries no prev")
                failures += 1
            elif (
                prev.get("treeSize") != prev_seen["treeSize"]
                or prev.get("rootHashB64u") != prev_seen["rootHashB64u"]
            ):
                print(
                    f"  FAIL: prev does not match the previous checkpoint "
                    f"({prev.get('treeSize')}, {prev.get('rootHashB64u')})"
                )
                failures += 1
            elif int(tree_size) < int(prev_seen["treeSize"]):
                print("  FAIL: the log shrank")
                failures += 1
            elif (
                tree_size == prev_seen["treeSize"]
                and root_hash != prev_seen["rootHashB64u"]
            ):
                # Same size, different root. The log did not grow, it was
                # replaced. `prev` chaining cannot catch this on its own,
                # because the new checkpoint honestly names the old one; only
                # comparing the pair says the contents changed underneath.
                print(
                    "  FAIL: same treeSize as the previous checkpoint but a "
                    "different root, so this log was replaced, not extended"
                )
                failures += 1
            else:
                print("  ok: prev chains to the previous checkpoint in this log")
        else:
            print("  first checkpoint seen for this log")

        if root_hash == local_root and tree_size == str(len(entries)):
            print("  ok: root recomputes from the local entries")
            matched_root = True

        prev_by_log[log_key] = {"treeSize": tree_size, "rootHashB64u": root_hash}
        print()

    if not matched_root:
        print(
            f"FAIL: no checkpoint on the topic publishes the root of the "
            f"{len(entries)} local entries"
        )
        failures += 1

    if failures:
        print(f"FAIL: {failures} problem(s) across {len(checkpoints)} checkpoint(s)")
        return 1

    print(
        f"PASS: {len(checkpoints)} checkpoint(s) read off the ledger, "
        f"root recomputed independently, chain intact"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--topic-id", required=True, help="e.g. 0.0.12345")
    parser.add_argument("--network", default="testnet", choices=sorted(MIRROR))
    parser.add_argument(
        "--entries",
        default=str(DEFAULT_ENTRIES),
        help="entries the published root should commit to (default: the committed vector)",
    )
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    entries = json.loads(Path(args.entries).read_text(encoding="utf-8"))
    return check_topic(args.network, args.topic_id, entries, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
