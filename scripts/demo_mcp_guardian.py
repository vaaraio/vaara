#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vaara as the guardian in front of an MCP agent, anchored to Hedera.

The demo this exists to give: an agent asks to do a series of things, Vaara
decides each one, the decisions become a hash-chained trail, and the head of
that trail is committed to a public Hedera topic as an HCS-27 checkpoint.
A stranger then reads the topic and checks what the agent was allowed to do,
without installing Vaara and without trusting anyone who ran it.

    scripts/demo_mcp_guardian.py --topic-id 0.0.NNNNN

Run it with `--offline` to do everything except the ledger submit, which is
the version that works on a machine with no account.

Why this composes with HCS-27 rather than duplicating it
---------------------------------------------------------
The Hedera Agent Kit already governs value movement: spend caps, recipient
allowlists, time windows, enforced by smart contract. Its own tracker (issue
#1001, open since July) says even that is unfinished, and it has no notion of
tool scope, argument content, or a decision reason.

Vaara governs the act. Most of what an agent does is not a payment, and a
transaction history cannot record a tool call that was refused, because a
refused call never becomes a transaction. That absence is the whole problem:
the actions worth auditing are exactly the ones that left no trace on chain.

So the checkpoint commits to decisions, including the denials, and HCS-27
supplies the append-only commitment underneath. The two do not overlap.

Chaining
--------
Every checkpoint after the first carries `prev`, the last accepted
(treeSize, rootHashB64u). That state lives in `.shared/hedera/` and is not in
the repository, because it is per-deployment and not per-checkout. Losing it
does not lose the trail; it means the next checkpoint has to be re-linked by
reading the last one back off the topic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

STATE = REPO / ".shared" / "hedera" / "checkpoint-state.json"

#: What an agent connected through an MCP proxy actually asks for. Ordinary
#: work first, then the ones that should not go through.
#:
#: Every name here is checked against `create_default_registry()`, which knows
#: 23 action types. That check is not cosmetic: an unregistered name classifies
#: as `unknown` with a local blast radius and scores LOW, so a demo built on a
#: made-up tool name would show Vaara waving through the very thing it exists
#: to catch. The first draft of this file did exactly that with a plausible
#: looking `id.credential.issue`, which is not an action Vaara knows.
CALLS: list[tuple[str, dict[str, Any]]] = [
    ("data.read", {"resource": "customer_records", "count": 25}),
    ("data.read", {"table": "orders", "filter": "status=open"}),
    ("tx.transfer", {"amount": 25.0, "currency": "EUR", "to": "supplier_a"}),
    ("data.export", {"resource": "customer_records", "destination": "s3://vendor-bucket", "count": 50000}),
    ("tx.transfer", {"amount": 250000.0, "currency": "EUR", "to": "unknown_account"}),
    ("id.grant_permission", {"subject": "contractor_9", "scope": "admin", "role": "root"}),
    ("data.read", {"resource": "public_catalogue", "count": 3}),
]

AGENT = "mcp-agent-demo"


def _load_state(topic_id: str) -> dict[str, Any]:
    if not STATE.exists():
        return {}
    return json.loads(STATE.read_text(encoding="utf-8")).get(topic_id, {})


def _save_state(topic_id: str, tree_size: int, root_b64u: str) -> None:
    all_state = {}
    if STATE.exists():
        all_state = json.loads(STATE.read_text(encoding="utf-8"))
    all_state[topic_id] = {"treeSize": tree_size, "rootHashB64u": root_b64u}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(all_state, indent=2) + "\n", encoding="utf-8")


def run_agent(db_path: Path) -> tuple[Any, list[Any]]:
    """Drive the real pipeline against a trail that survives between runs.

    The trail is deliberately persistent. An in-memory trail would make every
    run produce a fresh log of the same length, and successive checkpoints
    would then publish different roots at the same ``treeSize``: a log that was
    replaced rather than extended, which is precisely the failure
    ``hcs27_mirror_check.py`` exists to catch. It caught this script's first
    version doing it.
    """
    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.pipeline import InterceptionPipeline

    from vaara.taxonomy.actions import create_default_registry

    registry = create_default_registry()
    unknown = [t for t, _ in CALLS if registry.classify(t, {}).name == "unknown"]
    if unknown:
        raise SystemExit(
            "refusing to run: these tool names are not in the taxonomy and would "
            f"classify as 'unknown', which scores low and allows: {sorted(set(unknown))}"
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    trail = SQLiteAuditBackend(str(db_path)).load_trail()
    pipeline = InterceptionPipeline(trail=trail)

    before = len(trail.snapshot())
    print(f"trail before this run: {before} records ({db_path.name})\n")
    print("The agent asks. Vaara answers.\n")
    print(f"  {'TOOL':<26} {'DECISION':<10} {'DETAIL':<10} WHY")
    print(f"  {'-' * 26} {'-' * 10} {'-' * 10} {'-' * 34}")

    for tool, params in CALLS:
        result = pipeline.intercept(
            agent_id=AGENT, tool_name=tool, parameters=params, session_id="demo-1"
        )
        detail = getattr(result, "decision_detail", None) or ""
        reason = (getattr(result, "reason", "") or "")[:34]
        mark = "ok " if result.allowed else "STOP"
        print(f"  {mark} {tool:<22} {result.decision:<10} {detail:<10} {reason}")

    return trail, trail.snapshot()


def build_checkpoint(records: list[Any], topic_id: str, memo: str) -> tuple[dict, list, bytes]:
    from vaara.audit.hcs27 import checkpoint_for_records

    prev = _load_state(topic_id)
    prev_size = prev.get("treeSize")
    prev_root = None
    if prev_size is not None:
        import base64

        padding = "=" * (-len(prev["rootHashB64u"]) % 4)
        prev_root = base64.urlsafe_b64decode(prev["rootHashB64u"] + padding)

    return checkpoint_for_records(
        records,
        registry="vaara",
        log_id="mcp-guardian",
        prev_tree_size=prev_size,
        prev_root_hash=prev_root,
        memo=memo,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--topic-id", default="0.0.10245892")
    parser.add_argument(
        "--reset", action="store_true",
        help="delete the persistent demo trail and start the log over",
    )
    parser.add_argument("--network", default="testnet")
    parser.add_argument(
        "--offline", action="store_true", help="do everything except the ledger submit"
    )
    args = parser.parse_args()

    if args.reset:
        for f in (REPO / ".shared" / "hedera").glob("mcp-guardian-trail.db*"):
            f.unlink()
        print("reset: removed the persistent demo trail\n")

    print("=" * 72)
    print("VAARA AS MCP GUARDIAN, ANCHORED TO HEDERA")
    print("=" * 72)
    print()

    db_path = REPO / ".shared" / "hedera" / "mcp-guardian-trail.db"
    trail, records = run_agent(db_path)

    print()
    print(f"  trail: {len(records)} records, chain intact: {trail.chain_intact}")
    print()

    memo = f"vaara mcp guardian, {len(records)} records"
    message, entries, root = build_checkpoint(records, args.topic_id, memo)

    from vaara.audit.hcs27 import b64u, message_bytes

    payload = message_bytes(message)
    root_b64u = b64u(root)

    print("-" * 72)
    print("CHECKPOINT")
    print("-" * 72)
    print(f"  entries    {len(entries)}")
    print(f"  treeSize   {message['metadata']['root']['treeSize']}")
    print(f"  root       {root_b64u}")
    prev = message["metadata"].get("prev")
    print(f"  prev       {prev['treeSize'] + ' / ' + prev['rootHashB64u'] if prev else 'genesis'}")
    print(f"  bytes      {len(payload)}")
    print()

    if args.offline:
        print("OFFLINE: not submitted. The bytes that would go on the ledger:")
        print()
        print(payload.decode("utf-8"))
        return 0

    import importlib.util as u

    spec = u.spec_from_file_location("pub", str(REPO / "scripts" / "hcs27_publish.py"))
    pub = u.module_from_spec(spec)
    spec.loader.exec_module(pub)

    from hiero_sdk_python import TopicId, TopicMessageSubmitTransaction

    client, _key, network, operator = pub._client()
    receipt = TopicMessageSubmitTransaction(
        topic_id=TopicId.from_string(args.topic_id), message=payload
    ).execute(client)

    print("-" * 72)
    print("SUBMITTED")
    print("-" * 72)
    print(f"  network    {network}")
    print(f"  topic      {args.topic_id}")
    print(f"  sequence   {getattr(receipt, 'topic_sequence_number', None)}")
    print()

    _save_state(args.topic_id, len(entries), root_b64u)

    entries_path = REPO / ".shared" / "hedera" / f"entries-{args.topic_id}.json"
    entries_path.parent.mkdir(parents=True, exist_ok=True)
    entries_path.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("Anyone can now check this, with no Vaara and no Hedera SDK:")
    print()
    print(f"  scripts/hcs27_mirror_check.py --topic-id {args.topic_id} \\")
    print(f"      --network {network} --entries {entries_path}")
    print()
    print(f"  https://{network}.mirrornode.hedera.com/api/v1/topics/{args.topic_id}/messages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
