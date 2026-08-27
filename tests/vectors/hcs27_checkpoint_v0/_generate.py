#!/usr/bin/env python3
"""Generate the hcs27_checkpoint_v0 conformance vector.

A Vaara audit trail, committed as an HCS-27 Transparency Log Checkpoint, with
the proofs a stranger needs to check it. Everything here is deterministic:
record ids and timestamps are fixed literals, so regenerating produces
byte-identical output and any diff is a real change.

What the vector demonstrates:

  1. A Vaara trail head can be published as an HCS-27 checkpoint without
     inventing a new shape and without changing Vaara's hash chain.
  2. The RFC 9162 Merkle root Vaara computes is the one HCS-27 expects, at a
     non-power-of-two tree size where the two constructions differ most
     visibly in shape.
  3. Inclusion and consistency proofs serialise into HCS-27's field names and
     encodings and verify against the published root.

Seven records, because seven is not a power of two: the last leaf is promoted
across two levels in Vaara's bottom-up fold and lands as a right-hand subtree
in HCS-27's recursive split. Same root either way.

Run: ``python tests/vectors/hcs27_checkpoint_v0/_generate.py``
"""

from __future__ import annotations

import json
from pathlib import Path

from vaara.attestation.transparency_log import InProcessTransparencyLog
from vaara.audit.hcs27 import (
    b64,
    build_checkpoint_metadata,
    canonical_json,
    checkpoint_message,
    consistency_proof_wire,
    entry_for_record,
    inclusion_proof_wire,
)
from vaara.audit.trail import _CURRENT_CHAIN_VERSION, AuditRecord, EventType

HERE = Path(__file__).resolve().parent

#: The checkpoint published at this earlier size, so the vector carries a real
#: prev linkage and a real consistency proof rather than only a genesis root.
PREV_SIZE = 4

EVENTS = [
    (EventType.ACTION_REQUESTED, "Bash", "act-1"),
    (EventType.RISK_SCORED, "Bash", "act-1"),
    (EventType.DECISION_MADE, "Bash", "act-1"),
    (EventType.ACTION_EXECUTED, "Bash", "act-1"),
    (EventType.ACTION_REQUESTED, "Write", "act-2"),
    (EventType.DECISION_MADE, "Write", "act-2"),
    (EventType.ACTION_BLOCKED, "Write", "act-2"),
]


def build_records() -> list[AuditRecord]:
    """Seven chained records, hashed by Vaara's own ``compute_hash``.

    Chaining mirrors ``AuditTrail._append_chained._stamp`` exactly: stamp the
    previous hash, stamp the current chain version, then hash. The trail class
    itself is bypassed only because it generates ids and wall-clock timestamps,
    which a fixed vector cannot have.

    ``agent_id`` carries a Finnish umlaut on purpose. It is the character that
    exposes an ``ensure_ascii=True`` canonicalisation, and Vaara is a Finnish
    product, so it belongs in the vector rather than in a footnote.
    """
    records: list[AuditRecord] = []
    previous_hash = ""
    for i, (event_type, tool, action_id) in enumerate(EVENTS):
        record = AuditRecord(
            record_id=f"rec-{i:02d}",
            action_id=action_id,
            event_type=event_type,
            timestamp=1756220400.0 + i,
            agent_id="agentti-ä",
            tool_name=tool,
            data={"seq": i, "note": "ääkkönen"},
            regulatory_articles=[{"framework": "EU AI Act", "article": "12"}],
            tenant_id="tenant-1",
        )
        record.previous_hash = previous_hash
        record.chain_version = _CURRENT_CHAIN_VERSION
        record.record_hash = record.compute_hash()
        previous_hash = record.record_hash
        records.append(record)
    return records


def _write(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    records = build_records()
    entries = [entry_for_record(r) for r in records]

    log = InProcessTransparencyLog()
    for entry in entries:
        log.append(canonical_json(entry))

    size = len(entries)
    prev_root = log.root_at(PREV_SIZE)
    root = log.root_at(size)

    prev_message = checkpoint_message(
        build_checkpoint_metadata(
            registry="vaara", log_id="trail-demo", tree_size=PREV_SIZE, root_hash=prev_root
        ),
        memo="vaara trail checkpoint, 4 records",
    )
    message = checkpoint_message(
        build_checkpoint_metadata(
            registry="vaara",
            log_id="trail-demo",
            tree_size=size,
            root_hash=root,
            prev_tree_size=PREV_SIZE,
            prev_root_hash=prev_root,
        ),
        memo="vaara trail checkpoint, 7 records",
    )

    proofs = {
        "inclusion": [
            inclusion_proof_wire(
                entry=entries[i], proof=log.inclusion_proof(i), root_hash=root
            )
            for i in range(size)
        ],
        "consistency": consistency_proof_wire(
            proof=log.consistency_proof(PREV_SIZE, size),
            old_root=prev_root,
            new_root=root,
        ),
    }

    _write(HERE / "records.json", [
        {
            "record_id": r.record_id,
            "action_id": r.action_id,
            "event_type": r.event_type.value,
            "timestamp": r.timestamp,
            "agent_id": r.agent_id,
            "tool_name": r.tool_name,
            "data": r.data,
            "regulatory_articles": r.regulatory_articles,
            "previous_hash": r.previous_hash,
            "record_hash": r.record_hash,
            "tenant_id": r.tenant_id,
            "chain_version": r.chain_version,
        }
        for r in records
    ])
    _write(HERE / "entries.json", entries)
    _write(HERE / "checkpoints.json", {"previous": prev_message, "current": message})
    _write(HERE / "proofs.json", proofs)
    _write(HERE / "expected.json", {
        "treeSize": size,
        "prevTreeSize": PREV_SIZE,
        "rootHashB64": b64(root),
        "prevRootHashB64": b64(prev_root),
        "chainHead": records[-1].record_hash,
        "verdicts": {
            "chainRecomputes": True,
            "entriesBindToRecords": True,
            "rootRecomputes": True,
            "prevRootRecomputes": True,
            "allInclusionProofsVerify": True,
            "consistencyProofVerifies": True,
            "checkpointShapeConforms": True,
        },
    })

    print(f"wrote {size} records, {size} entries, 2 checkpoints, "
          f"{size} inclusion proofs and 1 consistency proof")
    print(f"  root      {b64(root)}")
    print(f"  prev root {b64(prev_root)}")
    print(f"  chain head {records[-1].record_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
