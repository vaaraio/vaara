# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""HCS-27 Transparency Log Checkpoints: publish a Vaara trail head to Hedera.

HCS-27 anchors an append-only log to a Hedera Consensus Service topic by
publishing periodic Merkle checkpoints. It proves a log only ever grew. It
deliberately says nothing about what any record inside that log means:

    "This standard does not define: ... log entry schemas or registry event
    structures."

That sentence is the seam this module sits in. HCS-27 supplies the
append-only commitment; Vaara supplies the record that says a specific act
was permitted, by whom, under what scope. The two compose: a stranger checks
inclusion with a stock HCS-27 client that imports no Vaara, then re-derives
``record_hash`` to check the chain.

Why there is no Merkle implementation here
------------------------------------------
Because Vaara already had one. ``vaara.attestation.transparency_log`` builds
an RFC 6962 / RFC 9162 tree, and it turns out to be **byte-identical** to the
HCS-27 Merkle v1 profile. That was established by running the upstream
``standards-sdk/src/hcs-27/merkle.ts`` against it, not by reading it: roots
agree for every tree size 0..64 and at every 2^k +/- 1 boundary up to 1025.

The two construct the tree differently and arrive at the same root. Vaara
folds bottom-up, promoting an unpaired node; HCS-27 splits at the largest
power of two below n and recurses. Those are equivalent. Do not "align"
either one to look like the other, and do not re-derive this by inspection.

Canonicalisation
----------------
The HCS-27 leaf preimage is JCS (RFC 8785). Python reproduces it exactly with
``sort_keys=True, ensure_ascii=False, separators=(",", ":")``.

``ensure_ascii=False`` is load-bearing. Python's default escapes every ``a``
with an umlaut to ``\\uXXXX`` and silently yields a different leaf hash than
every other implementation in the ecosystem. Finnish text makes that a
certainty, not an edge case.

Two residual divergences from JavaScript are excluded by construction rather
than handled: floats (Python writes ``1.0`` where JS writes ``1``) are
rejected outright, and astral-plane object keys (which sort by UTF-16 code
unit in JS, code point in Python) cannot occur because entry keys are fixed
ASCII literals defined below.

Note this is NOT the canonicalisation used by ``AuditRecord.compute_hash``,
which keeps the ``ensure_ascii=True`` default. The chain digest and the
HCS-27 leaf are different functions over different inputs, on purpose: the
chain digest is Vaara's own tamper evidence and must stay stable for every
trail already on disk.

Profile naming, and a known wart in the standard
------------------------------------------------
``metadata.type`` is typed ``z.literal('ans-checkpoint-v1')`` in the upstream
schema, so that string is the only value a stock HCS-27 validator accepts.
The standard delegates entry schemas to consuming profiles but gives a second
profile no way to name itself. Vaara therefore emits the accepted literal and
carries its own identity in ``stream.registry`` and in the declared
``log.leaf`` formula, both of which are free strings. The schema is
``.passthrough()``, so ``vaaraProfile`` rides along for readers who look.

This is recorded here as a defect in the standard rather than routed around
silently, so that a reader who hits the same wall knows it was seen.
"""

from __future__ import annotations

import base64
import json
from typing import Any, Iterable, Optional, Sequence

from vaara.attestation.transparency_log import (
    ConsistencyProof,
    InclusionProof,
    _hash_leaf,
    _root_from_leaves,
)

PROTOCOL = "hcs-27"
OP_REGISTER = "register"

#: The only value a stock HCS-27 validator accepts. See the module docstring.
CHECKPOINT_TYPE = "ans-checkpoint-v1"

#: Vaara's own profile identifier, carried in the passthrough surface.
VAARA_PROFILE = "vaara-trail-checkpoint-v1"

#: Declared leaf preimage formula. `log.leaf` is a free string in the schema,
#: and this is the whole point of the delegation HCS-27 leaves open.
LEAF_FORMULA = "sha256(jcs(vaara.trail-entry/v1))"

MERKLE_PROFILE = "rfc9162"
HASH_ALG = "sha-256"

#: Hedera caps a topic message at 1024 bytes. Past this the metadata object is
#: replaced by an `hcs://1/<topicId>` inscription reference plus a digest.
MAX_MESSAGE_BYTES = 1024

#: `m` is capped at 299 characters by the upstream schema.
MAX_MEMO_CHARS = 299


class HCS27Error(ValueError):
    """Raised when a checkpoint or entry cannot be built conformantly."""


def canonical_json(value: Any) -> bytes:
    """Serialise ``value`` to JCS-canonical UTF-8 bytes.

    Rejects floats rather than emitting a digest that JavaScript would not
    reproduce. Callers that genuinely need a real number should carry it as a
    string, which is what every field in a trail entry already does.
    """
    _reject_floats(value)
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _reject_floats(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        raise HCS27Error(
            f"{path}: float {value!r} cannot be canonicalised portably "
            f"(Python writes '1.0' where JavaScript writes '1'); carry it as a string"
        )
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_floats(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _reject_floats(item, f"{path}[{i}]")


def b64u(data: bytes) -> str:
    """Unpadded base64url, which is what ``rootHashB64u`` fields require."""
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64(data: bytes) -> str:
    """Padded standard base64, which is what proof path nodes require.

    The two alphabets genuinely differ by field in HCS-27: checkpoint roots
    are base64url, while ``verifyInclusionProof`` and
    ``verifyConsistencyProof`` compare against ``Buffer.toString('base64')``.
    Mixing them produces a proof that fails for no visible reason.
    """
    return base64.b64encode(data).decode("ascii")


def entry_for_record(record: Any) -> dict[str, Any]:
    """Project an ``AuditRecord`` into the HCS-27 leaf entry.

    The entry carries identity and ``recordHash`` rather than the record's
    ``data`` payload. Three reasons, in order of weight:

    1. ``recordHash`` already commits to ``data``, so nothing is given up.
    2. It keeps arbitrary caller-supplied values out of the canonicalisation
       path, where a single float would break portability.
    3. It keeps the entry small and free of business content, so publishing
       the checkpoint to a public topic does not publish the payload.

    A verifier fetches the full record out of band, recomputes
    ``compute_hash()``, and compares against ``recordHash`` here.

    Two fields are deliberately absent. ``regulatory_articles`` is a list of
    dicts of caller-shaped content: copying it here would publish Vaara's
    compliance attribution to a public topic and would drag arbitrary values
    back into the canonicalisation path that point 2 above exists to keep
    clean. ``data`` is absent for the same reasons. Both remain committed
    through ``recordHash``, so a verifier who holds the record can still
    prove exactly what was in it.

    ``timestamp`` is ``AuditRecord.timestamp``, a float epoch, rendered with
    Python's ``repr`` and carried as a **string**. It is descriptive only;
    ``recordHash`` is what binds the time, because ``compute_hash`` hashes
    the float itself. Never let the float into the entry: it is precisely the
    value JavaScript would render differently.
    """
    event_type = getattr(record, "event_type", None)
    entry = {
        "v": 1,
        "recordId": str(record.record_id),
        "actionId": str(record.action_id),
        "eventType": str(getattr(event_type, "value", event_type)),
        "timestamp": str(record.timestamp),
        "agentId": str(record.agent_id),
        "toolName": str(record.tool_name) if record.tool_name is not None else "",
        "previousHash": str(record.previous_hash or ""),
        "recordHash": str(record.record_hash or ""),
        "chainVersion": int(getattr(record, "chain_version", 1) or 1),
    }
    tenant_id = getattr(record, "tenant_id", None)
    if tenant_id is not None:
        entry["tenantId"] = str(tenant_id)
    if not entry["recordHash"]:
        raise HCS27Error(
            f"record {entry['recordId']} has no record_hash; it was never appended to a trail"
        )
    return entry


def leaf_hash(entry: dict[str, Any]) -> bytes:
    """SHA-256(0x00 || jcs(entry)), the HCS-27 Merkle v1 leaf."""
    return _hash_leaf(canonical_json(entry))


def root_for_entries(entries: Sequence[dict[str, Any]]) -> bytes:
    """Merkle root over ``entries``, byte-identical to the HCS-27 SDK."""
    return _root_from_leaves([leaf_hash(e) for e in entries])


def build_checkpoint_metadata(
    *,
    registry: str,
    log_id: str,
    tree_size: int,
    root_hash: bytes,
    prev_tree_size: Optional[int] = None,
    prev_root_hash: Optional[bytes] = None,
    signature: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Build the ``metadata`` object for an ``op=register`` checkpoint.

    ``prev`` is required for every non-genesis checkpoint and MUST equal the
    last accepted ``(treeSize, rootHashB64u)`` pair. Passing neither marks
    this as genesis; passing one without the other is an error rather than a
    silently half-built chain.
    """
    if tree_size < 0:
        raise HCS27Error(f"tree_size must be non-negative, got {tree_size}")
    if (prev_tree_size is None) != (prev_root_hash is None):
        raise HCS27Error("prev_tree_size and prev_root_hash must be given together")

    metadata: dict[str, Any] = {
        "type": CHECKPOINT_TYPE,
        "stream": {"registry": registry, "log_id": log_id},
        "log": {"alg": HASH_ALG, "leaf": LEAF_FORMULA, "merkle": MERKLE_PROFILE},
        "root": {"treeSize": str(tree_size), "rootHashB64u": b64u(root_hash)},
        # Passthrough surface: the schema preserves unknown keys, so a reader
        # who cares can tell this is a Vaara trail and not an ANS registry.
        "vaaraProfile": VAARA_PROFILE,
    }
    if prev_tree_size is not None and prev_root_hash is not None:
        if prev_tree_size > tree_size:
            raise HCS27Error(
                f"prev.treeSize ({prev_tree_size}) must be <= root.treeSize ({tree_size}): "
                f"a log cannot shrink"
            )
        metadata["prev"] = {
            "treeSize": str(prev_tree_size),
            "rootHashB64u": b64u(prev_root_hash),
        }
    if signature is not None:
        missing = {"alg", "kid", "b64u"} - set(signature)
        if missing:
            raise HCS27Error(f"signature is missing {sorted(missing)}")
        metadata["sig"] = dict(signature)
    return metadata


def checkpoint_message(
    metadata: dict[str, Any], memo: Optional[str] = None
) -> dict[str, Any]:
    """Wrap ``metadata`` in the HCS-27 topic message envelope.

    Raises if the encoded message exceeds Hedera's 1024-byte cap. The spec's
    answer is an HCS-1 inscription reference, which this module does not
    build: a Vaara checkpoint is a fixed handful of short fields and cannot
    reach 1 KB. Failing loudly beats silently emitting an oversized message
    that the network rejects at submit time.
    """
    if memo is not None and len(memo) > MAX_MEMO_CHARS:
        raise HCS27Error(f"memo is {len(memo)} chars, over the {MAX_MEMO_CHARS} cap")

    message: dict[str, Any] = {"p": PROTOCOL, "op": OP_REGISTER, "metadata": metadata}
    if memo:
        message["m"] = memo

    size = len(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    if size > MAX_MESSAGE_BYTES:
        raise HCS27Error(
            f"checkpoint message is {size} bytes, over Hedera's {MAX_MESSAGE_BYTES}-byte cap; "
            f"HCS-1 overflow inscription is not implemented because a Vaara checkpoint "
            f"should never reach it"
        )
    return message


def message_bytes(message: dict[str, Any]) -> bytes:
    """Exact bytes to submit to the topic."""
    return json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def inclusion_proof_wire(
    *, entry: dict[str, Any], proof: InclusionProof, root_hash: bytes
) -> dict[str, Any]:
    """Serialise an inclusion proof in HCS-27's field names and encodings.

    ``leafHash`` is hex, ``path`` and ``rootHash`` are padded standard base64,
    and the integers are canonical decimal strings. That mix is upstream's,
    not ours.
    """
    return {
        "leafHash": leaf_hash(entry).hex(),
        "leafIndex": str(proof.log_index),
        "treeSize": str(proof.tree_size),
        "path": [b64(s) for s in proof.siblings],
        "rootHash": b64(root_hash),
        "treeVersion": 1,
    }


def consistency_proof_wire(
    *, proof: ConsistencyProof, old_root: bytes, new_root: bytes
) -> dict[str, Any]:
    """Serialise a consistency proof in HCS-27's field names and encodings."""
    return {
        "oldTreeSize": str(proof.first_size),
        "newTreeSize": str(proof.second_size),
        "oldRootHash": b64(old_root),
        "newRootHash": b64(new_root),
        "consistencyPath": [b64(h) for h in proof.hashes],
        "treeVersion": 1,
    }


def checkpoint_for_records(
    records: Iterable[Any],
    *,
    registry: str = "vaara",
    log_id: str = "trail",
    prev_tree_size: Optional[int] = None,
    prev_root_hash: Optional[bytes] = None,
    memo: Optional[str] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    """Build a full checkpoint over ``records``.

    Returns ``(message, entries, root_hash)``. ``entries`` is retained because
    the leaves are not recoverable from the checkpoint: HCS-27 publishes only
    the commitment, and inclusion proofs are served off-ledger by whoever
    holds the log.
    """
    entries = [entry_for_record(r) for r in records]
    root = _root_from_leaves([leaf_hash(e) for e in entries])
    metadata = build_checkpoint_metadata(
        registry=registry,
        log_id=log_id,
        tree_size=len(entries),
        root_hash=root,
        prev_tree_size=prev_tree_size,
        prev_root_hash=prev_root_hash,
    )
    return checkpoint_message(metadata, memo), entries, root


__all__ = [
    "CHECKPOINT_TYPE",
    "HCS27Error",
    "LEAF_FORMULA",
    "MAX_MESSAGE_BYTES",
    "MERKLE_PROFILE",
    "PROTOCOL",
    "VAARA_PROFILE",
    "b64",
    "b64u",
    "build_checkpoint_metadata",
    "canonical_json",
    "checkpoint_for_records",
    "checkpoint_message",
    "consistency_proof_wire",
    "entry_for_record",
    "inclusion_proof_wire",
    "leaf_hash",
    "message_bytes",
    "root_for_entries",
]
