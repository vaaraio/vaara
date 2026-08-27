"""Tests for HCS-27 Transparency Log Checkpoint publication.

Three layers:

  unit         the checkpoint builder's structural rules and the
               canonicalisation contract that everything else rests on.
  vector       the shipped ``hcs27_checkpoint_v0`` conformance vector
               regenerates byte-identically and passes its own independent
               checker.
  tamper       the checker actually fails when the bytes change. A conformance
               vector whose checker cannot fail proves nothing.

Cross-implementation agreement with the upstream ``standards-sdk`` TypeScript
is not asserted here, because that needs node and a network fetch. It was
established separately and is recorded in the vector's README: roots agree for
every tree size 0..64 and at every 2^k +/- 1 boundary to 1025, our proofs
verify in their verifier, and our checkpoints pass their zod schema.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.attestation.transparency_log import InProcessTransparencyLog
from vaara.audit.hcs27 import (
    CHECKPOINT_TYPE,
    LEAF_FORMULA,
    MAX_MEMO_CHARS,
    HCS27Error,
    b64,
    b64u,
    build_checkpoint_metadata,
    canonical_json,
    checkpoint_for_records,
    checkpoint_message,
    entry_for_record,
    leaf_hash,
    message_bytes,
    root_for_entries,
)
from vaara.audit.trail import _CURRENT_CHAIN_VERSION, AuditRecord, EventType

VECTOR = Path(__file__).resolve().parent / "vectors" / "hcs27_checkpoint_v0"


def _record(index: int = 0, previous_hash: str = "") -> AuditRecord:
    record = AuditRecord(
        record_id=f"rec-{index:02d}",
        action_id="act-1",
        event_type=EventType.DECISION_MADE,
        timestamp=1756220400.0 + index,
        agent_id="agentti-ä",
        tool_name="Bash",
        data={"seq": index},
        tenant_id="tenant-1",
    )
    record.previous_hash = previous_hash
    record.chain_version = _CURRENT_CHAIN_VERSION
    record.record_hash = record.compute_hash()
    return record


# --- canonicalisation --------------------------------------------------------

def test_canonical_json_does_not_escape_non_ascii():
    """The single rule that silently breaks interop if it is ever lost.

    Python's default ``ensure_ascii=True`` turns every Finnish umlaut into
    ``\\uXXXX`` and produces a leaf hash no other implementation reproduces.
    """
    assert canonical_json({"k": "ääkkönen"}) == '{"k":"ääkkönen"}'.encode("utf-8")
    assert b"\\u" not in canonical_json({"k": "ö"})


def test_canonical_json_sorts_keys_and_omits_whitespace():
    assert canonical_json({"b": 1, "a": "x"}) == b'{"a":"x","b":1}'


def test_canonical_json_matches_rfc8785():
    """Agreement with a third-party JCS implementation, not just with itself."""
    rfc8785 = pytest.importorskip("rfc8785")
    for value in (
        {"b": 1, "a": "x"},
        {"k": "ääkkönen ö"},
        {"z": [1, "a", None, {"q": True}], "a": {}},
        {"v": 1, "recordId": "rec-00", "chainVersion": 2},
    ):
        assert canonical_json(value) == rfc8785.dumps(value)


def test_canonical_json_rejects_floats():
    """Floats are the one value JS and Python render differently."""
    with pytest.raises(HCS27Error, match="float"):
        canonical_json({"x": 1.5})
    with pytest.raises(HCS27Error, match="float"):
        canonical_json({"nested": [{"deep": 2.0}]})


def test_b64u_is_unpadded_and_url_safe():
    raw = bytes(range(32))
    assert "=" not in b64u(raw)
    assert "+" not in b64u(raw) and "/" not in b64u(raw)
    # Proof nodes use the other alphabet, and the difference is real.
    assert b64(raw).endswith("=")


# --- entry projection --------------------------------------------------------

def test_entry_carries_record_hash_but_not_payload():
    record = _record()
    record.regulatory_articles = [{"framework": "EU AI Act", "article": "12"}]
    entry = entry_for_record(record)
    assert entry["recordHash"] == record.record_hash
    # data and regulatory_articles stay off the public topic; recordHash
    # still commits to both.
    assert "data" not in entry
    assert "regulatoryArticles" not in entry


def test_entry_timestamp_is_a_string_never_a_float():
    entry = entry_for_record(_record())
    assert isinstance(entry["timestamp"], str)
    canonical_json(entry)  # would raise if a float leaked in


def test_entry_requires_an_appended_record():
    record = _record()
    record.record_hash = ""
    with pytest.raises(HCS27Error, match="never appended"):
        entry_for_record(record)


# --- checkpoint construction -------------------------------------------------

def test_genesis_checkpoint_has_no_prev():
    metadata = build_checkpoint_metadata(
        registry="vaara", log_id="trail", tree_size=0,
        root_hash=root_for_entries([]),
    )
    assert "prev" not in metadata
    assert metadata["type"] == CHECKPOINT_TYPE
    assert metadata["log"]["leaf"] == LEAF_FORMULA
    assert metadata["log"]["merkle"] == "rfc9162"
    assert metadata["root"]["treeSize"] == "0"


def test_prev_pair_must_be_given_together():
    with pytest.raises(HCS27Error, match="together"):
        build_checkpoint_metadata(
            registry="vaara", log_id="trail", tree_size=2,
            root_hash=b"\x00" * 32, prev_tree_size=1,
        )


def test_log_cannot_shrink():
    with pytest.raises(HCS27Error, match="cannot shrink"):
        build_checkpoint_metadata(
            registry="vaara", log_id="trail", tree_size=2, root_hash=b"\x00" * 32,
            prev_tree_size=5, prev_root_hash=b"\x01" * 32,
        )


def test_memo_cap_is_enforced():
    metadata = build_checkpoint_metadata(
        registry="vaara", log_id="trail", tree_size=1, root_hash=b"\x00" * 32
    )
    with pytest.raises(HCS27Error, match="over the 299"):
        checkpoint_message(metadata, memo="x" * (MAX_MEMO_CHARS + 1))


def test_checkpoint_fits_the_hedera_message_cap():
    records, previous = [], ""
    for i in range(7):
        record = _record(i, previous)
        previous = record.record_hash
        records.append(record)
    message, entries, root = checkpoint_for_records(records)
    assert len(message_bytes(message)) < 1024
    assert message["metadata"]["root"]["treeSize"] == "7"
    assert len(entries) == 7
    assert b64u(root) == message["metadata"]["root"]["rootHashB64u"]


def test_tree_size_is_a_canonical_decimal_string():
    metadata = build_checkpoint_metadata(
        registry="vaara", log_id="trail", tree_size=1024, root_hash=b"\x00" * 32
    )
    assert metadata["root"]["treeSize"] == "1024"


# --- the shipped vector ------------------------------------------------------

def _load_vector(name: str):
    return json.loads((VECTOR / f"{name}.json").read_text(encoding="utf-8"))


def test_vector_regenerates_byte_identically():
    """A changed byte here is either a real change or a portability bug."""
    before = {n: (VECTOR / f"{n}.json").read_bytes()
              for n in ("records", "entries", "checkpoints", "proofs", "expected")}
    spec = importlib.util.spec_from_file_location("_gen", VECTOR / "_generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main() == 0
    after = {n: (VECTOR / f"{n}.json").read_bytes() for n in before}
    assert after == before


def test_vector_passes_its_independent_checker():
    pytest.importorskip("rfc8785")
    result = subprocess.run(
        [sys.executable, str(VECTOR / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_vector_root_recomputes_from_entries():
    entries = _load_vector("entries")
    expected = _load_vector("expected")
    assert b64(root_for_entries(entries)) == expected["rootHashB64"]


def test_vector_inclusion_proofs_bind_to_their_entries():
    entries = _load_vector("entries")
    proofs = _load_vector("proofs")["inclusion"]
    for entry, proof in zip(entries, proofs):
        assert proof["leafHash"] == leaf_hash(entry).hex()


# --- tamper detection --------------------------------------------------------

def test_changing_one_entry_changes_the_root():
    entries = _load_vector("entries")
    original = root_for_entries(entries)
    tampered = copy.deepcopy(entries)
    tampered[3]["agentId"] = "agentti-a"  # umlaut dropped, one byte
    assert root_for_entries(tampered) != original


def test_reordering_entries_changes_the_root():
    entries = _load_vector("entries")
    swapped = copy.deepcopy(entries)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    assert root_for_entries(swapped) != root_for_entries(entries)


def test_independent_checker_fails_on_a_tampered_record(tmp_path: Path):
    """The checker must be able to fail, or its passing means nothing."""
    pytest.importorskip("rfc8785")
    for name in ("records", "entries", "checkpoints", "proofs", "expected"):
        (tmp_path / f"{name}.json").write_bytes((VECTOR / f"{name}.json").read_bytes())
    checker = tmp_path / "_check_independent.py"
    checker.write_bytes((VECTOR / "_check_independent.py").read_bytes())

    records = json.loads((tmp_path / "records.json").read_text(encoding="utf-8"))
    records[2]["data"]["seq"] = 999
    (tmp_path / "records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")

    result = subprocess.run([sys.executable, str(checker)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "chainRecomputes" in result.stdout
    assert "hash mismatch" in result.stdout


def test_inclusion_proof_fails_against_the_wrong_root():
    entries = _load_vector("entries")
    log = InProcessTransparencyLog()
    for entry in entries:
        log.append(canonical_json(entry))
    proof = log.inclusion_proof(2)
    from vaara.attestation.transparency_log import verify_inclusion
    assert verify_inclusion(
        leaf_data=canonical_json(entries[2]), proof=proof, expected_root=log.root_hash
    )
    assert not verify_inclusion(
        leaf_data=canonical_json(entries[2]), proof=proof, expected_root=b"\x00" * 32
    )
