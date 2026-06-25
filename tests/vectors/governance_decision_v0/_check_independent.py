#!/usr/bin/env python3
"""Independent checker for the governance_decision_v0 conformance vectors.

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
NOT import Vaara. The corpus pins the bytes of the CrewAI ``GovernanceDecision``
/ ``GovernanceOutcome`` contract (crewAIInc/crewAI#6030) and the completeness
layer over it; a passing run here means each verdict is a property of those
bytes, recomputable by anyone with the public key.

What it reproduces, from the held records alone:

* every wrapped ``{record, signature}`` verifies under ES256 over ``JCS(record)``;
* the four *derived* content refs recompute from their stated member sets
  (``intent_digest``, ``intent_ref``, ``decision_context_hash``, ``receipt_ref``).
  ``params_hash`` and ``target_state_digest`` are the committed leaves the record
  carries in place of the raw params / target state (the receipt commits to a
  hash, not the cleartext), so they are taken as given, not re-hashed;
* ``sealed_contiguity`` per stream case: ``dropped`` is a provable mid-gap,
  ``tail_sealed`` is a suffix drop the seal still catches, ``tail_unsealed`` is
  the irreducible residual where, with no seal, the held prefix looks whole;
* the four fail-closed verdicts, each forced by the recomputed mismatch;
* the non-ASCII case, where ``intent_ref`` recomputes over a unicode
  ``normalized_scope`` that ``json.dumps(sort_keys=True)`` would not canonicalize
  the same way RFC 8785 does.

Run: tests/vectors/governance_decision_v0/_check_independent.py
Exit 0 means every verdict matched expected.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.primitives.serialization import load_pem_public_key

HERE = Path(__file__).resolve().parent
_STREAM_CASES = ("complete", "dropped", "tail_sealed", "tail_unsealed")
_SEALED = {"complete", "dropped", "tail_sealed"}  # tail_unsealed deliberately has no seal


def _sha(obj) -> str:
    return "sha256:" + hashlib.sha256(rfc8785.dumps(obj)).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_sig(pub, wrapped: dict) -> bool:
    """ES256 over JCS(record); signature is 128 hex == r||s, 32 bytes each."""
    record = wrapped.get("record")
    sig = wrapped.get("signature", "")
    if not isinstance(record, dict) or len(sig) != 128:
        return False
    raw = bytes.fromhex(sig)
    der = encode_dss_signature(
        int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
    )
    try:
        pub.verify(der, rfc8785.dumps(record), ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


def _recompute_refs(record: dict) -> dict:
    """Recompute the four DERIVED refs from the record's own fields.

    ``params_hash`` and ``target_state_digest`` are committed leaves carried in
    the record (the raw params / target state are not disclosed), so they feed
    the recompute as given. Every key here is read off the record, so the check
    is self-describing: no constant is hardcoded into the checker.
    """
    intent_digest = _sha(
        {
            "action_type": record["action_type"],
            "normalized_scope": record["normalized_scope"],
            "params_hash": record["params_hash"],
        }
    )
    intent_ref = _sha(
        {
            "schema": record["schema"],
            "agent_id": record["agent_id"],
            "action_type": record["action_type"],
            "normalized_scope": record["normalized_scope"],
            "intent_digest": intent_digest,
        }
    )
    decision_context_hash = _sha(
        {
            "policy_refs": record["policy_refs"],
            "target_state_digest": record["target_state_digest"],
            "continuation_id": record["continuation_id"],
            "normalization_id": record["normalization_id"],
        }
    )
    receipt_ref = _sha(
        {
            "intent_ref": intent_ref,
            "target_state_digest": record["target_state_digest"],
            "continuation_id": record["continuation_id"],
            "seq": record["completeness"]["seq"],
            "timestamp_ms": record["timestamp_ms"],
            "idempotency_key": record["idempotency_key"],
        }
    )
    return {
        "intent_digest": intent_digest,
        "intent_ref": intent_ref,
        "decision_context_hash": decision_context_hash,
        "receipt_ref": receipt_ref,
    }


def _derivations_consistent(record: dict) -> bool:
    refs = _recompute_refs(record)
    return all(record[k] == v for k, v in refs.items())


# --- contiguity, reproduced from scratch --------------------------------------


def _contiguity(records: list[dict]) -> dict:
    blocks = [r["record"]["completeness"] for r in records]
    seqs = [int(b["seq"]) for b in blocks]
    counts = [int(b["runningCount"]) for b in blocks]
    expected = max((max(seqs) + 1 if seqs else 0), (max(counts) if counts else 0))
    missing = sorted(set(range(expected)) - set(seqs))
    mismatch = any(int(b["runningCount"]) != int(b["seq"]) + 1 for b in blocks)
    present = len(blocks)
    ok = not missing and not mismatch and len(seqs) == len(set(seqs)) and present == expected
    return {"ok": ok, "present": present, "expected": expected, "missingSeqs": missing}


def _sealed_contiguity(records: list[dict], seal: dict | None) -> dict:
    base = _contiguity(records)
    if seal is None:
        return base
    total = int(seal["record"]["total"])
    seqs = {int(r["record"]["completeness"]["seq"]) for r in records}
    missing = sorted(set(range(total)) - seqs)
    ok = not missing and base["ok"] and base["expected"] == total
    return {"ok": ok, "present": len(records), "expected": total, "missingSeqs": missing}


def _load_stream(case: str) -> tuple[list[dict], dict | None]:
    case_dir = HERE / "stream" / case
    records = [_load(p) for p in sorted(case_dir.glob("*-decision.json"))]
    seal_path = case_dir / "seal.json"
    seal = _load(seal_path) if seal_path.exists() else None
    return records, seal


# --- fail-closed verdicts, recomputed from the case bytes ----------------------


def _verdict_exact_intent_mismatch(case: dict) -> str:
    a = _recompute_refs(case["approved"]["record"])["intent_ref"]
    c = _recompute_refs(case["candidate"]["record"])["intent_ref"]
    return "deny" if a != c else "allow"


def _verdict_target_state_drift(case: dict) -> str:
    ar, cr = case["approved"]["record"], case["candidate"]["record"]
    same_intent = _recompute_refs(ar)["intent_ref"] == _recompute_refs(cr)["intent_ref"]
    drift = ar["target_state_digest"] != cr["target_state_digest"]
    return "revise" if (same_intent and drift) else "deny"


def _verdict_continuation_mismatch(case: dict) -> str:
    ar, cr = case["approved"]["record"], case["candidate"]["record"]
    same_intent = _recompute_refs(ar)["intent_ref"] == _recompute_refs(cr)["intent_ref"]
    moved = (
        ar["continuation_id"] != cr["continuation_id"]
        and _recompute_refs(ar)["decision_context_hash"]
        != _recompute_refs(cr)["decision_context_hash"]
    )
    return "deny" if (same_intent and moved) else "allow"


def _verdict_duplicate_outcome(case: dict) -> str:
    f, s = case["first"]["record"], case["second"]["record"]
    replay = f["receipt_ref"] == s["receipt_ref"] and f["idempotency_key"] == s["idempotency_key"]
    return "deny" if replay else "allow"


_VERDICTS = {
    "exact_intent_mismatch": _verdict_exact_intent_mismatch,
    "target_state_drift": _verdict_target_state_drift,
    "continuation_mismatch": _verdict_continuation_mismatch,
    "duplicate_outcome": _verdict_duplicate_outcome,
}
_CASE_WRAPPED = {  # which wrapped records each case file carries, for the sig pass
    "exact_intent_mismatch": ("approved", "candidate"),
    "target_state_drift": ("approved", "candidate"),
    "continuation_mismatch": ("approved", "candidate"),
    "duplicate_outcome": ("first", "second"),
}


def main() -> int:
    pub = load_pem_public_key((HERE / "keys" / "es256_public.pem").read_bytes())
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))

    results: list[tuple[str, bool]] = []

    def check(label: str, ok: bool) -> None:
        results.append((label, ok))
        print(f"[{'OK' if ok else 'FAIL'}] {label}")

    # 1+3. stream signatures, derivation consistency, sealed contiguity.
    for case in _STREAM_CASES:
        records, seal = _load_stream(case)
        sigs_ok = all(_verify_sig(pub, r) for r in records)
        check(f"stream.{case}.all_signatures_ok",
              sigs_ok == expected["stream"][case]["all_signatures_ok"])
        check(f"stream.{case}.derivations_consistent",
              all(_derivations_consistent(r["record"]) for r in records))
        got = _sealed_contiguity(records, seal if case in _SEALED else None)
        check(f"stream.{case}.sealed_contiguity",
              got == expected["stream"][case]["sealed_contiguity"])

    # 2+4. fail-closed verdicts, each forced by the recomputed mismatch.
    for name, fn in _VERDICTS.items():
        case = _load(HERE / "cases" / f"{name}.json")
        sigs_ok = all(_verify_sig(pub, case[k]) for k in _CASE_WRAPPED[name])
        check(f"cases.{name}.signatures_ok",
              sigs_ok == expected["cases"][name]["signatures_ok"])
        check(f"cases.{name}.verdict",
              fn(case) == expected["cases"][name]["expected_verdict"])

    # 5. unicode normalized_scope: sig ok and intent_ref recomputes over the
    # non-ASCII bytes; params_hash matches its committed leaf.
    uni = _load(HERE / "cases" / "unicode_scope.json")
    urec = uni["record"]
    check("unicode_scope.signature_ok",
          _verify_sig(pub, uni) == expected["unicode_scope"]["signature_ok"])
    check("unicode_scope.intent_ref",
          _recompute_refs(urec)["intent_ref"] == expected["unicode_scope"]["intent_ref"]
          == urec["intent_ref"])
    check("unicode_scope.params_hash",
          urec["params_hash"] == expected["unicode_scope"]["params_hash"])

    ok = all(v for _, v in results)
    print(f"\n{'all verdicts matched expected' if ok else 'MISMATCH vs expected'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
