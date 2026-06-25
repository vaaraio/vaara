#!/usr/bin/env python3
"""Regenerate the governance_decision_v0 conformance vectors.

These vectors pin the *bytes* of the CrewAI ``GovernanceDecision`` /
``GovernanceOutcome`` contract (crewAIInc/crewAI#6030) and the completeness
layer that ``vaara.integrations.crewai`` (vaaraio/vaara#283) emits over it, as
one reproducible corpus. The contract envelope is CrewAI's; the recomputable
derivations, the gap-evident sequence, and the terminal seal are what an auditor
checks. The sibling ``_check_independent.py`` reproduces every verdict with no
Vaara import, so a passing check is a property of the bytes, not of this script.

Pinned, per record:

* the five content derivations, each a SHA-256 over the JCS (RFC 8785) canonical
  bytes of a named member set documented in README.md: ``params_hash``,
  ``intent_digest``, ``intent_ref`` (stable, no timestamp), ``receipt_ref``
  (per-record, timestamped), ``decision_context_hash``;
* a 0-indexed ``seq`` with ``running_count == seq + 1`` and a terminal
  ``GovernanceSeal`` carrying the boundary total, so a dropped record is a
  provable gap from the held set alone;
* the four fail-closed contract cases as verifier-recomputable mismatches:
  exact-intent mismatch -> deny, target-state drift -> revise, continuation
  mismatch -> deny, duplicate outcome -> deny;
* a non-ASCII ``normalized_scope`` case, where a ``json.dumps(sort_keys=True)``
  shortcut diverges from RFC 8785 and a conformant producer must not.

Run: tests/vectors/governance_decision_v0/_generate.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import rfc8785
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

HERE = Path(__file__).resolve().parent

# Fixed scalar -> stable public key across regenerations. Test-only key.
_SCALAR = 0x67074E2AC0FFEE5EA10DDF00DABCDEF0123456789998877665544332211000F
SCHEMA_DECISION = "crewai.governance.decision/v1"
SCHEMA_OUTCOME = "crewai.governance.outcome/v1"
SCHEMA_SEAL = "crewai.governance.seal/v1"
NORMALIZATION_ID = "jcs-rfc8785-v1"
BOUNDARY = "crew:checkout-run/2026-06-25"
AGENT = "agent:crewai/checkout-bot"
POLICY_REFS = ["policy:spend-limit/v3", "policy:eu-ai-act-art14/v1"]
STREAM_LEN = 4
DROPPED_SEQ = 2


def _sha(obj) -> str:
    return "sha256:" + hashlib.sha256(rfc8785.dumps(obj)).hexdigest()


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _derivations(
    *,
    action_type: str,
    normalized_scope: str,
    params: dict,
    target_state: dict,
    continuation_id: str,
    seq: int,
    timestamp_ms: int,
    idempotency_key: str,
) -> dict:
    """The five pinned content derivations. Member sets are documented in README.

    intent_digest / intent_ref carry NO timestamp: the same authorized intent
    recomputes to the same identity on a retry. receipt_ref is per-record and
    carries seq + timestamp, so it is unique per execution attempt.
    """
    params_hash = _sha(params)
    intent_digest = _sha(
        {
            "action_type": action_type,
            "normalized_scope": normalized_scope,
            "params_hash": params_hash,
        }
    )
    intent_ref = _sha(
        {
            "schema": SCHEMA_DECISION,
            "agent_id": AGENT,
            "action_type": action_type,
            "normalized_scope": normalized_scope,
            "intent_digest": intent_digest,
        }
    )
    target_state_digest = _sha(target_state)
    decision_context_hash = _sha(
        {
            "policy_refs": POLICY_REFS,
            "target_state_digest": target_state_digest,
            "continuation_id": continuation_id,
            "normalization_id": NORMALIZATION_ID,
        }
    )
    receipt_ref = _sha(
        {
            "intent_ref": intent_ref,
            "target_state_digest": target_state_digest,
            "continuation_id": continuation_id,
            "seq": seq,
            "timestamp_ms": timestamp_ms,
            "idempotency_key": idempotency_key,
        }
    )
    return {
        "params_hash": params_hash,
        "intent_digest": intent_digest,
        "intent_ref": intent_ref,
        "target_state_digest": target_state_digest,
        "decision_context_hash": decision_context_hash,
        "receipt_ref": receipt_ref,
    }


def _decision_record(
    *,
    seq: int,
    action_type: str,
    normalized_scope: str,
    params: dict,
    target_state: dict,
    continuation_id: str,
    decision: str,
    idempotency_key: str,
) -> dict:
    timestamp_ms = 1779200000000 + seq * 1000
    der = _derivations(
        action_type=action_type,
        normalized_scope=normalized_scope,
        params=params,
        target_state=target_state,
        continuation_id=continuation_id,
        seq=seq,
        timestamp_ms=timestamp_ms,
        idempotency_key=idempotency_key,
    )
    return {
        "schema": SCHEMA_DECISION,
        "decision_id": f"{BOUNDARY}#{seq}",
        "agent_id": AGENT,
        "action_type": action_type,
        "normalization_id": NORMALIZATION_ID,
        "normalized_scope": normalized_scope,
        "params_hash": der["params_hash"],
        "intent_digest": der["intent_digest"],
        "intent_ref": der["intent_ref"],
        "target_state_digest": der["target_state_digest"],
        "decision_context_hash": der["decision_context_hash"],
        "receipt_ref": der["receipt_ref"],
        "continuation_id": continuation_id,
        "policy_refs": POLICY_REFS,
        "decision": decision,
        "idempotency_key": idempotency_key,
        "timestamp_ms": timestamp_ms,
        "completeness": {"boundaryId": BOUNDARY, "seq": seq, "runningCount": seq + 1},
    }


def _outcome_record(decision: dict, *, status: str, result: dict) -> dict:
    return {
        "schema": SCHEMA_OUTCOME,
        "decision_id": decision["decision_id"],
        "intent_ref": decision["intent_ref"],
        "receipt_ref": decision["receipt_ref"],
        "status": status,
        "result_digest": _sha(result),
        "idempotency_key": decision["idempotency_key"],
        "completeness": dict(decision["completeness"]),
    }


def _seal(total: int, *, max_class: str | None = None) -> dict:
    seal = {"schema": SCHEMA_SEAL, "boundaryId": BOUNDARY, "sealed": True, "total": total}
    if max_class is not None:
        seal["maxClass"] = max_class
    return seal


def _sign(priv, record: dict) -> str:
    raw = priv.sign(_jcs(record), ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(raw)
    return r.to_bytes(32, "big").hex() + s.to_bytes(32, "big").hex()


def _wrap(priv, record: dict) -> dict:
    return {"record": record, "signature": _sign(priv, record)}


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    """Contiguity against the seal total when a seal is held.

    A terminal seal carries the boundary total, so a suffix drop that does not
    also remove the seal is a provable gap. With no seal, a suffix drop is the
    irreducible residual: the held prefix looks whole.
    """
    base = _contiguity(records)
    if seal is None:
        return base
    total = int(seal["record"]["total"])
    seqs = {int(r["record"]["completeness"]["seq"]) for r in records}
    missing = sorted(set(range(total)) - seqs)
    ok = not missing and base["ok"] and base["expected"] == total
    return {"ok": ok, "present": len(records), "expected": total, "missingSeqs": missing}


def main() -> int:
    priv = ec.derive_private_key(_SCALAR, ec.SECP256R1())
    (HERE / "keys").mkdir(parents=True, exist_ok=True)
    (HERE / "keys" / "es256_public.pem").write_bytes(
        priv.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    target_state = {"cart_total": "42.00", "currency": "USDC", "merchant": "acme"}
    continuation = "session:abc-123"

    stream = []
    for seq in range(STREAM_LEN):
        dec = _decision_record(
            seq=seq, action_type="purchase.fulfill",
            normalized_scope=f"merchant:acme/order:INV-{seq}",
            params={"amount": "42.00", "currency": "USDC", "order": f"INV-{seq}"},
            target_state=target_state, continuation_id=continuation,
            decision="allow", idempotency_key=f"idem-{seq}",
        )
        out = _outcome_record(dec, status="executed", result={"order": f"INV-{seq}", "state": "fulfilled"})
        stream.append((dec, out))
    wrapped_seal = _wrap(priv, _seal(STREAM_LEN, max_class="confidential"))

    complete_records = []
    for seq, (dec, out) in enumerate(stream):
        wd, wo = _wrap(priv, dec), _wrap(priv, out)
        complete_records.append(wd)
        _write(HERE / "stream" / "complete" / f"{seq:04d}-decision.json", wd)
        _write(HERE / "stream" / "complete" / f"{seq:04d}-outcome.json", wo)
        if seq != DROPPED_SEQ:
            _write(HERE / "stream" / "dropped" / f"{seq:04d}-decision.json", wd)
            _write(HERE / "stream" / "dropped" / f"{seq:04d}-outcome.json", wo)
        if seq != STREAM_LEN - 1:
            _write(HERE / "stream" / "tail_sealed" / f"{seq:04d}-decision.json", wd)
            _write(HERE / "stream" / "tail_sealed" / f"{seq:04d}-outcome.json", wo)
            _write(HERE / "stream" / "tail_unsealed" / f"{seq:04d}-decision.json", wd)
            _write(HERE / "stream" / "tail_unsealed" / f"{seq:04d}-outcome.json", wo)
    _write(HERE / "stream" / "complete" / "seal.json", wrapped_seal)
    _write(HERE / "stream" / "dropped" / "seal.json", wrapped_seal)
    _write(HERE / "stream" / "tail_sealed" / "seal.json", wrapped_seal)

    base = _decision_record(seq=0, action_type="purchase.fulfill",
        normalized_scope="merchant:acme/order:INV-1",
        params={"amount": "42.00", "currency": "USDC", "order": "INV-1"},
        target_state=target_state, continuation_id=continuation,
        decision="allow", idempotency_key="idem-A")
    mismatch = _decision_record(seq=0, action_type="purchase.fulfill",
        normalized_scope="merchant:acme/order:INV-999",
        params={"amount": "42.00", "currency": "USDC", "order": "INV-999"},
        target_state=target_state, continuation_id=continuation,
        decision="allow", idempotency_key="idem-A")
    _write(HERE / "cases" / "exact_intent_mismatch.json",
        {"approved": _wrap(priv, base), "candidate": _wrap(priv, mismatch), "expected_verdict": "deny"})
    drift = _decision_record(seq=0, action_type="purchase.fulfill",
        normalized_scope="merchant:acme/order:INV-1",
        params={"amount": "42.00", "currency": "USDC", "order": "INV-1"},
        target_state={"cart_total": "88.00", "currency": "USDC", "merchant": "acme"},
        continuation_id=continuation, decision="revise", idempotency_key="idem-A")
    _write(HERE / "cases" / "target_state_drift.json",
        {"approved": _wrap(priv, base), "candidate": _wrap(priv, drift), "expected_verdict": "revise"})
    cont = _decision_record(seq=0, action_type="purchase.fulfill",
        normalized_scope="merchant:acme/order:INV-1",
        params={"amount": "42.00", "currency": "USDC", "order": "INV-1"},
        target_state=target_state, continuation_id="session:hijacked-999",
        decision="allow", idempotency_key="idem-A")
    _write(HERE / "cases" / "continuation_mismatch.json",
        {"approved": _wrap(priv, base), "candidate": _wrap(priv, cont), "expected_verdict": "deny"})
    out1 = _outcome_record(base, status="executed", result={"order": "INV-1", "state": "fulfilled"})
    out2 = _outcome_record(base, status="executed", result={"order": "INV-1", "state": "fulfilled"})
    _write(HERE / "cases" / "duplicate_outcome.json",
        {"first": _wrap(priv, out1), "second": _wrap(priv, out2), "expected_verdict": "deny"})
    unicode_dec = _decision_record(seq=0, action_type="purchase.fulfill",
        normalized_scope="merchant:café/order:Ünïcode-€",
        params={"amount": "42.00", "merchant": "café", "note": "résumé"},
        target_state=target_state, continuation_id=continuation,
        decision="allow", idempotency_key="idem-U")
    _write(HERE / "cases" / "unicode_scope.json", _wrap(priv, unicode_dec))

    expected = {
        "stream": {
            "complete": {"all_signatures_ok": True, "sealed_contiguity": _sealed_contiguity(complete_records, wrapped_seal)},
            "dropped": {"all_signatures_ok": True, "sealed_contiguity": _sealed_contiguity([r for i, r in enumerate(complete_records) if i != DROPPED_SEQ], wrapped_seal)},
            "tail_sealed": {"all_signatures_ok": True, "sealed_contiguity": _sealed_contiguity(complete_records[:-1], wrapped_seal)},
            "tail_unsealed": {"all_signatures_ok": True, "sealed_contiguity": _sealed_contiguity(complete_records[:-1], None)},
        },
        "cases": {
            "exact_intent_mismatch": {"expected_verdict": "deny", "signatures_ok": True},
            "target_state_drift": {"expected_verdict": "revise", "signatures_ok": True},
            "continuation_mismatch": {"expected_verdict": "deny", "signatures_ok": True},
            "duplicate_outcome": {"expected_verdict": "deny", "signatures_ok": True},
        },
        "unicode_scope": {"signature_ok": True, "intent_ref": unicode_dec["intent_ref"], "params_hash": unicode_dec["params_hash"]},
    }
    _write(HERE / "expected.json", expected)
    print("wrote governance_decision_v0 vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
