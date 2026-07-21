#!/usr/bin/env python3
"""Independent checker for the crewai_enforcement_v0 conformance vectors.

Imports only the standard library plus ``cryptography`` and ``rfc8785``. It does
NOT import Vaara. The corpus pins the bytes of the enforcement claim over the
CrewAI ``GovernanceDecision`` / ``GovernanceOutcome`` contract
(crewAIInc/crewAI#6030): a ``deny`` is enforced only when a linked outcome
attests the tool did not run; a ``deny`` paired with execution is a
``violation``. A passing run here means each verdict is a property of the
committed bytes, recomputable by anyone with the public key.

What it reproduces, from the held records alone:

* both wrapped ``{record, signature}`` pairs verify under ES256 over ``JCS(record)``;
* the decision's ``intent_ref`` and ``receipt_ref`` recompute from their member sets;
* the outcome links to the decision by ``decision_id``, ``intent_ref``, and ``receipt_ref``;
* the enforcement verdict: ``enforced`` when the outcome shows the tool was
  blocked and did not execute, ``violation`` when a deny still executed.

Run: tests/vectors/crewai_enforcement_v0/_check_independent.py
Exit 0 means every verdict matched expected.json; exit 1 on any mismatch.
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
SCHEMA_DECISION = "crewai.governance.decision/v1"
AGENT = "agent:crewai/checkout-bot"


def _sha(obj) -> str:
    return "sha256:" + hashlib.sha256(rfc8785.dumps(obj)).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_sig(pub, wrapped: dict) -> bool:
    """ES256 over JCS(record); signature is 128 hex == r||s, 32 bytes each."""
    record = wrapped.get("record")
    sig = wrapped.get("signature", "")
    if record is None or len(sig) != 128:
        return False
    try:
        r = int(sig[:64], 16)
        s = int(sig[64:], 16)
    except ValueError:
        return False
    try:
        pub.verify(
            encode_dss_signature(r, s),
            rfc8785.dumps(record),
            ec.ECDSA(hashes.SHA256()),
        )
        return True
    except InvalidSignature:
        return False


def _refs_consistent(dec: dict) -> bool:
    intent_digest = _sha(
        {"action_type": dec["action_type"], "normalized_scope": dec["normalized_scope"],
         "params_hash": dec["params_hash"]}
    )
    intent_ref = _sha(
        {"schema": SCHEMA_DECISION, "agent_id": AGENT, "action_type": dec["action_type"],
         "normalized_scope": dec["normalized_scope"], "intent_digest": intent_digest}
    )
    receipt_ref = _sha(
        {"decision_id": dec["decision_id"], "intent_ref": intent_ref, "seq": dec["seq"]}
    )
    return (
        dec["intent_digest"] == intent_digest
        and dec["intent_ref"] == intent_ref
        and dec["receipt_ref"] == receipt_ref
    )


def _verdict(dec: dict, out: dict) -> str:
    linked = (
        out["decision_id"] == dec["decision_id"]
        and out["intent_ref"] == dec["intent_ref"]
        and out["receipt_ref"] == dec["receipt_ref"]
    )
    if not linked:
        return "unlinked"
    if dec["decision"] != "deny":
        return "out_of_scope"
    if out["executed"] is False and out["status"] == "blocked":
        return "enforced"
    return "violation"


def main() -> int:
    pub = load_pem_public_key((HERE / "keys" / "es256_public.pem").read_bytes())
    cases = _load(HERE / "cases.json")
    expected = _load(HERE / "expected.json")

    failures: list[str] = []
    for name, case in cases.items():
        dec_w, out_w = case["decision"], case["outcome"]
        dec, out = dec_w["record"], out_w["record"]

        sigs_ok = _verify_sig(pub, dec_w) and _verify_sig(pub, out_w)
        if not _refs_consistent(dec):
            failures.append(f"{name}: decision refs do not recompute")
        verdict = _verdict(dec, out)

        want = expected.get(name, {})
        if sigs_ok != want.get("signatures_ok"):
            failures.append(f"{name}: signatures_ok {sigs_ok} != {want.get('signatures_ok')}")
        if verdict != want.get("verdict"):
            failures.append(f"{name}: verdict {verdict!r} != {want.get('verdict')!r}")

    if failures:
        for f in failures:
            print("FAIL", f, file=sys.stderr)
        return 1
    print(f"crewai_enforcement_v0: {len(cases)} cases OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
