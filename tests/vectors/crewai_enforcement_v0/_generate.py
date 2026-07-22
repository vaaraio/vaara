#!/usr/bin/env python3
"""Regenerate the crewai_enforcement_v0 conformance vectors.

These vectors pin the *bytes* of the enforcement claim behind the CrewAI
``GovernanceDecision`` / ``GovernanceOutcome`` contract
(crewAIInc/crewAI#6030): a ``deny`` decision counts as enforced only when a
linked outcome attests the tool did not execute. A ``deny`` paired with an
outcome that shows execution is a ``violation``: the record says deny but the
side effect happened anyway. That gap is the P0 the reducer work has to close,
and this corpus makes it testable rather than asserted.

The sibling ``_check_independent.py`` recomputes every verdict with no Vaara
import (only ``rfc8785`` and ``cryptography``), so a passing check is a property
of the committed bytes, not of this script. The signing key is a fixed test
scalar, so the bytes are stable across regenerations.

Run: tests/vectors/crewai_enforcement_v0/_generate.py
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
_SCALAR = 0x5EC0FFEE11EE22DD33CC44BB55AA66990088776655443322110ABCDEF0F1E2D3
SCHEMA_DECISION = "crewai.governance.decision/v1"
SCHEMA_OUTCOME = "crewai.governance.outcome/v1"
BOUNDARY = "crew:enforcement-demo/2026-07-21"
AGENT = "agent:crewai/checkout-bot"
POLICY_REFS = ["policy:vaara/deny-untrusted-fs-write@v1"]


def _sha(obj) -> str:
    return "sha256:" + hashlib.sha256(rfc8785.dumps(obj)).hexdigest()


def _jcs(obj) -> bytes:
    return rfc8785.dumps(obj)


def _sign(priv, record: dict) -> str:
    raw = priv.sign(_jcs(record), ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(raw)
    return f"{r:064x}{s:064x}"


def _wrap(priv, record: dict) -> dict:
    return {"record": record, "signature": _sign(priv, record)}


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _decision(seq: int, action_type: str, normalized_scope: str, params: dict) -> dict:
    params_hash = _sha(params)
    intent_digest = _sha(
        {"action_type": action_type, "normalized_scope": normalized_scope,
         "params_hash": params_hash}
    )
    intent_ref = _sha(
        {"schema": SCHEMA_DECISION, "agent_id": AGENT, "action_type": action_type,
         "normalized_scope": normalized_scope, "intent_digest": intent_digest}
    )
    decision_id = f"{BOUNDARY}#{seq}"
    receipt_ref = _sha({"decision_id": decision_id, "intent_ref": intent_ref, "seq": seq})
    return {
        "schema": SCHEMA_DECISION,
        "decision_id": decision_id,
        "agent_id": AGENT,
        "action_type": action_type,
        "normalized_scope": normalized_scope,
        "params_hash": params_hash,
        "intent_digest": intent_digest,
        "intent_ref": intent_ref,
        "receipt_ref": receipt_ref,
        "policy_refs": POLICY_REFS,
        "decision": "deny",
        "seq": seq,
    }


def _outcome(decision: dict, *, executed: bool, status: str) -> dict:
    return {
        "schema": SCHEMA_OUTCOME,
        "decision_id": decision["decision_id"],
        "intent_ref": decision["intent_ref"],
        "receipt_ref": decision["receipt_ref"],
        "executed": executed,
        "status": status,
        "seq": decision["seq"],
    }


def _verdict(decision: dict, outcome: dict) -> str:
    """The enforcement rule, recomputable from the two records alone."""
    linked = (
        outcome["decision_id"] == decision["decision_id"]
        and outcome["intent_ref"] == decision["intent_ref"]
        and outcome["receipt_ref"] == decision["receipt_ref"]
    )
    if not linked:
        return "unlinked"
    if decision["decision"] != "deny":
        return "out_of_scope"
    if outcome["executed"] is False and outcome["status"] == "blocked":
        return "enforced"
    return "violation"


def main() -> int:
    priv = ec.derive_private_key(_SCALAR, ec.SECP256R1())
    (HERE / "keys").mkdir(exist_ok=True)
    (HERE / "keys" / "es256_public.pem").write_bytes(
        priv.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

    params = {"content": "root:x:0:0:root:/root:/bin/bash\n", "path": "/etc/passwd"}
    dec = _decision(0, "fs.write_file", "fs.write_file:/etc/*", params)
    honored = _outcome(dec, executed=False, status="blocked")
    violated = _outcome(dec, executed=True, status="completed")

    cases = {
        "enforced_deny": {"decision": _wrap(priv, dec), "outcome": _wrap(priv, honored)},
        "violated_deny": {"decision": _wrap(priv, dec), "outcome": _wrap(priv, violated)},
    }
    _write(HERE / "cases.json", cases)

    expected = {
        name: {
            "signatures_ok": True,
            "verdict": _verdict(c["decision"]["record"], c["outcome"]["record"]),
        }
        for name, c in cases.items()
    }
    _write(HERE / "expected.json", expected)
    print(f"wrote {len(cases)} cases to {HERE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
