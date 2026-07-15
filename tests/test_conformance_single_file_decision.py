import argparse
import copy
import json

from vaara.attestation._decision_emit import build_decision_basis
from vaara.attestation.zk._circuit import to_fixed
from vaara.cli import _cmd_verify_record

POLICY = {"version": 1, "deny": 0.8, "escalate": 0.5}
INPUTS = {"riskScore": 0.2, "deny": 0.8, "escalate": 0.5}
INTENT = "refund <= 50 EUR to verified customer"


def _record():
    basis = build_decision_basis(
        rule="R1",
        reason="under threshold",
        declared_intent=INTENT,
        canonical_policy=POLICY,
        canonical_inputs=INPUTS,
        verdict="allow",
        score_fp=to_fixed(0.2),
        deny_fp=to_fixed(0.8),
        escalate_fp=to_fixed(0.5),
        with_proof=True,
    )
    return {
        "version": 1,
        "alg": "HS256",
        "backLink": {"attestationDigest": "sha256:" + "aa" * 32, "attestationNonce": "n"},
        "decisionDerived": {
            "decision": "allow",
            "decidedAt": "2026-07-15T00:00:00Z",
            "rationale": basis["rationale"],
            "binding": basis["binding"],
            "decisionProof": basis["decisionProof"],
        },
        "issuerAsserted": {
            "iss": "x", "sub": "y", "iat": "2026-07-15T00:00:00Z",
            "nonce": "n", "secretVersion": "1", "alg": "HS256",
        },
        "signature": "ab",
    }


def _run(tmp_path, record):
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(record), encoding="utf-8")
    args = argparse.Namespace(record=str(p), attestation=None, json=True)
    return _cmd_verify_record(args)


def test_single_file_decision_proof_passes(tmp_path):
    assert _run(tmp_path, _record()) == 0


def test_single_file_tampered_verdict_fails(tmp_path):
    rec = copy.deepcopy(_record())
    rec["decisionDerived"]["decision"] = "block"
    rec["decisionDerived"]["decisionProof"]["publicInputs"]["verdict"] = "block"
    assert _run(tmp_path, rec) == 1
