import copy

from vaara.attestation._decision_binding import build_binding
from vaara.attestation._decision_proof_verify import verify_decision_proof
from vaara.attestation.zk._circuit import to_fixed
from vaara.attestation.zk._prove import build_proof_envelope


def _record(score=0.2, deny=0.8, escalate=0.5, verdict="allow", intent="refund <= 50 EUR"):
    policy = {"version": 1, "deny": deny, "escalate": escalate}
    inputs = {"riskScore": score, "deny": deny, "escalate": escalate}
    binding = build_binding(policy, intent, inputs, verdict)
    env = build_proof_envelope(
        verdict, to_fixed(score), to_fixed(deny), to_fixed(escalate),
        binding["bindingDigest"],
    )
    return {
        "decision": verdict,
        "rationale": {"rule": "R1", "reason": "under threshold", "declaredIntent": intent},
        "binding": binding,
        "decisionProof": env,
    }


def test_valid_proof_verifies():
    ok, reason = verify_decision_proof(_record())
    assert ok is True, reason


def test_block_proof_verifies():
    ok, reason = verify_decision_proof(_record(score=0.9, verdict="block"))
    assert ok is True, reason


def test_tampered_verdict_fails():
    rec = _record(score=0.2, verdict="allow")
    rec["decisionProof"]["publicInputs"]["verdict"] = "block"
    rec["decision"] = "block"
    ok, _ = verify_decision_proof(rec)
    assert ok is False


def test_wrong_params_digest_fails():
    rec = _record()
    rec["decisionProof"]["verifierParamsDigest"] = "sha256:" + "00" * 32
    ok, _ = verify_decision_proof(rec)
    assert ok is False


def test_binding_mismatch_fails():
    rec = copy.deepcopy(_record())
    rec["binding"]["bindingDigest"] = "sha256:" + "11" * 32
    ok, _ = verify_decision_proof(rec)
    assert ok is False
