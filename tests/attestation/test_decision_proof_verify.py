"""Verification of the decisionProof, including forgery attempts.

These tests encode the attacks an adversarial review raised: a proof whose
committed values are not bound to the record, a proof lifted onto another
record, a relabelled verdict, and absent cross-check fields.
"""

from vaara.attestation._decision_emit import build_decision_basis
from vaara.attestation._decision_proof_verify import verify_decision_proof
from vaara.attestation.zk._circuit import to_fixed

POLICY = {"version": 1}
INTENT = "refund <= 50 EUR to verified customer"


def _record(score=0.2, deny=0.8, escalate=0.5, verdict="allow"):
    basis = build_decision_basis(
        rule="R1",
        reason="under threshold",
        declared_intent=INTENT,
        canonical_policy=POLICY,
        canonical_inputs={},
        verdict=verdict,
        score_fp=to_fixed(score),
        deny_fp=to_fixed(deny),
        escalate_fp=to_fixed(escalate),
        with_proof=True,
    )
    return {
        "decision": verdict,
        "rationale": basis["rationale"],
        "binding": basis["binding"],
        "decisionProof": basis["decisionProof"],
    }


def test_valid_allow_verifies():
    ok, reason = verify_decision_proof(_record())
    assert ok is True, reason


def test_valid_block_verifies():
    ok, reason = verify_decision_proof(_record(score=0.9, verdict="block"))
    assert ok is True, reason


def test_valid_escalate_verifies():
    ok, reason = verify_decision_proof(_record(score=0.6, verdict="escalate"))
    assert ok is True, reason


# --- mandatory cross-checks (HIGH-2: no fail-open) ---

def test_missing_binding_rejected():
    rec = _record()
    del rec["binding"]
    ok, _ = verify_decision_proof(rec)
    assert ok is False


def test_binding_not_dict_rejected():
    rec = _record()
    rec["binding"] = "not-a-dict"
    ok, _ = verify_decision_proof(rec)
    assert ok is False


def test_missing_decision_rejected():
    rec = _record()
    del rec["decision"]
    ok, _ = verify_decision_proof(rec)
    assert ok is False


# --- forgery / lifting attacks (CRITICAL-1: commitments bound to the record) ---

def test_verdict_relabel_rejected():
    rec = _record(score=0.2, verdict="allow")
    rec["decision"] = "block"
    rec["decisionProof"]["publicInputs"]["verdict"] = "block"
    ok, _ = verify_decision_proof(rec)
    assert ok is False


def test_foreign_proof_bytes_rejected():
    # Splice a proof built for different committed values onto this record. Its
    # commitments will not match binding.inputsDigest.
    good = _record(score=0.2)
    other = _record(score=0.1)
    good["decisionProof"]["proof"] = other["decisionProof"]["proof"]
    ok, reason = verify_decision_proof(good)
    assert ok is False


def test_whole_proof_swap_rejected():
    good = _record(score=0.2)
    other = _record(score=0.1)
    good["decisionProof"] = other["decisionProof"]
    ok, _ = verify_decision_proof(good)
    assert ok is False


def test_inputs_digest_tamper_rejected():
    rec = _record()
    rec["binding"]["inputsDigest"] = "sha256:" + "00" * 32
    ok, _ = verify_decision_proof(rec)
    assert ok is False


def test_params_digest_tamper_rejected():
    rec = _record()
    rec["decisionProof"]["verifierParamsDigest"] = "sha256:" + "00" * 32
    ok, _ = verify_decision_proof(rec)
    assert ok is False
