"""Cryptographic verification of a SEP-2828 ``decisionProof`` (behind the extra).

The keyless conformance checker (``_decision_conformance``) validates the proof
envelope's shape. This module goes further and checks the proof itself: that the
zero-knowledge argument holds for the claimed verdict under the pinned parameters
and binding digest. It imports the P-256 engine and so requires the ``attestation``
extra, the same split signatures and time anchors use.
"""

from __future__ import annotations

from typing import Any

from .zk._params import PROOF_SYSTEM, params_digest
from .zk._verify import verify as _zk_verify


def verify_decision_proof(decision_derived: dict[str, Any]) -> tuple[bool, str]:
    """Return (ok, reason) for the decisionProof carried on a decision record.

    Fails closed on any missing field, parameter mismatch, or failed proof. Does
    not raise on malformed input.
    """
    p = decision_derived.get("decisionProof")
    if not isinstance(p, dict):
        return False, "no decisionProof present"
    if p.get("proofSystem") != PROOF_SYSTEM:
        return False, f"unsupported proofSystem {p.get('proofSystem')!r}"
    if p.get("verifierParamsDigest") != params_digest():
        return False, "verifierParamsDigest does not match this engine"
    pi = p.get("publicInputs")
    if not isinstance(pi, dict):
        return False, "publicInputs missing"
    bd = pi.get("bindingDigest")
    verdict = pi.get("verdict")
    if not isinstance(bd, str) or not isinstance(verdict, str):
        return False, "publicInputs incomplete"
    binding = decision_derived.get("binding")
    if isinstance(binding, dict) and binding.get("bindingDigest") != bd:
        return False, "publicInputs.bindingDigest does not match binding.bindingDigest"
    decision = decision_derived.get("decision")
    if isinstance(decision, str) and decision != verdict:
        return False, "publicInputs.verdict does not match decisionDerived.decision"
    pf = p.get("proof")
    if not isinstance(pf, str):
        return False, "proof missing"
    try:
        raw = bytes.fromhex(pf)
    except ValueError:
        return False, "proof is not valid hex"
    ok = _zk_verify(raw, verdict, bd)
    return (ok, "proof verified" if ok else "proof failed verification")
