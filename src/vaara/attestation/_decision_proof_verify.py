# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cryptographic verification of a SEP-2828 ``decisionProof`` (behind the extra).

The keyless conformance checker (``_decision_conformance``) validates the proof
envelope's shape. This module goes further and checks the proof itself: that the
zero-knowledge argument holds for the claimed verdict under the pinned parameters
and binding digest. It imports the P-256 engine and so requires the ``attestation``
extra, the same split signatures and time anchors use.
"""

from __future__ import annotations

from typing import Any

from ._decision_binding import binding_digest as _binding_digest
from ._decision_binding import inputs_digest as _inputs_digest
from .zk._params import PROOF_SYSTEM, params_digest
from .zk._prove import commitments_from_proof
from .zk._verify import verify as _zk_verify


def verify_decision_proof(decision_derived: dict[str, Any]) -> tuple[bool, str]:
    """Return (ok, reason) for the decisionProof carried on a decision record.

    Fails closed on any missing field, parameter mismatch, or failed proof. The
    cross-checks are mandatory: an absent ``binding`` or ``decision`` is a
    rejection, not a skip. The proof's committed values are bound to the record by
    requiring ``binding.inputsDigest`` to be the digest of exactly those
    commitments, so a proof cannot be lifted onto a record it was not built for.
    Does not raise on malformed input.
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
    if not isinstance(binding, dict):
        return False, "binding block is required to verify a proof"
    decision = decision_derived.get("decision")
    if not isinstance(decision, str):
        return False, "decisionDerived.decision is required"
    if binding.get("bindingDigest") != bd:
        return False, "publicInputs.bindingDigest does not match binding.bindingDigest"
    if decision != verdict:
        return False, "publicInputs.verdict does not match decisionDerived.decision"

    pf = p.get("proof")
    if not isinstance(pf, str):
        return False, "proof missing"
    try:
        raw = bytes.fromhex(pf)
    except ValueError:
        return False, "proof is not valid hex"

    # Bind the proof's commitments to the signed record: inputsDigest MUST be the
    # digest of exactly the commitments the proof carries, and bindingDigest MUST
    # be honestly derived from its components.
    try:
        vs, vd, ve = commitments_from_proof(raw)
    except (ValueError, IndexError):
        return False, "proof does not carry well-formed commitments"
    zk_inputs = {
        "scoreCommit": vs.to_bytes().hex(),
        "denyCommit": vd.to_bytes().hex(),
        "escalateCommit": ve.to_bytes().hex(),
    }
    if _inputs_digest(zk_inputs) != binding.get("inputsDigest"):
        return False, "binding.inputsDigest does not commit the proof's commitments"
    want_bd = _binding_digest(
        binding.get("policyDigest", ""),
        binding.get("intentDigest", ""),
        binding.get("inputsDigest", ""),
        verdict,
    )
    if want_bd != bd:
        return False, "bindingDigest is not consistent with its components"

    ok = _zk_verify(raw, verdict, bd)
    return (ok, "proof verified" if ok else "proof failed verification")
