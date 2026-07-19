#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the evidenceRef conformance vectors.

Writes signed fixtures under ``tests/vectors/evidence_ref_v0/``. Each case
holds a signed decision record whose ``decisionDerived`` carries an
``evidenceRef`` content address, the external drift record that address
points at, and an ``expected.json`` of verdicts an independent verifier
MUST reproduce from the committed bytes alone.

The property the vectors pin is the two-implementation recompute contract
from ``docs/design/evidence-ref-mapping-spec.md``: a detector emits a
drift record and computes its content address; the decision issuer signs a
record citing that address; a third party who trusts neither side
recomputes the address from the referenced bytes and verifies the
signature. Two verdicts capture it:

- ``decision_signature_ok``: the signature verifies over the canonical
  ``(version, alg, backLink, decisionDerived, issuerAsserted)`` blocks.
  Because ``evidenceRef`` sits inside ``decisionDerived``, a swapped or
  stripped citation breaks this. The binding is not advisory.
- ``evidence_ref_resolves``: ``sha256`` over the JCS-canonical drift
  record equals ``decisionDerived.evidenceRef.digest``. The citation
  points at exactly the committed bytes and nothing else.

The two are independent. ``tampered_drift_record`` keeps a valid signature
while the evidence stops resolving, which is the case a signature alone
cannot catch.

Usage: python scripts/generate_evidence_ref_vectors.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from vaara.attestation.decision import (
    DecisionDerived, EvidenceRef, emit_decision_record, make_back_link,
)
from vaara.attestation.tool_call_attestation import (
    PayloadDerived, PlannerDeclared, ToolCallBinding,
    emit_attestation, make_args_digest,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "vectors" / "evidence_ref_v0"
HS = bytes.fromhex("42" * 32)
IAT = "2026-06-01T10:00:00Z"
COMMON = dict(iss="issuer://test", sub="agent:reader",
              secret_version="v1", iat=IAT)

def _jcs_address(obj: dict) -> str:
    import hashlib

    import rfc8785
    return "sha256:" + hashlib.sha256(rfc8785.dumps(obj)).hexdigest()


# The two tool surfaces the drift record's hashes are computed over. These
# are Interlock's schema (the detector owns the surface shape); they ship
# so the surface hashes in the drift record are real sha256 over published
# bytes, not placeholders. send_invoice gains an external network effect
# after approval: effects.network goes from [] to billing.example.com.
APPROVED_SURFACE = {
    "schema": "interlock.tool-surface/v0",
    "tool": "send_invoice",
    "inputs": {"invoice": "string"},
    "effects": {"network": [], "filesystem": []},
}
CURRENT_SURFACE = {
    "schema": "interlock.tool-surface/v0",
    "tool": "send_invoice",
    "inputs": {"invoice": "string"},
    "effects": {"network": ["https://billing.example.com"], "filesystem": []},
}

# The drift record from the worked example in
# docs/design/evidence-ref-mapping-spec.md. approvedSurfaceHash and
# currentSurfaceHash are the JCS content addresses of the two surfaces
# above, so the whole chain (surface bytes -> surface hash -> drift record
# -> evidenceRef address) recomputes end to end.
DRIFT_RECORD = {
    "schema": "interlock.drift-record/v0",
    "tool": "send_invoice",
    "approvedSurfaceHash": _jcs_address(APPROVED_SURFACE),
    "currentSurfaceHash": _jcs_address(CURRENT_SURFACE),
    "classifiedDelta": {
        "kind": "external-reach-added",
        "field": "effects.network",
        "from": [],
        "to": ["https://billing.example.com"],
    },
    "policyId": "policy:tool-surface/2",
    "observedAt": "2026-06-01T10:00:00Z",
}


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _attestation(nonce="fixed-attestation-nonce-000"):
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="send_invoice", server_fingerprint="sha256:" + "1" * 64,
        args=make_args_digest({"invoice": "INV-1"})),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="send the invoice"),
        payload_derived=payload, iss="issuer://test", sub="agent:reader",
        secret_version="v1", alg="HS256", signing_material=HS,
        nonce=nonce, iat=IAT)


def _decision(bl, ref: EvidenceRef, *, key, alg, nonce):
    return emit_decision_record(
        back_link=bl, decision_derived=DecisionDerived(
            decision="escalate", decided_at=IAT,
            reason="post-approval external-reach drift on send_invoice",
            risk_score="0.74", threshold_allow="0.30", threshold_block="0.80",
            policy_id="policy:tool-surface/2", evidence_ref=ref),
        alg=alg, signing_material=key, nonce=nonce, **COMMON)


def main() -> None:
    es = ec.generate_private_key(ec.SECP256R1())
    keys = OUT / "keys"
    keys.mkdir(parents=True, exist_ok=True)
    (keys / "hs256_secret.bin").write_bytes(HS)
    (keys / "es256_private.pem").write_bytes(es.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption()))
    (keys / "es256_public.pem").write_bytes(es.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo))

    att = _attestation()
    bl = make_back_link(att)
    address = _jcs_address(DRIFT_RECORD)
    ref = EvidenceRef(
        digest=address, canonicalization="JCS",
        schema="interlock.drift-record/v0", ref="ipfs://bafy-drift-record-cid")

    # Publish the two surfaces the drift record's hashes are computed over,
    # so a second implementation can recompute approvedSurfaceHash and
    # currentSurfaceHash from real bytes, not just the evidenceRef address.
    valid_dir = OUT / "normative" / "valid_evidence_ref_resolves"
    _write(valid_dir / "approved_surface.json", APPROVED_SURFACE)
    _write(valid_dir / "current_surface.json", CURRENT_SURFACE)

    def case(name, *, decision, drift, expected):
        d = OUT / "normative" / name
        _write(d / "decision.json", decision)
        _write(d / "drift_record.json", drift)
        _write(d / "expected.json", expected)

    # 1. Valid ES256: the decision cites the drift record's address and the
    #    committed bytes hash back to it. Signature verifies, citation
    #    resolves.
    valid = _decision(bl, ref, key=es, alg="ES256", nonce="d1").to_dict()
    case("valid_evidence_ref_resolves",
         decision=valid, drift=DRIFT_RECORD,
         expected={"decision_signature_ok": True,
                   "evidence_ref_resolves": True})

    # 2. Valid HS256: the same contract under a symmetric key, so a verifier
    #    with the shared secret reproduces both verdicts.
    valid_hs = _decision(bl, ref, key=HS, alg="HS256", nonce="d2").to_dict()
    case("hs256_valid_resolves",
         decision=valid_hs, drift=DRIFT_RECORD,
         expected={"decision_signature_ok": True,
                   "evidence_ref_resolves": True})

    # 3. Swapped evidence digest: the signed citation is repointed at a
    #    different address after signing. evidenceRef is inside the signed
    #    block, so the signature breaks; the committed drift record no longer
    #    hashes to the stored digest, so the citation does not resolve either.
    swapped = copy.deepcopy(valid)
    swapped["decisionDerived"]["evidenceRef"]["digest"] = "sha256:" + "c" * 64
    case("swapped_evidence_digest",
         decision=swapped, drift=DRIFT_RECORD,
         expected={"decision_signature_ok": False,
                   "evidence_ref_resolves": False,
                   "note": "evidenceRef.digest repointed after signing: the "
                           "citation is inside the signed decisionDerived "
                           "block, so the signature breaks and the committed "
                           "drift record no longer resolves to it"})

    # 4. Stripped evidence ref: the citation is removed after signing. The
    #    signature covered it, so verification fails; with no reference there
    #    is nothing to resolve.
    stripped = copy.deepcopy(valid)
    del stripped["decisionDerived"]["evidenceRef"]
    case("stripped_evidence_ref",
         decision=stripped, drift=DRIFT_RECORD,
         expected={"decision_signature_ok": False,
                   "evidence_ref_resolves": False,
                   "note": "evidenceRef stripped after signing: a verifier "
                           "cannot recover the omitted citation, so the "
                           "signature fails and no reference resolves"})

    # 5. Tampered drift record: the decision is untouched and verifies, but
    #    the committed evidence bytes were altered. The citation no longer
    #    resolves. This is the substitution a valid signature alone cannot
    #    catch: the two verdicts are independent.
    tampered_drift = copy.deepcopy(DRIFT_RECORD)
    tampered_drift["classifiedDelta"]["to"] = ["https://attacker.example.com"]
    case("tampered_drift_record",
         decision=valid, drift=tampered_drift,
         expected={"decision_signature_ok": True,
                   "evidence_ref_resolves": False,
                   "note": "decision unmodified and signature valid, but the "
                           "referenced drift record bytes were altered: the "
                           "content address no longer matches, so the cited "
                           "evidence does not resolve"})

    print(f"wrote vectors under {OUT}")


if __name__ == "__main__":
    main()
