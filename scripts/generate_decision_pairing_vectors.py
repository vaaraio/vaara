#!/usr/bin/env python3
"""Generate the SEP-2828 decision/outcome pairing conformance vectors.

Writes signed fixtures under ``tests/vectors/decision_pairing_v0/``. Each
case holds the attestation, decision record, optional execution receipt,
and an ``expected.json`` of verdicts an independent verifier MUST
reproduce from the committed wire bytes alone. These are the SEP-owned
fixtures requested on modelcontextprotocol/modelcontextprotocol#2828.

Pairing model (SEP-2828, v0.51) is two checks. Check A is the instance
anchor: a decision and a receipt share the same attestation back-link
(attestationDigest constant-time equal AND attestationNonce equal).
Check B is the normative outcome-to-decision binding: the receipt's
``outcomeDerived.decisionDigest`` equals ``sha256`` over the full signed
decision wire bytes. ``records_paired`` requires both, so a receipt that
shares an attestation but commits to a different decision does not pair.

Usage: python scripts/generate_decision_pairing_vectors.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from vaara.attestation._sep2787_canonical import canonical_json
from vaara.attestation.decision import (
    DecisionDerived, decision_digest, emit_decision_record, make_back_link,
)
from vaara.attestation.receipt import (
    BackLink, OutcomeDerived, emit_receipt, make_result_digest,
)
from vaara.attestation.sep2787 import (
    PayloadDerived, PlannerDeclared, ToolCallBinding,
    emit_attestation, make_args_digest,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "vectors" / "decision_pairing_v0"
HS = bytes.fromhex("42" * 32)
IAT = "2026-06-01T10:00:00Z"
RESULT = {"rows": 10, "table": "employees"}
ARGS = {"table": "employees", "limit": 10}
COMMON = dict(iss="issuer://test", sub="agent:reader",
              secret_version="v1", iat=IAT)


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _attestation(nonce="fixed-attestation-nonce-000"):
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="query_table", server_fingerprint="sha256:" + "1" * 64,
        args=make_args_digest(ARGS)),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="show 10 employees"),
        payload_derived=payload, iss="issuer://test", sub="agent:reader",
        secret_version="v1", alg="HS256", signing_material=HS,
        nonce=nonce, iat=IAT)


def _decision(bl, verdict, *, key, alg, nonce):
    return emit_decision_record(
        back_link=bl, decision_derived=DecisionDerived(
            decision=verdict, decided_at=IAT, risk_score="0.21",
            threshold_allow="0.30", threshold_block="0.80",
            policy_id="policy:read-only/3"),
        alg=alg, signing_material=key, nonce=nonce, **COMMON)


def _receipt(bl, status, *, key, alg, nonce, commit=True, dec_digest=None):
    return emit_receipt(
        back_link=bl, outcome_derived=OutcomeDerived(
            status=status, completed_at=IAT,
            result_commitment=make_result_digest(RESULT) if commit else None,
            decision_digest=dec_digest),
        alg=alg, signing_material=key, nonce=nonce, **COMMON)


def _fallback_backlink() -> BackLink:
    # No SEP-2787 attestation: bind to SHA-256 over the JCS-canonical
    # observed request envelope plus a server nonce. The BackLink is an
    # opaque sha256: digest + nonce, so the fallback is what the
    # enforcement point commits to as the digest.
    envelope = {"method": "tools/call", "params": {
        "name": "query_table", "arguments": ARGS,
        "_meta": {"io.modelcontextprotocol/aiInvocation": {"turnId": "turn-7"}}}}
    digest = hashlib.sha256(canonical_json(envelope)).hexdigest()
    return BackLink(attestation_digest="sha256:" + digest,
                    attestation_nonce="server-chosen-nonce-001")


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

    def case(name, *, decision, receipt=None, expected):
        d = OUT / "normative" / name
        _write(d / "attestation.json", att.to_dict())
        _write(d / "decision.json", decision.to_dict())
        if receipt is not None:
            _write(d / "receipt.json", receipt.to_dict())
        _write(d / "expected.json", expected)

    # 1. Valid allow + executed: paired on the same attestation back-link
    #    (Check A) AND the receipt commits to the decision digest (Check B).
    dec1 = _decision(bl, "allow", key=HS, alg="HS256", nonce="d1")
    case("valid_pair_allow_executed",
         decision=dec1,
         receipt=_receipt(bl, "executed", key=HS, alg="HS256", nonce="r1",
                          dec_digest=decision_digest(dec1)),
         expected={"decision_signature_ok": True, "decision_back_link_ok": True,
                   "receipt_signature_ok": True, "receipt_back_link_ok": True,
                   "records_paired": True})

    # 2. Decision-only escalate: valid, no sibling outcome, not an error.
    case("decision_only_escalate",
         decision=_decision(bl, "escalate", key=es, alg="ES256", nonce="d2"),
         expected={"decision_signature_ok": True, "decision_back_link_ok": True,
                   "receipt_present": False, "outcome_required": False})

    # 3. Substituted attestation back-link: receipt binds a different
    #    attestation. Both verify alone; Check A rejects the pair.
    other = _attestation(nonce="other-attestation-nonce-999")
    dec3 = _decision(bl, "allow", key=HS, alg="HS256", nonce="d3")
    case("substituted_attestation_backlink",
         decision=dec3,
         receipt=_receipt(make_back_link(other), "executed", key=HS,
                          alg="HS256", nonce="r3", dec_digest=decision_digest(dec3)),
         expected={"decision_signature_ok": True, "decision_back_link_ok": True,
                   "receipt_signature_ok": True,
                   "receipt_back_link_ok_against_stored_attestation": False,
                   "records_paired": False})

    # 4. Substituted pairing nonce: same digest, mismatched nonce. The
    #    nonce is part of Check A, so the pair is rejected.
    tampered = BackLink(attestation_digest=bl.attestation_digest,
                        attestation_nonce="substituted-nonce")
    dec4 = _decision(bl, "allow", key=HS, alg="HS256", nonce="d4")
    case("substituted_pairing_nonce",
         decision=dec4,
         receipt=_receipt(tampered, "executed", key=HS, alg="HS256", nonce="r4",
                          dec_digest=decision_digest(dec4)),
         expected={"records_paired": False,
                   "note": "same attestationDigest, mismatched attestationNonce"})

    # 5. Equal-decidedAt supersession tie. Resolved in v0.51: latest
    #    decidedAt wins; on a tie, lowest issuerAsserted.nonce wins. Both
    #    share decidedAt, so the winner is the lower nonce, "d5a".
    d5 = OUT / "normative" / "supersession_equal_decidedat_tie"
    _write(d5 / "attestation.json", att.to_dict())
    _write(d5 / "decision_a.json",
           _decision(bl, "block", key=HS, alg="HS256", nonce="d5a").to_dict())
    _write(d5 / "decision_b.json",
           _decision(bl, "allow", key=HS, alg="HS256", nonce="d5b").to_dict())
    _write(d5 / "expected.json", {
        "both_signatures_ok": True, "both_back_links_ok": True, "winner": "d5a",
        "note": "equal decidedAt resolves by lexicographic issuerAsserted.nonce, "
                "lowest wins (d5a < d5b)"})

    # 6. Fallback request-envelope binding, replay/substitution. No
    #    SEP-2787 attestation, so Check A anchors on the request-envelope
    #    digest; Check B still binds the outcome to the decision.
    fb = _fallback_backlink()
    d6 = OUT / "normative" / "fallback_envelope_binding"
    dec6 = _decision(fb, "allow", key=HS, alg="HS256", nonce="d6")
    _write(d6 / "decision.json", dec6.to_dict())
    rec = _receipt(fb, "executed", key=HS, alg="HS256", nonce="r6",
                   dec_digest=decision_digest(dec6))
    _write(d6 / "receipt.json", rec.to_dict())
    replayed = rec.to_dict()
    replayed["backLink"]["attestationDigest"] = "sha256:" + "0" * 64
    _write(d6 / "receipt_replayed.json", replayed)
    _write(d6 / "expected.json", {
        "decision_signature_ok": True, "receipt_signature_ok": True,
        "records_paired": True, "replayed_receipt_signature_ok": False,
        "note": "fallback binding when no SEP-2787 attestation exists; the "
                "BackLink digest is over the JCS-canonical request envelope"})

    # 7. Check B in isolation: the receipt shares the attestation back-link
    #    (Check A passes) but commits to a DIFFERENT decision's digest, so
    #    presenting the substituted decision does not pair. This is the
    #    case instance-binding alone (Check A) cannot catch.
    dec7_bound = _decision(bl, "allow", key=HS, alg="HS256", nonce="d7a")
    dec7_other = _decision(bl, "block", key=HS, alg="HS256", nonce="d7b")
    case("substituted_decision_under_shared_attestation",
         decision=dec7_other,
         receipt=_receipt(bl, "executed", key=HS, alg="HS256", nonce="r7",
                          dec_digest=decision_digest(dec7_bound)),
         expected={"decision_signature_ok": True, "receipt_signature_ok": True,
                   "shared_back_link": True, "records_paired": False,
                   "note": "Check A (shared attestation back-link) passes; "
                           "Check B fails: the receipt's decisionDigest commits "
                           "to a different decision than the one presented"})

    print(f"wrote vectors under {OUT}")


if __name__ == "__main__":
    main()
