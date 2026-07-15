#!/usr/bin/env python3
"""Generate the evidence_bundle_v0 conformance vectors.

One ES256 receipt and a range of evidence bundles run through the single
``verify_evidence_bundle`` entrypoint, the 0.6 trust-plane capstone. Each
case commits the bundle and the verdict the reference implementation
produced: the overall ``ok``, whether authenticity was established, and per
lens whether it applied and passed. The independent checker reproduces the
same verdict over the committed bundle without importing Vaara.

ECDSA signatures are randomized; re-running overwrites the cases with fresh
but equivalent vectors. The committed JSON is the vector. Run from the repo
root: ``python tests/vectors/evidence_bundle_v0/_generate.py``.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ec

from vaara.attestation import InProcessTransparencyLog
from vaara.attestation.receipt import (
    BundleVerdict,
    EvidenceBundle,
    OutcomeDerived,
    RevocationEntry,
    RevocationRegistry,
    emit_receipt,
    make_back_link,
    receipt_leaf_bytes,
    verify_evidence_bundle,
)
from vaara.attestation.tool_call_attestation import (
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)

HERE = Path(__file__).resolve().parent
DID = "did:web:agents.example.com:billing"
KEYID = DID + "#key-2026"
SCALAR = 0x1F2E3D4C5B6A79887766554433221100FFEEDDCCBBAA99887766554433221101
WRONG_SCALAR = 0x0BADC0DE0BADC0DE0BADC0DE0BADC0DE0BADC0DE0BADC0DE0BADC0DE0BADC0DE1
IAT = "2026-05-29T10:00:00Z"
REVOKED_BEFORE = "2026-05-29T09:30:00Z"


def _b64u(value: int) -> str:
    return base64.urlsafe_b64encode(value.to_bytes(32, "big")).rstrip(b"=").decode()


def _pub_jwk(public_key: ec.EllipticCurvePublicKey) -> dict[str, str]:
    numbers = public_key.public_numbers()
    return {"kty": "EC", "crv": "P-256", "x": _b64u(numbers.x), "y": _b64u(numbers.y)}


def _did_document(public_key: ec.EllipticCurvePublicKey) -> dict[str, object]:
    return {
        "id": DID,
        "verificationMethod": [
            {
                "id": KEYID,
                "type": "JsonWebKey2020",
                "controller": DID,
                "publicKeyJwk": _pub_jwk(public_key),
            }
        ],
    }


def _attestation(intent: str, nonce: str, amount: int):
    payload = PayloadDerived(
        tool_calls=(
            ToolCallBinding(
                name="charge_card",
                server_fingerprint="sha256:" + "1" * 64,
                args=make_args_digest({"amount": amount}),
            ),
        )
    )
    return emit_attestation(
        planner_declared=PlannerDeclared(intent=intent),
        payload_derived=payload,
        iss="issuer://test",
        sub="agent:billing",
        secret_version="v1",
        alg="HS256",
        signing_material=b"\x42" * 32,
        nonce=nonce,
        iat="2026-05-29T09:59:59Z",
    )


def _inclusion_json(proof, root: bytes) -> dict[str, object]:
    return {
        "log_index": proof.log_index,
        "tree_size": proof.tree_size,
        "siblings_hex": [s.hex() for s in proof.siblings],
        "root_hex": root.hex(),
    }


def _consistency_json(proof, first_root: bytes, second_root: bytes) -> dict[str, object]:
    return {
        "first_size": proof.first_size,
        "second_size": proof.second_size,
        "hashes_hex": [h.hex() for h in proof.hashes],
        "first_root_hex": first_root.hex(),
        "second_root_hex": second_root.hex(),
    }


def _verdict_json(verdict: BundleVerdict) -> dict[str, object]:
    return {
        "ok": verdict.ok,
        "authenticity_established": verdict.authenticity_established,
        "lenses": {
            r.lens: {"applicable": r.applicable, "ok": r.ok} for r in verdict.lenses
        },
    }


def main() -> None:
    key = ec.derive_private_key(SCALAR, ec.SECP256R1())
    pub = key.public_key()
    wrong_pub = ec.derive_private_key(WRONG_SCALAR, ec.SECP256R1()).public_key()
    doc = _did_document(pub)

    att = _attestation("settle invoice", "att-nonce-fixed-0001", 4200)
    receipt = emit_receipt(
        back_link=make_back_link(att),
        outcome_derived=OutcomeDerived(status="executed", completed_at=IAT),
        iss=DID,
        sub=DID,
        secret_version="v1",
        alg="ES256",
        signing_material=key,
        nonce="rcpt-nonce-fixed-0001",
        iat=IAT,
    )
    receipt_dict = receipt.to_dict()
    other_att = _attestation("buy something else", "att-nonce-other-0002", 99)

    log = InProcessTransparencyLog()
    log.append(b"filler-0")
    log.append(b"filler-1")
    entry = log.append(receipt_leaf_bytes(receipt))
    log.append(b"filler-3")
    first_size = log.tree_size
    first_root = log.root_at(first_size)
    inc = log.inclusion_proof(entry.log_index)
    log.append(b"filler-4")
    log.append(b"filler-5")
    second_size = log.tree_size
    second_root = log.root_at(second_size)
    con = log.consistency_proof(first_size, second_size)
    bad_root = bytes(b ^ 0xFF for b in first_root)
    bad_second_root = bytes(b ^ 0xFF for b in second_root)
    clean = RevocationRegistry([])
    revoked = RevocationRegistry([RevocationEntry("key", KEYID, REVOKED_BEFORE)])

    cases: list[dict[str, object]] = []

    def add(name: str, bundle: EvidenceBundle, bundle_json: dict[str, object]) -> None:
        verdict = verify_evidence_bundle(bundle)
        cases.append(
            {"name": name, "bundle": bundle_json, "expected": _verdict_json(verdict)}
        )

    # 1. Full clean bundle: every lens applies and passes.
    add(
        "all_lenses_pass",
        EvidenceBundle(
            receipt=receipt, did_document=doc, expected_keyid=KEYID, attestation=att,
            inclusion=inc, log_root=first_root, consistency=con,
            consistency_first_root=first_root, consistency_second_root=second_root,
            registry=clean,
        ),
        {
            "receipt": receipt_dict, "did_document": doc, "expected_keyid": KEYID,
            "attestation": att.to_dict(), "inclusion": _inclusion_json(inc, first_root),
            "consistency": _consistency_json(con, first_root, second_root),
            "registry": clean.to_dict(),
        },
    )

    # 2. Same evidence, but the signing key was revoked before issuance.
    add(
        "revoked_in_time",
        EvidenceBundle(
            receipt=receipt, did_document=doc, expected_keyid=KEYID, attestation=att,
            inclusion=inc, log_root=first_root, registry=revoked,
        ),
        {
            "receipt": receipt_dict, "did_document": doc, "expected_keyid": KEYID,
            "attestation": att.to_dict(), "inclusion": _inclusion_json(inc, first_root),
            "registry": revoked.to_dict(),
        },
    )

    # 3. Inclusion proof against a tampered log root.
    add(
        "tampered_inclusion",
        EvidenceBundle(
            receipt=receipt, did_document=doc, attestation=att,
            inclusion=inc, log_root=bad_root, registry=clean,
        ),
        {
            "receipt": receipt_dict, "did_document": doc,
            "attestation": att.to_dict(), "inclusion": _inclusion_json(inc, bad_root),
            "registry": clean.to_dict(),
        },
    )

    # 4. Consistency proof against a forked second tree head.
    add(
        "forked_consistency",
        EvidenceBundle(
            receipt=receipt, did_document=doc, consistency=con,
            consistency_first_root=first_root, consistency_second_root=bad_second_root,
        ),
        {
            "receipt": receipt_dict, "did_document": doc,
            "consistency": _consistency_json(con, first_root, bad_second_root),
        },
    )

    # 5. Back-link to an attestation the receipt does not answer.
    add(
        "broken_back_link",
        EvidenceBundle(receipt=receipt, did_document=doc, attestation=other_att),
        {
            "receipt": receipt_dict, "did_document": doc,
            "attestation": other_att.to_dict(),
        },
    )

    # 6. Signature lens only, no DID document: authenticity via supplied key.
    add(
        "signature_only",
        EvidenceBundle(receipt=receipt, verifying_material=pub),
        {"receipt": receipt_dict, "verifying_jwk": _pub_jwk(pub)},
    )

    # 7. Included and not revoked, but the signature is never checked.
    add(
        "unauthenticated_in_log",
        EvidenceBundle(
            receipt=receipt, inclusion=inc, log_root=first_root, registry=clean
        ),
        {
            "receipt": receipt_dict, "inclusion": _inclusion_json(inc, first_root),
            "registry": clean.to_dict(),
        },
    )

    # 8. Signature lens with the wrong public key.
    add(
        "wrong_signature_key",
        EvidenceBundle(receipt=receipt, verifying_material=wrong_pub),
        {"receipt": receipt_dict, "verifying_jwk": _pub_jwk(wrong_pub)},
    )

    expected = {c["name"]: c["expected"] for c in cases}
    (HERE / "cases.json").write_text(
        json.dumps({"did": DID, "keyid": KEYID, "cases": cases}, indent=2, sort_keys=True)
        + "\n"
    )
    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {len(cases)} cases to cases.json, expected.json")


if __name__ == "__main__":
    main()
