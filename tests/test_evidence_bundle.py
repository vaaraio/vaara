"""One-call evidence-bundle verification: the 0.6 trust-plane capstone.

Covers `verify_evidence_bundle` over `EvidenceBundle`: the six lenses run
only when their evidence is present, the identity-resolved keyid threads
into the revocation lens, and `ok` is fail-closed on authenticity (an
unauthenticated receipt that is merely in a log is not `ok`). Also drives
the `evidence_bundle_v0` conformance vectors through both Vaara and the
standalone Vaara-free checker.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation import InProcessTransparencyLog  # noqa: E402
from vaara.attestation.receipt import (  # noqa: E402
    EvidenceBundle,
    OutcomeDerived,
    RevocationEntry,
    RevocationRegistry,
    emit_receipt,
    make_back_link,
    parse_receipt,
    receipt_leaf_bytes,
    verify_evidence_bundle,
)
from vaara.attestation.tool_call_attestation import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
    parse_attestation,
)
from vaara.attestation.transparency_log import (  # noqa: E402
    ConsistencyProof,
    InclusionProof,
)

VECTORS = Path(__file__).resolve().parent / "vectors" / "evidence_bundle_v0"
DID = "did:web:agents.example.com:billing"
KEYID = DID + "#key-2026"
IAT = "2026-05-29T10:00:00Z"
BEFORE = "2026-05-29T09:30:00Z"


def _b64u(n: int) -> str:
    return base64.urlsafe_b64encode(n.to_bytes(32, "big")).rstrip(b"=").decode()


def _jwk(public_key) -> dict:
    nums = public_key.public_numbers()
    return {"kty": "EC", "crv": "P-256", "x": _b64u(nums.x), "y": _b64u(nums.y)}


def _public_key_from_jwk(jwk: dict):
    def to_int(v: str) -> int:
        pad = "=" * (-len(v) % 4)
        return int.from_bytes(base64.urlsafe_b64decode(v + pad), "big")

    return ec.EllipticCurvePublicNumbers(
        to_int(jwk["x"]), to_int(jwk["y"]), ec.SECP256R1()
    ).public_key()


def _attestation():
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="settle invoice"),
        payload_derived=PayloadDerived(tool_calls=(ToolCallBinding(
            name="charge_card", server_fingerprint="sha256:" + "1" * 64,
            args=make_args_digest({"amount": 4200}),
        ),)),
        iss="issuer://test", sub="agent:billing", secret_version="v1",
        alg="HS256", signing_material=b"\x42" * 32,
        nonce="att-nonce-fixed-0001", iat="2026-05-29T09:59:59Z",
    )


def _receipt(key):
    return emit_receipt(
        back_link=make_back_link(_attestation()),
        outcome_derived=OutcomeDerived(status="executed", completed_at=IAT),
        iss=DID, sub=DID, secret_version="v1", alg="ES256",
        signing_material=key, nonce="rcpt-nonce-fixed-0001", iat=IAT,
    )


def _doc(key):
    return {
        "id": DID,
        "verificationMethod": [{
            "id": KEYID, "type": "JsonWebKey2020", "controller": DID,
            "publicKeyJwk": _jwk(key.public_key()),
        }],
    }


# ── Unit behaviour ───────────────────────────────────────────────────────────


def test_full_bundle_ok_and_keyid_threaded():
    key = ec.generate_private_key(ec.SECP256R1())
    receipt = _receipt(key)
    log = InProcessTransparencyLog()
    log.append(b"a")
    entry = log.append(receipt_leaf_bytes(receipt))
    verdict = verify_evidence_bundle(EvidenceBundle(
        receipt=receipt, did_document=_doc(key), attestation=_attestation(),
        inclusion=log.inclusion_proof(entry.log_index), log_root=log.root_hash,
        registry=RevocationRegistry([]),
    ))
    assert verdict.ok is True
    assert verdict.authenticity_established is True
    assert verdict.keyid == KEYID
    assert verdict.lens("consistency").applicable is False
    assert verdict.lens("revocation").applicable is True


def test_authenticity_fail_closed_for_record_only_in_log():
    # Inclusion and a clean registry pass, but nothing verified the signature.
    key = ec.generate_private_key(ec.SECP256R1())
    receipt = _receipt(key)
    log = InProcessTransparencyLog()
    entry = log.append(receipt_leaf_bytes(receipt))
    verdict = verify_evidence_bundle(EvidenceBundle(
        receipt=receipt, inclusion=log.inclusion_proof(entry.log_index),
        log_root=log.root_hash, registry=RevocationRegistry([]),
    ))
    assert verdict.lens("inclusion").ok is True
    assert verdict.lens("revocation").ok is True
    assert verdict.authenticity_established is False
    assert verdict.ok is False


def test_revocation_lens_uses_identity_resolved_keyid():
    # The key-scope revocation only bites because identity resolves the keyid.
    key = ec.generate_private_key(ec.SECP256R1())
    receipt = _receipt(key)
    registry = RevocationRegistry([RevocationEntry("key", KEYID, BEFORE)])
    verdict = verify_evidence_bundle(EvidenceBundle(
        receipt=receipt, did_document=_doc(key), registry=registry,
    ))
    assert verdict.lens("identity").ok is True
    assert verdict.lens("revocation").ok is False
    assert verdict.ok is False
    # Without the DID document the keyid is unknown, so the key-scope entry
    # cannot match and revocation passes.
    sig_only = verify_evidence_bundle(EvidenceBundle(
        receipt=receipt, verifying_material=key.public_key(), registry=registry,
    ))
    assert sig_only.lens("revocation").ok is True
    assert sig_only.ok is True


def test_signature_lens_alone_establishes_authenticity():
    key = ec.generate_private_key(ec.SECP256R1())
    verdict = verify_evidence_bundle(EvidenceBundle(
        receipt=_receipt(key), verifying_material=key.public_key(),
    ))
    assert verdict.ok is True
    assert verdict.lens("identity").applicable is False
    assert verdict.lens("signature").ok is True


def test_wrong_key_fails_authenticity():
    key = ec.generate_private_key(ec.SECP256R1())
    other = ec.generate_private_key(ec.SECP256R1())
    verdict = verify_evidence_bundle(EvidenceBundle(
        receipt=_receipt(key), verifying_material=other.public_key(),
    ))
    assert verdict.lens("signature").ok is False
    assert verdict.ok is False


def test_lens_lookup_rejects_unknown_name():
    key = ec.generate_private_key(ec.SECP256R1())
    verdict = verify_evidence_bundle(EvidenceBundle(receipt=_receipt(key)))
    with pytest.raises(KeyError):
        verdict.lens("nonsense")


# ── Conformance vectors ──────────────────────────────────────────────────────


def _bundle_from_json(b: dict) -> EvidenceBundle:
    inc = b.get("inclusion")
    con = b.get("consistency")
    jwk = b.get("verifying_jwk")
    registry = b.get("registry")
    return EvidenceBundle(
        receipt=parse_receipt(b["receipt"]),
        did_document=b.get("did_document"),
        expected_keyid=b.get("expected_keyid"),
        verifying_material=_public_key_from_jwk(jwk) if jwk else None,
        attestation=parse_attestation(b["attestation"]) if b.get("attestation") else None,
        inclusion=InclusionProof(
            log_index=inc["log_index"], tree_size=inc["tree_size"],
            siblings=tuple(bytes.fromhex(h) for h in inc["siblings_hex"]),
        ) if inc else None,
        log_root=bytes.fromhex(inc["root_hex"]) if inc else None,
        consistency=ConsistencyProof(
            first_size=con["first_size"], second_size=con["second_size"],
            hashes=tuple(bytes.fromhex(h) for h in con["hashes_hex"]),
        ) if con else None,
        consistency_first_root=bytes.fromhex(con["first_root_hex"]) if con else None,
        consistency_second_root=bytes.fromhex(con["second_root_hex"]) if con else None,
        registry=RevocationRegistry.from_dict(registry) if registry is not None else None,
    )


def _cases():
    return json.loads((VECTORS / "cases.json").read_text())["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_vaara_reproduces_vector_verdict(case):
    want = case["expected"]
    verdict = verify_evidence_bundle(_bundle_from_json(case["bundle"]))
    assert verdict.ok is want["ok"]
    assert verdict.authenticity_established is want["authenticity_established"]
    got = {r.lens: {"applicable": r.applicable, "ok": r.ok} for r in verdict.lenses}
    assert got == want["lenses"]


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
