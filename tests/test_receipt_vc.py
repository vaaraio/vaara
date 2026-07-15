"""W3C VC opt-in serialization for execution receipts.

The VC is a lossless view of the native receipt: round-trip identity holds,
and trust always routes through the unchanged receipt-signature verifier.
See ``docs/design/w3c-vc-receipt-spec.md``.
"""

from __future__ import annotations

import importlib.util

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from cryptography.hazmat.primitives.asymmetric import ec, rsa  # noqa: E402

from vaara.attestation.receipt import (  # noqa: E402
    VAARA_RECEIPT_CONTEXT_URL,
    OutcomeDerived,
    emit_receipt,
    load_receipt_context,
    make_back_link,
    make_result_digest,
    receipt_from_vc,
    receipt_to_vc,
    verify_receipt_signature,
)
from vaara.attestation.tool_call_attestation import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)

HS_SECRET = b"\x42" * 32
RESULT = {"deleted": True, "path": "/archive/2024-Q3.md"}


def _attestation():
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="delete_file",
        server_fingerprint="sha256:" + "1" * 64,
        args=make_args_digest({"path": "/archive/2024-Q3.md"}),
    ),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="archive obsolete report"),
        payload_derived=payload,
        iss="issuer://test",
        sub="agent:archiver",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )


def _emit(commit_result=True, **overrides):
    commitment = make_result_digest(RESULT) if commit_result else None
    kwargs = dict(
        back_link=make_back_link(_attestation()),
        outcome_derived=OutcomeDerived(
            status="executed",
            completed_at="2026-05-29T10:00:00Z",
            result_commitment=commitment,
        ),
        iss="issuer://test",
        sub="agent:archiver",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )
    kwargs.update(overrides)
    return emit_receipt(**kwargs)


# ── Round-trip identity ─────────────────────────────────────────────────────

@pytest.mark.parametrize("commit_result", [True, False])
def test_hs256_round_trip_identity(commit_result):
    r = _emit(commit_result=commit_result)
    assert receipt_from_vc(receipt_to_vc(r)) == r


def test_es256_round_trip_identity():
    priv = ec.generate_private_key(ec.SECP256R1())
    r = _emit(alg="ES256", signing_material=priv)
    assert receipt_from_vc(receipt_to_vc(r)) == r


def test_rs256_round_trip_identity():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    r = _emit(alg="RS256", signing_material=priv)
    assert receipt_from_vc(receipt_to_vc(r)) == r


# ── Signature parity: trust routes through the unchanged verifier ───────────

def test_unwrapped_vc_verifies_like_native():
    r = _emit()
    assert verify_receipt_signature(r, verifying_material=HS_SECRET) is True
    recovered = receipt_from_vc(receipt_to_vc(r))
    assert verify_receipt_signature(recovered, verifying_material=HS_SECRET) is True


def test_tampered_credential_subject_fails_after_unwrap():
    r = _emit()
    vc = receipt_to_vc(r)
    vc["credentialSubject"]["outcomeDerived"]["status"] = "refused"
    recovered = receipt_from_vc(vc)
    assert verify_receipt_signature(recovered, verifying_material=HS_SECRET) is False


def test_tampered_proof_value_fails_after_unwrap():
    r = _emit()
    vc = receipt_to_vc(r)
    pv = vc["proof"]["proofValue"]
    vc["proof"]["proofValue"] = ("0" if pv[0] != "0" else "1") + pv[1:]
    recovered = receipt_from_vc(vc)
    assert verify_receipt_signature(recovered, verifying_material=HS_SECRET) is False


# ── Wire shape + offline ────────────────────────────────────────────────────

def test_vcdm_required_fields_present():
    vc = receipt_to_vc(_emit())
    for field in ("@context", "type", "issuer", "validFrom", "credentialSubject", "proof"):
        assert field in vc, field
    assert vc["@context"][0] == "https://www.w3.org/ns/credentials/v2"
    assert vc["@context"][1] == VAARA_RECEIPT_CONTEXT_URL
    assert vc["type"] == ["VerifiableCredential", "VaaraExecutionReceipt"]
    assert vc["proof"]["type"] == "VaaraSep2787DetachedSignature2026"
    assert vc["proof"]["cryptosuite"] == "jcs-hs256"
    assert "signature" not in vc["credentialSubject"]


def test_issuer_and_validfrom_from_receipt():
    r = _emit()
    vc = receipt_to_vc(r)
    assert vc["issuer"] == r.receipt_asserted.iss
    assert vc["validFrom"] == r.receipt_asserted.iat


def test_context_resolves_offline():
    ctx = load_receipt_context()
    assert "@context" in ctx
    assert ctx["@context"]["VaaraExecutionReceipt"].startswith(VAARA_RECEIPT_CONTEXT_URL)


def test_from_vc_rejects_malformed():
    from vaara.attestation._attest_types import AttestationError
    with pytest.raises(AttestationError):
        receipt_from_vc({"credentialSubject": {}})  # no proof
    with pytest.raises(AttestationError):
        receipt_from_vc({"proof": {"proofValue": "x"}})  # no subject
