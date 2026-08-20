# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The release condition: a receipt gates a payment, not the other way round.

Everything else in the tree runs one direction: a payment gates access and the
settlement evidence lands inside a receipt. These tests pin the inversion. Money
is held, a signed condition names what must be proved, and a Vaara receipt
proving the authorised action happened is what releases it.

The property under test, above all the individual cases: a verifier that proved
nothing must never read as green, and must never read as the same false as a
genuine failure. ``HELD`` because no receipt arrived and ``REFUSED`` because a
receipt was tampered with are different states, and the reason space is
partitioned so they cannot collapse into each other.
"""

from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec, ed25519  # noqa: E402

from vaara.attestation._attest_canonical import canonical_json  # noqa: E402
from vaara.attestation._attest_types import AttestationError  # noqa: E402
from vaara.audit.signer import Ed25519Signer, Ed25519Verifier  # noqa: E402
from vaara.credential import (  # noqa: E402
    Capability,
    GrantBinding,
    GrantScope,
    GrantVerdict,
    emit_grant,
    mint_authorization_receipt,
)
from vaara.settlement.release import (  # noqa: E402
    REASON_STATE,
    SCHEMA,
    HeldValue,
    ReleaseBundle,
    ReleaseReason,
    ReleaseRequirements,
    ReleaseState,
    condition_digest,
    emit_release_condition,
    evaluate,
    receipt_key_fingerprint,
    verify_condition_signature,
)

TENANT = "mcp-tenant-01"
TOOL = "transfer_funds"
ATT_DIGEST = "sha256:" + "a" * 64
ATT_NONCE = "nonce-release-v0-01"
ISSUER = "escrow.example"
RECEIPT_ISS = "gateway.example"
RUNTIME_ARGS = {"amount": 400, "vendor": "acme"}
NOT_AFTER = "2026-09-01T00:00:00Z"
BEFORE = "2026-08-21T12:00:00Z"
AFTER = "2026-09-02T00:00:00Z"


# --- corpus keys, derived from constants so every run is reproducible --------


def _receipt_key():
    return ec.derive_private_key(0x5EC0DE_5EC0DE_5EC0DE_5EC0DE, ec.SECP256R1())


def _other_receipt_key():
    return ec.derive_private_key(0xBADBEEF_BADBEEF_BADBEEF, ec.SECP256R1())


def _condition_key():
    return ed25519.Ed25519PrivateKey.from_private_bytes(bytes(range(32)))


def _public_pem(private_key) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


# --- fixtures ---------------------------------------------------------------


def _grant():
    return emit_grant(
        scope=GrantScope(tool_name=TOOL, args_commitment="", tenant_id=TENANT),
        binding=GrantBinding(
            attestation_digest=ATT_DIGEST, attestation_nonce=ATT_NONCE
        ),
        iss=RECEIPT_ISS,
        sub=TENANT,
        secret_version="corpus-key-v0",
        alg="ES256",
        signing_material=_receipt_key(),
        exp_seconds=300,
        capabilities=(Capability("amount", "le", "500"),),
        iat="2026-08-20T09:00:00Z",
        nonce="grant-nonce-release-v0",
    )


def _receipt(*, decision_ok=True, args=None, key=None, grant=None):
    """Mint a real authorization receipt: envelope plus its evidence record."""
    minted = mint_authorization_receipt(
        credential=grant or _grant(),
        runtime_args=RUNTIME_ARGS if args is None else args,
        verdict=GrantVerdict(decision_ok, "ok" if decision_ok else "capability_exceeded"),
        iss=RECEIPT_ISS,
        sub=TENANT,
        secret_version="corpus-key-v0",
        alg="ES256",
        signing_material=key or _receipt_key(),
        decided_at="2026-08-20T09:00:05Z",
        nonce="receipt-nonce-release-v0",
    )
    return minted.record.to_dict(), minted.evidence


def _requirements(evidence, *, key=None):
    return ReleaseRequirements(
        action_digest=evidence["argsCommitment"],
        grant_fingerprint=evidence["grantFingerprint"],
        receipt_issuer=RECEIPT_ISS,
        receipt_key_fingerprint=receipt_key_fingerprint(
            _public_pem(key or _receipt_key())
        ),
    )


def _condition(requires, *, not_after=NOT_AFTER):
    return emit_release_condition(
        signer=Ed25519Signer(_condition_key()),
        issuer=ISSUER,
        condition_id="release-condition-v0-fixture",
        held=HeldValue(
            amount="1000000", asset="USDC", network="base", payee="0x" + "1" * 40
        ),
        requires=requires,
        not_after=not_after,
    )


def _bundle(**kwargs):
    defaults = {
        "now": BEFORE,
        "condition_verifier": Ed25519Verifier(
            _condition_key().public_key().public_bytes_raw()
        ),
        "receipt_public_key_pem": _public_pem(_receipt_key()),
    }
    defaults.update(kwargs)
    return ReleaseBundle(**defaults)


# --- the reason space is a partition ----------------------------------------


def test_every_reason_maps_to_exactly_one_state():
    assert set(REASON_STATE) == set(ReleaseReason)
    assert set(REASON_STATE.values()) == set(ReleaseState)


def test_absence_and_refusal_never_share_a_state():
    # The SCITT property, made structural: nothing-proved and proved-wrong are
    # partitioned, so no third boolean can collapse them.
    assert REASON_STATE[ReleaseReason.RECEIPT_ABSENT] is ReleaseState.HELD
    assert (
        REASON_STATE[ReleaseReason.RECEIPT_SIGNATURE_INVALID] is ReleaseState.REFUSED
    )
    assert REASON_STATE[ReleaseReason.CONDITION_EXPIRED] is ReleaseState.EXPIRED
    held = {r for r, s in REASON_STATE.items() if s is ReleaseState.HELD}
    refused = {r for r, s in REASON_STATE.items() if s is ReleaseState.REFUSED}
    assert not held & refused


def test_only_one_reason_releases():
    releasing = [r for r, s in REASON_STATE.items() if s is ReleaseState.RELEASED]
    assert releasing == [ReleaseReason.RECEIPT_MATCHES]


# --- the condition document -------------------------------------------------


def test_condition_is_signed_content_addressed_and_verifies():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    assert condition["schema"] == SCHEMA
    assert condition["signature"]
    verifier = Ed25519Verifier(_condition_key().public_key().public_bytes_raw())
    assert verify_condition_signature(condition, verifier=verifier)
    digest = condition_digest(condition)
    assert digest == "sha256:" + hashlib.sha256(canonical_json(condition)).hexdigest()


def test_condition_signature_does_not_cover_itself_but_covers_the_requirements():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    verifier = Ed25519Verifier(_condition_key().public_key().public_bytes_raw())
    tampered = json.loads(json.dumps(condition))
    tampered["requires"]["actionDigest"] = "sha256:" + "0" * 64
    assert not verify_condition_signature(tampered, verifier=verifier)


def test_condition_rejects_floats_in_the_held_value():
    # Floats are the standard source of cross-stack signature drift; amounts are
    # decimal strings everywhere else in the tree and here too.
    with pytest.raises(AttestationError):
        HeldValue(amount=1.5, asset="USDC", network="base", payee="0x1")  # type: ignore[arg-type]


def test_requirements_reject_a_digest_that_is_not_one():
    with pytest.raises(AttestationError):
        ReleaseRequirements(
            action_digest="not-a-digest",
            grant_fingerprint="sha256:" + "0" * 64,
            receipt_issuer=RECEIPT_ISS,
            receipt_key_fingerprint="sha256:" + "0" * 64,
        )


# --- the six vectors, as unit tests -----------------------------------------


def test_matching_receipt_releases():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    decision = evaluate(condition, _bundle(receipt=receipt, evidence=evidence))
    assert decision.state is ReleaseState.RELEASED
    assert decision.reason is ReleaseReason.RECEIPT_MATCHES
    assert decision.released is True
    assert decision.condition_digest == condition_digest(condition)
    assert decision.receipt_digest == "sha256:" + hashlib.sha256(
        canonical_json(receipt)
    ).hexdigest()


def test_absent_receipt_holds_and_names_absence():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    decision = evaluate(condition, _bundle())
    assert decision.state is ReleaseState.HELD
    assert decision.reason is ReleaseReason.RECEIPT_ABSENT
    assert decision.receipt_digest is None


def test_authorization_mismatch_holds():
    # A sound receipt, but it proves a different authorization than the one the
    # condition names. Nothing is wrong with the evidence; it is insufficient.
    receipt, evidence = _receipt()
    other_grant = emit_grant(
        scope=GrantScope(tool_name=TOOL, args_commitment="", tenant_id=TENANT),
        binding=GrantBinding(
            attestation_digest="sha256:" + "b" * 64, attestation_nonce="other"
        ),
        iss=RECEIPT_ISS,
        sub=TENANT,
        secret_version="corpus-key-v0",
        alg="ES256",
        signing_material=_receipt_key(),
        exp_seconds=300,
        capabilities=(Capability("amount", "le", "500"),),
        iat="2026-08-20T09:00:00Z",
        nonce="other-grant-nonce",
    )
    other_receipt, other_evidence = _receipt(grant=other_grant)
    condition = _condition(_requirements(evidence))
    decision = evaluate(
        condition, _bundle(receipt=other_receipt, evidence=other_evidence)
    )
    assert decision.state is ReleaseState.HELD
    assert decision.reason is ReleaseReason.AUTHORIZATION_MISMATCH


def test_receipt_for_a_different_action_holds():
    receipt, evidence = _receipt()
    other_receipt, other_evidence = _receipt(args={"amount": 401, "vendor": "acme"})
    condition = _condition(_requirements(evidence))
    decision = evaluate(
        condition, _bundle(receipt=other_receipt, evidence=other_evidence)
    )
    assert decision.state is ReleaseState.HELD
    assert decision.reason is ReleaseReason.ACTION_DIGEST_MISMATCH


def test_expired_condition_expires_and_does_not_read_as_refused():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    decision = evaluate(
        condition, _bundle(now=AFTER, receipt=receipt, evidence=evidence)
    )
    assert decision.state is ReleaseState.EXPIRED
    assert decision.reason is ReleaseReason.CONDITION_EXPIRED
    assert decision.state is not ReleaseState.REFUSED
    assert decision.state is not ReleaseState.RELEASED


def test_expiry_is_inclusive_of_the_named_instant():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    decision = evaluate(
        condition, _bundle(now=NOT_AFTER, receipt=receipt, evidence=evidence)
    )
    assert decision.state is ReleaseState.RELEASED


def test_tampered_receipt_refuses_and_names_the_broken_signature():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    tampered = json.loads(json.dumps(receipt))
    tampered["decisionDerived"]["decidedAt"] = "2026-08-20T09:00:06Z"
    decision = evaluate(condition, _bundle(receipt=tampered, evidence=evidence))
    assert decision.state is ReleaseState.REFUSED
    assert decision.reason is ReleaseReason.RECEIPT_SIGNATURE_INVALID


# --- the rest of the closed set ---------------------------------------------


def test_receipt_under_an_unpinned_key_refuses():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    other_receipt, other_evidence = _receipt(key=_other_receipt_key())
    decision = evaluate(
        condition,
        _bundle(
            receipt=other_receipt,
            evidence=other_evidence,
            receipt_public_key_pem=_public_pem(_other_receipt_key()),
        ),
    )
    assert decision.state is ReleaseState.REFUSED
    assert decision.reason is ReleaseReason.RECEIPT_KEY_UNTRUSTED


def test_a_receipt_that_proves_a_refusal_does_not_release():
    # The deny receipt is real evidence and it verifies. It proves the action was
    # blocked, which is exactly not what the condition requires.
    receipt, evidence = _receipt()
    denied_receipt, denied_evidence = _receipt(decision_ok=False)
    condition = _condition(_requirements(evidence))
    decision = evaluate(
        condition, _bundle(receipt=denied_receipt, evidence=denied_evidence)
    )
    assert decision.state is ReleaseState.HELD
    assert decision.reason is ReleaseReason.DECISION_NOT_ACCEPTED


def test_evidence_that_does_not_resolve_to_the_signed_digest_refuses():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    swapped = json.loads(json.dumps(evidence))
    swapped["argsCommitment"] = "sha256:" + "c" * 64
    decision = evaluate(condition, _bundle(receipt=receipt, evidence=swapped))
    assert decision.state is ReleaseState.REFUSED
    assert decision.reason is ReleaseReason.EVIDENCE_DIGEST_MISMATCH


def test_receipt_without_its_evidence_refuses():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    decision = evaluate(condition, _bundle(receipt=receipt))
    assert decision.state is ReleaseState.REFUSED
    assert decision.reason is ReleaseReason.EVIDENCE_DIGEST_MISMATCH


def test_receipt_from_an_unaccepted_issuer_holds():
    receipt, evidence = _receipt()
    requires = ReleaseRequirements(
        action_digest=evidence["argsCommitment"],
        grant_fingerprint=evidence["grantFingerprint"],
        receipt_issuer="someone.else.example",
        receipt_key_fingerprint=receipt_key_fingerprint(_public_pem(_receipt_key())),
    )
    decision = evaluate(
        _condition(requires), _bundle(receipt=receipt, evidence=evidence)
    )
    assert decision.state is ReleaseState.HELD
    assert decision.reason is ReleaseReason.ISSUER_NOT_ACCEPTED


def test_condition_without_a_verifier_refuses_rather_than_holding():
    # An unverifiable condition is not "waiting for evidence"; it is an artifact
    # nobody can stand behind, and it must never sit in the same state as a
    # condition that is simply still open.
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    decision = evaluate(
        condition,
        _bundle(receipt=receipt, evidence=evidence, condition_verifier=None),
    )
    assert decision.state is ReleaseState.REFUSED
    assert decision.reason is ReleaseReason.CONDITION_KEY_ABSENT


def test_forged_condition_refuses():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    forged = json.loads(json.dumps(condition))
    forged["holds"]["amount"] = "9000000"
    decision = evaluate(forged, _bundle(receipt=receipt, evidence=evidence))
    assert decision.state is ReleaseState.REFUSED
    assert decision.reason is ReleaseReason.CONDITION_SIGNATURE_INVALID


def test_malformed_condition_refuses():
    decision = evaluate({"schema": "something.else/v0"}, _bundle())
    assert decision.state is ReleaseState.REFUSED
    assert decision.reason is ReleaseReason.CONDITION_MALFORMED


def test_malformed_receipt_refuses():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    decision = evaluate(
        condition, _bundle(receipt={"not": "a receipt"}, evidence=evidence)
    )
    assert decision.state is ReleaseState.REFUSED
    assert decision.reason is ReleaseReason.RECEIPT_MALFORMED


def test_receipt_presented_without_its_key_refuses():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    decision = evaluate(
        condition,
        _bundle(receipt=receipt, evidence=evidence, receipt_public_key_pem=None),
    )
    assert decision.state is ReleaseState.REFUSED
    assert decision.reason is ReleaseReason.RECEIPT_KEY_ABSENT


# --- ordering: an expired window must not swallow a tampering finding -------


def test_tampering_outranks_expiry():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    tampered = json.loads(json.dumps(receipt))
    tampered["decisionDerived"]["decidedAt"] = "2026-08-20T09:00:06Z"
    decision = evaluate(
        condition, _bundle(now=AFTER, receipt=tampered, evidence=evidence)
    )
    assert decision.state is ReleaseState.REFUSED
    assert decision.reason is ReleaseReason.RECEIPT_SIGNATURE_INVALID


def test_expiry_outranks_insufficiency():
    # Past the window, an insufficient receipt reports the closed window rather
    # than a mismatch: the window is why the money is not moving.
    receipt, evidence = _receipt()
    other_receipt, other_evidence = _receipt(args={"amount": 401, "vendor": "acme"})
    condition = _condition(_requirements(evidence))
    decision = evaluate(
        condition, _bundle(now=AFTER, receipt=other_receipt, evidence=other_evidence)
    )
    assert decision.state is ReleaseState.EXPIRED
    assert decision.reason is ReleaseReason.CONDITION_EXPIRED


def test_absent_receipt_past_the_window_expires_rather_than_holding():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    decision = evaluate(condition, _bundle(now=AFTER))
    assert decision.state is ReleaseState.EXPIRED
    assert decision.reason is ReleaseReason.CONDITION_EXPIRED


# --- the decision is a portable, recomputable artifact ----------------------


def test_decision_serialises_to_the_wire_shape():
    receipt, evidence = _receipt()
    condition = _condition(_requirements(evidence))
    decision = evaluate(condition, _bundle(receipt=receipt, evidence=evidence))
    wire = decision.to_dict()
    assert wire["state"] == "released"
    assert wire["reason"] == "receipt_matches"
    assert wire["conditionDigest"] == condition_digest(condition)
    assert wire["evaluatedAt"] == BEFORE
    # Round-trips through JCS, so a decision is itself content-addressable.
    assert json.loads(canonical_json(wire).decode()) == wire
