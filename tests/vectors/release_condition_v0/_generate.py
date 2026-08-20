"""Regenerate the release_condition_v0 conformance vectors.

Eight cases pin the inversion: a Vaara receipt gates a payment, instead of a
payment gating access. Money is held against a signed release condition, and the
question is whether what has been presented releases it.

pos_matching_receipt      the authorised action is proved  -> released
neg_absent_receipt        nothing presented yet            -> held
neg_authorization_mismatch a sound receipt, another grant  -> held
neg_other_action          a sound receipt, another action  -> held
neg_blocked_decision      a sound receipt proving a REFUSAL-> held
neg_expired_condition     the window closed                -> expired
neg_tampered_receipt      the signature is broken          -> refused
neg_untrusted_key         signed by a key nobody pinned    -> refused

The three negatives that matter most are the last four together: absence, a
closed window, and two kinds of broken evidence must all be distinguishable from
each other and from green. A corpus with only the positive case would prove
nothing about that.

Each fixture is self-contained JSON referencing a committed public key by path.
``_check_independent.py`` reproduces every verdict with no Vaara import, using
only rfc8785 and cryptography, so a passing check is a property of the bytes and
not of this script.

Run: python3 tests/vectors/release_condition_v0/_generate.py
"""
from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

from vaara.audit.signer import Ed25519Signer
from vaara.credential import (
    Capability,
    GrantBinding,
    GrantScope,
    GrantVerdict,
    emit_grant,
    mint_authorization_receipt,
)
from vaara.settlement.release import (
    HeldValue,
    ReleaseRequirements,
    emit_release_condition,
    receipt_key_fingerprint,
)

HERE = Path(__file__).resolve().parent

# Corpus keys, derived from fixed constants so the suite regenerates from
# nothing. Not deployed keys, and only the public halves are committed.
RECEIPT_KEY_SECRET = 0x5EC0DE5EC0DE5EC0DE5EC0DE
OTHER_KEY_SECRET = 0xBADBEEFBADBEEFBADBEEF
CONDITION_KEY_SEED = bytes(range(32))

TENANT = "mcp-tenant-01"
TOOL = "transfer_funds"
ATT_DIGEST = "sha256:" + "a" * 64
ATT_NONCE = "nonce-release-v0-01"
OTHER_ATT_DIGEST = "sha256:" + "b" * 64
ISSUER = "escrow.example"
RECEIPT_ISS = "gateway.example"
SECRET_VERSION = "corpus-key-v0"

RUNTIME_ARGS = {"amount": 400, "vendor": "acme"}
OTHER_ARGS = {"amount": 401, "vendor": "acme"}

GRANT_IAT = "2026-08-20T09:00:00Z"
RECEIPT_IAT = "2026-08-20T09:00:05Z"
DECIDED_AT = "2026-08-20T09:00:05Z"
NOT_AFTER = "2026-09-01T00:00:00Z"
NOW_OPEN = "2026-08-21T12:00:00Z"
NOW_CLOSED = "2026-09-02T00:00:00Z"


def _receipt_key():
    return ec.derive_private_key(RECEIPT_KEY_SECRET, ec.SECP256R1())


def _other_receipt_key():
    return ec.derive_private_key(OTHER_KEY_SECRET, ec.SECP256R1())


def _condition_key():
    return ed25519.Ed25519PrivateKey.from_private_bytes(CONDITION_KEY_SEED)


def _public_pem(private_key) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _grant(*, attestation_digest=ATT_DIGEST, nonce="grant-nonce-release-v0"):
    return emit_grant(
        scope=GrantScope(tool_name=TOOL, args_commitment="", tenant_id=TENANT),
        binding=GrantBinding(
            attestation_digest=attestation_digest, attestation_nonce=ATT_NONCE
        ),
        iss=RECEIPT_ISS,
        sub=TENANT,
        secret_version=SECRET_VERSION,
        alg="ES256",
        signing_material=_receipt_key(),
        exp_seconds=300,
        capabilities=(Capability("amount", "le", "500"),),
        iat=GRANT_IAT,
        nonce=nonce,
    )


def _receipt(*, grant=None, args=None, allowed=True, key=None, nonce="receipt-nonce-release-v0"):
    minted = mint_authorization_receipt(
        credential=grant or _grant(),
        runtime_args=RUNTIME_ARGS if args is None else args,
        verdict=GrantVerdict(allowed, "ok" if allowed else "capability_exceeded"),
        iss=RECEIPT_ISS,
        sub=TENANT,
        secret_version=SECRET_VERSION,
        alg="ES256",
        signing_material=key or _receipt_key(),
        decided_at=DECIDED_AT,
        nonce=nonce,
        iat=RECEIPT_IAT,
    )
    return minted.record.to_dict(), minted.evidence


def _condition(evidence, *, not_after=NOT_AFTER, condition_id="release-condition-v0"):
    requires = ReleaseRequirements(
        action_digest=evidence["argsCommitment"],
        grant_fingerprint=evidence["grantFingerprint"],
        receipt_issuer=RECEIPT_ISS,
        receipt_key_fingerprint=receipt_key_fingerprint(_public_pem(_receipt_key())),
    )
    return emit_release_condition(
        signer=Ed25519Signer(_condition_key()),
        issuer=ISSUER,
        condition_id=condition_id,
        held=HeldValue(
            amount="1000000", asset="USDC", network="base", payee="0x" + "1" * 40
        ),
        requires=requires,
        not_after=not_after,
    )


def _case(
    *,
    condition,
    state,
    reason,
    receipt=None,
    evidence=None,
    now=NOW_OPEN,
    receipt_key="keys/es256_public.pem",
):
    return {
        "condition": condition,
        "condition_key": "keys/ed25519_public.pem",
        "evidence": evidence,
        "expected_reason": reason,
        "expected_state": state,
        "now": now,
        "receipt": receipt,
        "receipt_key": receipt_key if receipt is not None else None,
    }


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    keys_dir = HERE / "keys"
    keys_dir.mkdir(parents=True, exist_ok=True)
    (keys_dir / "es256_public.pem").write_bytes(_public_pem(_receipt_key()))
    (keys_dir / "es256_other_public.pem").write_bytes(_public_pem(_other_receipt_key()))
    (keys_dir / "ed25519_public.pem").write_bytes(_public_pem(_condition_key()))

    cases_dir = HERE / "cases"
    receipt, evidence = _receipt()
    condition = _condition(evidence)

    _write(
        cases_dir / "pos_matching_receipt.json",
        _case(
            condition=condition,
            receipt=receipt,
            evidence=evidence,
            state="released",
            reason="receipt_matches",
        ),
    )

    _write(
        cases_dir / "neg_absent_receipt.json",
        _case(condition=condition, state="held", reason="receipt_absent"),
    )

    other_grant_receipt, other_grant_evidence = _receipt(
        grant=_grant(attestation_digest=OTHER_ATT_DIGEST, nonce="other-grant-nonce"),
        nonce="receipt-nonce-other-grant",
    )
    _write(
        cases_dir / "neg_authorization_mismatch.json",
        _case(
            condition=condition,
            receipt=other_grant_receipt,
            evidence=other_grant_evidence,
            state="held",
            reason="authorization_mismatch",
        ),
    )

    other_action_receipt, other_action_evidence = _receipt(
        args=OTHER_ARGS, nonce="receipt-nonce-other-action"
    )
    _write(
        cases_dir / "neg_other_action.json",
        _case(
            condition=condition,
            receipt=other_action_receipt,
            evidence=other_action_evidence,
            state="held",
            reason="action_digest_mismatch",
        ),
    )

    blocked_receipt, blocked_evidence = _receipt(
        allowed=False, nonce="receipt-nonce-blocked"
    )
    _write(
        cases_dir / "neg_blocked_decision.json",
        _case(
            condition=condition,
            receipt=blocked_receipt,
            evidence=blocked_evidence,
            state="held",
            reason="decision_not_accepted",
        ),
    )

    _write(
        cases_dir / "neg_expired_condition.json",
        _case(
            condition=condition,
            receipt=receipt,
            evidence=evidence,
            now=NOW_CLOSED,
            state="expired",
            reason="condition_expired",
        ),
    )

    # Tampered after signing: the decision time is moved by one second, which is
    # exactly the edit that would make a stale receipt look current.
    tampered = json.loads(json.dumps(receipt))
    tampered["decisionDerived"]["decidedAt"] = "2026-08-20T09:00:06Z"
    _write(
        cases_dir / "neg_tampered_receipt.json",
        _case(
            condition=condition,
            receipt=tampered,
            evidence=evidence,
            state="refused",
            reason="receipt_signature_invalid",
        ),
    )

    untrusted_receipt, untrusted_evidence = _receipt(
        key=_other_receipt_key(), nonce="receipt-nonce-untrusted-key"
    )
    _write(
        cases_dir / "neg_untrusted_key.json",
        _case(
            condition=condition,
            receipt=untrusted_receipt,
            evidence=untrusted_evidence,
            state="refused",
            reason="receipt_key_untrusted",
            receipt_key="keys/es256_other_public.pem",
        ),
    )

    expected_cases = {}
    for path in sorted(cases_dir.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        expected_cases[path.stem] = {
            "expected_state": case["expected_state"],
            "expected_reason": case["expected_reason"],
        }
    _write(HERE / "expected.json", {"cases": expected_cases})

    print(f"wrote {len(expected_cases)} cases + expected.json")


if __name__ == "__main__":
    main()
