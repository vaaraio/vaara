"""Authorization-receipt tests: proof-carrying enforcement.

A gateway verdict on its own is ephemeral. These tests pin the property that
makes Vaara's authority layer different from a kernel egress filter: every
decision, and above all every refusal, becomes a signed, content-addressed,
independently recomputable receipt. The deny case is the headline: a refused
call leaves a portable proof of the non-action that verifies against the
issuer's public key alone, with zero trust in the producer.
"""

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402

from vaara.attestation._sep2787_canonical import (  # noqa: E402
    canonical_json,
    iso8601_to_epoch,
    make_args_digest,
)
from vaara.attestation.decision import (  # noqa: E402
    parse_decision_record,
    verify_decision_signature,
)
from vaara.credential import (  # noqa: E402
    Capability,
    GrantBinding,
    GrantScope,
    emit_grant,
    verify_grant,
)
from vaara.credential._authorization_receipt import (  # noqa: E402
    AUTHORIZATION_SCHEMA,
    mint_authorization_receipt,
)

DIGEST = "sha256:" + "ab" * 32
NONCE = "att-nonce-xyz"
IAT = "2026-06-18T12:00:00Z"
IAT_EPOCH = iso8601_to_epoch(IAT)

MINT_ARGS = {"amount": 100, "vendor": "acme", "destination": "0xABC"}
COMMIT = make_args_digest(MINT_ARGS).projection_digest
CAPS = (
    Capability("amount", "le", "500"),
    Capability("vendor", "in", ("acme", "globex")),
    Capability("destination", "eq", "0xABC"),
)
RUNTIME_OK = {"amount": 400, "vendor": "globex", "destination": "0xABC"}
RUNTIME_OVER = {"amount": 600, "vendor": "acme", "destination": "0xABC"}


def _digest(obj) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj)).hexdigest()


@pytest.fixture
def key():
    return ec.generate_private_key(ec.SECP256R1())


def _grant(priv):
    return emit_grant(
        scope=GrantScope(tool_name="pay.send", args_commitment=COMMIT, tenant_id="tenant-a"),
        binding=GrantBinding(attestation_digest=DIGEST, attestation_nonce=NONCE),
        iss="vaara-mcp-proxy",
        sub="tenant-a/upstream",
        secret_version="key-v1",
        alg="ES256",
        signing_material=priv,
        exp_seconds=60,
        iat=IAT,
        capabilities=CAPS,
    )


def _verdict(cred, priv, runtime_args):
    return verify_grant(
        cred,
        verifying_material=priv.public_key(),
        runtime_tool_name="pay.send",
        runtime_args=runtime_args,
        runtime_tenant_id="tenant-a",
        known_attestation_digests=frozenset({DIGEST}),
        now=IAT_EPOCH + 5,
    )


def _mint(cred, priv, verdict, runtime_args):
    return mint_authorization_receipt(
        credential=cred,
        runtime_args=runtime_args,
        verdict=verdict,
        iss="vaara-mcp-proxy",
        sub="tenant-a/upstream",
        secret_version="key-v1",
        alg="ES256",
        signing_material=priv,
        decided_at="2026-06-18T12:00:05Z",
    )


def test_allow_mints_a_signed_recomputable_receipt(key):
    cred = _grant(key)
    verdict = _verdict(cred, key, RUNTIME_OK)
    assert verdict.ok and verdict.reason == "ok"

    auth = _mint(cred, key, verdict, RUNTIME_OK)
    record = auth.record

    # Envelope is the SPEC decision shape; allow maps to the allow verdict.
    assert record.decision_derived.decision == "allow"
    assert record.decision_derived.reason == "ok"
    # Signature verifies against the issuer's public key alone.
    assert verify_decision_signature(record, verifying_material=key.public_key())
    # Evidence is content-addressed: the receipt pins its JCS digest.
    assert record.decision_derived.evidence_ref.digest == _digest(auth.evidence)
    assert record.decision_derived.evidence_ref.schema == AUTHORIZATION_SCHEMA
    assert auth.evidence["verdict"] == "allow"
    # Back-link joins to the same attestation the grant is bound to.
    assert record.back_link.attestation_digest == DIGEST
    assert record.back_link.attestation_nonce == NONCE


def test_deny_leaves_a_portable_proof_of_the_refusal(key):
    cred = _grant(key)
    verdict = _verdict(cred, key, RUNTIME_OVER)
    assert not verdict.ok and verdict.reason == "capability_exceeded"

    auth = _mint(cred, key, verdict, RUNTIME_OVER)
    record = auth.record

    # A refusal maps to the block verdict, carrying the precise machine reason.
    assert record.decision_derived.decision == "block"
    assert record.decision_derived.reason == "capability_exceeded"
    assert auth.evidence["verdict"] == "deny"
    assert auth.evidence["reason"] == "capability_exceeded"
    # The deny receipt is itself a valid, verifiable signed record.
    assert verify_decision_signature(record, verifying_material=key.public_key())
    assert record.decision_derived.evidence_ref.digest == _digest(auth.evidence)


def test_arguments_stay_private_only_the_commitment_travels(key):
    cred = _grant(key)
    verdict = _verdict(cred, key, RUNTIME_OK)
    auth = _mint(cred, key, verdict, RUNTIME_OK)

    # The raw argument map is never embedded as a value; only its commitment is.
    # (A substring scan would be fragile: capability bounds legitimately carry
    # argument values, and a hash hex can contain any digit run by chance.)
    assert RUNTIME_OK not in auth.evidence.values()
    assert "args" not in auth.evidence
    # The commitment is a hash that binds the exact arguments and recomputes.
    assert auth.evidence["argsCommitment"].startswith("sha256:")
    assert auth.evidence["argsCommitment"] == _digest(RUNTIME_OK)


def test_tamper_breaks_the_signature(key):
    cred = _grant(key)
    verdict = _verdict(cred, key, RUNTIME_OVER)
    auth = _mint(cred, key, verdict, RUNTIME_OVER)

    tampered = auth.record.to_dict()
    # Flip the refusal into an approval; the signature must reject it.
    tampered["decisionDerived"]["decision"] = "allow"
    reloaded = parse_decision_record(tampered)
    assert not verify_decision_signature(reloaded, verifying_material=key.public_key())


def test_evidence_names_the_capabilities_the_verdict_used(key):
    cred = _grant(key)
    verdict = _verdict(cred, key, RUNTIME_OK)
    auth = _mint(cred, key, verdict, RUNTIME_OK)
    assert auth.evidence["toolName"] == "pay.send"
    assert auth.evidence["tenantId"] == "tenant-a"
    caps = auth.evidence["capabilities"]
    assert {c["arg"] for c in caps} == {"amount", "vendor", "destination"}
