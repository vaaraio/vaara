"""Emit and verify-signature for decision-record envelopes.

Internal module. Public surface is in ``vaara.attestation.decision``.

Reuses the SEP-2787 canonicalization (RFC 8785 JCS) and signing stack
(HS256 / ES256 / RS256) unchanged. The only new wire shape is the
envelope layout; the cryptographic primitives are shared so a verifier
that already handles SEP-2787 signatures handles decision records with
no new crypto code.
"""

from __future__ import annotations

from typing import Any, Optional

from vaara.attestation._decision_types import (
    DecisionDerived,
    DecisionRecord,
    IssuerAsserted,
    decision_to_dict,
)
from vaara.attestation._receipt_types import BackLink, back_link_to_dict, receipt_asserted_to_dict
from vaara.attestation._attest_canonical import (
    canonical_json,
    new_nonce,
    now_iso8601,
)
from vaara.attestation._attest_signing import (
    sign_es256,
    sign_hs256,
    sign_rs256,
    verify_es256,
    verify_hs256,
    verify_rs256,
)
from vaara.attestation._attest_types import (
    VALID_ALGS,
    Algorithm,
    AttestationError,
)


def build_decision_basis(
    *,
    rule: str,
    reason: str,
    declared_intent: str,
    canonical_policy: dict[str, Any],
    canonical_inputs: dict[str, Any],
    verdict: str,
    intent_satisfied: Optional[bool] = None,
    score_fp: Optional[int] = None,
    deny_fp: Optional[int] = None,
    escalate_fp: Optional[int] = None,
    with_proof: bool = False,
) -> dict[str, Any]:
    """Build the native decision basis: ``rationale`` and ``binding`` always,
    and the zero-knowledge ``decisionProof`` envelope when ``with_proof`` is set.

    Keyless by default. ``rationale`` and ``binding`` are pure stdlib. The proof
    path is opt-in and imported lazily, so this stays importable without the
    ``attestation`` extra; requesting a proof without the extra raises.
    """
    from vaara.attestation._decision_binding import build_binding, build_rationale

    rationale = build_rationale(rule, reason, declared_intent, intent_satisfied)

    if not with_proof:
        return {
            "rationale": rationale,
            "binding": build_binding(
                canonical_policy, declared_intent, canonical_inputs, verdict
            ),
        }

    # Proof mode. The commitments are created first, hashed into the binding as the
    # inputs (so the record's signed bindingDigest pins exactly these commitments),
    # then the proof is produced against the same commitments. This binds the values
    # the proof opens to the record; without it the committed values would float free
    # of the record and the proof would attest a vacuous statement.
    if score_fp is None or deny_fp is None or escalate_fp is None:
        raise ValueError("with_proof requires score_fp, deny_fp, and escalate_fp")
    from vaara.attestation.zk._commit import commit, random_scalar
    from vaara.attestation.zk._prove import build_proof_envelope

    gs, gd, ge = random_scalar(), random_scalar(), random_scalar()
    vs, vd, ve = commit(score_fp, gs), commit(deny_fp, gd), commit(escalate_fp, ge)
    zk_inputs = {
        "scoreCommit": vs.to_bytes().hex(),
        "denyCommit": vd.to_bytes().hex(),
        "escalateCommit": ve.to_bytes().hex(),
    }
    binding = build_binding(canonical_policy, declared_intent, zk_inputs, verdict)
    envelope = build_proof_envelope(
        verdict, score_fp, deny_fp, escalate_fp, binding["bindingDigest"],
        blinds=(gs, gd, ge),
    )
    return {"rationale": rationale, "binding": binding, "decisionProof": envelope}


def _signing_payload(
    *,
    version: int,
    alg: Algorithm,
    back_link: BackLink,
    decision_derived: DecisionDerived,
    issuer_asserted: IssuerAsserted,
) -> bytes:
    """JCS-canonical encoding of the decision blocks, signature excluded."""
    body = {
        "version": version,
        "alg": alg,
        "backLink": back_link_to_dict(back_link),
        "decisionDerived": decision_to_dict(decision_derived),
        "issuerAsserted": receipt_asserted_to_dict(issuer_asserted),
    }
    return canonical_json(body)


def emit_decision_record(
    *,
    back_link: BackLink,
    decision_derived: DecisionDerived,
    iss: str,
    sub: str,
    secret_version: str,
    alg: Algorithm,
    signing_material: Any,
    nonce: Optional[str] = None,
    iat: Optional[str] = None,
    version: int = 1,
) -> DecisionRecord:
    """Build, JCS-canonicalize, and sign a DecisionRecord envelope.

    ``back_link`` joins the decision to the SEP-2787 attestation it
    governs (build it with ``make_back_link``). ``decision_derived``
    carries the verdict, its risk basis, and the decision time. Any
    float in the risk basis is rejected at the JCS boundary; the risk
    fields MUST be decimal strings.

    ``signing_material`` is either a bytes shared secret (HS256) or a
    private-key object from ``cryptography.hazmat`` (ES256 / RS256).
    """
    if alg not in VALID_ALGS:
        raise AttestationError(f"unsupported alg: {alg!r}")
    if not back_link.attestation_digest.startswith("sha256:"):
        raise AttestationError(
            "backLink.attestationDigest MUST be a 'sha256:' digest"
        )
    if not back_link.attestation_nonce:
        raise AttestationError("backLink.attestationNonce MUST be non-empty")
    er = decision_derived.evidence_ref
    if er is not None and not er.digest.startswith("sha256:"):
        raise AttestationError("evidenceRef.digest MUST be a 'sha256:' digest")

    issuer_asserted = IssuerAsserted(
        iss=iss,
        sub=sub,
        iat=iat or now_iso8601(),
        nonce=nonce or new_nonce(),
        secret_version=secret_version,
        alg=alg,
    )

    payload = _signing_payload(
        version=version,
        alg=alg,
        back_link=back_link,
        decision_derived=decision_derived,
        issuer_asserted=issuer_asserted,
    )

    if alg == "HS256":
        if not isinstance(signing_material, (bytes, bytearray)):
            raise AttestationError("HS256 requires bytes shared_secret")
        signature_hex = sign_hs256(payload, shared_secret=bytes(signing_material))
    elif alg == "ES256":
        signature_hex = sign_es256(payload, private_key=signing_material)
    elif alg == "RS256":
        signature_hex = sign_rs256(payload, private_key=signing_material)
    else:
        raise AttestationError(f"unreachable alg: {alg!r}")

    return DecisionRecord(
        version=version,
        alg=alg,
        back_link=back_link,
        decision_derived=decision_derived,
        issuer_asserted=issuer_asserted,
        signature=signature_hex,
    )


def verify_decision_signature(
    record: DecisionRecord,
    *,
    verifying_material: Any,
) -> bool:
    """Verify the decision-record signature only.

    Returns True iff the signature matches the JCS-canonical encoding of
    the record blocks under ``verifying_material``. Back-link and pairing
    checks are composed separately via ``verify_decision_back_link`` and
    ``records_paired``; a decision record is a durable record so there is
    no TTL to enforce.

    ``verifying_material`` is either a bytes shared secret (HS256) or a
    public-key object from ``cryptography.hazmat`` (ES256 / RS256).
    """
    payload = _signing_payload(
        version=record.version,
        alg=record.alg,
        back_link=record.back_link,
        decision_derived=record.decision_derived,
        issuer_asserted=record.issuer_asserted,
    )

    if record.alg == "HS256":
        if not isinstance(verifying_material, (bytes, bytearray)):
            return False
        return verify_hs256(
            payload,
            signature_hex=record.signature,
            shared_secret=bytes(verifying_material),
        )
    if record.alg == "ES256":
        return verify_es256(
            payload,
            signature_hex=record.signature,
            public_key=verifying_material,
        )
    if record.alg == "RS256":
        return verify_rs256(
            payload,
            signature_hex=record.signature,
            public_key=verifying_material,
        )
    return False
