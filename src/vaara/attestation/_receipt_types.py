"""Dataclasses and serialization for execution-receipt envelopes.

Internal module. Public surface is in ``vaara.attestation.receipt``.

The execution receipt is the post-execution sibling of the SEP-2787
request attestation. SEP-2787 binds an *observed* ``tools/call``
request before execution: issuer, subject, target, intent, nonce,
time, and an argument commitment. It deliberately says nothing about
whether the call ran or what came back. The receipt covers exactly
that deferred half: it binds the *outcome* of one attested request
and links back to the attestation it answers.

Three blocks plus the signature, mirroring the SEP-2787 trust-surface
layout so the two envelopes verify with the same canonicalization and
signing stack:

1. ``backLink`` joins the receipt to its request attestation. It
   carries the attestation's nonce (fast correlation) and a digest
   over the full SEP-2787 wire envelope including its signature, which
   pins the exact attestation instance the receipt answers.
2. ``receiptAsserted`` is the issuer block, set by whoever observed
   the outcome (the executing server, or an intermediary such as a
   governance proxy). Its signature is the proof the values were bound
   together at receipt time.
3. ``outcomeDerived`` carries the execution status, completion time,
   and an optional commitment over the result. The result commitment
   reuses the SEP-2787 argument-commitment shapes (``ArgsRef`` /
   ``ArgsProjection``) verbatim: structurally they are a commitment
   over a JSON value, which is what a result commitment also is.

A receipt is a durable record, not a time-bounded capability, so it
carries no ``exp`` and the verifier enforces no TTL. This is the same
pre-execution-capability versus post-execution-record distinction the
SEP-2787 thread keeps drawing: the attestation can expire, the record
of what happened does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from vaara.attestation._attest_types import (
    VALID_ALGS,
    Algorithm,
    ArgsCommitment,
    AttestationError,
    args_from_dict,
    args_to_dict,
)

# Result commitments are structurally identical to argument commitments:
# a commitment over a JSON-serialisable value. Reuse the shapes rather
# than duplicate them.
ResultCommitment = ArgsCommitment


def _reject_unknown_keys(d: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    """Fail closed on any key not in the closed schema for a signed block.

    The signature covers the JCS encoding of the *modeled* fields. If parsing
    silently dropped an unrecognized key, a verifier that re-derives the
    preimage from the model would exclude bytes that a byte-exact verifier (or
    the independent checker, which canonicalizes the raw wire) includes, so the
    two would disagree and Vaara could report a record as signed when the
    injected bytes were never covered. A closed schema with a hard reject keeps
    the modeled preimage byte-exact to the wire. Extending the schema is an
    explicit version bump, not a silently-tolerated extra field.
    """
    extra = set(d) - allowed
    if extra:
        raise AttestationError(
            f"{where} carries unrecognized field(s) {sorted(extra)!r}; "
            "the signed schema is closed"
        )

ReceiptStatus = Literal["executed", "refused", "errored"]
VALID_STATUSES: frozenset[str] = frozenset({"executed", "refused", "errored"})


@dataclass(frozen=True)
class BackLink:
    """Join from a receipt to the SEP-2787 attestation it answers.

    ``attestation_nonce`` echoes the attestation's
    ``issuerAsserted.nonce`` for fast correlation. ``attestation_digest``
    is ``sha256:<hex>`` over the JCS-canonical encoding of the full
    attestation wire envelope (signature included), pinning the exact
    attestation instance.

    ``fallback_projection`` is set only in the no-SEP-2787 fallback case
    (``fallbackProjection`` on the wire): it names the projection version
    the ``attestation_digest`` was computed under, so a verifier
    reconstructs the same named projection deterministically from the
    signed record rather than inferring it from the observed envelope, and
    a later projection revision is an explicit version rather than a silent
    reinterpretation. It is absent on the attestation path.
    """

    attestation_digest: str
    attestation_nonce: str
    fallback_projection: Optional[str] = None


@dataclass(frozen=True)
class ReceiptAsserted:
    """Issuer block: what the receipt issuer binds at receipt time.

    Set by the party that observed the outcome (executing server or an
    intermediary). The signature over the envelope is the proof these
    values were bound together.

    ``sig_suite`` (``sigSuite`` on the wire) is the optional post-quantum
    hybrid commitment. Absent means classical-only: the receipt's ``alg``
    is the whole story and the envelope is byte-for-byte what it was before
    this field existed. When present it names an allowlisted hybrid suite
    (e.g. ``"ES256+ML-DSA-65"``), committing the issuer's intent *inside*
    the signed preimage so a stripped ``pqSignature`` is a detectable
    downgrade. See ``docs/design/pq-hybrid-signing-spec.md``.

    ``crypto_posture`` (``cryptoPosture`` on the wire) is the optional
    CycloneDX-CBOM crypto-posture block: the signature algorithms protecting
    the record and the effective NIST post-quantum security level they reach.
    Like ``sig_suite`` it rides inside the signed preimage, so an inflated
    quantum-resistance claim or a stripped ML-DSA leg is recomputable from the
    signed bytes. Absent means the record predates the block and the envelope
    is byte-for-byte what it was before. See
    ``docs/design/cbom-crypto-posture-spec.md``.
    """

    iss: str
    sub: str
    iat: str
    nonce: str
    secret_version: str
    alg: Algorithm
    sig_suite: Optional[str] = None
    crypto_posture: Optional["CryptoPosture"] = None


@dataclass(frozen=True)
class OutcomeDerived:
    """Facts about what happened to the attested request.

    ``status`` is one of ``executed`` / ``refused`` / ``errored``.
    ``result_commitment`` is a commitment over the result payload and
    is optional: a refused call has no result, so the commitment is
    absent. An executed or errored call commits to the result or the
    error object respectively.

    ``decision_digest`` is the SEP-2828 Check B (outcome-to-decision)
    binding: ``sha256:<hex>`` over the full signed decision-record wire
    bytes the outcome was produced under. It is optional on the type so
    pre-v0.51 receipts and the no-attestation fallback still parse, but
    v0.51 emitters MUST set it and pairing (``records_paired``) fails
    without it. Where ``backLink`` pins the call instance (Check A), this
    pins which decision's content the outcome answers.
    """

    status: ReceiptStatus
    completed_at: str
    result_commitment: Optional[ResultCommitment] = None
    decision_digest: Optional[str] = None


@dataclass(frozen=True)
class PqSignature:
    """A parallel post-quantum signature over the receipt preimage.

    ``alg`` is the PQC scheme (``"ML-DSA-65"`` in v0). ``keyid`` names the
    DID verification method whose ML-DSA public key the ``sig`` verifies
    under. ``sig`` is the hex-encoded signature over the **same** JCS
    preimage the classical ``signature`` covers, so the two signatures
    bind identical bytes. This block rides outside the signed preimage (it
    signs the preimage and so cannot be a field of it); the downgrade it
    would otherwise allow is closed by ``receiptAsserted.sigSuite``, which
    is inside the preimage. See ``docs/design/pq-hybrid-signing-spec.md``.
    """

    alg: str
    keyid: str
    sig: str


def pq_signature_to_dict(pq: PqSignature) -> dict[str, Any]:
    return {"alg": pq.alg, "keyid": pq.keyid, "sig": pq.sig}


def pq_signature_from_dict(d: dict[str, Any]) -> PqSignature:
    for required in ("alg", "keyid", "sig"):
        value = d.get(required)
        if not isinstance(value, str) or not value:
            raise AttestationError(
                f"pqSignature.{required} must be a non-empty string"
            )
    return PqSignature(alg=d["alg"], keyid=d["keyid"], sig=d["sig"])


@dataclass(frozen=True)
class CryptoAlgorithm:
    """One cryptographic algorithm protecting a receipt, CycloneDX-CBOM shaped.

    ``algorithm`` is the JOSE / Vaara suite name (``"ES256"``, ``"ML-DSA-65"``).
    ``primitive`` is the CycloneDX cryptographic primitive: ``"signature"`` for
    the asymmetric suites, ``"mac"`` for the HS256 keyed-MAC path.
    ``nist_quantum_security_level`` is the NIST post-quantum security category
    the algorithm reaches: 0 for the classical suites (no quantum resistance),
    3 for ML-DSA-65 (FIPS 204, Category 3). Field names track CycloneDX 1.6
    ``cryptoProperties.algorithmProperties`` (ECMA-424).
    """

    algorithm: str
    primitive: str
    nist_quantum_security_level: int


@dataclass(frozen=True)
class CryptoPosture:
    """CycloneDX-CBOM crypto posture committed inside the receipt preimage.

    A signed, tamper-evident declaration of which signature algorithms protect
    the record and the effective NIST post-quantum security level they reach.
    It rides inside ``receiptAsserted`` (so inside the signed preimage) next to
    ``sigSuite``: a verifier recomputes the expected posture from the receipt's
    own ``alg`` + ``sigSuite``, so an inflated quantum-resistance claim, or a
    stripped ML-DSA leg, is a signed statement the record cannot back.

    ``nist_quantum_security_level`` is the effective floor for the record: the
    max over ``algorithms``. A hybrid reaches its ML-DSA leg's category because
    an attacker must still forge that leg; a classical-only receipt reports 0.
    """

    asset_type: str
    nist_quantum_security_level: int
    algorithms: tuple[CryptoAlgorithm, ...]


def crypto_algorithm_to_dict(ca: CryptoAlgorithm) -> dict[str, Any]:
    return {
        "algorithm": ca.algorithm,
        "primitive": ca.primitive,
        "nistQuantumSecurityLevel": ca.nist_quantum_security_level,
    }


def crypto_posture_to_dict(cp: CryptoPosture) -> dict[str, Any]:
    return {
        "assetType": cp.asset_type,
        "nistQuantumSecurityLevel": cp.nist_quantum_security_level,
        "algorithms": [crypto_algorithm_to_dict(a) for a in cp.algorithms],
    }


_CRYPTO_ALGORITHM_KEYS = frozenset(
    {"algorithm", "primitive", "nistQuantumSecurityLevel"}
)
_CRYPTO_POSTURE_KEYS = frozenset(
    {"assetType", "nistQuantumSecurityLevel", "algorithms"}
)


def _crypto_level(value: Any, where: str) -> int:
    """A NIST quantum-security level: an integer 0..5, rejecting bool.

    ``bool`` is an ``int`` subclass, so ``True``/``False`` would otherwise slip
    through and canonicalize to ``1``/``0``, letting a boolean masquerade as a
    level. Reject it explicitly.
    """
    if not isinstance(value, int) or isinstance(value, bool) or not (0 <= value <= 5):
        raise AttestationError(f"{where} must be an integer 0..5")
    return value


def crypto_algorithm_from_dict(d: dict[str, Any]) -> CryptoAlgorithm:
    _reject_unknown_keys(d, _CRYPTO_ALGORITHM_KEYS, "cryptoPosture.algorithms[]")
    for required in ("algorithm", "primitive", "nistQuantumSecurityLevel"):
        if required not in d:
            raise AttestationError(
                f"cryptoPosture algorithm missing required field {required!r}"
            )
    algorithm = d["algorithm"]
    primitive = d["primitive"]
    if not isinstance(algorithm, str) or not algorithm:
        raise AttestationError(
            "cryptoPosture.algorithms[].algorithm must be a non-empty string"
        )
    if not isinstance(primitive, str) or not primitive:
        raise AttestationError(
            "cryptoPosture.algorithms[].primitive must be a non-empty string"
        )
    return CryptoAlgorithm(
        algorithm=algorithm,
        primitive=primitive,
        nist_quantum_security_level=_crypto_level(
            d["nistQuantumSecurityLevel"],
            "cryptoPosture.algorithms[].nistQuantumSecurityLevel",
        ),
    )


def crypto_posture_from_dict(d: dict[str, Any]) -> CryptoPosture:
    _reject_unknown_keys(d, _CRYPTO_POSTURE_KEYS, "cryptoPosture")
    for required in ("assetType", "nistQuantumSecurityLevel", "algorithms"):
        if required not in d:
            raise AttestationError(
                f"cryptoPosture missing required field {required!r}"
            )
    if d["assetType"] != "algorithm":
        raise AttestationError("cryptoPosture.assetType must be 'algorithm'")
    algorithms = d["algorithms"]
    if not isinstance(algorithms, list) or not algorithms:
        raise AttestationError(
            "cryptoPosture.algorithms must be a non-empty array"
        )
    return CryptoPosture(
        asset_type="algorithm",
        nist_quantum_security_level=_crypto_level(
            d["nistQuantumSecurityLevel"], "cryptoPosture.nistQuantumSecurityLevel"
        ),
        algorithms=tuple(crypto_algorithm_from_dict(a) for a in algorithms),
    )


@dataclass(frozen=True)
class ExecutionReceipt:
    """Execution-receipt envelope.

    ``backLink`` plus two trust-surface blocks plus the signature. The
    signature is computed over the JCS-canonical encoding of
    ``{version, alg, backLink, outcomeDerived, receiptAsserted}`` and
    does not cover itself.

    ``pq_signature`` (``pqSignature`` on the wire) is the optional
    post-quantum hybrid signature, a sibling of ``signature`` that also
    sits outside the preimage. It is present only for hybrid receipts.
    """

    version: int
    alg: Algorithm
    back_link: BackLink
    receipt_asserted: ReceiptAsserted
    outcome_derived: OutcomeDerived
    signature: str
    pq_signature: Optional[PqSignature] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "version": self.version,
            "alg": self.alg,
            "backLink": back_link_to_dict(self.back_link),
            "outcomeDerived": outcome_to_dict(self.outcome_derived),
            "receiptAsserted": receipt_asserted_to_dict(self.receipt_asserted),
            "signature": self.signature,
        }
        if self.pq_signature is not None:
            out["pqSignature"] = pq_signature_to_dict(self.pq_signature)
        return out


def back_link_to_dict(bl: BackLink) -> dict[str, Any]:
    out: dict[str, Any] = {
        "attestationDigest": bl.attestation_digest,
        "attestationNonce": bl.attestation_nonce,
    }
    if bl.fallback_projection is not None:
        out["fallbackProjection"] = bl.fallback_projection
    return out


def receipt_asserted_to_dict(ra: ReceiptAsserted) -> dict[str, Any]:
    out: dict[str, Any] = {
        "alg": ra.alg,
        "iat": ra.iat,
        "iss": ra.iss,
        "nonce": ra.nonce,
        "secretVersion": ra.secret_version,
        "sub": ra.sub,
    }
    if ra.sig_suite is not None:
        out["sigSuite"] = ra.sig_suite
    if ra.crypto_posture is not None:
        out["cryptoPosture"] = crypto_posture_to_dict(ra.crypto_posture)
    return out


def outcome_to_dict(od: OutcomeDerived) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": od.status,
        "completedAt": od.completed_at,
    }
    if od.result_commitment is not None:
        out["resultCommitment"] = args_to_dict(od.result_commitment)
    if od.decision_digest is not None:
        out["decisionDigest"] = od.decision_digest
    return out


_BACK_LINK_KEYS = frozenset(
    {"attestationDigest", "attestationNonce", "fallbackProjection"}
)
_RECEIPT_ASSERTED_KEYS = frozenset(
    {"alg", "iat", "iss", "nonce", "secretVersion", "sub", "sigSuite",
     "cryptoPosture"}
)
_OUTCOME_KEYS = frozenset(
    {"status", "completedAt", "resultCommitment", "decisionDigest"}
)
_RECEIPT_KEYS = frozenset(
    {"version", "alg", "backLink", "outcomeDerived", "receiptAsserted",
     "signature", "pqSignature"}
)


def back_link_from_dict(d: dict[str, Any]) -> BackLink:
    _reject_unknown_keys(d, _BACK_LINK_KEYS, "backLink")
    for required in ("attestationDigest", "attestationNonce"):
        if required not in d:
            raise AttestationError(f"backLink missing required field {required!r}")
    return BackLink(
        attestation_digest=d["attestationDigest"],
        attestation_nonce=d["attestationNonce"],
        fallback_projection=d.get("fallbackProjection"),
    )


def receipt_asserted_from_dict(d: dict[str, Any]) -> ReceiptAsserted:
    _reject_unknown_keys(d, _RECEIPT_ASSERTED_KEYS, "receiptAsserted")
    for required in ("alg", "iat", "iss", "nonce", "secretVersion", "sub"):
        if required not in d:
            raise AttestationError(
                f"receiptAsserted missing required field {required!r}"
            )
    if d["alg"] not in VALID_ALGS:
        raise AttestationError(f"unsupported alg {d['alg']!r}")
    sig_suite = d.get("sigSuite")
    if sig_suite is not None and not isinstance(sig_suite, str):
        raise AttestationError("receiptAsserted.sigSuite must be a string or absent")
    crypto_posture_raw = d.get("cryptoPosture")
    crypto_posture = None
    if crypto_posture_raw is not None:
        if not isinstance(crypto_posture_raw, dict):
            raise AttestationError(
                "receiptAsserted.cryptoPosture must be an object or absent"
            )
        crypto_posture = crypto_posture_from_dict(crypto_posture_raw)
    return ReceiptAsserted(
        alg=d["alg"],
        iat=d["iat"],
        iss=d["iss"],
        nonce=d["nonce"],
        secret_version=d["secretVersion"],
        sub=d["sub"],
        sig_suite=sig_suite,
        crypto_posture=crypto_posture,
    )


def outcome_from_dict(d: dict[str, Any]) -> OutcomeDerived:
    _reject_unknown_keys(d, _OUTCOME_KEYS, "outcomeDerived")
    for required in ("status", "completedAt"):
        if required not in d:
            raise AttestationError(
                f"outcomeDerived missing required field {required!r}"
            )
    if d["status"] not in VALID_STATUSES:
        raise AttestationError(f"invalid status {d['status']!r}")
    commitment = (
        args_from_dict(d["resultCommitment"])
        if "resultCommitment" in d
        else None
    )
    decision_digest = d.get("decisionDigest")
    if decision_digest is not None and not decision_digest.startswith("sha256:"):
        raise AttestationError(
            "outcomeDerived.decisionDigest MUST be a 'sha256:' digest"
        )
    return OutcomeDerived(
        status=d["status"],
        completed_at=d["completedAt"],
        result_commitment=commitment,
        decision_digest=decision_digest,
    )


def receipt_from_dict(d: dict[str, Any]) -> ExecutionReceipt:
    """Reconstruct an ExecutionReceipt from its wire JSON dict.

    Inverse of ``ExecutionReceipt.to_dict()``. Field-presence
    validation only; signature verification still requires the
    caller's keying material.
    """
    _reject_unknown_keys(d, _RECEIPT_KEYS, "receipt")
    for required in (
        "version", "alg", "backLink", "outcomeDerived",
        "receiptAsserted", "signature",
    ):
        if required not in d:
            raise AttestationError(f"receipt missing required field {required!r}")
    if d["alg"] not in VALID_ALGS:
        raise AttestationError(f"unsupported alg {d['alg']!r}")
    pq_raw = d.get("pqSignature")
    pq_signature = None
    if pq_raw is not None:
        if not isinstance(pq_raw, dict):
            raise AttestationError("pqSignature must be an object or absent")
        pq_signature = pq_signature_from_dict(pq_raw)
    return ExecutionReceipt(
        version=d["version"],
        alg=d["alg"],
        back_link=back_link_from_dict(d["backLink"]),
        receipt_asserted=receipt_asserted_from_dict(d["receiptAsserted"]),
        outcome_derived=outcome_from_dict(d["outcomeDerived"]),
        signature=d["signature"],
        pq_signature=pq_signature,
    )
