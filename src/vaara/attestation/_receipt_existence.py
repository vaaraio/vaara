# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Qualified existence-in-time proof over a SEP-2828 execution record.

Internal module. Public surface is re-exported from
``vaara.attestation.receipt``.

The execution record proves what an attested call did and pins which decision
it answers. It does not prove *when* the record existed to a party who runs
none of the producer's software. ``existenceProof`` closes that: an RFC 3161
trusted timestamp over the record's own canonical bytes. A relying party
recomputes the record digest, checks the token imprints exactly that digest,
and checks the token's signer against a CA it pins from a trusted list. For
``rfc3161-eidas-qualified`` that list is the EU trusted list, so a match makes
the attested time a *qualified* eIDAS timestamp, recognised EU-wide and
checkable offline. Without a pin the time is self-asserted by whoever holds
the signing certificate, and the verifier reports it as such rather than as
evidence.

The proof imprints the signed record, so it is produced after signing and
rides outside the signed preimage, the same structural position as
``pqSignature``. The digest it commits to therefore excludes the
``existenceProof`` field itself.

The RFC 3161 verification is the ``timeanchor`` extra (``asn1crypto`` and
``cryptography``), imported lazily so the digest helper and record parsing
stay dependency-free in a base install.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._attest_canonical import canonical_json


def existence_record_digest(record: dict[str, Any]) -> str:
    """The ``sha256:`` digest an existence proof imprints.

    Computed over the JCS-canonical encoding of the record with the
    ``existenceProof`` field removed, since the proof cannot commit to itself.
    Every other field, the signature included, is covered: the proof pins the
    exact signed record as delivered.
    """
    covered = {k: v for k, v in record.items() if k != "existenceProof"}
    return "sha256:" + hashlib.sha256(canonical_json(covered)).hexdigest()


@dataclass(frozen=True)
class ExistenceResult:
    """Verdict of an existence-in-time check on one execution record.

    ``ok`` is the offline validity of the timestamp: the token parses, its
    message imprint equals the recomputed record digest, and its signature
    verifies under the certificate it carries. ``qualified`` is the stronger
    tier: ``ok`` *and* the signer chains to a CA the caller pinned from a
    trusted list, so the attested time is an eIDAS qualified timestamp.
    ``basis`` is ``"qualified"``, ``"self_asserted"`` (valid token, no pinned
    authority), or ``None`` when there is nothing to judge.
    """

    present: bool
    backend: Optional[str]
    hash_algorithm: Optional[str]
    record_digest: Optional[str]
    attested_time: Optional[str]
    qualified: bool
    basis: Optional[str]
    ok: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "backend": self.backend,
            "hashAlgorithm": self.hash_algorithm,
            "recordDigest": self.record_digest,
            "attestedTime": self.attested_time,
            "qualified": self.qualified,
            "basis": self.basis,
            "ok": self.ok,
            "reason": self.reason,
        }


def _absent(reason: str) -> ExistenceResult:
    return ExistenceResult(
        present=False, backend=None, hash_algorithm=None, record_digest=None,
        attested_time=None, qualified=False, basis=None, ok=False, reason=reason,
    )


def verify_existence_proof(
    record: dict[str, Any],
    *,
    trusted_issuer_cert: Optional[bytes] = None,
    trusted_signer_cert: Optional[bytes] = None,
) -> ExistenceResult:
    """Verify a record's ``existenceProof`` and grade it qualified / self-asserted.

    Recomputes the record digest, requires the proof's ``recordDigest`` to
    match it (so a proof stapled to a mutated record fails), then verifies the
    RFC 3161 token imprints that digest and is internally consistent. If a
    trusted-list pin is supplied and the token's signer chains to it, the
    attested time is qualified; otherwise it is self-asserted. A wrong pin is
    not a forged proof: the token stays valid, it is simply not qualified.

    Requires the ``timeanchor`` extra. Callers that must degrade gracefully
    (e.g. the base install) should catch :class:`ImportError`.
    """
    ep = record.get("existenceProof")
    if not isinstance(ep, dict):
        return _absent("record carries no existenceProof")

    backend = ep.get("backend")
    hash_algorithm = ep.get("hashAlgorithm", "sha256")
    recomputed = existence_record_digest(record)

    claimed = ep.get("recordDigest")
    if claimed != recomputed:
        return ExistenceResult(
            present=True, backend=backend, hash_algorithm=hash_algorithm,
            record_digest=recomputed, attested_time=None, qualified=False,
            basis=None, ok=False,
            reason="existenceProof.recordDigest does not match the record; "
                   "the proof was stapled to different bytes",
        )

    token_b64 = ep.get("token")
    if not isinstance(token_b64, str):
        return ExistenceResult(
            present=True, backend=backend, hash_algorithm=hash_algorithm,
            record_digest=recomputed, attested_time=None, qualified=False,
            basis=None, ok=False, reason="existenceProof.token is missing",
        )
    try:
        token_der = base64.b64decode(token_b64)
    except (ValueError, TypeError) as exc:
        return ExistenceResult(
            present=True, backend=backend, hash_algorithm=hash_algorithm,
            record_digest=recomputed, attested_time=None, qualified=False,
            basis=None, ok=False, reason=f"existenceProof.token is not base64: {exc}",
        )

    from vaara.audit.timeanchor import TimeAnchorError, verify_timestamp_token

    expected_digest = bytes.fromhex(recomputed.split(":", 1)[1])

    # Internal validity + imprint match, with no trust assumption.
    try:
        attested = verify_timestamp_token(
            token_der, expected_digest, hash_algorithm=hash_algorithm
        )
    except TimeAnchorError as exc:
        return ExistenceResult(
            present=True, backend=backend, hash_algorithm=hash_algorithm,
            record_digest=recomputed, attested_time=None, qualified=False,
            basis=None, ok=False, reason=f"timestamp token invalid: {exc}",
        )
    attested_time = attested.isoformat()

    # Qualified only when a pin is supplied and the signer chains to it.
    qualified = False
    basis = "self_asserted"
    if trusted_issuer_cert is not None or trusted_signer_cert is not None:
        try:
            verify_timestamp_token(
                token_der, expected_digest, hash_algorithm=hash_algorithm,
                trusted_issuer_cert=trusted_issuer_cert,
                trusted_signer_cert=trusted_signer_cert,
            )
            qualified = True
            basis = "qualified"
        except TimeAnchorError:
            qualified = False
            basis = "self_asserted"

    reason = (
        "existence attested by a qualified timestamp over the record"
        if qualified
        else "existence attested by a self-asserted timestamp over the record; "
             "pin a trusted-list CA to qualify it"
    )
    return ExistenceResult(
        present=True, backend=backend, hash_algorithm=hash_algorithm,
        record_digest=recomputed, attested_time=attested_time, qualified=qualified,
        basis=basis, ok=True, reason=reason,
    )


def attach_existence_proof(
    record: dict[str, Any],
    *,
    tsa_url: str,
    transport: Any = None,
    timeout: float = 10.0,
    hash_algorithm: str = "sha256",
    backend: str = "rfc3161-eidas-qualified",
) -> dict[str, Any]:
    """Return a copy of ``record`` carrying an existence proof from a TSA.

    Timestamps the record digest directly as the RFC 3161 message imprint, so
    the returned token attests this exact record. ``transport`` is injectable
    (``(url, der_request, timeout) -> der_response``) for offline use; the
    default performs the HTTP round trip. The token is verified against the
    digest before it is attached, so a malformed or mis-imprinted reply raises
    rather than producing a bad proof. Requires the ``timeanchor`` extra.
    """
    from vaara.audit import timeanchor as _ta

    rd = existence_record_digest(record)
    digest = bytes.fromhex(rd.split(":", 1)[1])
    request = _ta.build_timestamp_request(digest, hash_algorithm=hash_algorithm)
    tr = transport or _ta._urllib_transport
    response = tr(tsa_url, request, timeout)
    token_der = _ta.extract_token_from_response(response)
    _ta.verify_timestamp_token(token_der, digest, hash_algorithm=hash_algorithm)

    out = dict(record)
    out["existenceProof"] = {
        "backend": backend,
        "hashAlgorithm": hash_algorithm,
        "recordDigest": rd,
        "token": base64.b64encode(token_der).decode("ascii"),
    }
    return out


__all__ = [
    "ExistenceResult",
    "attach_existence_proof",
    "existence_record_digest",
    "verify_existence_proof",
]
