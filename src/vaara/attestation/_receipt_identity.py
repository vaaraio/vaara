"""Resolvable agent identity (did:web) for execution receipts.

Internal module. Public surface is re-exported from
``vaara.attestation.receipt``.

Level-2 pinned-resolvable verification, per
``docs/design/resolvable-agent-identity-spec.md``: given a receipt whose
``receiptAsserted.iss`` is a ``did:web`` identifier and a DID document the
caller already holds, confirm the receipt signature was made by a key the
document lists. No network: the caller supplies the document, so the check
is offline and reproducible. Live resolution (fetching the document over
HTTPS) is a thin wrapper a deployer adds on top and is out of scope here so
the core stays dependency-free and offline-verifiable.

This is purely additive. It composes the unchanged
``verify_receipt_signature`` and touches neither the receipt envelope nor
the canonicalization, so every existing conformance vector verifies exactly
as before. An opaque-string ``iss`` is never failed for lack of a DID:
resolution is opt-in by the verifier.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._receipt_emit import verify_receipt_signature
from vaara.attestation._receipt_types import ExecutionReceipt
from vaara.attestation._attest_types import AttestationError

_DID_WEB_PREFIX = "did:web:"


def _b64u_to_int(value: str) -> int:
    pad = "=" * (-len(value) % 4)
    return int.from_bytes(base64.urlsafe_b64decode(value + pad), "big")


def did_web_to_url(did: str) -> str:
    """Map a ``did:web`` identifier to its DID-document HTTPS URL.

    Follows the W3C did:web method: the first colon-separated label after
    the method is the host (a percent-encoded ``%3A`` carries an optional
    port), remaining labels are path segments. A bare host resolves to
    ``/.well-known/did.json``; a host with a path resolves to
    ``<path>/did.json``.
    """
    if not did.startswith(_DID_WEB_PREFIX):
        raise AttestationError(f"not a did:web identifier: {did!r}")
    ident = did[len(_DID_WEB_PREFIX):]
    if not ident:
        raise AttestationError("empty did:web identifier")
    labels = ident.split(":")
    host = labels[0].replace("%3A", ":").replace("%3a", ":")
    if not host:
        raise AttestationError(f"did:web has no host: {did!r}")
    path = labels[1:]
    if path:
        return "https://" + host + "/" + "/".join(path) + "/did.json"
    return "https://" + host + "/.well-known/did.json"


def _jwk_to_public_key(jwk: dict[str, Any]) -> Any:
    """Convert an EC P-256 or RSA JWK to a cryptography public-key object."""
    from cryptography.hazmat.primitives.asymmetric import ec, rsa

    kty = jwk.get("kty")
    if kty == "EC":
        if jwk.get("crv") != "P-256":
            raise AttestationError(f"unsupported EC curve: {jwk.get('crv')!r}")
        numbers = ec.EllipticCurvePublicNumbers(
            _b64u_to_int(jwk["x"]), _b64u_to_int(jwk["y"]), ec.SECP256R1()
        )
        return numbers.public_key()
    if kty == "RSA":
        return rsa.RSAPublicNumbers(
            _b64u_to_int(jwk["e"]), _b64u_to_int(jwk["n"])
        ).public_key()
    raise AttestationError(f"unsupported JWK kty: {kty!r}")


# A verification key is usable for a receipt only when its key type matches
# the receipt's signature algorithm. HS256 is symmetric and can never appear
# in a published DID document.
_ALG_KTY = {"ES256": "EC", "RS256": "RSA"}


@dataclass(frozen=True)
class IdentityResult:
    """Verdict of a level-2 pinned-resolvable identity check.

    ``resolved`` is True when ``iss`` is a ``did:web`` that matches the
    document's ``id`` and the algorithm is resolvable. ``bound`` is True
    when the receipt signature verifies under a verification key the
    document lists; ``keyid`` names that key. ``reason`` is a short human
    string for logs and audit reports.
    """

    resolved: bool
    bound: bool
    keyid: Optional[str]
    reason: str


def _verification_methods(did_document: dict[str, Any]) -> list[dict[str, Any]]:
    methods = did_document.get("verificationMethod")
    if not isinstance(methods, list):
        return []
    return [m for m in methods if isinstance(m, dict)]


def verify_receipt_identity(
    receipt: ExecutionReceipt,
    did_document: dict[str, Any],
    *,
    expected_keyid: Optional[str] = None,
) -> IdentityResult:
    """Level-2 pinned-resolvable identity check.

    Confirms the receipt's ``iss`` is a ``did:web`` matching
    ``did_document['id']`` and that the receipt signature verifies under a
    verification key the document lists. When ``expected_keyid`` is given,
    only that verification method is tried. HS256 receipts carry no
    resolvable public key and return ``resolved=False``.
    """
    iss = receipt.receipt_asserted.iss
    if not iss.startswith(_DID_WEB_PREFIX):
        return IdentityResult(False, False, None, "iss is not a did:web identifier")
    doc_id = did_document.get("id")
    if doc_id != iss:
        return IdentityResult(
            False, False, None,
            f"document id {doc_id!r} does not match iss {iss!r}",
        )
    want_kty = _ALG_KTY.get(receipt.alg)
    if want_kty is None:
        return IdentityResult(
            False, False, None,
            f"alg {receipt.alg!r} carries no resolvable public key",
        )

    methods = _verification_methods(did_document)
    if not methods:
        return IdentityResult(True, False, None, "document lists no verificationMethod")

    saw_keyid = False
    for method in methods:
        keyid = method.get("id")
        if expected_keyid is not None and keyid != expected_keyid:
            continue
        saw_keyid = True
        jwk = method.get("publicKeyJwk")
        if not isinstance(jwk, dict) or jwk.get("kty") != want_kty:
            continue
        try:
            public_key = _jwk_to_public_key(jwk)
        except (AttestationError, KeyError, ValueError):
            continue
        if verify_receipt_signature(receipt, verifying_material=public_key):
            return IdentityResult(True, True, keyid, "signature bound to a document key")
    if expected_keyid is not None and not saw_keyid:
        return IdentityResult(
            True, False, None, f"no verification method with id {expected_keyid!r}"
        )
    return IdentityResult(True, False, None, "signature does not match any document key")
