"""W3C Verifiable Credential serialization for execution receipts (opt-in).

Internal module. Public surface re-exports ``receipt_to_vc`` and
``receipt_from_vc`` from ``vaara.attestation.receipt``.

A consumer who standardizes on W3C VCs can ingest Vaara execution receipts
without a custom parser. The VC is a lossless *view* of the native receipt,
not a second trust surface:

- ``credentialSubject`` carries the native receipt blocks verbatim (the same
  dict ``ExecutionReceipt.to_dict()`` produces, minus the detached signature).
- ``proof.proofValue`` carries the existing SEP-2787 detached signature, typed
  so a verifier knows it is a detached JCS signature over the receipt blocks,
  not a standard VC Data Integrity proof.
- ``receipt_from_vc`` recovers the exact ``ExecutionReceipt`` and verification
  routes through the unchanged ``verify_receipt_signature`` stack. The signed
  bytes stay the JCS-canonical receipt; no new crypto, no re-canonicalization.

The round-trip invariant the design hangs on::

    receipt_from_vc(receipt_to_vc(r)) == r

We do not claim VC Data Integrity / JWT-VC conformance for the proof; we claim
a VCDM 2.0 envelope whose proof is the receipt's own detached signature. A true
``DataIntegrityProof`` variant is a separate, larger track (out of scope).

See ``docs/design/w3c-vc-receipt-spec.md``.
"""

from __future__ import annotations

from typing import Any

from vaara.attestation._receipt_types import ExecutionReceipt, receipt_from_dict
from vaara.attestation._attest_types import AttestationError

# VCDM 2.0 base context plus the Vaara term-definition document. Both are
# declared in the credential; verification never fetches them (the trust
# decision routes through the receipt signature), so this is offline-safe.
VC_V2_CONTEXT = "https://www.w3.org/ns/credentials/v2"
VAARA_RECEIPT_CONTEXT_URL = "https://vaara.io/credentials/execution-receipt/v1"

PROOF_TYPE = "VaaraSep2787DetachedSignature2026"
CREDENTIAL_TYPE = "VaaraExecutionReceipt"

# Vendored term-definition document for VAARA_RECEIPT_CONTEXT_URL. Held as a
# module constant rather than a fetched URL so that any verifier resolving the
# context can do so entirely offline.
VAARA_RECEIPT_CONTEXT_DOCUMENT: dict[str, Any] = {
    "@context": {
        "@protected": True,
        "VaaraExecutionReceipt": VAARA_RECEIPT_CONTEXT_URL + "#VaaraExecutionReceipt",
        "version": "https://schema.org/version",
        "alg": VAARA_RECEIPT_CONTEXT_URL + "#alg",
        "backLink": VAARA_RECEIPT_CONTEXT_URL + "#backLink",
        "outcomeDerived": VAARA_RECEIPT_CONTEXT_URL + "#outcomeDerived",
        "receiptAsserted": VAARA_RECEIPT_CONTEXT_URL + "#receiptAsserted",
    }
}


def load_receipt_context() -> dict[str, Any]:
    """Return the vendored Vaara receipt JSON-LD context document.

    Offline only; no network resolution. Provided so a JSON-LD consumer can
    resolve ``VAARA_RECEIPT_CONTEXT_URL`` from the package itself.
    """
    return VAARA_RECEIPT_CONTEXT_DOCUMENT


def receipt_to_vc(receipt: ExecutionReceipt) -> dict[str, Any]:
    """Wrap an ``ExecutionReceipt`` as a W3C Verifiable Credential.

    The credential is a lossless view: ``credentialSubject`` holds the native
    receipt blocks verbatim and ``proof.proofValue`` holds the receipt's
    existing detached signature. No re-signing, no re-canonicalization.

    Args:
        receipt: The native execution receipt to present as a VC.

    Returns:
        A VCDM 2.0 credential dict.
    """
    blocks = receipt.to_dict()
    signature = blocks.pop("signature")
    return {
        "@context": [VC_V2_CONTEXT, VAARA_RECEIPT_CONTEXT_URL],
        "type": ["VerifiableCredential", CREDENTIAL_TYPE],
        "issuer": receipt.receipt_asserted.iss,
        "validFrom": receipt.receipt_asserted.iat,
        # version, alg, backLink, outcomeDerived, receiptAsserted (verbatim)
        "credentialSubject": blocks,
        "proof": {
            "type": PROOF_TYPE,
            "cryptosuite": f"jcs-{receipt.alg.lower()}",
            "verificationMethod": receipt.receipt_asserted.secret_version,
            "proofValue": signature,
        },
    }


def receipt_from_vc(vc: dict[str, Any]) -> ExecutionReceipt:
    """Recover the exact ``ExecutionReceipt`` from its VC wrapper.

    Inverse of :func:`receipt_to_vc`. The recovered receipt is byte-for-byte
    the payload that was signed at emit time, so it verifies with the
    unchanged ``verify_receipt_signature`` stack.

    Args:
        vc: A credential previously produced by :func:`receipt_to_vc`.

    Returns:
        The reconstructed execution receipt.

    Raises:
        AttestationError: If the VC is missing ``credentialSubject`` or a
            ``proof`` with a ``proofValue``.
    """
    if not isinstance(vc, dict):
        raise AttestationError("VC must be a JSON object")
    subject = vc.get("credentialSubject")
    proof = vc.get("proof")
    if not isinstance(subject, dict):
        raise AttestationError("VC missing credentialSubject object")
    if not isinstance(proof, dict) or "proofValue" not in proof:
        raise AttestationError("VC missing proof with proofValue")
    # Rebuild the native to_dict() shape: receipt blocks + detached signature.
    blocks = dict(subject)
    blocks["signature"] = proof["proofValue"]
    return receipt_from_dict(blocks)
