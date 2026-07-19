# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""COSE Receipt (SCITT-compatible) emission over Vaara's transparency log.

A SCITT / COSE *Receipt* (draft-ietf-cose-merkle-tree-proofs) is a cryptographic
inclusion proof against an append-only Merkle log. Vaara already ships exactly
that log: ``transparency_log`` is an RFC 6962 / RFC 9162 SHA-256 tree with a
recompute-from-bytes ``verify_inclusion``. So this module is a thin, additive
wire-shaping layer, not a new mechanism: it serialises an existing inclusion
proof into the COSE Receipt CBOR shape so a SCITT-aware relying party can consume
Vaara evidence, without changing the receipt format or adding a second Merkle
implementation.

The design is **keyless first**. draft-ietf-cose-merkle-tree-proofs registers the
RFC 9162 SHA-256 tree as verifiable-data-structure (VDS) type ``1`` and lets a
verifier recompute the tree head from (leaf, inclusion path) alone. That is
Vaara's whole posture, so the default receipt carries the proof and *no*
signature: verification is ``verify_inclusion`` and nothing else. A COSE_Sign1
signature is *optional* (``sign_cose_receipt``) for deployments that need the
strict operator-signed SCITT wire form; it never replaces the keyless
recomputation, it only adds a second, independent check on top.

Scope honesty: the CBOR here is modelled on draft-ietf-cose-merkle-tree-proofs
(VDS 1; inclusion proof = tree-size, leaf-index, path) and reuses Vaara's own
RFC 6962 Merkle maths, so a Vaara COSE receipt and a native Vaara inclusion proof
are byte-for-byte the same computation. Byte-exact interop against a third-party
reference SCITT verifier is the next validation step and is NOT asserted here.

Requires ``cbor2`` (the ``attestation`` extra). Optional signing reuses the
``cryptography`` ES256 path already in the receipt stack.
"""

from __future__ import annotations

from typing import Any

from vaara.attestation.transparency_log import InclusionProof, verify_inclusion


class CoseReceiptError(RuntimeError):
    """Raised when a COSE receipt cannot be built, decoded, or verified."""


# COSE Verifiable Data Structures registry: 1 = RFC 9162 SHA-256 Merkle tree
# (Certificate Transparency), the exact tree ``transparency_log`` implements.
VDS_RFC9162_SHA256 = 1

# COSE algorithm id for ES256 (ECDSA w/ SHA-256), used by the optional signature.
_COSE_ES256 = -7


def _cbor():
    try:
        import cbor2
    except ImportError as exc:  # pragma: no cover - exercised via the install hint
        raise CoseReceiptError(
            "cbor2 not installed. Install with: pip install 'vaara[attestation]'"
        ) from exc
    return cbor2


def cose_inclusion_receipt(
    *,
    leaf_data: bytes,
    proof: InclusionProof,
    tree_root: bytes,
) -> bytes:
    """Serialise an inclusion proof as a keyless COSE Receipt (canonical CBOR).

    ``leaf_data`` is the exact bytes appended to the log for this entry; the
    verifier re-hashes them, and the leaf hash is deliberately *not* carried, so
    the receipt cannot lie about which bytes it proves. ``proof`` is a
    ``transparency_log.InclusionProof``; ``tree_root`` is the head the proof
    recomputes to (the detached COSE payload).

    No signature: the receipt is verified purely by recomputation (see
    ``verify_cose_inclusion_receipt``).
    """
    if not isinstance(leaf_data, (bytes, bytearray)):
        raise CoseReceiptError("leaf_data must be bytes")
    if not isinstance(tree_root, (bytes, bytearray)):
        raise CoseReceiptError("tree_root must be bytes")
    cbor2 = _cbor()
    body = {
        "vds": VDS_RFC9162_SHA256,
        "inclusion": {
            "tree_size": int(proof.tree_size),
            "leaf_index": int(proof.log_index),
            "path": [bytes(s) for s in proof.siblings],
        },
        "tree_root": bytes(tree_root),
    }
    return cbor2.dumps(body, canonical=True)


def _decode_body(receipt_bytes: bytes) -> dict[str, Any]:
    cbor2 = _cbor()
    try:
        body = cbor2.loads(receipt_bytes)
    except Exception as exc:  # cbor2 raises assorted decode errors
        raise CoseReceiptError(f"malformed CBOR: {exc}") from exc
    if not isinstance(body, dict) or "inclusion" not in body:
        raise CoseReceiptError("not a Vaara COSE inclusion receipt")
    return body


def verify_cose_inclusion_receipt(
    receipt_bytes: bytes,
    *,
    leaf_data: bytes,
    expected_root: bytes,
) -> bool:
    """Keyless verification: recompute the head from (leaf, proof) and compare.

    Returns ``True`` iff the receipt's inclusion proof recomputes to
    ``expected_root`` over ``leaf_data`` using the RFC 6962 maths, *and* the head
    the receipt carries equals ``expected_root``. No key, no operator to trust.
    Any malformation, mismatch, or tamper returns ``False`` rather than raising.
    """
    try:
        body = _decode_body(receipt_bytes)
    except CoseReceiptError:
        return False
    incl = body.get("inclusion")
    if not isinstance(incl, dict):
        return False
    try:
        proof = InclusionProof(
            log_index=int(incl["leaf_index"]),
            tree_size=int(incl["tree_size"]),
            siblings=tuple(bytes(s) for s in incl["path"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    if bytes(body.get("tree_root", b"")) != bytes(expected_root):
        return False
    return verify_inclusion(
        leaf_data=bytes(leaf_data),
        proof=proof,
        expected_root=bytes(expected_root),
    )


def sign_cose_receipt(receipt_bytes: bytes, *, private_key: Any) -> bytes:
    """Wrap a keyless receipt in a COSE_Sign1 (ES256) for strict SCITT interop.

    Produces a COSE_Sign1 ``[protected, unprotected, payload, signature]`` with
    the receipt as a *detached* payload (``payload`` is nil; the verifier is
    handed ``receipt_bytes`` out of band) and a raw R||S ES256 signature over the
    COSE ``Sig_structure``. This does not replace keyless verification; it adds
    an operator-signed check for relying parties that require one.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, utils

    cbor2 = _cbor()
    protected = cbor2.dumps({1: _COSE_ES256}, canonical=True)  # {alg: ES256}
    sig_structure = ["Signature1", protected, b"", receipt_bytes]
    to_sign = cbor2.dumps(sig_structure, canonical=True)
    der = private_key.sign(to_sign, ec.ECDSA(hashes.SHA256()))
    r, s = utils.decode_dss_signature(der)
    raw = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return cbor2.dumps([protected, {}, None, raw], canonical=True)


def verify_cose_signature(
    cose_sign1_bytes: bytes,
    *,
    receipt_bytes: bytes,
    public_key: Any,
) -> bool:
    """Verify the optional COSE_Sign1 (ES256) over a detached receipt payload."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, utils

    cbor2 = _cbor()
    try:
        protected, _unprotected, _payload, raw = cbor2.loads(cose_sign1_bytes)
        if not isinstance(raw, (bytes, bytearray)) or len(raw) != 64:
            return False
        r = int.from_bytes(raw[:32], "big")
        s = int.from_bytes(raw[32:], "big")
        der = utils.encode_dss_signature(r, s)
        sig_structure = ["Signature1", protected, b"", receipt_bytes]
        to_sign = cbor2.dumps(sig_structure, canonical=True)
        public_key.verify(der, to_sign, ec.ECDSA(hashes.SHA256()))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False
