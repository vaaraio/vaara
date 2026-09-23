# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""COSE receipts for inclusion in Vaara's transparency log.

Two forms live here.

``rfc9942_receipt`` / ``verify_rfc9942_receipt`` produce and check a Receipt
for Inclusion as RFC 9942 (COSE Receipts, formerly
draft-ietf-cose-merkle-tree-proofs) defines it for the RFC9162_SHA256
verifiable data structure: a tagged COSE_Sign1 (tag 18) whose protected
header carries ``alg`` (1) and ``vds`` (395) = 1, whose unprotected header
carries ``vdp`` (396) with the inclusion proofs under label -1, each a
``bstr .cbor [tree_size, leaf_index, [path]]``, and whose payload is the
Merkle root, detached. A verifier recomputes the root from the entry and the
proof, then checks the operator's signature with that root as the payload.
The signature is what makes this a receipt: without it nothing binds the
root to the log operator.

``cose_inclusion_receipt`` / ``verify_cose_inclusion_receipt`` are the older
Vaara form: a text-keyed CBOR map holding the proof and the root. It is not
a COSE message and no RFC 9942 verifier accepts it. Its check is sound only
when the caller supplies an ``expected_root`` obtained independently of the
receipt, which is why that argument is required. ``sign_cose_receipt`` wraps
that map in an untagged COSE_Sign1 over the map bytes; it is likewise not an
RFC 9942 receipt and is kept for existing callers.

Requires ``cbor2`` (the ``attestation`` extra). Signing and signature checks
use ``cryptography`` (ES256, P-256).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vaara.attestation.transparency_log import InclusionProof, verify_inclusion


class CoseReceiptError(RuntimeError):
    """Raised when a COSE receipt cannot be built, decoded, or verified."""


# COSE Verifiable Data Structures registry: 1 = RFC 9162 SHA-256 Merkle tree
# (Certificate Transparency), the exact tree ``transparency_log`` implements.
VDS_RFC9162_SHA256 = 1

# COSE algorithm id for ES256 (ECDSA w/ SHA-256), used by the optional signature.
_COSE_ES256 = -7


def _cbor() -> Any:
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
    """Serialise an inclusion proof as Vaara's keyless CBOR map (not RFC 9942).

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
    return bytes(cbor2.dumps(body, canonical=True))


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
    the receipt carries equals ``expected_root``. The check means something
    only when ``expected_root`` was obtained independently of the receipt,
    for example from a tree head the log operator published. Any
    malformation, mismatch, or tamper returns ``False`` rather than raising.
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
    """Wrap Vaara's CBOR map in an untagged COSE_Sign1 (ES256) over the map bytes.

    Not an RFC 9942 receipt: the signed payload is the map, not the Merkle
    root, and the ``vds`` / ``vdp`` headers are absent. Use
    ``rfc9942_receipt`` for the standard form.

    Produces a COSE_Sign1 ``[protected, unprotected, payload, signature]`` with
    the receipt as a *detached* payload (``payload`` is nil; the verifier is
    handed ``receipt_bytes`` out of band) and a raw R||S ES256 signature over the
    COSE ``Sig_structure``. This does not replace keyless verification; it adds
    an operator-signed check for relying parties that require one.
    """
    cbor2 = _cbor()
    protected = cbor2.dumps({_HDR_ALG: _COSE_ES256}, canonical=True)
    sig_structure = ["Signature1", protected, b"", receipt_bytes]
    raw = _es256_sign(private_key, cbor2.dumps(sig_structure, canonical=True))
    return bytes(cbor2.dumps([protected, {}, None, raw], canonical=True))


def verify_cose_signature(
    cose_sign1_bytes: bytes,
    *,
    receipt_bytes: bytes,
    public_key: Any,
) -> bool:
    """Verify the optional COSE_Sign1 (ES256) over a detached receipt payload."""
    cbor2 = _cbor()
    try:
        protected, _unprotected, _payload, raw = cbor2.loads(cose_sign1_bytes)
        if not isinstance(raw, (bytes, bytearray)):
            return False
        sig_structure = ["Signature1", protected, b"", receipt_bytes]
        return _es256_verify(public_key, bytes(raw),
                             cbor2.dumps(sig_structure, canonical=True))
    except (ValueError, TypeError):
        return False


# RFC 9942 header labels and values (IANA COSE Header Parameters registry).
COSE_SIGN1_TAG = 18
_HDR_ALG = 1
_HDR_KID = 4
_HDR_VDS = 395
_HDR_VDP = 396
_VDP_INCLUSION = -1


def _es256_sign(private_key: Any, to_sign: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, utils

    der = private_key.sign(to_sign, ec.ECDSA(hashes.SHA256()))
    r, s = utils.decode_dss_signature(der)
    return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def _es256_verify(public_key: Any, signature: bytes, to_verify: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, utils

    if len(signature) != 64:
        return False
    der = utils.encode_dss_signature(int.from_bytes(signature[:32], "big"),
                                     int.from_bytes(signature[32:], "big"))
    try:
        public_key.verify(der, to_verify, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        return False
    return True


def rfc9942_receipt(
    *,
    proof: InclusionProof,
    tree_root: bytes,
    private_key: Any,
    kid: bytes | None = None,
) -> bytes:
    """Build an RFC 9942 Receipt for Inclusion (RFC9162_SHA256), signed ES256.

    ``proof`` and ``tree_root`` come from the log (``inclusion_proof`` and the
    root at that tree size). ``private_key`` is the log operator's P-256 key.
    The payload is detached, so the receipt carries no root: a verifier
    derives it from the entry it holds.
    """
    if not isinstance(tree_root, (bytes, bytearray)) or len(tree_root) != 32:
        raise CoseReceiptError("tree_root must be 32 bytes")
    cbor2 = _cbor()
    header: dict[int, Any] = {_HDR_ALG: _COSE_ES256, _HDR_VDS: VDS_RFC9162_SHA256}
    if kid is not None:
        header[_HDR_KID] = bytes(kid)
    protected = cbor2.dumps(header, canonical=True)
    proof_content = cbor2.dumps(
        [int(proof.tree_size), int(proof.log_index), [bytes(s) for s in proof.siblings]],
        canonical=True,
    )
    unprotected = {_HDR_VDP: {_VDP_INCLUSION: [proof_content]}}
    sig_structure = cbor2.dumps(["Signature1", protected, b"", bytes(tree_root)],
                                canonical=True)
    signature = _es256_sign(private_key, sig_structure)
    message = cbor2.CBORTag(COSE_SIGN1_TAG, [protected, unprotected, None, signature])
    return bytes(cbor2.dumps(message, canonical=True))


def verify_rfc9942_receipt(
    receipt_bytes: bytes,
    *,
    leaf_data: bytes,
    public_key: Any,
) -> bool:
    """Check an RFC 9942 Receipt for Inclusion against the entry it covers.

    Follows RFC 9942 Section 5.2: every inclusion proof in ``vdp`` is run over
    ``leaf_data``, each must yield the same root, and the COSE_Sign1 signature
    must verify with that root as the detached payload under ``public_key``.
    Returns ``False`` for anything malformed, a proof that does not describe a
    tree, a leaf index outside its tree, or a signature that fails.
    """
    from vaara.attestation.transparency_log import root_from_inclusion

    cbor2 = _cbor()
    try:
        msg = cbor2.loads(receipt_bytes)
    except Exception:  # cbor2 raises assorted decode errors
        return False
    if not isinstance(msg, cbor2.CBORTag) or msg.tag != COSE_SIGN1_TAG:
        return False
    # cbor2 6 decodes the contents of a tag immutably (tuple, frozendict) and
    # earlier versions mutably (list, dict), so the checks go by ABC.
    parts = msg.value
    if not isinstance(parts, Sequence) or isinstance(parts, (bytes, str)) or len(parts) != 4:
        return False
    protected_bytes, unprotected, payload, signature = parts
    if payload is not None or not isinstance(signature, bytes):
        return False
    try:
        protected = cbor2.loads(protected_bytes)
    except Exception:  # cbor2 raises assorted decode errors
        return False
    if not isinstance(protected, Mapping) or not isinstance(unprotected, Mapping):
        return False
    if protected.get(_HDR_ALG) != _COSE_ES256:
        return False
    if protected.get(_HDR_VDS) != VDS_RFC9162_SHA256:
        return False
    vdp = unprotected.get(_HDR_VDP)
    proofs = vdp.get(_VDP_INCLUSION) if isinstance(vdp, Mapping) else None
    if not isinstance(proofs, (list, tuple)) or not proofs:
        return False

    root: bytes | None = None
    for encoded in proofs:
        if not isinstance(encoded, bytes):
            return False
        try:
            tree_size, leaf_index, path = cbor2.loads(encoded)
            proof = InclusionProof(
                log_index=int(leaf_index),
                tree_size=int(tree_size),
                siblings=tuple(bytes(p) for p in path),
            )
        except Exception:  # malformed proof content of any shape
            return False
        this_root = root_from_inclusion(leaf_data=bytes(leaf_data), proof=proof)
        if this_root is None or (root is not None and this_root != root):
            return False
        root = this_root

    if root is None:
        return False
    sig_structure = cbor2.dumps(["Signature1", protected_bytes, b"", root],
                                canonical=True)
    return _es256_verify(public_key, signature, sig_structure)
