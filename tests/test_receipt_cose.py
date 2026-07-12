"""Keyless COSE Receipt emission over the RFC 6962 transparency log.

Verifies the recompute-from-bytes property (a Vaara COSE receipt checks out with
no key), tamper-evidence, and the optional operator-signed COSE_Sign1 wrapper.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cbor2")

from vaara.attestation._receipt_cose import (  # noqa: E402
    cose_inclusion_receipt,
    sign_cose_receipt,
    verify_cose_inclusion_receipt,
    verify_cose_signature,
)
from vaara.attestation.transparency_log import InProcessTransparencyLog  # noqa: E402


def _log(n: int) -> InProcessTransparencyLog:
    log = InProcessTransparencyLog()
    for i in range(n):
        log.append(f"leaf-{i}".encode())
    return log


def test_keyless_receipt_recomputes_root() -> None:
    log = _log(5)
    proof = log.inclusion_proof(2)
    root = log.root_hash
    receipt = cose_inclusion_receipt(leaf_data=b"leaf-2", proof=proof, tree_root=root)
    assert verify_cose_inclusion_receipt(
        receipt, leaf_data=b"leaf-2", expected_root=root
    )


def test_wrong_leaf_fails() -> None:
    log = _log(5)
    proof = log.inclusion_proof(2)
    root = log.root_hash
    receipt = cose_inclusion_receipt(leaf_data=b"leaf-2", proof=proof, tree_root=root)
    assert not verify_cose_inclusion_receipt(
        receipt, leaf_data=b"leaf-WRONG", expected_root=root
    )


def test_tampered_cbor_fails() -> None:
    log = _log(4)
    proof = log.inclusion_proof(1)
    root = log.root_hash
    receipt = bytearray(
        cose_inclusion_receipt(leaf_data=b"leaf-1", proof=proof, tree_root=root)
    )
    receipt[-1] ^= 0x01
    assert not verify_cose_inclusion_receipt(
        bytes(receipt), leaf_data=b"leaf-1", expected_root=root
    )


def test_wrong_root_fails() -> None:
    log = _log(6)
    proof = log.inclusion_proof(3)
    root = log.root_hash
    receipt = cose_inclusion_receipt(leaf_data=b"leaf-3", proof=proof, tree_root=root)
    assert not verify_cose_inclusion_receipt(
        receipt, leaf_data=b"leaf-3", expected_root=bytes(32)
    )


def test_optional_cose_signature_roundtrip() -> None:
    from cryptography.hazmat.primitives.asymmetric import ec

    log = _log(3)
    proof = log.inclusion_proof(0)
    root = log.root_hash
    receipt = cose_inclusion_receipt(leaf_data=b"leaf-0", proof=proof, tree_root=root)
    key = ec.generate_private_key(ec.SECP256R1())
    sign1 = sign_cose_receipt(receipt, private_key=key)
    assert verify_cose_signature(
        sign1, receipt_bytes=receipt, public_key=key.public_key()
    )
    # A different key must not verify.
    other = ec.generate_private_key(ec.SECP256R1())
    assert not verify_cose_signature(
        sign1, receipt_bytes=receipt, public_key=other.public_key()
    )
    # Keyless verification holds independently of the signature.
    assert verify_cose_inclusion_receipt(
        receipt, leaf_data=b"leaf-0", expected_root=root
    )
