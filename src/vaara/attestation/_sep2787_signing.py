"""Signing-mode helpers for SEP-2787 attestation envelopes.

Three signing modes matching the v1 SEP-2787 draft: HS256 (HMAC-SHA256
with shared secret), ES256 (ECDSA P-256, raw r||s concatenation, not
DER), RS256 (RSASSA-PKCS1-v1_5 SHA-256). All signatures are encoded as
hexadecimal strings to match the v1 draft's wire format.

This is an internal helper module. The public surface is in
``vaara.attestation.sep2787``.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any


def sign_hs256(payload: bytes, *, shared_secret: bytes) -> str:
    return hmac.new(shared_secret, payload, hashlib.sha256).hexdigest()


def verify_hs256(
    payload: bytes,
    *,
    signature_hex: str,
    shared_secret: bytes,
) -> bool:
    expected = hmac.new(shared_secret, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_hex)


def sign_es256(payload: bytes, *, private_key: Any) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import (
        decode_dss_signature,
    )

    der = private_key.sign(payload, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    return r.to_bytes(32, "big").hex() + s.to_bytes(32, "big").hex()


def verify_es256(
    payload: bytes,
    *,
    signature_hex: str,
    public_key: Any,
) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import (
        encode_dss_signature,
    )

    if len(signature_hex) != 128:
        return False
    try:
        raw = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    r = int.from_bytes(raw[:32], "big")
    s = int.from_bytes(raw[32:], "big")
    der = encode_dss_signature(r, s)
    try:
        public_key.verify(der, payload, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


def sign_rs256(payload: bytes, *, private_key: Any) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    sig = private_key.sign(payload, padding.PKCS1v15(), hashes.SHA256())
    return sig.hex()


def verify_rs256(
    payload: bytes,
    *,
    signature_hex: str,
    public_key: Any,
) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    try:
        sig = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    try:
        public_key.verify(sig, payload, padding.PKCS1v15(), hashes.SHA256())
        return True
    except InvalidSignature:
        return False
