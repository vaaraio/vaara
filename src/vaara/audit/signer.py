# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pluggable signers for the audit-trail export envelope.

Vaara's default signer is Ed25519 via ``cryptography``. v0.14.0 adds an
optional ML-DSA-65 (FIPS 204) signer via the pure-Python
``dilithium-py`` library, installed with ``pip install 'vaara[pq]'``.

Operators with retention horizons that cross the credible quantum
threshold (10+ year audit trails under the EU AI Act's technical-file
retention rules, financial-services or healthcare audit holds) can
swap the signer without changing the rest of the export pipeline. The
manifest carries the algorithm identifier so verifiers can dispatch.

The Signer / Verifier protocols are intentionally narrow — sign a
message, expose the raw public key bytes — so a future HSM-backed or
KMS-backed implementation drops in at the same call sites.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class Signer(Protocol):
    """Sign byte strings and expose the verifier-side public key bytes."""

    algorithm: str

    def sign(self, message: bytes) -> bytes: ...

    def public_key_bytes(self) -> bytes: ...

    def public_key_fingerprint(self) -> str: ...


@runtime_checkable
class Verifier(Protocol):
    """Verify a (message, signature) pair under a fixed public key."""

    algorithm: str

    def verify(self, message: bytes, signature: bytes) -> bool: ...


_ALGORITHM_ED25519 = "Ed25519"
_ALGORITHM_MLDSA65 = "ML-DSA-65"


def _fingerprint(public_key_bytes: bytes) -> str:
    return hashlib.sha256(public_key_bytes).hexdigest()


class Ed25519Signer:
    """Signer wrapping a ``cryptography`` Ed25519 private key."""

    algorithm = _ALGORITHM_ED25519

    def __init__(self, private_key) -> None:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError("Ed25519Signer requires an Ed25519PrivateKey")
        self._key = private_key

    def sign(self, message: bytes) -> bytes:
        return self._key.sign(message)

    def public_key_bytes(self) -> bytes:
        pub = self._key.public_key()
        if hasattr(pub, "public_bytes_raw"):
            return pub.public_bytes_raw()
        from cryptography.hazmat.primitives import serialization
        return pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def public_key_fingerprint(self) -> str:
        return _fingerprint(self.public_key_bytes())


class Ed25519Verifier:
    """Verifier wrapping a 32-byte raw Ed25519 public key."""

    algorithm = _ALGORITHM_ED25519

    def __init__(self, public_key_raw: bytes) -> None:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
        self._pub = Ed25519PublicKey.from_public_bytes(public_key_raw)

    def verify(self, message: bytes, signature: bytes) -> bool:
        from cryptography.exceptions import InvalidSignature
        try:
            self._pub.verify(signature, message)
            return True
        except (InvalidSignature, ValueError):
            return False


class MLDSA65Signer:
    """ML-DSA-65 (FIPS 204) signer via dilithium-py.

    Requires ``pip install 'vaara[pq]'``. Signatures are ~3.3 KB versus
    Ed25519's 64 bytes; the rest of the export envelope is unchanged.
    """

    algorithm = _ALGORITHM_MLDSA65

    def __init__(self, secret_key: bytes) -> None:
        if not isinstance(secret_key, (bytes, bytearray)):
            raise TypeError("MLDSA65Signer requires raw bytes secret key")
        self._sk = bytes(secret_key)

    def sign(self, message: bytes) -> bytes:
        from dilithium_py.ml_dsa import ML_DSA_65
        return ML_DSA_65.sign(self._sk, message)

    def public_key_bytes(self) -> bytes:
        from dilithium_py.ml_dsa import ML_DSA_65
        return ML_DSA_65.pk_from_sk(self._sk)

    def public_key_fingerprint(self) -> str:
        return _fingerprint(self.public_key_bytes())

    @classmethod
    def generate(cls) -> tuple["MLDSA65Signer", bytes]:
        """Generate a fresh keypair. Returns (signer, public_key_bytes)."""
        from dilithium_py.ml_dsa import ML_DSA_65
        pk, sk = ML_DSA_65.keygen()
        return cls(sk), pk


class MLDSA65Verifier:
    """ML-DSA-65 verifier via dilithium-py. Pairs with ``MLDSA65Signer``."""

    algorithm = _ALGORITHM_MLDSA65

    def __init__(self, public_key: bytes) -> None:
        if not isinstance(public_key, (bytes, bytearray)):
            raise TypeError("MLDSA65Verifier requires raw bytes public key")
        self._pk = bytes(public_key)

    def verify(self, message: bytes, signature: bytes) -> bool:
        from dilithium_py.ml_dsa import ML_DSA_65
        return bool(ML_DSA_65.verify(self._pk, message, signature))


def verifier_for(algorithm: str, public_key_bytes: bytes) -> Verifier:
    """Dispatch a verifier instance by manifest ``signature_algorithm``."""
    if algorithm == _ALGORITHM_ED25519:
        return Ed25519Verifier(public_key_bytes)
    if algorithm == _ALGORITHM_MLDSA65:
        return MLDSA65Verifier(public_key_bytes)
    raise ValueError(f"unknown signature_algorithm: {algorithm!r}")
