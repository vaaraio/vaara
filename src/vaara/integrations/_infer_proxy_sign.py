# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Signing key loader for the inference proxy.

Public surface is ``infer_proxy``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("vaara.infer_proxy")

_ISS = "vaara-infer-proxy"


class InferProxyConfigError(RuntimeError):
    """Operator-side inference-proxy config is incomplete or unusable."""


def load_signing_key(
    path: Path, secret_version: Optional[str]
) -> "tuple[Any, str, str]":
    """Load a signing key, returning ``(signing_material, alg, secret_version)``.

    PEM EC P-256 -> ES256, PEM RSA -> RS256, raw bytes -> HS256. Same
    autodetection as the MCP proxy so keys are interchangeable between them.
    """
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePrivateKey
        from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
    except ImportError as exc:
        raise InferProxyConfigError(
            "vaara-infer-proxy requires the attestation extra. "
            "Install with: pip install 'vaara[attestation]'"
        ) from exc

    key_path = Path(path).expanduser()
    if not key_path.is_file():
        raise InferProxyConfigError(f"--signing-key file not found: {key_path}")
    raw = key_path.read_bytes()

    if raw.lstrip().startswith(b"-----BEGIN"):
        try:
            key = serialization.load_pem_private_key(raw, password=None)
        except Exception as exc:
            raise InferProxyConfigError(
                f"--signing-key is not a usable PEM private key: {exc}"
            ) from exc
        if isinstance(key, EllipticCurvePrivateKey):
            if key.curve.name != "secp256r1":
                raise InferProxyConfigError(
                    "--signing-key EC key must use P-256 (secp256r1) for ES256; "
                    f"got {key.curve.name!r}."
                )
            alg = "ES256"
        elif isinstance(key, RSAPrivateKey):
            alg = "RS256"
        else:
            raise InferProxyConfigError(
                "--signing-key must be EC P-256 (ES256) or RSA (RS256). "
                "Generate: openssl ecparam -genkey -name prime256v1 | "
                "openssl pkcs8 -topk8 -nocrypt -out infer_key.pem"
            )
        signing_material: Any = key
        if secret_version is None:
            pub_der = key.public_key().public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            secret_version = hashlib.sha256(pub_der).hexdigest()[:8]
    else:
        if len(raw) < 16:
            raise InferProxyConfigError(
                f"--signing-key raw bytes must be at least 16 bytes; got {len(raw)}"
            )
        alg = "HS256"
        signing_material = raw
        if secret_version is None:
            secret_version = hashlib.sha256(raw).hexdigest()[:8]

    return signing_material, alg, secret_version
