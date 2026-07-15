"""RFC 8785 (JCS) canonicalization and args-commitment helpers.

Internal module. Public surface is in ``vaara.attestation.tool_call_attestation``.

JCS in principle handles floats per ECMAScript Number.toString, but
cross-stack float behaviour is the most common source of signature
drift. Floats are rejected at the boundary so callers must use
integers or decimal strings, which round-trip exactly. This matches
OVERT Protocol Profile 1.0 numeric discipline.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Any, Optional

from vaara.attestation._attest_types import (
    ArgsProjection,
    AttestationError,
)


def _reject_floats(value: Any, path: str = "") -> None:
    if isinstance(value, float):
        raise AttestationError(
            f"IEEE-754 float at {path or '<root>'} is prohibited. Use a "
            f"scaled integer or decimal string instead."
        )
    if isinstance(value, dict):
        for k, v in value.items():
            _reject_floats(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _reject_floats(v, f"{path}[{i}]")


def canonical_json(value: Any) -> bytes:
    """RFC 8785 (JCS) canonical JSON encoding.

    Pinned behaviour for sorted keys, whitespace, Unicode escaping, and
    duplicate-key handling. IEEE-754 floats are rejected at the
    boundary.
    """
    try:
        import rfc8785
    except ImportError as exc:
        raise AttestationError(
            "rfc8785 not installed. Install with: pip install "
            "'vaara[attestation]'"
        ) from exc
    _reject_floats(value)
    return rfc8785.dumps(value)


def now_iso8601() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def new_nonce(num_bytes: int = 18) -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(num_bytes)).rstrip(
        b"="
    ).decode("ascii")


def iso8601_to_epoch(iso: str) -> Optional[float]:
    try:
        if iso.endswith("Z"):
            iso = iso[:-1] + "+00:00"
        return datetime.fromisoformat(iso).timestamp()
    except (ValueError, TypeError):
        return None


def make_args_digest(args_obj: Any) -> ArgsProjection:
    """Build a commitment-only ArgsProjection from a JSON-serialisable args object.

    Computes the JCS-canonical encoding of ``args_obj``, takes its
    sha256, and ships the hash inside a hash-only-identity projection
    of the form ``{"digest": "sha256:<hex>"}``. The original payload
    never leaves the function. The verifier reconstructs the same
    digest from the runtime arguments and rejects on mismatch.

    Replaces the v1 ``ArgsDigest`` extension: per the v2 envelope
    shape, commitment-only audit is expressed as a hash-only-identity
    projection rather than a third commitment kind.
    """
    payload = canonical_json(args_obj)
    args_digest_hex = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    projection_obj = {"digest": args_digest_hex}
    projection_bytes = canonical_json(projection_obj)
    return ArgsProjection(
        projection=projection_bytes.decode("utf-8"),
        projection_digest=f"sha256:{hashlib.sha256(projection_bytes).hexdigest()}",
    )


def make_args_projection(projection_obj: dict[str, Any]) -> ArgsProjection:
    """Build an ArgsProjection (reviewed redaction) from a projection dict.

    The projection is JCS-canonicalised to bytes, decoded to UTF-8 to
    produce the wire-format projection string, and digested.
    """
    payload = canonical_json(projection_obj)
    return ArgsProjection(
        projection=payload.decode("utf-8"),
        projection_digest=f"sha256:{hashlib.sha256(payload).hexdigest()}",
    )
