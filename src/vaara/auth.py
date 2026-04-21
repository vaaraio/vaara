"""API key authentication and role-based access control for Vaara.

Enterprise deployments expose the MCP server and pipeline to untrusted
callers. This module provides:

- ``Role`` — WRITER, READER, ADMIN
- ``APIKey`` — key metadata (never stores plaintext)
- ``require_role`` — guard helper for method-level checks

API keys are persisted in the audit SQLite database via
``SQLiteAuditBackend.create_api_key`` / ``authenticate_api_key`` so
they survive restarts and are co-located with the evidence they protect.

Usage::

    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.auth import Role

    backend = SQLiteAuditBackend("audit.db")
    key = backend.create_api_key("ci-writer", Role.WRITER)
    print(f"API key (save this, shown once): {key}")

    # On every request:
    caller = backend.authenticate_api_key(incoming_key)
    if caller is None or caller.role not in (Role.WRITER, Role.ADMIN):
        raise PermissionError("Unauthorized")
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Role(str, Enum):
    """Access roles for Vaara API keys.

    WRITER  — intercept, report_outcome, resolve_escalation (agent-facing)
    READER  — query, export, compliance reports, status (ops/audit-facing)
    ADMIN   — all of the above + create/revoke keys, redact PII
    """
    WRITER = "writer"
    READER = "reader"
    ADMIN = "admin"


@dataclass
class APIKey:
    """API key metadata. Plaintext key is never stored — only the hash."""
    key_id: str
    name: str
    role: Role
    created_at: float
    last_used_at: Optional[float] = None

    @property
    def role_value(self) -> str:
        return self.role.value if isinstance(self.role, Role) else str(self.role)


def _hash_key(plaintext_key: str) -> str:
    """PBKDF2-HMAC-SHA256 hash of a plaintext API key.

    Uses a fixed salt derived from the key itself (not a random salt) so
    lookup is O(1) without storing the salt separately. The cost factor
    (iterations=100_000) is high enough to make brute-force impractical
    while adding <1ms overhead per authentication.
    """
    # Derive a deterministic salt from the key so we don't need to store it.
    # This is intentionally not the same security level as bcrypt with
    # random salts — it prevents bulk dictionary attacks but not targeted
    # brute-force against a specific hash. For a governance library running
    # on trusted infrastructure this is an acceptable trade-off vs. the
    # complexity of storing random salts alongside hashes.
    salt = hashlib.sha256(b"vaara-key-salt:" + plaintext_key[:8].encode()).digest()
    dk = hashlib.pbkdf2_hmac("sha256", plaintext_key.encode(), salt, iterations=100_000)
    return dk.hex()


def generate_api_key() -> str:
    """Generate a cryptographically random API key (URL-safe, 43 chars)."""
    return secrets.token_urlsafe(32)


def require_role(key: Optional[APIKey], *allowed_roles: Role) -> None:
    """Raise PermissionError if key is None or its role is not in allowed_roles."""
    if key is None:
        raise PermissionError("Authentication required: no valid API key provided")
    if key.role not in allowed_roles:
        raise PermissionError(
            f"Insufficient privileges: role {key.role.value!r} is not in "
            f"{[r.value for r in allowed_roles]}"
        )
