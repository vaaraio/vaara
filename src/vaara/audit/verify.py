"""Verifier for signed audit-trail exports produced by :mod:`vaara.audit.export`.

This module is intentionally self-contained — it imports only the standard
library plus ``cryptography``. No other Vaara modules are imported. A
regulator can copy ``scripts/verify_vaara_trail.py`` (which wraps this
logic) to any machine and run it without installing Vaara.
"""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey as _PK

logger = logging.getLogger(__name__)

_REQUIRED_FILES = {"trail.jsonl", "manifest.json", "trail.sig"}
_INSTALL_HINT = (
    "Signed-trail verification requires the 'cryptography' package. "
    "Install with: pip install 'vaara[export]' (or 'pip install cryptography' "
    "if using the standalone verifier script)."
)


@dataclass
class VerifyResult:
    """Return value of :func:`verify_signed`.

    ``ok``: whether every check passed.
    ``errors``: list of human-readable error strings (empty if ``ok``).
    ``manifest``: the manifest dict from the zip (``None`` if unreadable).
    """
    ok: bool
    errors: list[str] = field(default_factory=list)
    manifest: Optional[dict] = None


def _require_crypto() -> None:
    if not _HAS_CRYPTO:
        raise ImportError(_INSTALL_HINT)


def _load_public_key(
    key: Union[str, Path, bytes, "_PK"],
) -> "Ed25519PublicKey":
    _require_crypto()
    if isinstance(key, Ed25519PublicKey):
        return key
    if isinstance(key, (str, Path)):
        key = Path(key).read_bytes()
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError(
            f"public_key must be a Path, str, bytes, or Ed25519PublicKey — got {type(key).__name__}"
        )
    loaded = serialization.load_pem_public_key(bytes(key))
    if not isinstance(loaded, Ed25519PublicKey):
        raise ValueError("public_key must be an Ed25519 public key")
    return loaded


def _verify_chain_bytes(trail_bytes: bytes) -> Optional[str]:
    """Re-verify the hash chain from the exported JSONL bytes.

    Returns None if intact, else a human-readable error string. Uses only
    stdlib — mirrors :meth:`AuditTrail.verify_chain` without requiring
    vaara imports, so the standalone verifier can run it.
    """
    prev_hash = ""
    line_iter = (ln for ln in trail_bytes.splitlines() if ln.strip())
    for idx, raw in enumerate(line_iter):
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as e:
            return f"record {idx}: malformed JSON ({e})"
        if rec.get("previous_hash", "") != prev_hash:
            return (
                f"record {idx}: previous_hash mismatch "
                f"(expected {prev_hash!r}, got {rec.get('previous_hash')!r})"
            )
        # Recompute hash over the canonical content.
        content = {
            "record_id": rec.get("record_id"),
            "action_id": rec.get("action_id"),
            "event_type": rec.get("event_type"),
            "timestamp": rec.get("timestamp"),
            "agent_id": rec.get("agent_id"),
            "tool_name": rec.get("tool_name"),
            "data": rec.get("data", {}),
            "regulatory_articles": rec.get("regulatory_articles", []),
            "previous_hash": prev_hash,
        }
        canonical = json.dumps(
            content, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        if rec.get("record_hash") != expected:
            return f"record {idx}: hash mismatch (record tampered)"
        prev_hash = rec["record_hash"]
    return None


def verify_signed(
    zip_path: Union[str, Path],
    public_key: Optional[Union[str, Path, bytes, "_PK"]] = None,
) -> VerifyResult:
    """Verify a signed audit-trail export.

    Checks, in order:

    1. Zip contains the required files (``trail.jsonl``, ``manifest.json``,
       ``trail.sig``, and optionally ``signer_pubkey.pem``).
    2. Manifest parses as JSON and has the expected schema version.
    3. ``manifest.trail_sha256`` matches the actual SHA-256 of
       ``trail.jsonl``.
    4. Ed25519 signature over ``sha256(trail.jsonl || manifest.json)``
       verifies under ``public_key`` (if given) or the embedded
       ``signer_pubkey.pem``.
    5. The hash chain re-verifies from the exported bytes.
    6. Manifest endpoints (``first_hash``, ``last_hash``, ``record_count``)
       match the exported JSONL.

    Args:
        zip_path: Path to the zip produced by :func:`export_signed`.
        public_key: Ed25519 public key — Path, PEM bytes, or loaded key
            object. If omitted, uses ``signer_pubkey.pem`` from inside
            the zip (convenient for local verification; for production,
            always pass a trusted public key out-of-band).

    Returns:
        VerifyResult. On success, ``ok=True`` and ``errors`` is empty.
    """
    _require_crypto()

    zip_path = Path(zip_path)
    errors: list[str] = []
    manifest: Optional[dict] = None

    if not zip_path.exists():
        return VerifyResult(ok=False, errors=[f"file not found: {zip_path}"])

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = set(zf.namelist())
            missing = _REQUIRED_FILES - names
            if missing:
                return VerifyResult(
                    ok=False,
                    errors=[f"missing files in zip: {sorted(missing)}"],
                )
            trail_bytes = zf.read("trail.jsonl")
            manifest_bytes = zf.read("manifest.json")
            signature = zf.read("trail.sig")
            embedded_pubkey = (
                zf.read("signer_pubkey.pem")
                if "signer_pubkey.pem" in names
                else None
            )
    except zipfile.BadZipFile as e:
        return VerifyResult(ok=False, errors=[f"corrupt zip: {e}"])

    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as e:
        return VerifyResult(ok=False, errors=[f"manifest.json is not valid JSON: {e}"])

    if manifest.get("schema_version") != "1":
        errors.append(
            f"unsupported schema_version: {manifest.get('schema_version')!r}"
        )

    actual_sha = hashlib.sha256(trail_bytes).hexdigest()
    if manifest.get("trail_sha256") != actual_sha:
        errors.append(
            "trail.jsonl SHA-256 does not match manifest.trail_sha256 "
            "(trail was tampered after export)"
        )

    if public_key is None:
        if embedded_pubkey is None:
            return VerifyResult(
                ok=False,
                errors=errors + ["no public_key argument and no signer_pubkey.pem in zip"],
                manifest=manifest,
            )
        pk = _load_public_key(embedded_pubkey)
    else:
        pk = _load_public_key(public_key)

    to_verify = hashlib.sha256(trail_bytes + manifest_bytes).digest()
    try:
        pk.verify(signature, to_verify)
    except InvalidSignature:
        errors.append(
            "Ed25519 signature did not verify "
            "(wrong public key, tampered manifest, or tampered trail)"
        )

    chain_error = _verify_chain_bytes(trail_bytes)
    if chain_error is not None:
        errors.append(f"hash chain check failed: {chain_error}")

    lines = [ln for ln in trail_bytes.splitlines() if ln.strip()]
    if manifest.get("record_count") != len(lines):
        errors.append(
            f"record_count mismatch: manifest={manifest.get('record_count')}, "
            f"actual={len(lines)}"
        )
    if lines:
        try:
            first = json.loads(lines[0]).get("record_hash", "")
            last = json.loads(lines[-1]).get("record_hash", "")
            if manifest.get("first_hash") != first:
                errors.append("first_hash in manifest does not match first record")
            if manifest.get("last_hash") != last:
                errors.append("last_hash in manifest does not match last record")
        except json.JSONDecodeError:
            errors.append("could not parse first/last trail record")

    return VerifyResult(ok=not errors, errors=errors, manifest=manifest)
