"""Signed, regulator-handoff export of the audit trail.

Produces a single zip containing:

- ``trail.jsonl``   — canonical event log (one JSON record per line).
- ``manifest.json`` — schema version, record count, first/last hashes,
  signer public-key fingerprint, vaara version, UTC timestamp, and
  whether the hash chain verified intact at export time.
- ``trail.sig``     — Ed25519 detached signature over
  ``sha256(trail.jsonl_bytes || manifest.json_bytes)``.

A regulator verifies the package with ``vaara.audit.verify.verify_signed``
or the standalone ``scripts/verify_vaara_trail.py`` — no Vaara install
required by the verifier.

Requires ``pip install vaara[export]`` (pulls ``cryptography``). The
core library stays zero-dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False

if TYPE_CHECKING:
    from vaara.audit.trail import AuditTrail

logger = logging.getLogger(__name__)

EXPORT_SCHEMA_VERSION = "1"
_INSTALL_HINT = (
    "Signed export requires the 'cryptography' package. "
    "Install with: pip install 'vaara[export]'"
)


@dataclass(frozen=True)
class ExportResult:
    """Return value of :func:`export_signed`.

    ``path``: path to the written zip.
    ``manifest``: the manifest dict embedded in the zip.
    ``chain_intact``: whether the hash chain verified at export time.
    """
    path: Path
    manifest: dict
    chain_intact: bool


def _require_crypto() -> None:
    if not _HAS_CRYPTO:
        raise ImportError(_INSTALL_HINT)


def _load_private_key(
    key: Union[str, Path, bytes, "Ed25519PrivateKey"],
) -> "Ed25519PrivateKey":
    """Accept a PEM path, PEM bytes, or an already-loaded key object."""
    _require_crypto()
    if isinstance(key, Ed25519PrivateKey):
        return key
    if isinstance(key, (str, Path)):
        key = Path(key).read_bytes()
    if not isinstance(key, (bytes, bytearray)):
        raise TypeError(
            f"signer_key must be a Path, str, bytes, or Ed25519PrivateKey — got {type(key).__name__}"
        )
    loaded = serialization.load_pem_private_key(bytes(key), password=None)
    if not isinstance(loaded, Ed25519PrivateKey):
        raise ValueError("signer_key must be an Ed25519 private key")
    return loaded


def _pubkey_fingerprint(public_key: "Ed25519PublicKey") -> str:
    """SHA-256 fingerprint (hex, first 16 bytes) of the raw public key.

    Matches what ``vaara keygen`` prints so operators can cross-check which
    key signed a given export.
    """
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()[:32]


def _build_manifest(
    trail_jsonl_bytes: bytes,
    record_count: int,
    first_hash: str,
    last_hash: str,
    chain_intact: bool,
    public_key: "Ed25519PublicKey",
    vaara_version: str,
    agent_id: str = "",
) -> dict:
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "vaara_version": vaara_version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "record_count": record_count,
        "first_hash": first_hash,
        "last_hash": last_hash,
        "chain_intact_at_export": chain_intact,
        "trail_sha256": hashlib.sha256(trail_jsonl_bytes).hexdigest(),
        "signer_pubkey_fingerprint": _pubkey_fingerprint(public_key),
        "signature_algorithm": "Ed25519",
        "signature_over": "sha256(trail.jsonl_bytes || manifest.json_bytes_without_signature)",
        "agent_id": agent_id or "",
    }


def export_signed(
    trail: "AuditTrail",
    out_path: Union[str, Path],
    signer_key: Union[str, Path, bytes, "Ed25519PrivateKey"],
    agent_id: str = "",
) -> ExportResult:
    """Produce a signed, regulator-handoff zip of the audit trail.

    Args:
        trail: The :class:`AuditTrail` to export.
        out_path: Where to write the zip file. Parent directory must exist.
        signer_key: Ed25519 private key — accepts a PEM path, PEM bytes, or
            a loaded ``Ed25519PrivateKey`` object. For production, use keys
            from a KMS/HSM/Vault — see ``docs/signing-keys.md``.
        agent_id: Optional scope hint written to the manifest. Does not
            filter the exported records (export is whole-trail).

    Returns:
        ExportResult with the output path, manifest dict, and chain status.

    Raises:
        ImportError: If the ``cryptography`` package is not installed.
        FileNotFoundError: If ``out_path`` parent directory does not exist.
    """
    _require_crypto()

    # Local import avoids a package-level dep on the trail module for tooling
    # that only imports the export helpers.
    from vaara import __version__ as vaara_version
    from vaara.audit.trail import AuditTrail

    if not isinstance(trail, AuditTrail):
        raise TypeError(
            f"trail must be an AuditTrail instance, got {type(trail).__name__}"
        )

    out_path = Path(out_path)
    if not out_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {out_path.parent}"
        )

    private_key = _load_private_key(signer_key)
    public_key = private_key.public_key()

    # Snapshot under the trail's own lock by using its existing exporter.
    # Use a temp path next to the final zip so we can stream large trails
    # without holding everything in memory.
    tmp_jsonl = out_path.with_suffix(out_path.suffix + ".tmp.jsonl")
    try:
        record_count = trail.export_jsonl(tmp_jsonl)
        trail_bytes = tmp_jsonl.read_bytes()
    finally:
        tmp_jsonl.unlink(missing_ok=True)

    # Snapshot chain state and endpoints.
    chain_error = trail.verify_chain()
    chain_intact = chain_error is None
    first_hash = ""
    last_hash = ""
    if record_count > 0:
        # Read the first and last lines from the bytes we just wrote so the
        # manifest matches the exported file exactly, not a live re-read.
        lines = trail_bytes.splitlines()
        if lines:
            first_hash = json.loads(lines[0]).get("record_hash", "")
            last_hash = json.loads(lines[-1]).get("record_hash", "")

    manifest = _build_manifest(
        trail_jsonl_bytes=trail_bytes,
        record_count=record_count,
        first_hash=first_hash,
        last_hash=last_hash,
        chain_intact=chain_intact,
        public_key=public_key,
        vaara_version=vaara_version,
        agent_id=agent_id,
    )
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")

    to_sign = hashlib.sha256(trail_bytes + manifest_bytes).digest()
    signature = private_key.sign(to_sign)

    pubkey_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("trail.jsonl", trail_bytes)
        zf.writestr("manifest.json", manifest_bytes)
        zf.writestr("trail.sig", signature)
        zf.writestr("signer_pubkey.pem", pubkey_pem)

    logger.info(
        "Exported signed trail: %s (%d records, chain_intact=%s, fingerprint=%s)",
        out_path,
        record_count,
        chain_intact,
        manifest["signer_pubkey_fingerprint"],
    )

    return ExportResult(path=out_path, manifest=manifest, chain_intact=chain_intact)
