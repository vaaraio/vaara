# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Encrypted retention archives with a public, signed seal.

    rotate --encrypt-to R
      export_signed -> verify_signed          (as without encryption)
      -> encrypt the zip to recipient R       archive.zip.sealed
      -> sign a public seal                   archive.zip.seal.json
           zip sha256, ciphertext sha256, record count,
           first and last chain hash, recipient key fingerprint
      -> verify the seal, remove the plaintext zip, then purge

    verify-seal   anyone, no key: the signature, and the ciphertext's digest
    unseal        the recipient: an access record first, then decrypt, then
                  check the zip against the seal and verify it as an export

A retention archive has two readers with different rights. A data protection
officer or a court holds the decryption key and reads the records. Anyone
else, an auditor checking that retention happened or a counterparty
checking continuity, needs to know the archive exists, is intact, and ends
where the live trail's seam begins, without reading a record. The seal is
for them: it carries the chain's first and last hash and the record count in
the clear, signed by the trail's own key, and the ciphertext's digest binds
it to the file.

Cipher: an ephemeral X25519 key agreed with the recipient's key, HKDF-SHA256
to a 256-bit key, and AES-256-GCM over 1 MiB chunks. Each chunk's nonce is a
random 7-byte prefix, a 4-byte big-endian counter and a final-chunk flag
byte, the STREAM construction, so chunks cannot be reordered, dropped or
truncated without the tag failing. The file header is authenticated data on
every chunk. Requires the ``cryptography`` package (the ``export`` extra).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Union

SEAL_FORMAT = "vaara.sealed-archive/v0"
CIPHER = "X25519-HKDF-SHA256-AES-256-GCM-STREAM-1MiB"
MAGIC = b"VAARA-SEALED-v0\n"
CHUNK = 1 << 20
_HKDF_INFO = b"vaara sealed archive v0"
_TAG = 16
_MAX_HEADER = 4096
_MAX_CHUNKS = 0xFFFFFFFF


class SealError(Exception):
    """A seal could not be made, read, or did not match."""


def _crypto():
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric.x25519 import (
            X25519PrivateKey, X25519PublicKey,
        )
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    except ImportError as exc:  # pragma: no cover - extra not installed
        raise ImportError(
            "sealed archives need the cryptography package: pip install 'vaara[export]'"
        ) from exc
    return hashes, serialization, X25519PrivateKey, X25519PublicKey, AESGCM, HKDF


def _raw_public(pub: Any) -> bytes:
    _, serialization, *_ = _crypto()
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def fingerprint(public_raw: bytes) -> str:
    """SHA-256 of a raw public key, the first 32 hex characters."""
    return hashlib.sha256(public_raw).hexdigest()[:32]


def generate_recipient_key(private_out: Union[str, Path],
                           public_out: Union[str, Path]) -> str:
    """Write a new X25519 key pair as PEM files; return its fingerprint.

    The private key file is created with mode 0600 and never overwritten.
    """
    _, serialization, X25519PrivateKey, *_ = _crypto()
    key = X25519PrivateKey.generate()
    private_pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption())
    public_pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    fd = os.open(private_out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(private_pem)
    Path(public_out).write_bytes(public_pem)
    return fingerprint(_raw_public(key.public_key()))


def _load_recipient_public(pem: Union[str, Path, bytes]):
    _, serialization, _, X25519PublicKey, *_ = _crypto()
    data = Path(pem).read_bytes() if isinstance(pem, (str, Path)) else bytes(pem)
    try:
        key = serialization.load_pem_public_key(data)
    except ValueError as exc:
        raise SealError(f"recipient key is not a PEM public key: {exc}") from exc
    if not isinstance(key, X25519PublicKey):
        raise SealError("recipient key must be an X25519 public key")
    return key


def _load_recipient_private(pem: Union[str, Path, bytes]):
    _, serialization, X25519PrivateKey, *_ = _crypto()
    data = Path(pem).read_bytes() if isinstance(pem, (str, Path)) else bytes(pem)
    try:
        key = serialization.load_pem_private_key(data, password=None)
    except (ValueError, TypeError) as exc:
        raise SealError(f"recipient key is not a PEM private key: {exc}") from exc
    if not isinstance(key, X25519PrivateKey):
        raise SealError("recipient key must be an X25519 private key")
    return key


def _derive(shared: bytes, eph_raw: bytes, rcpt_raw: bytes) -> bytes:
    hashes, _, _, _, _, HKDF = _crypto()
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=eph_raw + rcpt_raw,
                info=_HKDF_INFO).derive(shared)


def _nonce(prefix: bytes, counter: int, last: bool) -> bytes:
    return prefix + struct.pack(">I", counter) + (b"\x01" if last else b"\x00")


def _sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def encrypt_file(plain: Path, sealed: Path, recipient_public) -> str:
    """Encrypt ``plain`` to ``sealed`` for ``recipient_public``; return the
    recipient fingerprint. The output is written whole or not at all."""
    _, _, X25519PrivateKey, _, AESGCM, _ = _crypto()
    eph = X25519PrivateKey.generate()
    eph_raw = _raw_public(eph.public_key())
    rcpt_raw = _raw_public(recipient_public)
    aead = AESGCM(_derive(eph.exchange(recipient_public), eph_raw, rcpt_raw))
    prefix = os.urandom(7)
    header = json.dumps({
        "cipher": CIPHER,
        "ephemeral_public": base64.b64encode(eph_raw).decode(),
        "nonce_prefix": base64.b64encode(prefix).decode(),
        "recipient_fingerprint": fingerprint(rcpt_raw),
    }, sort_keys=True, separators=(",", ":")).encode()
    tmp = sealed.with_name(sealed.name + f".{os.getpid()}.tmp")
    try:
        with open(plain, "rb") as src, open(tmp, "wb") as out:
            out.write(MAGIC + struct.pack(">H", len(header)) + header)
            counter = 0
            block = src.read(CHUNK)
            while True:
                nxt = src.read(CHUNK)
                last = not nxt
                out.write(aead.encrypt(_nonce(prefix, counter, last), block, header))
                if last:
                    break
                counter += 1
                if counter >= _MAX_CHUNKS:
                    raise SealError("archive too large for one sealed file")
                block = nxt
        os.replace(tmp, sealed)
    finally:
        if tmp.exists():
            tmp.unlink()
    return fingerprint(rcpt_raw)


def _read_header(fh: BinaryIO) -> tuple[bytes, dict]:
    if fh.read(len(MAGIC)) != MAGIC:
        raise SealError("not a Vaara sealed archive (bad magic)")
    raw_len = fh.read(2)
    if len(raw_len) != 2:
        raise SealError("sealed archive is truncated in its header")
    (length,) = struct.unpack(">H", raw_len)
    if length > _MAX_HEADER:
        raise SealError("sealed archive header is too large")
    header = fh.read(length)
    try:
        parsed = json.loads(header)
    except ValueError as exc:
        raise SealError(f"sealed archive header is not JSON: {exc}") from exc
    if not isinstance(parsed, dict) or parsed.get("cipher") != CIPHER:
        raise SealError(f"unsupported cipher {parsed.get('cipher') if isinstance(parsed, dict) else None!r}")
    return header, parsed


def decrypt_file(sealed: Path, plain: Path, recipient_private) -> None:
    """Decrypt ``sealed`` into ``plain``. Any tampering, reordering or
    truncation raises ``SealError`` and leaves no output file."""
    from cryptography.exceptions import InvalidTag

    _, _, _, X25519PublicKey, AESGCM, _ = _crypto()
    tmp = plain.with_name(plain.name + f".{os.getpid()}.tmp")
    try:
        with open(sealed, "rb") as fh, open(tmp, "wb") as out:
            header, parsed = _read_header(fh)
            try:
                eph_raw = base64.b64decode(parsed["ephemeral_public"], validate=True)
                prefix = base64.b64decode(parsed["nonce_prefix"], validate=True)
            except (KeyError, ValueError) as exc:
                raise SealError(f"sealed archive header is malformed: {exc}") from exc
            if len(prefix) != 7 or len(eph_raw) != 32:
                raise SealError("sealed archive header has the wrong key or nonce size")
            rcpt_raw = _raw_public(recipient_private.public_key())
            if parsed.get("recipient_fingerprint") != fingerprint(rcpt_raw):
                raise SealError("this archive was sealed to a different recipient key")
            aead = AESGCM(_derive(
                recipient_private.exchange(X25519PublicKey.from_public_bytes(eph_raw)),
                eph_raw, rcpt_raw))
            counter = 0
            block = fh.read(CHUNK + _TAG)
            while True:
                nxt = fh.read(CHUNK + _TAG)
                last = not nxt
                try:
                    out.write(aead.decrypt(_nonce(prefix, counter, last), block, header))
                except InvalidTag as exc:
                    raise SealError(
                        f"chunk {counter} failed authentication: the archive was "
                        "altered, reordered or truncated") from exc
                if last:
                    break
                counter += 1
                block = nxt
        os.replace(tmp, plain)
    finally:
        if tmp.exists():
            tmp.unlink()


@dataclass
class SealVerdict:
    ok: bool
    manifest: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def seal_archive(zip_path: Union[str, Path], recipient_public_pem: Union[str, Path, bytes],
                 signer_key: Union[str, Path, bytes]) -> tuple[Path, Path, dict]:
    """Encrypt a signed export zip and sign its public seal.

    Returns ``(sealed_path, seal_path, manifest)``. The plaintext zip is left
    in place; the caller removes it once the seal verifies.
    """
    import zipfile

    from vaara import __version__ as vaara_version
    from vaara.audit.export import _load_private_key
    from vaara.audit.signer import Ed25519Signer

    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        try:
            export_manifest = json.loads(zf.read("manifest.json"))
        except KeyError as exc:
            raise SealError(f"{zip_path} is not a signed Vaara export (no manifest.json)") from exc
    recipient = _load_recipient_public(recipient_public_pem)
    signer = _load_private_key(signer_key)
    sealed = zip_path.with_name(zip_path.name + ".sealed")
    seal = zip_path.with_name(zip_path.name + ".seal.json")
    rcpt_fp = encrypt_file(zip_path, sealed, recipient)
    zip_sha, zip_size = _sha256_file(zip_path)
    ct_sha, ct_size = _sha256_file(sealed)
    _, serialization, *_ = _crypto()
    signer_fp = Ed25519Signer(signer).public_key_fingerprint()[:32]
    if export_manifest.get("signer_pubkey_fingerprint") != signer_fp:
        sealed.unlink()
        raise SealError("the seal key is not the key that signed the export")
    manifest = {
        "format": SEAL_FORMAT,
        "cipher": CIPHER,
        "archive": {"name": zip_path.name, "sha256": zip_sha, "size": zip_size},
        "ciphertext": {"name": sealed.name, "sha256": ct_sha, "size": ct_size},
        "trail": {
            "record_count": export_manifest.get("record_count"),
            "first_hash": export_manifest.get("first_hash"),
            "last_hash": export_manifest.get("last_hash"),
            "chain_intact_at_export": export_manifest.get("chain_intact_at_export"),
            "trail_sha256": export_manifest.get("trail_sha256"),
        },
        "export_signer_fingerprint": export_manifest.get("signer_pubkey_fingerprint"),
        "recipient_fingerprint": rcpt_fp,
        "signature_algorithm": "Ed25519",
        "signer_pubkey_fingerprint": signer_fp,
        "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vaara_version": vaara_version,
    }
    manifest_text = json.dumps(manifest, sort_keys=True, indent=2)
    signature = signer.sign(manifest_text.encode("utf-8"))
    signer_pem = signer.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    tmp = seal.with_name(seal.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({
        "manifest": manifest_text,
        "signature": base64.b64encode(signature).decode(),
        "signer_pubkey_pem": signer_pem,
    }, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, seal)
    return sealed, seal, manifest


def verify_seal(seal_path: Union[str, Path], ciphertext: Union[str, Path, None] = None,
                *, expected_signer: Union[str, Path, bytes, None] = None) -> SealVerdict:
    """Check a seal without any decryption key.

    The signature is checked against the public key the seal carries, or
    against ``expected_signer`` (a PEM public key) when given, which is how a
    verifier pins whose seal it accepts. With ``ciphertext``, its SHA-256 and
    size must match the seal.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    _, serialization, *_ = _crypto()
    errors: list[str] = []
    try:
        outer = json.loads(Path(seal_path).read_text(encoding="utf-8"))
        manifest_text = outer["manifest"]
        signature = base64.b64decode(outer["signature"], validate=True)
        manifest = json.loads(manifest_text)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return SealVerdict(False, errors=[f"seal unreadable: {exc}"])
    if not isinstance(manifest, dict) or manifest.get("format") != SEAL_FORMAT:
        return SealVerdict(False, errors=[f"not a {SEAL_FORMAT} seal"])
    try:
        if expected_signer is None:
            data = str(outer.get("signer_pubkey_pem", "")).encode()
        elif isinstance(expected_signer, (bytes, bytearray)):
            data = bytes(expected_signer)
        else:
            data = Path(expected_signer).read_bytes()
        pub = serialization.load_pem_public_key(data)
        if not isinstance(pub, Ed25519PublicKey):
            raise ValueError("not an Ed25519 public key")
    except (OSError, ValueError, TypeError) as exc:
        return SealVerdict(False, manifest, [f"signer key unusable: {exc}"])
    from vaara.audit.signer import _fingerprint

    raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    if _fingerprint(raw)[:32] != manifest.get("signer_pubkey_fingerprint"):
        errors.append("signer key does not match the fingerprint the seal names")
    try:
        pub.verify(signature, manifest_text.encode("utf-8"))
    except InvalidSignature:
        errors.append("seal signature does not verify")
    if ciphertext is not None:
        try:
            ct_sha, ct_size = _sha256_file(Path(ciphertext))
        except OSError as exc:
            errors.append(f"ciphertext unreadable: {exc}")
        else:
            want = manifest.get("ciphertext") or {}
            if ct_sha != want.get("sha256") or ct_size != want.get("size"):
                errors.append("ciphertext does not match the seal (digest or size differs)")
    return SealVerdict(not errors, manifest, errors)


def unseal(sealed: Union[str, Path], seal_path: Union[str, Path],
           recipient_private_pem: Union[str, Path, bytes], out_zip: Union[str, Path],
           *, trail=None, on_behalf_of: str = "", basis: str = "",
           accessed_by: str = "vaara trail unseal") -> Path:
    """Decrypt a sealed archive after checking its seal.

    With ``trail``, an access record is written into it BEFORE decryption,
    naming the archive by its ciphertext digest and committing to the zip
    digest the seal states, so opening a past period's records is itself on
    the record. The decrypted zip must match the seal and verify as a signed
    export, or it is removed and ``SealError`` is raised.
    """
    from vaara.audit.verify import verify_signed

    verdict = verify_seal(seal_path, sealed)
    if not verdict.ok:
        raise SealError("; ".join(verdict.errors))
    manifest = verdict.manifest
    key = _load_recipient_private(recipient_private_pem)
    if trail is not None:
        from vaara.audit.access import record_access

        record_access(
            trail,
            subject=f"sealed-archive:sha256:{manifest['ciphertext']['sha256']}",
            accessed_by=accessed_by,
            on_behalf_of=on_behalf_of,
            basis=basis,
            returned_sha256=f"sha256:{manifest['archive']['sha256']}",
            returned_count=(manifest.get("trail") or {}).get("record_count"),
            channel="unseal",
        )
    out = Path(out_zip)
    decrypt_file(Path(sealed), out, key)
    got, _ = _sha256_file(out)
    if got != manifest["archive"]["sha256"]:
        out.unlink()
        raise SealError("decrypted archive does not match the zip digest in the seal")
    check = verify_signed(out)
    if not check.ok:
        out.unlink()
        raise SealError("decrypted archive does not verify as a signed export: "
                        + "; ".join(check.errors))
    return out
