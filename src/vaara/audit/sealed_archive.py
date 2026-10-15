# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Encrypted retention archives with a public, signed seal.

    rotate --encrypt-to R
      export_signed -> verify_signed          (as without encryption)
      -> encrypt the zip to recipient R       archive.zip.sealed
         and decrypt it back once to prove the file opens to the same zip
      -> sign a public seal                   archive.zip.seal.json
           zip sha256, ciphertext sha256, record count,
           first and last chain hash, recipient key fingerprint
      -> verify the seal, remove the plaintext zip, then purge

    verify-seal   anyone, no key: the signature, and the ciphertext's digest
    unseal        the recipient: checks first, an access record, then decrypt,
                  then the zip against the seal and as a signed export

A retention archive has two readers with different rights. A data protection
officer or a court holds the decryption key and reads the records. Anyone
else, an auditor checking that retention happened or a counterparty
checking continuity, needs to know the archive exists, is intact, and was
signed by the trail's key, without reading a record. The seal is for them: it
carries the chain's first and last hash and the record count in the clear,
signed by the key that signed the export, and the ciphertext's digest binds
it to the file. A seal proves who made it only to a verifier who pins that
key; checked against the key it carries, it proves only that it is
self-consistent, so ``unseal`` always takes the expected signer.

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
from typing import Any, BinaryIO, Optional, Union

SEAL_FORMAT = "vaara.sealed-archive/v0"
CIPHER = "X25519-HKDF-SHA256-AES-256-GCM-STREAM-1MiB"
MAGIC = b"VAARA-SEALED-v0\n"
CHUNK = 1 << 20
_HKDF_INFO = b"vaara sealed archive v0"
_TAG = 16
_MAX_HEADER = 4096
_MAX_CHUNKS = 0xFFFFFFFF

KeySource = Union[str, Path, bytes]


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


def _key_errors() -> tuple[type[BaseException], ...]:
    # cryptography 47+ raises UnsupportedAlgorithm where 46.x raised ValueError.
    from cryptography.exceptions import UnsupportedAlgorithm

    return (ValueError, TypeError, UnsupportedAlgorithm)


def _raw_public(pub: Any) -> bytes:
    _, serialization, *_ = _crypto()
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def fingerprint(public_raw: bytes) -> str:
    """The export manifest's fingerprint convention, for any raw public key."""
    from vaara.audit.export import _short_fp

    return _short_fp(public_raw)


def _read_key_bytes(source: KeySource) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    try:
        return Path(source).read_bytes()
    except OSError as exc:
        raise SealError(f"key file unreadable: {exc}") from exc


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


def load_recipient_public(source: KeySource):
    _, serialization, _, X25519PublicKey, *_ = _crypto()
    try:
        key = serialization.load_pem_public_key(_read_key_bytes(source))
    except _key_errors() as exc:
        raise SealError(f"recipient key is not a usable PEM public key: {exc}") from exc
    if not isinstance(key, X25519PublicKey):
        raise SealError("recipient key must be an X25519 public key")
    return key


def load_recipient_private(source: KeySource):
    _, serialization, X25519PrivateKey, *_ = _crypto()
    try:
        key = serialization.load_pem_private_key(_read_key_bytes(source), password=None)
    except _key_errors() as exc:
        raise SealError(f"recipient key is not a usable PEM private key: {exc}") from exc
    if not isinstance(key, X25519PrivateKey):
        raise SealError("recipient key must be an X25519 private key")
    return key


def _load_signer_public(source: KeySource):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    _, serialization, *_ = _crypto()
    try:
        key = serialization.load_pem_public_key(_read_key_bytes(source))
    except _key_errors() as exc:
        raise SealError(f"signer key is not a usable PEM public key: {exc}") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise SealError("signer key must be an Ed25519 public key")
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


@dataclass(frozen=True)
class _Header:
    raw: bytes
    ephemeral_public: bytes
    nonce_prefix: bytes
    recipient_fingerprint: str


def _read_header(fh: BinaryIO) -> _Header:
    """Parse and check a sealed file's header. Every defect is a SealError."""
    if fh.read(len(MAGIC)) != MAGIC:
        raise SealError("not a Vaara sealed archive (bad magic)")
    raw_len = fh.read(2)
    if len(raw_len) != 2:
        raise SealError("sealed archive is truncated in its header")
    (length,) = struct.unpack(">H", raw_len)
    if length > _MAX_HEADER:
        raise SealError("sealed archive header is too large")
    raw = fh.read(length)
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("not an object")
        if parsed.get("cipher") != CIPHER:
            raise ValueError(f"unsupported cipher {parsed.get('cipher')!r}")
        eph = base64.b64decode(str(parsed["ephemeral_public"]), validate=True)
        prefix = base64.b64decode(str(parsed["nonce_prefix"]), validate=True)
        rcpt = parsed["recipient_fingerprint"]
        if len(eph) != 32 or len(prefix) != 7 or not isinstance(rcpt, str):
            raise ValueError("wrong key, nonce or fingerprint size")
    except (ValueError, KeyError, TypeError) as exc:
        raise SealError(f"sealed archive header is malformed: {exc}") from exc
    return _Header(raw, eph, prefix, rcpt)


def _decrypt_stream(fh: BinaryIO, header: _Header, aead, sink) -> None:
    """Decrypt every chunk after the header into ``sink(bytes)``."""
    from cryptography.exceptions import InvalidTag

    counter = 0
    block = fh.read(CHUNK + _TAG)
    while True:
        nxt = fh.read(CHUNK + _TAG)
        last = not nxt
        try:
            sink(aead.decrypt(_nonce(header.nonce_prefix, counter, last), block, header.raw))
        except InvalidTag as exc:
            raise SealError(
                f"chunk {counter} failed authentication: the archive was "
                "altered, reordered or truncated") from exc
        if last:
            return
        counter += 1
        block = nxt


def encrypt_file(plain: Path, sealed: Path, recipient_public) -> tuple[str, str, int]:
    """Encrypt ``plain`` to ``sealed`` for ``recipient_public``.

    Returns ``(recipient_fingerprint, plaintext_sha256, plaintext_size)``.
    Before the file is moved into place, it is decrypted once with the same
    key and must hash to the plaintext read, so a write fault never becomes
    the only copy of an archive. The output is written whole or not at all.
    """
    _, _, X25519PrivateKey, _, AESGCM, _ = _crypto()
    eph = X25519PrivateKey.generate()
    eph_raw = _raw_public(eph.public_key())
    rcpt_raw = _raw_public(recipient_public)
    try:
        aead = AESGCM(_derive(eph.exchange(recipient_public), eph_raw, rcpt_raw))
    except ValueError as exc:
        raise SealError(f"recipient key cannot be used for key agreement: {exc}") from exc
    prefix = os.urandom(7)
    rcpt_fp = fingerprint(rcpt_raw)
    header = json.dumps({
        "cipher": CIPHER,
        "ephemeral_public": base64.b64encode(eph_raw).decode(),
        "nonce_prefix": base64.b64encode(prefix).decode(),
        "recipient_fingerprint": rcpt_fp,
    }, sort_keys=True, separators=(",", ":")).encode()
    plain_hash = hashlib.sha256()
    plain_size = 0
    tmp = sealed.with_name(sealed.name + f".{os.getpid()}.tmp")
    try:
        with open(plain, "rb") as src, open(tmp, "wb") as out:
            out.write(MAGIC + struct.pack(">H", len(header)) + header)
            counter = 0
            block = src.read(CHUNK)
            while True:
                plain_hash.update(block)
                plain_size += len(block)
                nxt = src.read(CHUNK)
                last = not nxt
                out.write(aead.encrypt(_nonce(prefix, counter, last), block, header))
                if last:
                    break
                counter += 1
                if counter >= _MAX_CHUNKS:
                    raise SealError("archive too large for one sealed file")
                block = nxt
            out.flush()
            os.fsync(out.fileno())
        check = hashlib.sha256()
        with open(tmp, "rb") as fh:
            parsed = _read_header(fh)
            if parsed.raw != header:
                raise SealError("sealed file header did not read back as written")
            _decrypt_stream(fh, parsed, aead, check.update)
        if check.hexdigest() != plain_hash.hexdigest():
            raise SealError("sealed file does not decrypt back to the archive")
        os.replace(tmp, sealed)
    finally:
        if tmp.exists():
            tmp.unlink()
    return rcpt_fp, plain_hash.hexdigest(), plain_size


def decrypt_file(sealed: Path, plain: Path, recipient_private) -> None:
    """Decrypt ``sealed`` into ``plain``. Any tampering, reordering or
    truncation raises ``SealError`` and leaves no output file."""
    _, _, _, X25519PublicKey, AESGCM, _ = _crypto()
    rcpt_raw = _raw_public(recipient_private.public_key())
    tmp = plain.with_name(plain.name + f".{os.getpid()}.tmp")
    try:
        with open(sealed, "rb") as fh:
            header = _read_header(fh)
            if header.recipient_fingerprint != fingerprint(rcpt_raw):
                raise SealError("this archive was sealed to a different recipient key")
            try:
                shared = recipient_private.exchange(
                    X25519PublicKey.from_public_bytes(header.ephemeral_public))
            except ValueError as exc:
                raise SealError(f"sealed archive ephemeral key is unusable: {exc}") from exc
            aead = AESGCM(_derive(shared, header.ephemeral_public, rcpt_raw))
            with open(tmp, "wb") as out:
                _decrypt_stream(fh, header, aead, out.write)
        os.replace(tmp, plain)
    finally:
        if tmp.exists():
            tmp.unlink()


@dataclass
class SealVerdict:
    ok: bool
    manifest: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _manifest_problems(manifest: Any) -> list[str]:
    """Shape check, so a self-signed seal with odd types reads as bad, not as a crash."""
    if not isinstance(manifest, dict) or manifest.get("format") != SEAL_FORMAT:
        return [f"not a {SEAL_FORMAT} seal"]
    problems = []
    for part in ("archive", "ciphertext"):
        entry = manifest.get(part)
        if not (isinstance(entry, dict) and isinstance(entry.get("sha256"), str)
                and isinstance(entry.get("size"), int)):
            problems.append(f"seal field {part!r} is missing its sha256 or size")
    if not isinstance(manifest.get("trail"), dict):
        problems.append("seal field 'trail' is missing")
    for key in ("recipient_fingerprint", "signer_pubkey_fingerprint", "export_signer_fingerprint"):
        if not isinstance(manifest.get(key), str):
            problems.append(f"seal field {key!r} is missing")
    return problems


def seal_archive(zip_path: Union[str, Path], recipient_public_pem: KeySource,
                 signer_key: KeySource) -> tuple[Path, Path, dict]:
    """Encrypt a signed export zip and sign its public seal.

    Returns ``(sealed_path, seal_path, manifest)``. The plaintext zip is left
    in place; the caller removes it once the seal verifies. On any failure
    nothing new is left beside the zip.
    """
    import zipfile

    from vaara import __version__ as vaara_version
    from vaara.audit.export import _load_private_key
    from vaara.audit.signer import Ed25519Signer

    zip_path = Path(zip_path)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            export_manifest = json.loads(zf.read("manifest.json"))
    except (KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise SealError(f"{zip_path} is not a signed Vaara export: {exc}") from exc
    if not isinstance(export_manifest, dict):
        raise SealError(f"{zip_path} has a malformed manifest.json")
    recipient = load_recipient_public(recipient_public_pem)
    signer = _load_private_key(signer_key)
    signer_fp = Ed25519Signer(signer).public_key_fingerprint()[:32]
    if export_manifest.get("signer_pubkey_fingerprint") != signer_fp:
        raise SealError("the seal key is not the key that signed the export")

    _, serialization, *_ = _crypto()
    sealed = zip_path.with_name(zip_path.name + ".sealed")
    seal = zip_path.with_name(zip_path.name + ".seal.json")
    try:
        rcpt_fp, zip_sha, zip_size = encrypt_file(zip_path, sealed, recipient)
        ct_sha, ct_size = _sha256_file(sealed)
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
        signer_pem = signer.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo).decode()
        tmp = seal.with_name(seal.name + f".{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps({
                "manifest": manifest_text,
                "signature": base64.b64encode(signer.sign(manifest_text.encode("utf-8"))).decode(),
                "signer_pubkey_pem": signer_pem,
            }, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, seal)
        finally:
            if tmp.exists():
                tmp.unlink()
    except BaseException:
        discard_sealed(sealed, seal)
        raise
    return sealed, seal, manifest


def discard_sealed(sealed: Path, seal: Path) -> None:
    """Remove a sealed file and its seal, so a failed rotation leaves no orphan."""
    for path in (sealed, seal):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def verify_seal(seal_path: Union[str, Path], ciphertext: Union[str, Path, None] = None,
                *, expected_signer: Optional[KeySource] = None) -> SealVerdict:
    """Check a seal without any decryption key.

    The signature is checked against ``expected_signer`` (an Ed25519 public
    key, PEM) when given, which is how a verifier pins whose seal it accepts;
    without it, against the key the seal carries, which proves only that the
    seal is self-consistent. With ``ciphertext``, its SHA-256 and size must
    match the seal.
    """
    from cryptography.exceptions import InvalidSignature

    try:
        outer = json.loads(Path(seal_path).read_text(encoding="utf-8"))
        manifest_text = outer["manifest"]
        signature = base64.b64decode(str(outer["signature"]), validate=True)
        manifest = json.loads(manifest_text)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return SealVerdict(False, errors=[f"seal unreadable: {exc}"])
    problems = _manifest_problems(manifest)
    if problems:
        return SealVerdict(False, manifest if isinstance(manifest, dict) else {}, problems)
    try:
        if expected_signer is None:
            pub = _load_signer_public(str(outer.get("signer_pubkey_pem", "")).encode())
        else:
            pub = _load_signer_public(expected_signer)
    except SealError as exc:
        return SealVerdict(False, manifest, [str(exc)])
    errors: list[str] = []
    signer_fp = fingerprint(_raw_public(pub))
    if signer_fp != manifest["signer_pubkey_fingerprint"]:
        errors.append("signer key does not match the fingerprint the seal names")
    if manifest["export_signer_fingerprint"] != manifest["signer_pubkey_fingerprint"]:
        errors.append("the seal was signed by a different key than the export it seals")
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
            want = manifest["ciphertext"]
            if ct_sha != want["sha256"] or ct_size != want["size"]:
                errors.append("ciphertext does not match the seal (digest or size differs)")
    return SealVerdict(not errors, manifest, errors)


def unseal(sealed: Union[str, Path], seal_path: Union[str, Path],
           recipient_private_pem: KeySource, out_zip: Union[str, Path],
           *, expected_signer: KeySource, trail=None, on_behalf_of: str = "",
           basis: str = "", accessed_by: str = "vaara trail unseal") -> Path:
    """Decrypt a sealed archive after checking its seal against a pinned signer.

    Every check that can fail without decrypting runs first: the seal against
    ``expected_signer`` and the ciphertext, the key against the seal's
    recipient, and the sealed file's header. Only then, with ``trail``, is an
    access record written, naming the archive by its ciphertext digest and
    committing to the zip digest the seal states, so opening a past period's
    records is on the record and a refused attempt does not read as an
    opening. The decrypted zip must match the seal, be signed by the pinned
    key, and verify as a signed export, or it is removed and ``SealError`` is
    raised.
    """
    import zipfile

    from vaara.audit.verify import verify_signed

    sealed = Path(sealed)
    verdict = verify_seal(seal_path, sealed, expected_signer=expected_signer)
    if not verdict.ok:
        raise SealError("; ".join(verdict.errors))
    manifest = verdict.manifest
    key = load_recipient_private(recipient_private_pem)
    if fingerprint(_raw_public(key.public_key())) != manifest["recipient_fingerprint"]:
        raise SealError("this archive was sealed to a different recipient key")
    try:
        with open(sealed, "rb") as fh:
            header = _read_header(fh)
    except OSError as exc:
        raise SealError(f"sealed archive unreadable: {exc}") from exc
    if header.recipient_fingerprint != manifest["recipient_fingerprint"]:
        raise SealError("the sealed file names a different recipient than its seal")
    if trail is not None:
        from vaara.audit.access import record_access

        count = manifest["trail"].get("record_count")
        record_access(
            trail,
            subject=f"sealed-archive:sha256:{manifest['ciphertext']['sha256']}",
            accessed_by=accessed_by,
            on_behalf_of=on_behalf_of,
            basis=basis,
            returned_sha256=f"sha256:{manifest['archive']['sha256']}",
            returned_count=count if isinstance(count, int) else None,
            channel="unseal",
        )
    out = Path(out_zip)
    decrypt_file(sealed, out, key)
    try:
        got, _ = _sha256_file(out)
        if got != manifest["archive"]["sha256"]:
            raise SealError("decrypted archive does not match the zip digest in the seal")
        try:
            with zipfile.ZipFile(out) as zf:
                inner = json.loads(zf.read("manifest.json"))
        except (KeyError, ValueError, zipfile.BadZipFile) as exc:
            raise SealError(f"decrypted archive is not a signed export: {exc}") from exc
        if not isinstance(inner, dict) or (
                inner.get("signer_pubkey_fingerprint") != manifest["signer_pubkey_fingerprint"]):
            raise SealError("decrypted archive was signed by a different key than the seal")
        check = verify_signed(out)
        if not check.ok:
            raise SealError("decrypted archive does not verify as a signed export: "
                            + "; ".join(check.errors))
    except BaseException:
        out.unlink()
        raise
    return out
