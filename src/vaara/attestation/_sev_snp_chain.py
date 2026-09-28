# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AMD SEV-SNP key chain: a report's signing key validated to AMD's root.

A SEV-SNP report is signed by a per-chip key, the VCEK, or by a per-cloud key,
the VLEK. AMD's Key Distribution Service (KDS) issues that key as an X.509
certificate signed by an intermediate (ASK for a VCEK, ASVK for a VLEK), which
is signed by the product's root, the ARK. All three are RSA-PSS SHA-384.

:func:`verify_sev_snp_chain` checks, with no network and no AMD software:

1. The ARK's public key is AMD's root for the product. The SHA-256 of each
   ARK's SubjectPublicKeyInfo is pinned below, read from
   ``https://kdsintf.amd.com/{vcek,vlek}/v1/{product}/cert_chain`` on
   2026-09-28. A chain built on any other root fails here.
2. The ARK is self-signed, the intermediate is signed by the ARK, and the
   signing key's certificate is signed by the intermediate.
3. The signing key's certificate describes this report: for a VCEK, its hwID
   extension equals the report's CHIP_ID (the first 8 bytes on Turin, where the
   rest must be zero), and its bootloader, TEE, SNP and microcode SPL
   extensions equal the report's REPORTED_TCB, decoded with the product's
   layout. A VLEK carries no hwID, so only the TCB is compared.
4. The report's ECDSA P-384 signature verifies under that key.

If all four hold, the report was produced by genuine AMD hardware of that
product line at that TCB. It says nothing about which guest image ran; pin
``measurement`` for that.

The certificates come from the guest itself (the extended report's certificate
table, see :func:`parse_certificate_table`), from the KDS
(:func:`fetch_amd_cert_chain`, :func:`fetch_vcek`), or from any cache: their
origin does not matter, because trust comes from the pinned root.
"""

from __future__ import annotations

import hashlib
import struct
import urllib.request
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Union

from vaara.attestation.tee import (
    SEVSNPReport,
    TEEAttestationError,
    verify_sev_snp_report_signature,
)

AMD_KDS_BASE = "https://kdsintf.amd.com"

# SHA-256 over the DER SubjectPublicKeyInfo of each product's ARK. The VCEK and
# VLEK chains of a product end in the same ARK key.
AMD_ARK_SPKI_SHA256 = {
    "Milan": "9f056bee44377e29308cb5ffa895bdfb62d18881fa6bed8d6f075b0204089cb9",
    "Genoa": "429a69c9422aa258ee4d8db5fcda9c6470ef15f8cd5a9cebd6cbc7d90b863831",
    "Turin": "4f125410563a2ab9a50356f9243f6fe0b6f73de98603f53f90339c70e9d7ad08",
}

# GUIDs of the extended report's certificate table (SEV-SNP GHCB spec).
SNP_CERT_GUIDS = {
    uuid.UUID("63da758d-e664-4564-adc5-f4b93be8accd"): "vcek",
    uuid.UUID("a8074bc2-a25a-483e-aae6-39c045a0b8a1"): "vlek",
    uuid.UUID("4ab7b379-bbac-4fe4-a02f-05aef327c782"): "ask",
    uuid.UUID("c0b406a4-a803-4952-9743-3fb6014cd0ae"): "ark",
}

_OID_PRODUCT_NAME = "1.3.6.1.4.1.3704.1.2"
_OID_BL_SPL = "1.3.6.1.4.1.3704.1.3.1"
_OID_TEE_SPL = "1.3.6.1.4.1.3704.1.3.2"
_OID_SNP_SPL = "1.3.6.1.4.1.3704.1.3.3"
_OID_UCODE_SPL = "1.3.6.1.4.1.3704.1.3.8"
_OID_HWID = "1.3.6.1.4.1.3704.1.4"

# Byte index of each SPL inside the 64-bit little-endian TCB_VERSION.
_TCB_LAYOUT = {
    "Milan": {"bl": 0, "tee": 1, "snp": 6, "ucode": 7},
    "Genoa": {"bl": 0, "tee": 1, "snp": 6, "ucode": 7},
    "Turin": {"fmc": 0, "bl": 1, "tee": 2, "snp": 3, "ucode": 7},
}
_TURIN_HWID_SIZE = 8

# SIGNING_KEY, bits 4:2 of the word at 0x48.
_SIGNER = {0: "vcek", 1: "vlek", 7: "none"}

CertInput = Union[bytes, Any]


def report_signer(report: SEVSNPReport) -> str:
    """Which key signed the report: ``vcek``, ``vlek``, ``none`` or ``unknown``."""
    return _SIGNER.get((report.author_key_en >> 2) & 0x7, "unknown")


def decode_tcb(tcb: int, product: str) -> dict[str, int]:
    """Split a 64-bit TCB_VERSION into its SPLs using the product's layout."""
    layout = _TCB_LAYOUT.get(product)
    if layout is None:
        raise TEEAttestationError(f"unknown SEV-SNP product {product!r}")
    return {name: (tcb >> (8 * index)) & 0xFF for name, index in layout.items()}


def _load_cert(data: CertInput):
    from cryptography import x509

    if isinstance(data, x509.Certificate):
        return data
    if not isinstance(data, (bytes, bytearray)):
        raise TEEAttestationError("certificate must be PEM or DER bytes")
    # KDS-issued VCEKs carry a serial number that cryptography warns about.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if bytes(data).lstrip().startswith(b"-----BEGIN"):
                return x509.load_pem_x509_certificate(bytes(data))
            return x509.load_der_x509_certificate(bytes(data))
        except ValueError as exc:
            raise TEEAttestationError(f"cannot load certificate: {exc}") from exc


def _spki_sha256(cert) -> str:
    from cryptography.hazmat.primitives import serialization

    spki = cert.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return hashlib.sha256(spki).hexdigest()


def _valid_at(cert, when: datetime) -> bool:
    # not_valid_*_utc arrived in cryptography 42; older releases return naive UTC.
    start = getattr(cert, "not_valid_before_utc", None)
    end = getattr(cert, "not_valid_after_utc", None)
    if start is None or end is None:
        start = cert.not_valid_before.replace(tzinfo=timezone.utc)
        end = cert.not_valid_after.replace(tzinfo=timezone.utc)
    return bool(start <= when <= end)


def _signed_by(child, issuer) -> bool:
    """True if ``child``'s signature verifies under ``issuer``'s public key."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric import ec, rsa

    key = issuer.public_key()
    params = child.signature_algorithm_parameters
    try:
        if isinstance(key, rsa.RSAPublicKey):
            key.verify(
                child.signature,
                child.tbs_certificate_bytes,
                params,
                child.signature_hash_algorithm,
            )
        elif isinstance(key, ec.EllipticCurvePublicKey):
            key.verify(child.signature, child.tbs_certificate_bytes, params)
        else:
            return False
    except (InvalidSignature, TypeError, ValueError):
        return False
    return True


def _ext_value(cert, oid: str) -> Optional[bytes]:
    from cryptography import x509

    try:
        ext = cert.extensions.get_extension_for_oid(x509.ObjectIdentifier(oid))
    except x509.ExtensionNotFound:
        return None
    value = ext.value
    return value.value if isinstance(value, x509.UnrecognizedExtension) else None


def _der_tlv(data: bytes, tag: int) -> Optional[bytes]:
    """Contents of one short-form DER TLV with the given tag, else None."""
    if len(data) < 2 or data[0] != tag or data[1] & 0x80:
        return None
    length = data[1]
    if len(data) != 2 + length:
        return None
    return data[2:]


def _ext_int(cert, oid: str) -> Optional[int]:
    raw = _ext_value(cert, oid)
    body = _der_tlv(raw, 0x02) if raw is not None else None
    return int.from_bytes(body, "big") if body else None


def _ext_hwid(cert) -> Optional[bytes]:
    raw = _ext_value(cert, _OID_HWID)
    if raw is None:
        return None
    # KDS writes the hwID bare, with no OCTET STRING tag; accept both.
    if len(raw) in (64, _TURIN_HWID_SIZE):
        return raw
    return _der_tlv(raw, 0x04)


def certificate_product(cert: CertInput) -> Optional[str]:
    """Product line named by a VCEK/VLEK/ASK/ARK certificate, or None.

    A signing key names it in the productName extension (``Milan-B0``); an ASK,
    ASVK or ARK in its common name (``SEV-Milan``, ``ARK-Genoa``).
    """
    from cryptography.x509.oid import NameOID

    cert = _load_cert(cert)
    raw = _ext_value(cert, _OID_PRODUCT_NAME)
    names: list[str] = []
    if raw is not None:
        body = _der_tlv(raw, 0x16)
        names.append((body if body is not None else raw).decode("ascii", "replace"))
    names.extend(
        str(a.value) for a in cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    )
    for name in names:
        for product in _TCB_LAYOUT:
            if product.lower() in name.lower():
                return product
    return None


@dataclass(frozen=True)
class SEVSNPChainVerdict:
    """Outcome of :func:`verify_sev_snp_chain`.

    ``ok`` is true only when every boolean below it is true. ``reason`` names
    the first failed check, or states what a pass proves.
    """

    ok: bool
    product: Optional[str]
    signer: str
    ark_pinned: bool
    ark_self_signed: bool
    intermediate_signed_by_ark: bool
    key_signed_by_intermediate: bool
    certificates_current: bool
    hwid_match: bool
    tcb_match: bool
    report_signature_valid: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "product": self.product,
            "signer": self.signer,
            "ark_pinned": self.ark_pinned,
            "ark_self_signed": self.ark_self_signed,
            "intermediate_signed_by_ark": self.intermediate_signed_by_ark,
            "key_signed_by_intermediate": self.key_signed_by_intermediate,
            "certificates_current": self.certificates_current,
            "hwid_match": self.hwid_match,
            "tcb_match": self.tcb_match,
            "report_signature_valid": self.report_signature_valid,
            "reason": self.reason,
        }


def verify_sev_snp_chain(
    report: SEVSNPReport,
    signing_cert: CertInput,
    intermediate_cert: CertInput,
    ark_cert: CertInput,
    *,
    product: Optional[str] = None,
    at: Optional[datetime] = None,
) -> SEVSNPChainVerdict:
    """Validate a report's signing key to AMD's pinned root, then the report.

    ``signing_cert`` is the VCEK or VLEK, ``intermediate_cert`` the ASK or ASVK,
    ``ark_cert`` the ARK, each PEM or DER. ``product`` (``Milan``, ``Genoa``,
    ``Turin``) defaults to the one the signing certificate names. ``at`` is the
    time the certificates must be valid at, default now.

    Never raises on a bad chain; a certificate that will not load raises
    :class:`TEEAttestationError`.
    """
    from cryptography.hazmat.primitives import serialization

    key_cert = _load_cert(signing_cert)
    mid = _load_cert(intermediate_cert)
    ark = _load_cert(ark_cert)
    signer = report_signer(report)
    product = product or certificate_product(key_cert)
    when = at or datetime.now(timezone.utc)

    pinned = AMD_ARK_SPKI_SHA256.get(product or "")
    ark_pinned = pinned is not None and _spki_sha256(ark) == pinned
    ark_self_signed = _signed_by(ark, ark)
    mid_ok = _signed_by(mid, ark)
    key_ok = _signed_by(key_cert, mid)
    current = all(_valid_at(c, when) for c in (key_cert, mid, ark))

    hwid_match = False
    tcb_match = False
    if product in _TCB_LAYOUT:
        if signer == "vlek":
            hwid_match = _ext_hwid(key_cert) is None
        else:
            hwid = _ext_hwid(key_cert)
            if hwid is not None and len(hwid) == 64:
                hwid_match = hwid == report.chip_id
            elif hwid is not None and product == "Turin":
                hwid_match = (
                    hwid == report.chip_id[:_TURIN_HWID_SIZE]
                    and not any(report.chip_id[_TURIN_HWID_SIZE:])
                )
        spl = decode_tcb(report.reported_tcb, product)
        tcb_match = all(
            _ext_int(key_cert, oid) == spl[name]
            for name, oid in (
                ("bl", _OID_BL_SPL),
                ("tee", _OID_TEE_SPL),
                ("snp", _OID_SNP_SPL),
                ("ucode", _OID_UCODE_SPL),
            )
        )

    report_signature_valid = False
    key_pem = key_cert.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    try:
        report_signature_valid = verify_sev_snp_report_signature(report, key_pem)
    except TEEAttestationError:
        report_signature_valid = False

    checks = [
        (signer in ("vcek", "vlek"), f"the report names signer {signer!r}"),
        (product in _TCB_LAYOUT, "the product line is unknown"),
        (ark_pinned, f"the ARK is not AMD's pinned {product} root"),
        (ark_self_signed, "the ARK is not self-signed"),
        (mid_ok, "the intermediate is not signed by the ARK"),
        (key_ok, f"the {signer.upper()} is not signed by the intermediate"),
        (current, "a certificate is outside its validity period"),
        (hwid_match, "the certificate's hwID does not match the report's CHIP_ID"),
        (tcb_match, "the certificate's SPLs do not match the report's REPORTED_TCB"),
        (report_signature_valid, "the report signature does not verify"),
    ]
    failed = [msg for passed, msg in checks if not passed]
    ok = not failed
    if ok:
        reason = (
            f"the report is signed by a {signer.upper()} that chains to AMD's "
            f"{product} root and matches the report's chip and TCB"
        )
    else:
        reason = failed[0]
    return SEVSNPChainVerdict(
        ok=ok,
        product=product,
        signer=signer,
        ark_pinned=ark_pinned,
        ark_self_signed=ark_self_signed,
        intermediate_signed_by_ark=mid_ok,
        key_signed_by_intermediate=key_ok,
        certificates_current=current,
        hwid_match=hwid_match,
        tcb_match=tcb_match,
        report_signature_valid=report_signature_valid,
        reason=reason + ".",
    )


def parse_certificate_table(blob: bytes) -> dict[str, bytes]:
    """Certificates in an extended report's table, keyed vcek/vlek/ask/ark.

    The table is a run of 24-byte entries (16-byte GUID, u32 offset, u32
    length, offsets from the table start) ended by an all-zero entry. Unknown
    GUIDs are skipped.
    """
    certs: dict[str, bytes] = {}
    pos = 0
    while pos + 24 <= len(blob):
        guid = uuid.UUID(bytes_le=bytes(blob[pos:pos + 16]))
        offset, length = struct.unpack_from("<II", blob, pos + 16)
        pos += 24
        if guid.int == 0 and offset == 0 and length == 0:
            break
        if offset + length > len(blob):
            raise TEEAttestationError("certificate table entry runs past the blob")
        name = SNP_CERT_GUIDS.get(guid)
        if name is not None:
            certs[name] = bytes(blob[offset:offset + length])
    return certs


def split_pem_chain(data: bytes) -> tuple[bytes, bytes, bytes]:
    """(signing cert, intermediate, ARK) from one PEM file holding the three."""
    end = b"-----END CERTIFICATE-----"
    blocks = [
        block[block.index(b"-----BEGIN"):].rstrip() + b"\n" + end + b"\n"
        for block in data.split(end)
        if b"-----BEGIN CERTIFICATE-----" in block
    ]
    if len(blocks) != 3:
        raise TEEAttestationError(
            f"AMD chain file holds {len(blocks)} certificates, not 3 "
            f"(VCEK or VLEK, ASK or ASVK, ARK)"
        )
    return blocks[0], blocks[1], blocks[2]


def _kds_get(path: str, timeout: float) -> bytes:
    # The scheme and host are the fixed https KDS base; only the path varies.
    url = AMD_KDS_BASE + path
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
        return bytes(resp.read())


def fetch_amd_cert_chain(
    product: str, *, signer: str = "vcek", timeout: float = 10.0
) -> tuple[bytes, bytes]:
    """(intermediate, ARK) PEM for a product from AMD's KDS."""
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    if product not in _TCB_LAYOUT or signer not in ("vcek", "vlek"):
        raise TEEAttestationError(f"no KDS chain for {signer} on {product!r}")
    certs = x509.load_pem_x509_certificates(
        _kds_get(f"/{signer}/v1/{product}/cert_chain", timeout)
    )
    if len(certs) != 2:
        raise TEEAttestationError("KDS cert_chain did not hold two certificates")
    pem = [c.public_bytes(serialization.Encoding.PEM) for c in certs]
    return pem[0], pem[1]


def fetch_vcek(report: SEVSNPReport, product: str, *, timeout: float = 10.0) -> bytes:
    """The DER VCEK for this report's chip and REPORTED_TCB from AMD's KDS."""
    spl = decode_tcb(report.reported_tcb, product)
    if product == "Turin":
        hwid = report.chip_id[:_TURIN_HWID_SIZE].hex()
        query = (
            f"fmcSPL={spl['fmc']}&blSPL={spl['bl']}&teeSPL={spl['tee']}"
            f"&snpSPL={spl['snp']}&ucodeSPL={spl['ucode']}"
        )
    else:
        hwid = report.chip_id.hex()
        query = (
            f"blSPL={spl['bl']}&teeSPL={spl['tee']}"
            f"&snpSPL={spl['snp']}&ucodeSPL={spl['ucode']}"
        )
    return _kds_get(f"/vcek/v1/{product}/{hwid}?{query}", timeout)


__all__ = [
    "AMD_ARK_SPKI_SHA256",
    "AMD_KDS_BASE",
    "SEVSNPChainVerdict",
    "SNP_CERT_GUIDS",
    "certificate_product",
    "decode_tcb",
    "fetch_amd_cert_chain",
    "fetch_vcek",
    "parse_certificate_table",
    "report_signer",
    "split_pem_chain",
    "verify_sev_snp_chain",
]
