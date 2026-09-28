# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hardware TEE attestation hook for OVERT 1.0 Base Envelopes.

Status: experimental. Adds an optional hardware-rooted attestation layer
alongside the Ed25519 (or ML-DSA-65) signature already on the OVERT 1.0
Base Envelope. Initial backend is AMD SEV-SNP, the natural fit for the
confidential-VM deployment model used in agent runtimes. Intel TDX and
Intel SGX backends would sit beside it.

Architectural framing
---------------------

The OVERT 1.0 Protocol Profile 1.0 Base Envelope (Annex B.6) is a closed
9-field schema. Hardware TEE attestation does NOT extend the envelope.
Instead, Vaara emits a sibling SEV-SNP attestation report and binds it
to the envelope by placing the SHA-512 of the envelope's canonical CBOR
encoding into the report's 64-byte REPORT_DATA field. SHA-512 fits the
slot exactly.

Verifiers therefore check two things independently:

1. The OVERT envelope's Ed25519 signature, as before.
2. The SEV-SNP report's ECDSA P-384 signature against the AMD VCEK, and
   that REPORT_DATA equals SHA-512 of the envelope's canonical CBOR.

If both hold, the attestation says "this OVERT envelope was emitted by
an arbiter running inside an AMD SEV-SNP confidential VM at the measured
launch state recorded in the report."

What this module holds
----------------------

- ``parse_sev_snp_report``: binary parser for the 1184-byte
  attestation-report structure (AMD SEV-SNP ABI Specification rev. 1.55,
  Table 22). Report versions 2 to 5 share these offsets.
- ``bind_overt_envelope_to_report_data``: computes the 64-byte
  REPORT_DATA value that binds a TEE report to a specific OVERT envelope.
- ``verify_sev_snp_report_signature``: validates the ECDSA P-384 over the
  report body against a VCEK or VLEK public key.
- ``verify_envelope_binding``: confirms the report's REPORT_DATA matches
  SHA-512 of the supplied envelope.
- ``MockSEVSNPAttester``: deterministic in-memory attester for tests and
  CI, building byte-compatible report blobs signed with a caller-supplied
  ECDSA P-384 key.
- ``SEVSNPHostAttester``: requests a real report inside a SEV-SNP guest,
  through configfs-tsm or the ``/dev/sev-guest`` ioctl.

The signing key's chain to AMD's root (VCEK or VLEK, ASK or ASVK, ARK) is
checked by :func:`vaara.attestation._sev_snp_chain.verify_sev_snp_chain`.
Intel TDX and Intel SGX would be further attester classes of the same shape.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from vaara.attestation.overt import BaseEnvelope


class TEEAttestationError(RuntimeError):
    """Raised on TEE attestation parse, verification, or binding failures."""


SEV_SNP_REPORT_SIZE = 1184
SEV_SNP_BODY_SIZE = 0x2A0  # 672 bytes signed payload
SEV_SNP_SIG_SIZE = 512
SEV_SNP_REPORT_DATA_SIZE = 64

SIGNATURE_ALGO_INVALID = 0
SIGNATURE_ALGO_ECDSA_P384_SHA384 = 1


@dataclass(frozen=True)
class SEVSNPReport:
    """Parsed AMD SEV-SNP attestation report.

    Field set and offsets match AMD SEV Secure Nested Paging Firmware ABI
    Specification, Revision 1.55, Table 22.
    """

    version: int
    guest_svn: int
    policy: int
    family_id: bytes
    image_id: bytes
    vmpl: int
    signature_algo: int
    current_tcb: int
    platform_info: int
    author_key_en: int
    report_data: bytes
    measurement: bytes
    host_data: bytes
    id_key_digest: bytes
    author_key_digest: bytes
    report_id: bytes
    report_id_ma: bytes
    reported_tcb: int
    chip_id: bytes
    committed_tcb: int
    current_build: int
    current_minor: int
    current_major: int
    committed_build: int
    committed_minor: int
    committed_major: int
    launch_tcb: int
    signature: bytes
    raw: bytes

    @property
    def body(self) -> bytes:
        """The 672-byte signed payload (offset 0 to SEV_SNP_BODY_SIZE)."""
        return self.raw[:SEV_SNP_BODY_SIZE]


def parse_sev_snp_report(report_bytes: bytes) -> SEVSNPReport:
    """Parse the AMD SEV-SNP attestation report binary structure.

    Little-endian throughout per AMD spec. Offsets follow Table 22 of the
    AMD SEV Secure Nested Paging Firmware ABI Specification, rev 1.55.
    """
    if len(report_bytes) != SEV_SNP_REPORT_SIZE:
        raise TEEAttestationError(
            f"SEV-SNP report must be {SEV_SNP_REPORT_SIZE} bytes; "
            f"got {len(report_bytes)}"
        )

    version = struct.unpack_from("<I", report_bytes, 0x000)[0]
    guest_svn = struct.unpack_from("<I", report_bytes, 0x004)[0]
    policy = struct.unpack_from("<Q", report_bytes, 0x008)[0]
    vmpl = struct.unpack_from("<I", report_bytes, 0x030)[0]
    signature_algo = struct.unpack_from("<I", report_bytes, 0x034)[0]
    current_tcb = struct.unpack_from("<Q", report_bytes, 0x038)[0]
    platform_info = struct.unpack_from("<Q", report_bytes, 0x040)[0]
    author_key_en = struct.unpack_from("<I", report_bytes, 0x048)[0]
    reported_tcb = struct.unpack_from("<Q", report_bytes, 0x180)[0]
    committed_tcb = struct.unpack_from("<Q", report_bytes, 0x1E0)[0]
    launch_tcb = struct.unpack_from("<Q", report_bytes, 0x1F0)[0]

    return SEVSNPReport(
        version=version,
        guest_svn=guest_svn,
        policy=policy,
        family_id=bytes(report_bytes[0x010:0x020]),
        image_id=bytes(report_bytes[0x020:0x030]),
        vmpl=vmpl,
        signature_algo=signature_algo,
        current_tcb=current_tcb,
        platform_info=platform_info,
        author_key_en=author_key_en,
        report_data=bytes(report_bytes[0x050:0x090]),
        measurement=bytes(report_bytes[0x090:0x0C0]),
        host_data=bytes(report_bytes[0x0C0:0x0E0]),
        id_key_digest=bytes(report_bytes[0x0E0:0x110]),
        author_key_digest=bytes(report_bytes[0x110:0x140]),
        report_id=bytes(report_bytes[0x140:0x160]),
        report_id_ma=bytes(report_bytes[0x160:0x180]),
        reported_tcb=reported_tcb,
        chip_id=bytes(report_bytes[0x1A0:0x1E0]),
        committed_tcb=committed_tcb,
        current_build=report_bytes[0x1E8],
        current_minor=report_bytes[0x1E9],
        current_major=report_bytes[0x1EA],
        committed_build=report_bytes[0x1EC],
        committed_minor=report_bytes[0x1ED],
        committed_major=report_bytes[0x1EE],
        launch_tcb=launch_tcb,
        signature=bytes(
            report_bytes[SEV_SNP_BODY_SIZE:SEV_SNP_BODY_SIZE + SEV_SNP_SIG_SIZE]
        ),
        raw=bytes(report_bytes),
    )


def bind_overt_envelope_to_report_data(envelope: BaseEnvelope) -> bytes:
    """64-byte REPORT_DATA that binds a SEV-SNP report to an OVERT envelope.

    REPORT_DATA = SHA-512(canonical_cbor(envelope_full_dict_including_signature))

    The hash covers all 9 envelope fields, including the Ed25519 signature.
    Any change to the envelope produces a different REPORT_DATA, so a TEE
    attestation carrying a given REPORT_DATA value can only correspond to
    that one specific envelope.
    """
    from vaara.attestation.iap import envelope_to_canonical_cbor

    cbor_bytes = envelope_to_canonical_cbor(envelope)
    return hashlib.sha512(cbor_bytes).digest()


def verify_sev_snp_report_signature(report: SEVSNPReport, vcek_pem: bytes) -> bool:
    """Verify the ECDSA P-384 signature on a SEV-SNP report against a VCEK.

    The VCEK (Versioned Chip Endorsement Key) is AMD's per-CPU signing key
    used to sign attestation reports (a VLEK public key works the same way).
    This checks the signature only; whether the key is AMD's is
    :func:`vaara.attestation._sev_snp_chain.verify_sev_snp_chain`.
    """
    if report.signature_algo != SIGNATURE_ALGO_ECDSA_P384_SHA384:
        raise TEEAttestationError(
            f"Unsupported SEV-SNP signature algo {report.signature_algo}; "
            f"only ECDSA P-384 SHA-384 (=1) is supported"
        )

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import (
            encode_dss_signature,
        )
    except ImportError as exc:
        raise TEEAttestationError(
            "cryptography not installed. Install with: "
            "pip install 'vaara[attestation]'"
        ) from exc

    try:
        vcek = serialization.load_pem_public_key(vcek_pem)
    except ValueError as exc:
        raise TEEAttestationError(f"Failed to load VCEK PEM: {exc}") from exc

    if not isinstance(vcek, ec.EllipticCurvePublicKey):
        raise TEEAttestationError("VCEK is not an EC public key")
    if not isinstance(vcek.curve, ec.SECP384R1):
        raise TEEAttestationError(
            f"VCEK curve is {vcek.curve.name}; expected secp384r1 (P-384)"
        )

    r_le = report.signature[:72]
    s_le = report.signature[72:144]
    r_int = int.from_bytes(r_le[:48], "little")
    s_int = int.from_bytes(s_le[:48], "little")

    der_sig = encode_dss_signature(r_int, s_int)

    try:
        vcek.verify(der_sig, report.body, ec.ECDSA(hashes.SHA384()))
        return True
    except InvalidSignature:
        return False


def verify_envelope_binding(report: SEVSNPReport, envelope: BaseEnvelope) -> bool:
    """Check that the report's REPORT_DATA matches SHA-512 of the envelope."""
    expected = bind_overt_envelope_to_report_data(envelope)
    return report.report_data == expected


class TEEAttester(Protocol):
    """Protocol for TEE attesters. Implementations emit attestation reports."""

    def emit(self, report_data: bytes) -> bytes:
        """Emit a binary attestation report carrying the supplied REPORT_DATA."""
        ...


class MockSEVSNPAttester:
    """Deterministic in-memory SEV-SNP attester for tests and CI.

    Builds a well-formed 1184-byte SEV-SNP attestation report with caller-
    supplied REPORT_DATA, signed by a caller-supplied ECDSA P-384 key. The
    resulting blob is byte-compatible with ``parse_sev_snp_report`` and
    ``verify_sev_snp_report_signature``.

    Not a substitute for real hardware attestation. The signing key has no
    AMD provenance, so any real-world verifier validating the chain to
    AMD's ARK will (correctly) reject reports from this attester.
    """

    def __init__(
        self,
        signing_key,
        *,
        measurement: bytes = b"\x00" * 48,
        version: int = 2,
        policy: int = 0,
    ):
        try:
            from cryptography.hazmat.primitives.asymmetric import ec
        except ImportError as exc:
            raise TEEAttestationError(
                "cryptography not installed. Install with: "
                "pip install 'vaara[attestation]'"
            ) from exc
        if not isinstance(signing_key, ec.EllipticCurvePrivateKey):
            raise TEEAttestationError("signing_key must be an EC private key")
        if not isinstance(signing_key.curve, ec.SECP384R1):
            raise TEEAttestationError("signing_key must be on secp384r1 (P-384)")
        if len(measurement) != 48:
            raise TEEAttestationError("measurement must be 48 bytes (SHA-384)")
        self._signing_key = signing_key
        self._measurement = measurement
        self._version = version
        self._policy = policy

    def emit(self, report_data: bytes) -> bytes:
        if len(report_data) != SEV_SNP_REPORT_DATA_SIZE:
            raise TEEAttestationError(
                f"report_data must be exactly {SEV_SNP_REPORT_DATA_SIZE} bytes; "
                f"got {len(report_data)}"
            )

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import (
            decode_dss_signature,
        )

        body = bytearray(SEV_SNP_BODY_SIZE)
        struct.pack_into("<I", body, 0x000, self._version)
        struct.pack_into("<Q", body, 0x008, self._policy)
        struct.pack_into("<I", body, 0x034, SIGNATURE_ALGO_ECDSA_P384_SHA384)
        body[0x050:0x090] = report_data
        body[0x090:0x0C0] = self._measurement

        der_sig = self._signing_key.sign(bytes(body), ec.ECDSA(hashes.SHA384()))
        r_int, s_int = decode_dss_signature(der_sig)
        r_bytes = r_int.to_bytes(48, "little")
        s_bytes = s_int.to_bytes(48, "little")

        sig_field = bytearray(SEV_SNP_SIG_SIZE)
        sig_field[:48] = r_bytes
        sig_field[72:72 + 48] = s_bytes

        return bytes(body) + bytes(sig_field)


class SEVSNPHostAttester:
    """Live SEV-SNP attester for a Linux guest in an AMD SEV-SNP confidential VM.

    Asks the AMD secure processor for a report through the kernel. It uses the
    configfs-tsm interface (``/sys/kernel/config/tsm/report``, Linux 6.7+) when
    the kernel offers it, else the ``SNP_GET_REPORT`` ioctl on
    ``/dev/sev-guest``. Both need root or a group with access to them.

    ``privlevel`` is the VMPL the report is requested for; ``None`` asks for the
    lowest the guest is allowed (``privlevel_floor``, 0 without an SVSM).

    :meth:`emit_with_certificates` also returns the certificates the host
    published for the extended report (VCEK or VLEK, ASK, ARK), when it did.
    """

    def __init__(
        self,
        device: str = "/dev/sev-guest",
        *,
        tsm_root: str = "/sys/kernel/config/tsm/report",
        privlevel: Optional[int] = None,
    ):
        self._device = Path(device)
        self._tsm_root = Path(tsm_root)
        self._privlevel = privlevel

    def emit(self, report_data: bytes) -> bytes:
        return self.emit_with_certificates(report_data)[0]

    def emit_with_certificates(self, report_data: bytes) -> tuple[bytes, bytes]:
        """(report, certificate table); the table is empty when none was given.

        Read the table with
        :func:`vaara.attestation._sev_snp_chain.parse_certificate_table`.
        """
        if len(report_data) != SEV_SNP_REPORT_DATA_SIZE:
            raise TEEAttestationError(
                f"report_data must be exactly {SEV_SNP_REPORT_DATA_SIZE} bytes"
            )
        # The sev-guest driver creates /dev/sev-guest in every SEV-SNP guest;
        # configfs-tsm can exist without it (other TEEs, or none at all).
        if not self._device.exists():
            raise TEEAttestationError(
                f"{self._device} not present. This host is not an SEV-SNP "
                f"guest. Use MockSEVSNPAttester for non-SEV-SNP test "
                f"environments, or capture a report from a real SEV-SNP "
                f"guest out of band."
            )
        if self._tsm_root.is_dir():
            return self._emit_configfs(report_data)
        return self._emit_ioctl(report_data), b""

    def _emit_configfs(self, report_data: bytes) -> tuple[bytes, bytes]:
        import os

        entry = self._tsm_root / f"vaara-{os.getpid()}-{os.urandom(8).hex()}"
        try:
            entry.mkdir()
        except OSError as exc:
            raise TEEAttestationError(f"cannot create {entry}: {exc}") from exc
        try:
            provider = (entry / "provider").read_text().strip()
            if provider != "sev_guest":
                raise TEEAttestationError(
                    f"configfs-tsm provider is {provider!r}, not sev_guest"
                )
            level = self._privlevel
            floor_file = entry / "privlevel_floor"
            if level is None and floor_file.exists():
                level = int(floor_file.read_text().strip() or 0)
            if level is not None and (entry / "privlevel").exists():
                (entry / "privlevel").write_text(str(level))
            (entry / "inblob").write_bytes(report_data)
            written = int((entry / "generation").read_text().strip())
            report = (entry / "outblob").read_bytes()
            certs = b""
            if (entry / "auxblob").exists():
                certs = (entry / "auxblob").read_bytes()
            # A write by anyone else between ours and the reads bumps generation.
            if int((entry / "generation").read_text().strip()) != written:
                raise TEEAttestationError(
                    "configfs-tsm entry changed while the report was read"
                )
        except OSError as exc:
            raise TEEAttestationError(f"configfs-tsm report failed: {exc}") from exc
        finally:
            try:
                entry.rmdir()
            except OSError:
                pass
        if len(report) != SEV_SNP_REPORT_SIZE:
            raise TEEAttestationError(
                f"configfs-tsm returned {len(report)} bytes, not a "
                f"{SEV_SNP_REPORT_SIZE}-byte SEV-SNP report"
            )
        return report, certs

    def _emit_ioctl(self, report_data: bytes) -> bytes:
        import ctypes
        import fcntl
        import os

        level = self._privlevel or 0
        # struct snp_report_req: user_data[64], u32 vmpl, rsvd[28].
        req = ctypes.create_string_buffer(
            report_data + struct.pack("<I", level) + bytes(28), 96
        )
        # struct snp_report_resp: the firmware's MSG_REPORT_RSP in 4000 bytes.
        resp = ctypes.create_string_buffer(4000)
        # struct snp_guest_request_ioctl: u8 msg_version, u64 req, u64 resp,
        # u64 exitinfo2, naturally aligned.
        arg = bytearray(
            struct.pack(
                "<B7xQQQ", 1, ctypes.addressof(req), ctypes.addressof(resp), 0
            )
        )
        try:
            fd = os.open(self._device, os.O_RDWR)
        except OSError as exc:
            raise TEEAttestationError(f"cannot open {self._device}: {exc}") from exc
        try:
            fcntl.ioctl(fd, _SNP_GET_REPORT, arg, True)
        except OSError as exc:
            exitinfo2 = struct.unpack_from("<Q", arg, 24)[0]
            raise TEEAttestationError(
                f"SNP_GET_REPORT failed: {exc} (firmware error "
                f"{exitinfo2 & 0xFFFFFFFF:#x}, VMM error {exitinfo2 >> 32:#x})"
            ) from exc
        finally:
            os.close(fd)
        # MSG_REPORT_RSP: u32 status, u32 report_size, 24 reserved, report.
        status, size = struct.unpack_from("<II", resp.raw, 0)
        if status != 0:
            raise TEEAttestationError(f"SNP_GET_REPORT status {status:#x}")
        if size != SEV_SNP_REPORT_SIZE:
            raise TEEAttestationError(
                f"SNP_GET_REPORT returned a {size}-byte report, not "
                f"{SEV_SNP_REPORT_SIZE}"
            )
        return resp.raw[32:32 + SEV_SNP_REPORT_SIZE]


# _IOWR('S', 0x0, struct snp_guest_request_ioctl), linux/sev-guest.h.
_SNP_GET_REPORT = (3 << 30) | (32 << 16) | (ord("S") << 8) | 0x0


__all__ = [
    "MockSEVSNPAttester",
    "SEVSNPHostAttester",
    "SEVSNPReport",
    "SEV_SNP_BODY_SIZE",
    "SEV_SNP_REPORT_DATA_SIZE",
    "SEV_SNP_REPORT_SIZE",
    "SEV_SNP_SIG_SIZE",
    "SIGNATURE_ALGO_ECDSA_P384_SHA384",
    "TEEAttestationError",
    "TEEAttester",
    "bind_overt_envelope_to_report_data",
    "parse_sev_snp_report",
    "verify_envelope_binding",
    "verify_sev_snp_report_signature",
]
