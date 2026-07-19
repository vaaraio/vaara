# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hardware TEE attestation hook for OVERT 1.0 Base Envelopes.

Status: experimental. Adds an optional hardware-rooted attestation layer
alongside the Ed25519 (or ML-DSA-65) signature already on the OVERT 1.0
Base Envelope. Initial backend is AMD SEV-SNP, the natural fit for the
confidential-VM deployment model used in agent runtimes. Intel TDX and
Intel SGX backends are tracked for later releases.

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

What ships in v0.18.0
---------------------

- ``parse_sev_snp_report``: binary parser for the 1184-byte
  attestation-report structure (AMD SEV-SNP ABI Specification rev. 1.55,
  Table 22).
- ``bind_overt_envelope_to_report_data``: computes the 64-byte
  REPORT_DATA value that binds a TEE report to a specific OVERT envelope.
- ``verify_sev_snp_report_signature``: validates the ECDSA P-384 over the
  report body against a caller-supplied VCEK PEM.
- ``verify_envelope_binding``: confirms the report's REPORT_DATA matches
  SHA-512 of the supplied envelope.
- ``MockSEVSNPAttester``: deterministic in-memory attester for tests and
  CI, building byte-compatible report blobs signed with a caller-supplied
  ECDSA P-384 key.
- ``SEVSNPHostAttester``: skeleton that wraps ``/dev/sev-guest``. Raises
  a clear error when not on an SEV-SNP host; the real ioctl path is
  tracked for v0.19+.

What does NOT ship in v0.18.0
-----------------------------

- AMD KDS-based cert-chain validation (VCEK to ASK to ARK). Validating a
  VCEK against AMD's Key Distribution Service requires a network fetch
  against https://kdsintf.amd.com/ and is tracked for v0.19+. For now,
  callers must obtain a trusted VCEK out of band and supply it directly.
- ``/dev/sev-guest`` ioctl emission. The SNP_GET_REPORT ioctl is
  well-defined in linux/sev-guest.h but not exercised here until a tested
  SEV-SNP guest host is available.
- Intel TDX, Intel SGX backends. Same module shape will accommodate them
  via additional attester classes.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
    used to sign attestation reports. v0.18.0 only validates the report
    signature against a supplied VCEK; full chain validation against AMD's
    Key Distribution Service is tracked for v0.19+.
    """
    if report.signature_algo != SIGNATURE_ALGO_ECDSA_P384_SHA384:
        raise TEEAttestationError(
            f"Unsupported SEV-SNP signature algo {report.signature_algo}; "
            f"only ECDSA P-384 SHA-384 (=1) is supported in v0.18.0"
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
    """Live SEV-SNP attester that requests reports from /dev/sev-guest.

    Only functional inside an actual SEV-SNP confidential VM. Raises a
    clear error on non-SEV-SNP hosts. The SNP_GET_REPORT ioctl path is
    documented in linux/sev-guest.h and is tracked for v0.19+ once a
    tested SEV-SNP guest is available for integration testing.
    """

    def __init__(self, device: str = "/dev/sev-guest"):
        self._device = Path(device)

    def emit(self, report_data: bytes) -> bytes:
        if len(report_data) != SEV_SNP_REPORT_DATA_SIZE:
            raise TEEAttestationError(
                f"report_data must be exactly {SEV_SNP_REPORT_DATA_SIZE} bytes"
            )
        if not self._device.exists():
            raise TEEAttestationError(
                f"{self._device} not present. This host is not an SEV-SNP "
                f"guest. Use MockSEVSNPAttester for non-SEV-SNP test "
                f"environments, or capture a report from a real SEV-SNP "
                f"guest out of band."
            )
        raise TEEAttestationError(
            "SEV-SNP /dev/sev-guest ioctl emission is not implemented in "
            "v0.18.0. Tracked for v0.19+ once a tested SEV-SNP host is "
            "available. Use MockSEVSNPAttester or pre-captured reports."
        )


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
