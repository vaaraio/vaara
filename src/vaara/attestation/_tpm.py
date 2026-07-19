# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""TPM 2.0 quote structures, IMA log replay, and a mock quoter.

The vendor-neutral, commodity-hardware twin of :mod:`vaara.attestation.tee`.
Where ``tee.py`` reads an AMD SEV-SNP report, this module reads a TPM 2.0
``TPM2_Quote`` and the kernel's IMA runtime-measurement log. The pairing is the
Phase-0 piece of the hardware-governance binding: a TPM quote proves a measured,
un-tampered platform state at a point in time, and its ``extraData`` slot carries
``SHA-256(jcs(record))`` so the quote binds to one specific SEP-2828 record. The
verdict lives in :mod:`vaara.attestation._tpm_binding`; this module is the wire
layer it stands on.

What this module parses
-----------------------

``parse_tpms_attest``: the ``TPMS_ATTEST`` structure a TPM emits for a quote
(magic, attest type, ``extraData``, clock info, firmware version, the quoted PCR
selection and the digest over those PCRs). TPM marshalling is big-endian, the
mirror of SEV-SNP's little-endian layout, so the integer unpacks here all read
``>``.

``parse_tpmt_signature``: the ``TPMT_SIGNATURE`` an ECDSA attestation key (AK)
produces. v0 reads the ECDSA case (``r``/``s`` as two ``TPM2B``), which covers
the vendor-neutral NIST P-256 / P-384 AKs; an RSA AK is a marshalling-only
addition tracked for a later release. The scheme hash is read from the signature
itself, so the verifier does not have to be told it out of band.

``replay_ima_pcr``: reconstructs the IMA PCR (10) by extending from zero with each
entry's template hash, the standard IMA aggregation. Lets a verifier confirm a
supplied IMA log is exactly the one folded into the quoted PCR digest, so the log
cannot be edited after the fact without breaking the chain to the AK signature.

Deferred (the honest boundary, mirrored from ``tee.py``)
--------------------------------------------------------

- AK provenance. The AK public key is consumed as supplied; its EK-certificate
  chain to a TPM vendor root, and the credential-activation step that proves the
  AK shares a TPM with that EK, are not validated here. A self-generated key is
  byte-identical to a real AK at this layer, exactly as a ``MockSEVSNPAttester``
  report is to a real SEV-SNP report.
- ``/dev/tpm0`` emission. Producing a live quote (``tpm2_createak`` /
  ``tpm2_quote``) needs the TPM tooling and ``tss``-group access; that lives in
  the capture script under ``scripts/tpm/``, not here. This module stays pure so it
  parses and verifies a pre-captured quote with no hardware present.
- RSA AKs, and PCR banks other than the one carried in the quote's own selection.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass


class TPMAttestationError(RuntimeError):
    """Raised on TPM quote / IMA parse, replay, or marshalling failures."""


# TPM 2.0 constants (Trusted Platform Module Library, Part 2: Structures).
TPM_GENERATED_VALUE = 0xFF544347  # "\xffTCG", the magic on a genuine attest.
TPM_ST_ATTEST_QUOTE = 0x8018

TPM_ALG_SHA1 = 0x0004
TPM_ALG_SHA256 = 0x000B
TPM_ALG_SHA384 = 0x000C
TPM_ALG_SHA512 = 0x000D
TPM_ALG_ECDSA = 0x0018

# extraData/qualifyingData is a TPM2B_DATA; its buffer maxes at sizeof(TPMU_HA),
# the TPM's largest IMPLEMENTED hash. That is only 64 on a TPM that implements
# SHA-512; most fTPMs cap at SHA-384 (48) and reject a 64-byte nonce TPM_RC_SIZE
# (confirmed on an AMD fTPM 2026-06-12). The binding nonce is therefore SHA-256
# (32 bytes), which fits every TPM 2.0 since SHA-256 is mandatory.
EXTRA_DATA_SIZE = 32

# IMA extends PCR 10 by convention (CONFIG_IMA_MEASURE_PCR_IDX default).
IMA_PCR = 10

_ALG_TO_HASHLIB = {
    TPM_ALG_SHA1: "sha1",
    TPM_ALG_SHA256: "sha256",
    TPM_ALG_SHA384: "sha384",
    TPM_ALG_SHA512: "sha512",
}
_DIGEST_LEN = {"sha1": 20, "sha256": 32, "sha384": 48, "sha512": 64}


@dataclass(frozen=True)
class PCRSelection:
    """One ``TPMS_PCR_SELECTION``: a hash bank and the PCR indices in it."""

    hash_alg: int
    pcrs: tuple[int, ...]


@dataclass(frozen=True)
class TPMSAttest:
    """A parsed ``TPMS_ATTEST`` of type ``TPM_ST_ATTEST_QUOTE``.

    ``raw`` is the exact attestation bytes the AK signed; the signature is
    verified over ``raw``, never over re-marshalled fields, so a parser that
    normalised anything could not silently change what was checked.
    """

    magic: int
    attest_type: int
    qualified_signer: bytes
    extra_data: bytes
    clock: int
    reset_count: int
    restart_count: int
    safe: int
    firmware_version: int
    pcr_selections: tuple[PCRSelection, ...]
    pcr_digest: bytes
    raw: bytes

    @property
    def magic_ok(self) -> bool:
        return self.magic == TPM_GENERATED_VALUE

    @property
    def is_quote(self) -> bool:
        return self.attest_type == TPM_ST_ATTEST_QUOTE


def _read_tpm2b(buf: bytes, off: int) -> tuple[bytes, int]:
    """Read a ``TPM2B`` (uint16 size, then that many bytes). Returns (data, off)."""
    if off + 2 > len(buf):
        raise TPMAttestationError("truncated TPM2B size field")
    (size,) = struct.unpack_from(">H", buf, off)
    off += 2
    if off + size > len(buf):
        raise TPMAttestationError("TPM2B length runs past end of buffer")
    return buf[off:off + size], off + size


def pcr_selection_to_indices(size_of_select: int, bitmap: bytes) -> tuple[int, ...]:
    """Decode a PCR-selection bitmap into ascending PCR indices.

    PCR ``i`` is selected when bit ``i % 8`` of byte ``i // 8`` is set, the
    little-endian-within-byte layout the TPM uses for ``pcrSelect``.
    """
    out: list[int] = []
    for byte_index in range(size_of_select):
        bits = bitmap[byte_index]
        for bit in range(8):
            if bits & (1 << bit):
                out.append(byte_index * 8 + bit)
    return tuple(out)


def parse_tpms_attest(attest_bytes: bytes) -> TPMSAttest:
    """Parse a ``TPMS_ATTEST`` quote. Big-endian throughout.

    Raises :class:`TPMAttestationError` on any structural problem (short buffer, a
    ``TPM2B`` whose length overruns, an unparseable PCR selection). Does *not*
    raise on a wrong magic or a non-quote type: those are returned as fields so
    the verifier can report them as a failed check rather than a crash, the way
    ``parse_sev_snp_report`` surfaces a bad version.
    """
    if len(attest_bytes) < 17:
        raise TPMAttestationError(
            f"TPMS_ATTEST too short: {len(attest_bytes)} bytes"
        )
    magic, attest_type = struct.unpack_from(">IH", attest_bytes, 0)
    off = 6
    qualified_signer, off = _read_tpm2b(attest_bytes, off)
    extra_data, off = _read_tpm2b(attest_bytes, off)

    # TPMS_CLOCK_INFO: clock(u64) resetCount(u32) restartCount(u32) safe(u8).
    if off + 17 > len(attest_bytes):
        raise TPMAttestationError("truncated clockInfo")
    clock, reset_count, restart_count, safe = struct.unpack_from(
        ">QIIB", attest_bytes, off
    )
    off += 17

    if off + 8 > len(attest_bytes):
        raise TPMAttestationError("truncated firmwareVersion")
    (firmware_version,) = struct.unpack_from(">Q", attest_bytes, off)
    off += 8

    # TPMS_QUOTE_INFO: TPML_PCR_SELECTION then a TPM2B_DIGEST pcrDigest.
    if off + 4 > len(attest_bytes):
        raise TPMAttestationError("truncated pcrSelect count")
    (count,) = struct.unpack_from(">I", attest_bytes, off)
    off += 4
    selections: list[PCRSelection] = []
    for _ in range(count):
        if off + 3 > len(attest_bytes):
            raise TPMAttestationError("truncated TPMS_PCR_SELECTION header")
        hash_alg, size_of_select = struct.unpack_from(">HB", attest_bytes, off)
        off += 3
        if off + size_of_select > len(attest_bytes):
            raise TPMAttestationError("pcrSelect bitmap runs past end of buffer")
        bitmap = attest_bytes[off:off + size_of_select]
        off += size_of_select
        selections.append(
            PCRSelection(hash_alg, pcr_selection_to_indices(size_of_select, bitmap))
        )

    pcr_digest, off = _read_tpm2b(attest_bytes, off)

    return TPMSAttest(
        magic=magic,
        attest_type=attest_type,
        qualified_signer=qualified_signer,
        extra_data=extra_data,
        clock=clock,
        reset_count=reset_count,
        restart_count=restart_count,
        safe=safe,
        firmware_version=firmware_version,
        pcr_selections=tuple(selections),
        pcr_digest=pcr_digest,
        raw=bytes(attest_bytes),
    )


@dataclass(frozen=True)
class TPMTSignature:
    """A parsed ECDSA ``TPMT_SIGNATURE``: scheme hash and the (r, s) integers."""

    sig_alg: int
    hash_alg: int
    r: int
    s: int


def parse_tpmt_signature(sig_bytes: bytes) -> TPMTSignature:
    """Parse a ``TPMT_SIGNATURE``. v0 reads the ECDSA case only.

    Layout: ``sigAlg`` (u16) then, for ECDSA, ``hash`` (u16) and two ``TPM2B``
    integers ``signatureR`` / ``signatureS``. Raises
    :class:`TPMAttestationError` for a non-ECDSA algorithm (RSA AKs are a tracked
    addition) or a malformed structure.
    """
    if len(sig_bytes) < 4:
        raise TPMAttestationError("TPMT_SIGNATURE too short")
    (sig_alg,) = struct.unpack_from(">H", sig_bytes, 0)
    if sig_alg != TPM_ALG_ECDSA:
        raise TPMAttestationError(
            f"unsupported signature algorithm 0x{sig_alg:04x}; v0 reads ECDSA "
            f"(0x{TPM_ALG_ECDSA:04x}) only"
        )
    (hash_alg,) = struct.unpack_from(">H", sig_bytes, 2)
    r_bytes, off = _read_tpm2b(sig_bytes, 4)
    s_bytes, _ = _read_tpm2b(sig_bytes, off)
    return TPMTSignature(
        sig_alg=sig_alg,
        hash_alg=hash_alg,
        r=int.from_bytes(r_bytes, "big"),
        s=int.from_bytes(s_bytes, "big"),
    )


def hash_name(hash_alg: int) -> str:
    """Map a ``TPM_ALG_*`` hash id to its hashlib name, or raise."""
    name = _ALG_TO_HASHLIB.get(hash_alg)
    if name is None:
        raise TPMAttestationError(f"unsupported hash algorithm 0x{hash_alg:04x}")
    return name


def pcr_digest_over(
    selections: "tuple[PCRSelection, ...] | list[PCRSelection]",
    pcr_values: dict[int, bytes],
    digest_alg: int,
) -> bytes:
    """Recompute the quoted ``pcrDigest`` from supplied PCR values.

    The TPM computes ``pcrDigest = H(PCR[i0] || PCR[i1] || ...)`` over every
    selected PCR, in selection order and ascending PCR index within each
    selection. Crucially ``H`` is the *signing scheme's* hash (``digest_alg``),
    not the PCR bank's hash: a P-384 AK quoting the SHA-256 bank folds 32-byte
    PCR values through SHA-384. Each value's length is still checked against its
    own bank. Recomputing this and comparing to the value the AK signed proves
    the supplied ``pcr_values`` are the ones folded into the quote.

    ``pcr_values`` is keyed by PCR index; v0 assumes a single bank per quote (the
    usual ``tpm2_quote -l sha256:10``), so one index maps to one value.
    """
    h = hashlib.new(hash_name(digest_alg))
    for selection in selections:
        bank_len = _DIGEST_LEN[hash_name(selection.hash_alg)]
        for pcr in selection.pcrs:
            if pcr not in pcr_values:
                raise TPMAttestationError(
                    f"PCR {pcr} is in the quote selection but not in the "
                    f"supplied PCR values"
                )
            value = pcr_values[pcr]
            if len(value) != bank_len:
                raise TPMAttestationError(
                    f"PCR {pcr} value is {len(value)} bytes; expected "
                    f"{bank_len} for its bank"
                )
            h.update(value)
    return h.digest()


def parse_ima_template_hashes(ima_log: str, hash_name_: str) -> tuple[bytes, ...]:
    """Pull the per-entry template hashes out of an IMA ascii log.

    The ascii ``runtime_measurements`` format is whitespace-separated, one entry
    per line, with the template hash in the second column:
    ``<pcr> <template-hash-hex> <template-name> <...>``. Blank lines are skipped.
    Each template hash must be the bank's digest length, so a log from the wrong
    bank (sha1 hashes against a sha256 quote) is rejected rather than silently
    mis-replayed.
    """
    want = _DIGEST_LEN[hash_name_]
    hashes: list[bytes] = []
    for lineno, line in enumerate(ima_log.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise TPMAttestationError(
                f"IMA log line {lineno} has too few fields"
            )
        try:
            digest = bytes.fromhex(parts[1])
        except ValueError as exc:
            raise TPMAttestationError(
                f"IMA log line {lineno}: template hash is not hex"
            ) from exc
        if len(digest) != want:
            raise TPMAttestationError(
                f"IMA log line {lineno}: template hash is {len(digest)} bytes; "
                f"expected {want} for {hash_name_} (wrong measurement bank?)"
            )
        hashes.append(digest)
    return tuple(hashes)


def replay_ima_pcr(ima_log: str, hash_alg: int) -> bytes:
    """Replay an IMA ascii log into its PCR (10) aggregate.

    PCR 10 resets to all zeros, then each measurement extends it:
    ``PCR = H(PCR || template_hash)``. One documented IMA quirk is preserved: an
    entry whose template hash is all zeros (a measurement violation) extends the
    PCR with all ``0xFF`` instead, so a log containing violations still replays to
    the real PCR value. On a clean log the substitution never fires.
    """
    name = hash_name(hash_alg)
    width = _DIGEST_LEN[name]
    acc = bytes(width)
    zero = bytes(width)
    ones = b"\xff" * width
    for template_hash in parse_ima_template_hashes(ima_log, name):
        measured = ones if template_hash == zero else template_hash
        acc = hashlib.new(name, acc + measured).digest()
    return acc


def _marshal_tpm2b(data: bytes) -> bytes:
    return struct.pack(">H", len(data)) + data


def _marshal_pcr_selection(selection: PCRSelection) -> bytes:
    if not selection.pcrs:
        size_of_select = 3
        bitmap = bytearray(size_of_select)
    else:
        size_of_select = max(3, (max(selection.pcrs) // 8) + 1)
        bitmap = bytearray(size_of_select)
        for pcr in selection.pcrs:
            bitmap[pcr // 8] |= 1 << (pcr % 8)
    return struct.pack(">HB", selection.hash_alg, size_of_select) + bytes(bitmap)


class MockTPMQuoter:
    """Build and sign a ``TPMS_ATTEST`` quote with a software EC key.

    The TPM-side counterpart of :class:`~vaara.attestation.tee.MockSEVSNPAttester`:
    it lets the test and conformance vectors exercise the full wire path
    (marshal a quote, ECDSA-sign the exact attest bytes, emit a ``TPMT_SIGNATURE``)
    with no hardware. It is honestly not a TPM: nothing here roots in an EK or a
    vendor certificate, which is precisely the provenance the verifier records as
    unestablished.
    """

    def __init__(
        self,
        signing_key,
        *,
        bank: int = TPM_ALG_SHA256,
        scheme_hash: int = TPM_ALG_SHA256,
        firmware_version: int = 0x0001_0002_0003_0004,
    ) -> None:
        try:
            from cryptography.hazmat.primitives.asymmetric import ec
        except ImportError as exc:  # pragma: no cover - extra not installed
            raise TPMAttestationError(
                "cryptography not installed. Install with: "
                "pip install 'vaara[attestation]'"
            ) from exc
        if not isinstance(signing_key, ec.EllipticCurvePrivateKey):
            raise TPMAttestationError("signing_key must be an EC private key")
        self._key = signing_key
        self._bank = bank
        self._scheme_hash = scheme_hash
        self._firmware_version = firmware_version

    def attest_bytes(
        self,
        extra_data: bytes,
        pcr_values: dict[int, bytes],
        *,
        clock: int = 0,
        reset_count: int = 0,
        restart_count: int = 0,
    ) -> bytes:
        """Marshal a ``TPMS_ATTEST`` over the given nonce and PCR values.

        ``clock``, ``reset_count`` and ``restart_count`` fill the ``TPMS_CLOCK_INFO``
        block. They default to zero (one quote in isolation, as Phase 0 uses), but
        a continuous-attestation chain advances ``clock`` per quote and holds the
        reset counters fixed across a single boot, which is exactly what the chain
        verifier checks for. They let the mock reproduce both a clean chain and the
        reboot / clock-rollback negatives with no hardware.
        """
        if len(extra_data) > EXTRA_DATA_SIZE:
            raise TPMAttestationError(
                f"extra_data must be at most {EXTRA_DATA_SIZE} bytes"
            )
        selection = PCRSelection(self._bank, tuple(sorted(pcr_values)))
        pcr_digest = pcr_digest_over((selection,), pcr_values, self._scheme_hash)
        out = bytearray()
        out += struct.pack(">IH", TPM_GENERATED_VALUE, TPM_ST_ATTEST_QUOTE)
        out += _marshal_tpm2b(b"")  # qualifiedSigner (empty for the mock)
        out += _marshal_tpm2b(extra_data)
        # clockInfo: clock(u64) resetCount(u32) restartCount(u32) safe(u8=1).
        out += struct.pack(">QIIB", clock, reset_count, restart_count, 1)
        out += struct.pack(">Q", self._firmware_version)
        out += struct.pack(">I", 1)  # one PCR selection
        out += _marshal_pcr_selection(selection)
        out += _marshal_tpm2b(pcr_digest)
        return bytes(out)

    def sign(self, attest_bytes: bytes) -> bytes:
        """ECDSA-sign the attest bytes and emit a ``TPMT_SIGNATURE``."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives.asymmetric.utils import (
            decode_dss_signature,
        )

        hash_cls = {
            TPM_ALG_SHA256: hashes.SHA256,
            TPM_ALG_SHA384: hashes.SHA384,
            TPM_ALG_SHA512: hashes.SHA512,
        }[self._scheme_hash]
        der = self._key.sign(attest_bytes, ec.ECDSA(hash_cls()))
        r, s = decode_dss_signature(der)
        size = (self._key.curve.key_size + 7) // 8
        out = struct.pack(">HH", TPM_ALG_ECDSA, self._scheme_hash)
        out += _marshal_tpm2b(r.to_bytes(size, "big"))
        out += _marshal_tpm2b(s.to_bytes(size, "big"))
        return out

    def quote(
        self,
        extra_data: bytes,
        pcr_values: dict[int, bytes],
        *,
        clock: int = 0,
        reset_count: int = 0,
        restart_count: int = 0,
    ) -> tuple[bytes, bytes]:
        """Return ``(attest_bytes, tpmt_signature_bytes)`` for one quote."""
        attest = self.attest_bytes(
            extra_data,
            pcr_values,
            clock=clock,
            reset_count=reset_count,
            restart_count=restart_count,
        )
        return attest, self.sign(attest)


__all__ = [
    "EXTRA_DATA_SIZE",
    "IMA_PCR",
    "MockTPMQuoter",
    "PCRSelection",
    "TPMAttestationError",
    "TPMSAttest",
    "TPMTSignature",
    "TPM_ALG_ECDSA",
    "TPM_ALG_SHA1",
    "TPM_ALG_SHA256",
    "TPM_ALG_SHA384",
    "TPM_ALG_SHA512",
    "TPM_GENERATED_VALUE",
    "TPM_ST_ATTEST_QUOTE",
    "hash_name",
    "parse_ima_template_hashes",
    "parse_tpms_attest",
    "parse_tpmt_signature",
    "pcr_digest_over",
    "pcr_selection_to_indices",
    "replay_ima_pcr",
]
