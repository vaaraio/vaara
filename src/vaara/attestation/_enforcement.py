# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Attested enforcement: bind an AMD SEV-SNP report to a SEP-2828 record.

The enforcement point that writes execution records can itself run inside an
AMD SEV-SNP confidential VM. This module verifies that a *specific signed
execution record* was hashed inside such a VM, by checking a sibling SEV-SNP
attestation report whose 64-byte ``REPORT_DATA`` field carries
``SHA-512(canonical_json(record))``. It is the verify side; the report arrives
pre-captured (the enforcement point requests it at runtime via the chip).

What a passing check proves
---------------------------

``bound`` (or ``measurement_pinned`` when a measurement is pinned): an
ECDSA-P384 SEV-SNP report carrying ``SHA-512(jcs(record))`` in ``REPORT_DATA``
verifies against the VCEK the caller supplied, so this exact record's bytes
were hashed inside *some* SEV-SNP CVM whose VCEK the caller chose to trust.

``attested``: the caller also supplied the AMD chain (``amd_chain``), the VCEK
chains to AMD's pinned root for the product and matches the report's chip and
TCB (:func:`vaara.attestation._sev_snp_chain.verify_sev_snp_chain`), and the
measurement is pinned. This record's bytes were hashed inside a genuine AMD
SEV-SNP guest running the pinned image.

What it does NOT prove
----------------------

1. That the enforcement *decision logic* executed in the enclave. ``REPORT_DATA``
   only shows that something inside the measured VM hashed the record and asked
   the chip to attest. ``enforcement_logic_basis`` is therefore always
   ``not_established``.
2. That the chip is a genuine AMD part, unless ``amd_chain`` is supplied. Without
   it a :class:`~vaara.attestation.tee.MockSEVSNPAttester` report with no AMD
   provenance is byte-identical and passes the same check, and
   ``vcek_chain_basis`` reads ``caller_supplied_unverified``. With it the basis
   reads ``kds_verified`` or ``chain_failed``.
3. *Which* image ran, unless ``expected_measurement`` pins ``report.measurement``
   against an independently vetted launch measurement.
4. *When* enforcement happened. A SEV-SNP report has no timestamp or nonce, so a
   captured report can be re-presented against the same record; v0 makes no
   freshness claim.

Stated plainly: until ``vcek_chain_basis`` is ``kds_verified`` *and*
``measurement_basis`` is ``pinned``, this verdict has no component the submitter
cannot forge. AMD's ARK is the un-forgeable root here, as the eIDAS RFC 3161
anchor is in the cross-org handoff, so a ``bound`` verdict is necessary but not
sufficient for genuine AMD hardware.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._attest_canonical import canonical_json
from vaara.attestation._sev_snp_chain import _load_cert, verify_sev_snp_chain
from vaara.attestation.tee import (
    SIGNATURE_ALGO_ECDSA_P384_SHA384,
    TEEAttestationError,
    parse_sev_snp_report,
    verify_sev_snp_report_signature,
)

ENFORCEMENT_SCHEMA = "vaara.enforcement-attestation/v0"

# parse_sev_snp_report reads the AMD ABI rev 1.55 (Table 22) field offsets.
# Versions 3 to 5 only fill reserved bytes (CPUID at 0x188, mitigation vectors at
# 0x1F8), so every field read here sits at the same offset. Any other version
# fails closed rather than be misread.
_SUPPORTED_REPORT_VERSIONS = frozenset({2, 3, 4, 5})


def bind_record_to_report_data(record: dict[str, Any]) -> bytes:
    """The 64-byte ``REPORT_DATA`` that binds a SEV-SNP report to a record.

    ``REPORT_DATA = SHA-512(canonical_json(record))`` over the FULL on-disk
    record dict, *including* its top-level ``signature`` field. SHA-512 is 64
    bytes, exactly the ``REPORT_DATA`` slot; SHA-256 would under-fill it. This is
    the deliberate divergence from the handoff anchor imprint
    (``sha256(jcs(record))``): same record bytes, different digest, because the
    carriers differ (a 64-byte hardware slot vs an RFC 3161 imprint).

    The full record is hashed, not the five signed blocks alone, because the
    signed-block subset is signature-malleable: ``rfc8785`` over
    ``{version, alg, backLink, outcomeDerived, receiptAsserted}`` is byte-identical
    when only ``signature`` changes, so a subset binding would let a report bound
    to a genuinely-signed record equally bind a variant carrying a stripped or
    forged signature. Hashing the whole record closes that.
    """
    return hashlib.sha512(canonical_json(record)).digest()


@dataclass(frozen=True)
class EnforcementVerdict:
    """Verdict over a SEV-SNP attestation bound to a SEP-2828 record.

    ``tier`` is the single label, one of ``unverified`` (signature or binding
    failed), ``bound`` (signature verifies against the supplied VCEK and
    ``REPORT_DATA`` binds to this record), or ``measurement_pinned`` (``bound``
    and the report's measurement matches a caller-supplied vetted value), or
    ``attested`` (``measurement_pinned`` and the VCEK chains to AMD's root).

    ``vcek_chain_basis`` and ``measurement_basis`` are the two honesty fields:
    they record whether the VCEK was checked against AMD's chain and whether the
    measurement was pinned. ``amd_chain`` is the chain check's own verdict, or
    None when no chain was supplied. ``enforcement_logic_basis`` is a constant disclaimer
    that binding a report to a record does not prove the record came from the
    image's decision path. The boolean sub-results (``signature_valid``,
    ``bound``, ...) make the tier reconstructable. ``report_context`` surfaces
    raw report fields (policy, vmpl, tcb, ...) without asserting anything about
    them. ``reason`` is non-normative prose.

    ``ok`` is the overall answer: in default mode, the signature verifies, the
    report binds to the record, and any supplied measurement matched. In
    ``strict`` mode it requires the ``attested`` tier.
    """

    schema: str
    tier: str
    parsed: bool
    report_version: Optional[int]
    signature_algo_ok: bool
    signature_valid: bool
    bound: bool
    report_data_expected: str
    report_data_actual: Optional[str]
    measurement: Optional[str]
    expected_measurement: Optional[str]
    measurement_basis: str
    vcek_chain_basis: str
    enforcement_logic_basis: str
    report_context: dict[str, Any]
    strict: bool
    ok: bool
    record: dict[str, Any]
    reason: str
    amd_chain: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "tier": self.tier,
            "ok": self.ok,
            "strict": self.strict,
            "parsed": self.parsed,
            "report_version": self.report_version,
            "signature_algo_ok": self.signature_algo_ok,
            "signature_valid": self.signature_valid,
            "bound": self.bound,
            "report_data_expected": self.report_data_expected,
            "report_data_actual": self.report_data_actual,
            "measurement": self.measurement,
            "expected_measurement": self.expected_measurement,
            "measurement_basis": self.measurement_basis,
            "vcek_chain_basis": self.vcek_chain_basis,
            "enforcement_logic_basis": self.enforcement_logic_basis,
            "report_context": self.report_context,
            "record": self.record,
            "reason": self.reason,
            "amd_chain": self.amd_chain,
        }


def _normalize_hex(value: Optional[str]) -> Optional[bytes]:
    """Parse a hex string to bytes; None on absent or malformed input."""
    if not isinstance(value, str):
        return None
    try:
        return bytes.fromhex(value.strip())
    except ValueError:
        return None


def _same_key(vcek_cert: bytes, vcek_pem: bytes) -> bool:
    """True if the VCEK certificate carries the public key in ``vcek_pem``."""
    from cryptography.hazmat.primitives import serialization

    try:
        pem = serialization.load_pem_public_key(vcek_pem)
        cert_key = _load_cert(vcek_cert).public_key()
    except (TEEAttestationError, ValueError):
        return False
    fmt = (serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return bool(pem.public_bytes(*fmt) == cert_key.public_bytes(*fmt))


def _report_context(report: Any) -> dict[str, Any]:
    """Raw report fields surfaced for inspection. None are gated on in v0.

    These describe the platform and guest at attestation time. ``policy``
    (debug / migration / SMT bits), ``vmpl`` (the privilege level that requested
    the report), and the TCB / SVN values matter for a vetted-image policy, but
    pinning them needs a deployment model, so v0 reports them without judgement.
    """
    return {
        "vmpl": report.vmpl,
        "policy": report.policy,
        "guest_svn": report.guest_svn,
        "reported_tcb": report.reported_tcb,
        "launch_tcb": report.launch_tcb,
        "chip_id": report.chip_id.hex(),
    }


def _enforcement_reason(
    *,
    tier: str,
    parsed: bool,
    report_version: Optional[int],
    signature_algo_ok: bool,
    signature_valid: bool,
    bound: bool,
    measurement_basis: str,
    vcek_chain_basis: str,
    chain_reason: Optional[str],
    strict: bool,
    ok: bool,
) -> str:
    """A non-normative explanation that always carries the trust caveats."""
    parts: list[str] = []
    if not parsed:
        parts.append("the report did not parse as a 1184-byte SEV-SNP report")
    elif report_version not in _SUPPORTED_REPORT_VERSIONS:
        parts.append(
            f"report version {report_version} is not supported (versions 2 "
            f"to 5 are read)"
        )
    elif not signature_algo_ok:
        parts.append("the report signature algorithm is not ECDSA-P384-SHA384")
    elif not signature_valid:
        parts.append("the report signature did not verify against the supplied VCEK")
    elif not bound:
        parts.append(
            "REPORT_DATA does not equal sha512(jcs(record)): the report does not "
            "bind to this record"
        )
    else:
        parts.append(
            "the report verifies against the supplied VCEK and REPORT_DATA binds "
            "to this record"
        )
        if measurement_basis == "pinned":
            parts.append("the measurement matches the pinned reference")
        elif measurement_basis == "pin_mismatch":
            parts.append(
                "but the measurement does NOT match the pinned reference "
                "(a different image ran)"
            )
        else:
            parts.append(
                "the measurement is reported but not pinned; supply "
                "expected_measurement from a reproducible build or a trusted "
                "channel to learn which image ran"
            )
    if vcek_chain_basis == "kds_verified":
        parts.append("the VCEK chains to AMD's root and matches the report")
    elif vcek_chain_basis == "chain_failed":
        parts.append(f"the AMD chain check failed: {chain_reason}")
    else:
        parts.append(
            "the VCEK was trusted as supplied and not validated to AMD's ARK "
            "(no amd_chain given), so a report with no AMD provenance passes "
            "the same check"
        )
    parts.append(
        "binding a report to a record does not prove the enforcement decision "
        "logic ran in the enclave"
    )
    if strict and not ok:
        parts.append(
            "strict mode requires a VCEK validated to AMD's ARK and a pinned "
            "measurement"
        )
    return "; ".join(parts) + "."


def verify_enforcement(
    record: dict[str, Any],
    report_bytes: bytes,
    vcek_pem: bytes,
    *,
    expected_measurement: Optional[str] = None,
    strict: bool = False,
    amd_chain: Optional[tuple[bytes, bytes, bytes]] = None,
) -> EnforcementVerdict:
    """Verify a SEV-SNP attestation binds to a SEP-2828 record. One verdict.

    ``record`` is the on-disk record dict; ``report_bytes`` the binary SEV-SNP
    attestation report; ``vcek_pem`` the PEM-encoded VCEK to check the report
    signature against. ``amd_chain`` is ``(vcek_cert, ask_cert, ark_cert)``, PEM
    or DER; when given, the VCEK certificate must carry the same key as
    ``vcek_pem`` and chain to AMD's pinned root. ``expected_measurement``
    optionally pins ``report.measurement`` (hex) against an independently
    vetted launch measurement. ``strict`` requires the ``attested`` tier.

    A malformed report yields ``tier='unverified'`` (``parsed=False``), never a
    traceback. Raises :class:`ValueError` if ``record`` is not a JSON object or
    cannot be canonicalised. Propagates :class:`TEEAttestationError` only for a
    bad VCEK input (unloadable PEM, wrong curve), which is a verifier-side error.
    """
    if not isinstance(record, dict):
        raise ValueError(
            f"record must be a JSON object, got {type(record).__name__}"
        )
    try:
        expected_report_data = bind_record_to_report_data(record)
    except Exception as exc:  # noqa: BLE001 - canonical_json raises on bad shapes
        raise ValueError(f"cannot canonicalise record: {exc}") from exc
    report_data_expected = expected_report_data.hex()

    parsed = False
    report_version: Optional[int] = None
    version_ok = False
    signature_algo_ok = False
    signature_valid = False
    bound = False
    report_data_actual: Optional[str] = None
    measurement: Optional[str] = None
    report_context: dict[str, Any] = {}

    try:
        report = parse_sev_snp_report(report_bytes)
        parsed = True
    except TEEAttestationError:
        report = None

    if report is not None:
        report_version = report.version
        report_data_actual = report.report_data.hex()
        measurement = report.measurement.hex()
        report_context = _report_context(report)
        version_ok = report.version in _SUPPORTED_REPORT_VERSIONS
        signature_algo_ok = (
            report.signature_algo == SIGNATURE_ALGO_ECDSA_P384_SHA384
        )
        # Constant-time compare of all 64 REPORT_DATA bytes.
        bound = hmac.compare_digest(report.report_data, expected_report_data)
        if version_ok and signature_algo_ok:
            # A bad VCEK (unloadable PEM, non-EC, wrong curve) raises; a genuine
            # signature mismatch returns False. Let the input error propagate.
            signature_valid = verify_sev_snp_report_signature(report, vcek_pem)

    # Measurement pin (independent of the binding result).
    if expected_measurement is None:
        measurement_basis = "unpinned"
    else:
        expected_bytes = _normalize_hex(expected_measurement)
        if (
            report is not None
            and expected_bytes is not None
            and hmac.compare_digest(report.measurement, expected_bytes)
        ):
            measurement_basis = "pinned"
        else:
            measurement_basis = "pin_mismatch"

    vcek_chain_basis = "caller_supplied_unverified"
    chain_dict: Optional[dict[str, Any]] = None
    chain_reason: Optional[str] = None
    if amd_chain is not None:
        vcek_chain_basis = "chain_failed"
        if report is None:
            chain_reason = "the report did not parse"
        else:
            chain = verify_sev_snp_chain(report, *amd_chain)
            chain_dict = chain.to_dict()
            chain_reason = chain.reason.rstrip(".")
            if chain.ok and not _same_key(amd_chain[0], vcek_pem):
                chain_reason = "the chain's VCEK is not the key in vcek_pem"
            elif chain.ok:
                vcek_chain_basis = "kds_verified"
    # Binding a report to a record never proves the enclave's decision logic.
    enforcement_logic_basis = "not_established"

    crypto_ok = bool(
        parsed and version_ok and signature_algo_ok and signature_valid and bound
    )

    if not crypto_ok:
        tier = "unverified"
    elif measurement_basis == "pinned" and vcek_chain_basis == "kds_verified":
        tier = "attested"
    elif measurement_basis == "pinned":
        tier = "measurement_pinned"
    else:
        # 'bound' even on pin_mismatch: the binding holds; the pin gates ``ok``.
        tier = "bound"

    if strict:
        ok = tier == "attested"
    else:
        ok = bool(
            crypto_ok
            and measurement_basis != "pin_mismatch"
            and vcek_chain_basis != "chain_failed"
        )

    reason = _enforcement_reason(
        tier=tier,
        parsed=parsed,
        report_version=report_version,
        signature_algo_ok=signature_algo_ok,
        signature_valid=signature_valid,
        bound=bound,
        measurement_basis=measurement_basis,
        vcek_chain_basis=vcek_chain_basis,
        chain_reason=chain_reason,
        strict=strict,
        ok=ok,
    )

    return EnforcementVerdict(
        schema=ENFORCEMENT_SCHEMA,
        tier=tier,
        parsed=parsed,
        report_version=report_version,
        signature_algo_ok=signature_algo_ok,
        signature_valid=signature_valid,
        bound=bound,
        report_data_expected=report_data_expected,
        report_data_actual=report_data_actual,
        measurement=measurement,
        expected_measurement=expected_measurement,
        measurement_basis=measurement_basis,
        vcek_chain_basis=vcek_chain_basis,
        enforcement_logic_basis=enforcement_logic_basis,
        report_context=report_context,
        strict=strict,
        ok=ok,
        record=record,
        reason=reason,
        amd_chain=chain_dict,
    )


__all__ = [
    "ENFORCEMENT_SCHEMA",
    "EnforcementVerdict",
    "bind_record_to_report_data",
    "verify_enforcement",
]
