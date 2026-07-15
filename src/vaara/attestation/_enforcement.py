"""Attested enforcement: bind an AMD SEV-SNP report to a SEP-2828 record.

The enforcement point that writes execution records can itself run inside an
AMD SEV-SNP confidential VM. This module verifies that a *specific signed
execution record* was hashed inside such a VM, by checking a sibling SEV-SNP
attestation report whose 64-byte ``REPORT_DATA`` field carries
``SHA-512(canonical_json(record))``. It is the verify side; the report arrives
pre-captured (the enforcement point requests it at runtime via the chip).

What a passing check proves
---------------------------

At the top tier reachable today (``bound``, or ``measurement_pinned`` when a
measurement is pinned): an ECDSA-P384 SEV-SNP report carrying
``SHA-512(jcs(record))`` in ``REPORT_DATA`` verifies against the VCEK the caller
supplied, so this exact record's bytes were hashed inside *some* SEV-SNP CVM
whose VCEK the caller chose to trust.

What it does NOT prove
----------------------

1. That the enforcement *decision logic* executed in the enclave. ``REPORT_DATA``
   only shows that something inside the measured VM hashed the record and asked
   the chip to attest. ``enforcement_logic_basis`` is therefore always
   ``not_established``.
2. That the chip is a genuine AMD part. The VCEK -> ASK -> ARK chain to AMD's
   Key Distribution Service is not validated here (deferred, like
   :mod:`vaara.attestation.tee`); a :class:`~vaara.attestation.tee.MockSEVSNPAttester`
   report with no AMD provenance is byte-identical and passes the same check.
   ``vcek_chain_basis`` is therefore always ``caller_supplied_unverified`` in v0.
3. *Which* image ran, unless ``expected_measurement`` pins ``report.measurement``
   against an independently vetted launch measurement.
4. *When* enforcement happened. A SEV-SNP report has no timestamp or nonce, so a
   captured report can be re-presented against the same record; v0 makes no
   freshness claim.

The honest summary, stated plainly: until ``vcek_chain_basis`` is
``kds_verified`` *and* ``measurement_basis`` is ``pinned``, this verdict has no
component the submitter cannot forge. This is the deliberate contrast with the
cross-org handoff, where the eIDAS RFC 3161 anchor is the one un-forgeable
component. AMD's ARK is the analogous root here, and it is exactly the part v0
does not yet check, so a ``bound`` verdict is necessary but not sufficient for
genuine AMD hardware. The word ``attested`` is reserved for the future tier that
validates the AMD chain.

Install: ``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._attest_canonical import canonical_json
from vaara.attestation.tee import (
    SIGNATURE_ALGO_ECDSA_P384_SHA384,
    TEEAttestationError,
    parse_sev_snp_report,
    verify_sev_snp_report_signature,
)

ENFORCEMENT_SCHEMA = "vaara.enforcement-attestation/v0"

# parse_sev_snp_report reads the AMD ABI rev 1.55 (Table 22) field offsets, which
# match VERSION 2. v0 fails closed on any other version rather than misread a
# differently-laid-out report; widening to VERSION 3 is additive once the offset
# delta for REPORT_DATA / MEASUREMENT is confirmed against the AMD ABI.
_SUPPORTED_REPORT_VERSIONS = frozenset({2})


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
    and the report's measurement matches a caller-supplied vetted value). The
    tier ``attested`` is reserved for a future release that validates the VCEK
    chain to AMD's ARK and is never emitted in v0.

    ``vcek_chain_basis`` and ``measurement_basis`` are the two honesty fields:
    they record that the VCEK was trusted without a chain check and whether the
    measurement was pinned. ``enforcement_logic_basis`` is a constant disclaimer
    that binding a report to a record does not prove the record came from the
    image's decision path. The boolean sub-results (``signature_valid``,
    ``bound``, ...) make the tier reconstructable. ``report_context`` surfaces
    raw report fields (policy, vmpl, tcb, ...) without asserting anything about
    them. ``reason`` is non-normative prose.

    ``ok`` is the overall answer: in default mode, the signature verifies, the
    report binds to the record, and any supplied measurement matched. In
    ``strict`` mode it additionally requires the chain-rooted ``attested`` tier,
    which v0 cannot reach, so a strict pass is honestly unavailable until the
    AMD KDS chain is validated.
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
        }


def _normalize_hex(value: Optional[str]) -> Optional[bytes]:
    """Parse a hex string to bytes; None on absent or malformed input."""
    if not isinstance(value, str):
        return None
    try:
        return bytes.fromhex(value.strip())
    except ValueError:
        return None


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
    strict: bool,
    ok: bool,
) -> str:
    """A non-normative explanation that always carries the trust caveats."""
    parts: list[str] = []
    if not parsed:
        parts.append("the report did not parse as a 1184-byte SEV-SNP report")
    elif report_version not in _SUPPORTED_REPORT_VERSIONS:
        parts.append(
            f"report version {report_version} is not supported (v0 reads the "
            f"AMD ABI rev 1.55 / VERSION 2 layout)"
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
    # The always-on honesty caveats.
    parts.append(
        "the VCEK was trusted as supplied and not validated to AMD's ARK "
        "(KDS chain deferred), so a report with no AMD provenance passes the "
        "same check"
    )
    parts.append(
        "binding a report to a record does not prove the enforcement decision "
        "logic ran in the enclave"
    )
    if strict and not ok:
        parts.append(
            "strict mode requires a VCEK validated to AMD's ARK (KDS chain) and "
            "a pinned measurement, which v0 cannot establish"
        )
    return "; ".join(parts) + "."


def verify_enforcement(
    record: dict[str, Any],
    report_bytes: bytes,
    vcek_pem: bytes,
    *,
    expected_measurement: Optional[str] = None,
    strict: bool = False,
) -> EnforcementVerdict:
    """Verify a SEV-SNP attestation binds to a SEP-2828 record. One verdict.

    ``record`` is the on-disk record dict; ``report_bytes`` the binary SEV-SNP
    attestation report; ``vcek_pem`` the PEM-encoded VCEK to check the report
    signature against (trusted as supplied; its AMD chain is not validated).
    ``expected_measurement`` optionally pins ``report.measurement`` (hex) against
    an independently vetted launch measurement. ``strict`` requires the
    chain-rooted ``attested`` tier (unreachable in v0).

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

    # v0 never validates the VCEK chain and never proves enclave decision logic.
    vcek_chain_basis = "caller_supplied_unverified"
    enforcement_logic_basis = "not_established"

    crypto_ok = bool(
        parsed and version_ok and signature_algo_ok and signature_valid and bound
    )

    if not crypto_ok:
        tier = "unverified"
    elif measurement_basis == "pinned":
        tier = "measurement_pinned"
    else:
        # 'bound' even on pin_mismatch: the binding holds; the pin gates ``ok``.
        tier = "bound"

    if strict:
        # Requires the chain-rooted top tier, unreachable until KDS validation.
        ok = bool(
            crypto_ok
            and measurement_basis == "pinned"
            and vcek_chain_basis == "kds_verified"
        )
    else:
        ok = bool(crypto_ok and measurement_basis != "pin_mismatch")

    reason = _enforcement_reason(
        tier=tier,
        parsed=parsed,
        report_version=report_version,
        signature_algo_ok=signature_algo_ok,
        signature_valid=signature_valid,
        bound=bound,
        measurement_basis=measurement_basis,
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
    )


__all__ = [
    "ENFORCEMENT_SCHEMA",
    "EnforcementVerdict",
    "bind_record_to_report_data",
    "verify_enforcement",
]
