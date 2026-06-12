"""The TPM evidence bundle: one self-contained file the verifier consumes.

Phase-0 binds three things a regulator must check together: the signed SEP-2828
record (WHAT was decided), a TPM 2.0 quote (proof it ran on un-tampered, measured
hardware), and the IMA log (WHICH software was measured). Rather than make an
auditor juggle a record, a quote blob, a signature blob, a PCR file, and an IMA
log, this module stitches them into one JSON document and reads it back.

The bundle is transport, not trust. Every field is re-derived and re-checked by
:func:`~vaara.attestation._tpm_binding.verify_tpm_binding`; a bundle that has been
edited fails the same way a tampered quote does, because the AK signature covers
the quote and the quote covers the PCR digest the IMA log must replay to.

Schema ``vaara.tpm-evidence-bundle/v0``.
"""

from __future__ import annotations

import base64
from typing import Any, Optional

from vaara.attestation._tpm import IMA_PCR
from vaara.attestation._tpm_binding import TPMBindingVerdict, verify_tpm_binding

TPM_BUNDLE_SCHEMA = "vaara.tpm-evidence-bundle/v0"


def build_tpm_bundle_document(
    record: dict[str, Any],
    attest_bytes: bytes,
    signature: bytes,
    ak_pub_pem: bytes,
    pcr_values: dict[int, bytes],
    ima_log: str,
    *,
    bank: str = "sha256",
    expected_ima_pcr: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble a ``vaara.tpm-evidence-bundle/v0`` document from raw pieces.

    Binary fields (the quote and its signature) are base64-encoded; PCR values are
    hex. The producer side of ``vaara verify-tpm-binding``; the capture script
    under ``scripts/tpm/`` calls this after reading a live quote off ``/dev/tpm0``.
    """
    return {
        "schema": TPM_BUNDLE_SCHEMA,
        "record": record,
        "quote": {
            "attest_b64": base64.b64encode(attest_bytes).decode("ascii"),
            "signature_b64": base64.b64encode(signature).decode("ascii"),
            "akPubPem": ak_pub_pem.decode("ascii")
            if isinstance(ak_pub_pem, bytes)
            else ak_pub_pem,
        },
        "pcrs": {
            "bank": bank,
            "values": {str(idx): val.hex() for idx, val in sorted(pcr_values.items())},
        },
        "imaLog": ima_log,
        "expectedImaPcr": expected_ima_pcr,
    }


def _require(doc: dict[str, Any], key: str, typ: type, label: str) -> Any:
    if key not in doc:
        raise ValueError(f"TPM bundle is missing {label!r}")
    value = doc[key]
    if not isinstance(value, typ):
        raise ValueError(
            f"TPM bundle {label!r} must be {typ.__name__}, got "
            f"{type(value).__name__}"
        )
    return value


def verify_tpm_bundle(
    doc: dict[str, Any], *, strict: bool = False
) -> TPMBindingVerdict:
    """Verify a TPM evidence bundle. Returns one :class:`TPMBindingVerdict`.

    Raises :class:`ValueError` on a structurally malformed bundle (missing keys,
    wrong types, non-base64 blobs, non-hex PCR values). A *well-formed* bundle that
    simply does not verify is not an error: it comes back as a verdict with
    ``ok=False`` and the failing link flagged, so a caller can tell "this evidence
    is broken" apart from "this evidence does not hold".
    """
    if not isinstance(doc, dict):
        raise ValueError(
            f"TPM bundle must be a JSON object, got {type(doc).__name__}"
        )
    schema = doc.get("schema")
    if schema != TPM_BUNDLE_SCHEMA:
        raise ValueError(
            f"unexpected bundle schema {schema!r}; expected {TPM_BUNDLE_SCHEMA!r}"
        )
    record = _require(doc, "record", dict, "record")
    quote = _require(doc, "quote", dict, "quote")
    pcrs = _require(doc, "pcrs", dict, "pcrs")
    ima_log = _require(doc, "imaLog", str, "imaLog")

    try:
        attest_bytes = base64.b64decode(
            _require(quote, "attest_b64", str, "quote.attest_b64"), validate=True
        )
        signature = base64.b64decode(
            _require(quote, "signature_b64", str, "quote.signature_b64"),
            validate=True,
        )
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError(f"TPM bundle has a non-base64 quote field: {exc}") from exc
    ak_pub_pem = _require(quote, "akPubPem", str, "quote.akPubPem").encode("ascii")

    raw_values = _require(pcrs, "values", dict, "pcrs.values")
    pcr_values: dict[int, bytes] = {}
    for idx_str, hexval in raw_values.items():
        try:
            idx = int(idx_str)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TPM bundle pcrs.values has a non-integer index {idx_str!r}"
            ) from exc
        if not isinstance(hexval, str):
            raise ValueError(
                f"TPM bundle PCR {idx} value must be a hex string"
            )
        try:
            pcr_values[idx] = bytes.fromhex(hexval)
        except ValueError as exc:
            raise ValueError(
                f"TPM bundle PCR {idx} value is not valid hex"
            ) from exc

    expected = doc.get("expectedImaPcr")
    if expected is not None and not isinstance(expected, str):
        raise ValueError("TPM bundle expectedImaPcr must be a hex string or null")

    return verify_tpm_binding(
        record,
        attest_bytes,
        signature,
        ak_pub_pem,
        pcr_values=pcr_values,
        ima_log=ima_log,
        expected_ima_pcr=expected,
        strict=strict,
    )


__all__ = [
    "IMA_PCR",
    "TPM_BUNDLE_SCHEMA",
    "build_tpm_bundle_document",
    "verify_tpm_bundle",
]
