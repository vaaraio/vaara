"""The TPM evidence-chain bundle: one file carrying a whole attestation loop.

The continuous-attestation counterpart of :mod:`vaara.attestation._tpm_bundle`.
Where that stitches one record + one quote + one IMA log into a
``vaara.tpm-evidence-bundle/v0``, this carries the record once and an ordered list
of links, each a quote + its AK signature + the quoted PCR values + the IMA log at
that tick. The bundle is transport, not trust: every link, and every cross-link
invariant, is re-derived and re-checked by
:func:`~vaara.attestation._tpm_chain.verify_tpm_chain`.

Each link stores its ``seq`` and ``prevDigest`` for human inspection, but the
verifier does not trust them: it recomputes the predecessor digest from the actual
previous link and folds it into the binding, so a link that lies about its position
or predecessor simply fails to verify. Storing the full IMA log per link keeps each
link independently replayable offline; a future revision may delta-encode the log
while preserving the same append-only check.

Schema ``vaara.tpm-evidence-chain/v0``.
"""

from __future__ import annotations

import base64
from typing import Any, Optional

from vaara.attestation._tpm_chain import (
    TPM_CHAIN_SCHEMA,
    GENESIS_PREV_DIGEST,
    TPMChainLink,
    TPMChainVerdict,
    link_digest,
    verify_tpm_chain,
)


def build_tpm_chain_document(
    record: dict[str, Any],
    links: "list[TPMChainLink]",
    *,
    bank: str = "sha256",
    expected_ima_pcr: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble a ``vaara.tpm-evidence-chain/v0`` document from ordered links.

    ``links`` is the sequence of :class:`TPMChainLink` ticks in chain order. The
    ``seq`` and ``prevDigest`` of each link are derived here (genesis ``prevDigest``
    is 32 zero bytes); binary fields are base64-encoded and PCR values hex.
    """
    out_links: list[dict[str, Any]] = []
    prev_digest = GENESIS_PREV_DIGEST
    for seq, link in enumerate(links):
        out_links.append(
            {
                "seq": seq,
                "prevDigest": prev_digest.hex(),
                "quote": {
                    "attest_b64": base64.b64encode(link.attest).decode("ascii"),
                    "signature_b64": base64.b64encode(link.signature).decode(
                        "ascii"
                    ),
                    "akPubPem": link.ak_pub_pem.decode("ascii")
                    if isinstance(link.ak_pub_pem, bytes)
                    else link.ak_pub_pem,
                },
                "pcrs": {
                    "bank": bank,
                    "values": {
                        str(idx): val.hex()
                        for idx, val in sorted(link.pcr_values.items())
                    },
                },
                "imaLog": link.ima_log,
            }
        )
        prev_digest = link_digest(link.attest)
    return {
        "schema": TPM_CHAIN_SCHEMA,
        "record": record,
        "links": out_links,
        "expectedImaPcr": expected_ima_pcr,
    }


def _require(doc: dict[str, Any], key: str, typ: type, label: str) -> Any:
    if key not in doc:
        raise ValueError(f"TPM chain is missing {label!r}")
    value = doc[key]
    if not isinstance(value, typ):
        raise ValueError(
            f"TPM chain {label!r} must be {typ.__name__}, got "
            f"{type(value).__name__}"
        )
    return value


def _parse_link(entry: dict[str, Any], idx: int) -> TPMChainLink:
    """Decode one bundle link into a :class:`TPMChainLink`."""
    if not isinstance(entry, dict):
        raise ValueError(f"TPM chain link {idx} must be a JSON object")
    quote = _require(entry, "quote", dict, f"links[{idx}].quote")
    try:
        attest = base64.b64decode(
            _require(quote, "attest_b64", str, f"links[{idx}].quote.attest_b64"),
            validate=True,
        )
        signature = base64.b64decode(
            _require(
                quote, "signature_b64", str, f"links[{idx}].quote.signature_b64"
            ),
            validate=True,
        )
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError(
            f"TPM chain link {idx} has a non-base64 quote field: {exc}"
        ) from exc
    ak_pub_pem = _require(
        quote, "akPubPem", str, f"links[{idx}].quote.akPubPem"
    ).encode("ascii")

    pcrs = _require(entry, "pcrs", dict, f"links[{idx}].pcrs")
    raw_values = _require(pcrs, "values", dict, f"links[{idx}].pcrs.values")
    pcr_values: dict[int, bytes] = {}
    for idx_str, hexval in raw_values.items():
        try:
            pcr_idx = int(idx_str)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"TPM chain link {idx} pcrs.values has a non-integer index "
                f"{idx_str!r}"
            ) from exc
        if not isinstance(hexval, str):
            raise ValueError(
                f"TPM chain link {idx} PCR {pcr_idx} value must be a hex string"
            )
        try:
            pcr_values[pcr_idx] = bytes.fromhex(hexval)
        except ValueError as exc:
            raise ValueError(
                f"TPM chain link {idx} PCR {pcr_idx} value is not valid hex"
            ) from exc

    ima_log = _require(entry, "imaLog", str, f"links[{idx}].imaLog")
    return TPMChainLink(
        attest=attest,
        signature=signature,
        ak_pub_pem=ak_pub_pem,
        pcr_values=pcr_values,
        ima_log=ima_log,
    )


def verify_tpm_chain_bundle(
    doc: dict[str, Any], *, strict: bool = False
) -> TPMChainVerdict:
    """Verify a TPM evidence-chain bundle. Returns one :class:`TPMChainVerdict`.

    Raises :class:`ValueError` on a structurally malformed bundle (missing keys,
    wrong types, non-base64 blobs, non-hex PCR values, empty link list). A
    *well-formed* bundle that simply does not verify is not an error: it comes back
    as a verdict with ``ok=False`` and the failing link or invariant flagged.
    """
    if not isinstance(doc, dict):
        raise ValueError(
            f"TPM chain must be a JSON object, got {type(doc).__name__}"
        )
    schema = doc.get("schema")
    if schema != TPM_CHAIN_SCHEMA:
        raise ValueError(
            f"unexpected chain schema {schema!r}; expected {TPM_CHAIN_SCHEMA!r}"
        )
    record = _require(doc, "record", dict, "record")
    raw_links = _require(doc, "links", list, "links")
    if not raw_links:
        raise ValueError("TPM chain has an empty links list")

    expected = doc.get("expectedImaPcr")
    if expected is not None and not isinstance(expected, str):
        raise ValueError("TPM chain expectedImaPcr must be a hex string or null")

    links = [_parse_link(entry, idx) for idx, entry in enumerate(raw_links)]
    return verify_tpm_chain(
        record, links, expected_ima_pcr=expected, strict=strict
    )


__all__ = [
    "TPM_CHAIN_SCHEMA",
    "build_tpm_chain_document",
    "verify_tpm_chain_bundle",
]
