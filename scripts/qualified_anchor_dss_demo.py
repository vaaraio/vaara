#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Anchor a receipt to a qualified TSA and validate the token in EU DSS.

Reproduces the 1.30.0 claim end to end, against live services: obtains an
RFC 3161 token from an eIDAS-qualified TSA (default: Sectigo's qualified
endpoint, listed as TSA/QTST with status granted on the Spanish trusted
list), pins the signer's issuing CA out of the first reply, records the
``rfc3161-eidas-qualified`` anchor, then submits the token plus the
receipt's JCS signed payload to the European Commission's DSS demo
validator and prints its verdict. Expected output ends with::

    DSS verdict: PASSED QTSA (Qualified timestamp)

Needs the network (the TSA and the DSS demo webapp) and the 'timeanchor'
extra. The TSA is a free public endpoint with no SLA; a refusal or
timeout is a service condition, not a Vaara failure.

    python scripts/qualified_anchor_dss_demo.py [path/to/receipt.json]
"""
from __future__ import annotations

import base64
import json
import sys
import urllib.request
from pathlib import Path

import rfc8785
from asn1crypto import cms

from vaara.audit.receipt_anchor import (
    _SIGNED_BLOCKS,
    QualifiedTSA,
    verify_receipt_anchor,
)
from vaara.audit.timeanchor import (
    _urllib_transport,
    build_timestamp_request,
    extract_token_from_response,
)

DEFAULT = (Path(__file__).resolve().parents[1]
           / "tests/vectors/x402_settlement_v0/generic/step1/receipt.json")
TSA_URL = "http://timestamp.sectigo.com/qualified"
DSS_URL = ("https://ec.europa.eu/digital-building-blocks/DSS/webapp-demo"
           "/services/rest/validation/validateSignature")


def issuing_ca_from_probe(tsa_url: str) -> bytes:
    """Fetch one token and return the embedded issuing CA certificate (DER).

    Trust-on-first-use for a demo. For production, pin the CA certificate
    obtained from the trusted list instead of from the endpoint itself.
    """
    import hashlib

    probe = build_timestamp_request(hashlib.sha256(b"vaara-dss-demo").digest())
    token = extract_token_from_response(
        _urllib_transport(tsa_url, probe, 20.0))
    signed_data = cms.ContentInfo.load(token)["content"]
    certs = [c.chosen for c in signed_data["certificates"]
             if c.name == "certificate"]
    subjects = {c["tbs_certificate"]["subject"].dump(): c for c in certs}
    for cert in certs:  # the CA: a cert that issued another embedded cert
        issuer = cert["tbs_certificate"]["issuer"].dump()
        if issuer in subjects and subjects[issuer] is not cert:
            return subjects[issuer].dump()
    raise SystemExit("no issuing CA certificate embedded in the TSA reply")


def dss_validate(token_der: bytes, payload: bytes) -> tuple[str, str, str]:
    body = json.dumps({
        "signedDocument": {
            "bytes": base64.b64encode(token_der).decode("ascii"),
            "name": "qualified-token.tst",
        },
        "originalDocuments": [{
            "bytes": base64.b64encode(payload).decode("ascii"),
            "name": "signed-payload.jcs",
        }],
        "policy": None,
    }).encode("ascii")
    req = urllib.request.Request(
        DSS_URL, data=body, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        report = json.load(resp)
    entry = report["SimpleReport"]["signatureOrTimestampOrEvidenceRecord"][0]
    ts = entry["Timestamp"]
    level = ts.get("TimestampLevel") or {}
    return (ts.get("Indication", "?"), level.get("value", "?"),
            level.get("description", "?"))


def main() -> int:
    receipt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    receipt = json.loads(receipt_path.read_text())

    ca_der = issuing_ca_from_probe(TSA_URL)
    qtsa = QualifiedTSA(TSA_URL, trusted_issuer_cert=ca_der, timeout=20.0)
    anchor = qtsa.anchor_receipt(receipt)
    print(f"anchor method:  {anchor['method']}")
    print(f"authority:      {anchor['authority']}")

    attested = verify_receipt_anchor(
        receipt, anchor, trusted_issuer_cert=ca_der)
    print(f"pinned verify:  OK, attested {attested.isoformat()}")

    payload = rfc8785.dumps({k: receipt[k] for k in _SIGNED_BLOCKS})
    indication, value, description = dss_validate(
        base64.b64decode(anchor["token"]), payload)
    print(f"DSS verdict: {indication} {value} ({description})")
    return 0 if (indication, value) == ("PASSED", "QTSA") else 1


if __name__ == "__main__":
    raise SystemExit(main())
