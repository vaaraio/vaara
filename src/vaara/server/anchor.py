# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Qualified-anchor helper for the Vaara server.

Wraps :class:`vaara.audit.receipt_anchor.QualifiedTSA` so a route turns a
receipt into an ``rfc3161-eidas-qualified`` anchor with one call. Defaults to
Sectigo's free qualified endpoint (validated PASSED/QTSA by the EU DSS demo
validator).

The QTSP's issuing CA is pinned. In production, pin it from the EU trusted list
(``VAARA_ANCHOR_CA_CERT`` = path to a PEM or DER CA certificate). Absent that,
the helper falls back to trust-on-first-use: one probe of the endpoint to learn
the CA. That is fine for self-hosting and demos but is not a trusted-list pin.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Optional

from vaara.audit.receipt_anchor import QualifiedTSA, verify_receipt_anchor
from vaara.audit.timeanchor import (
    _urllib_transport,
    build_timestamp_request,
    extract_token_from_response,
)

_DEFAULT_TSA = "http://timestamp.sectigo.com/qualified"


def _issuing_ca_from_probe(tsa_url: str, timeout: float = 20.0) -> bytes:
    """Trust-on-first-use: fetch one token and return its issuing CA (DER).

    For production, pin the CA obtained from the EU trusted list instead of
    from the endpoint itself.
    """
    from asn1crypto import cms

    probe = build_timestamp_request(
        hashlib.sha256(b"vaara-anchor-probe").digest()
    )
    token = extract_token_from_response(
        _urllib_transport(tsa_url, probe, timeout)
    )
    signed = cms.ContentInfo.load(token)["content"]
    certs = [
        c.chosen for c in signed["certificates"] if c.name == "certificate"
    ]
    subjects = {c["tbs_certificate"]["subject"].dump(): c for c in certs}
    for cert in certs:  # the CA is a cert that issued another embedded cert
        issuer = cert["tbs_certificate"]["issuer"].dump()
        if issuer in subjects and subjects[issuer] is not cert:
            return subjects[issuer].dump()
    raise RuntimeError("no issuing CA certificate embedded in the TSA reply")


class Anchorer:
    """Lazily-initialised qualified anchorer. Thread-safe, memoises the QTSA.

    Construction does no network I/O, so importing or instantiating the server
    stays offline; the first ``anchor`` call resolves the CA (pin or probe) and
    builds the pinned :class:`QualifiedTSA`.
    """

    def __init__(
        self,
        tsa_url: Optional[str] = None,
        ca_cert: Optional[bytes] = None,
    ) -> None:
        self.tsa_url = (
            tsa_url
            or os.environ.get("VAARA_ANCHOR_TSA_URL", _DEFAULT_TSA).strip()
        )
        self._ca = ca_cert
        if self._ca is None:
            path = os.environ.get("VAARA_ANCHOR_CA_CERT", "").strip()
            if path:
                self._ca = Path(path).read_bytes()
        self._qtsa: Optional[QualifiedTSA] = None
        self._lock = threading.Lock()

    def _ensure(self) -> QualifiedTSA:
        if self._qtsa is not None:
            return self._qtsa
        with self._lock:
            if self._qtsa is None:
                ca = self._ca or _issuing_ca_from_probe(self.tsa_url)
                self._ca = ca
                self._qtsa = QualifiedTSA(
                    self.tsa_url, trusted_issuer_cert=ca, timeout=20.0
                )
            return self._qtsa

    def anchor(self, receipt: dict) -> dict[str, Any]:
        """Return an ``rfc3161-eidas-qualified`` anchor for ``receipt``."""
        return self._ensure().anchor_receipt(receipt)

    def attested_time(self, receipt: dict, anchor: dict) -> str:
        """ISO-8601 UTC time the pinned QTSP attested, verified against the pin."""
        self._ensure()
        dt = verify_receipt_anchor(
            receipt, anchor, trusted_issuer_cert=self._ca
        )
        return dt.isoformat()
