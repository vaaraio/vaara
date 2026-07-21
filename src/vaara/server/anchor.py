# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Qualified-anchor helper for the Vaara server.

Wraps :class:`vaara.audit.receipt_anchor.QualifiedTSA` so a route turns a
receipt into an ``rfc3161-eidas-qualified`` anchor with one call. No timestamp
provider is baked in: the operator names their own QTSP endpoint via
``VAARA_ANCHOR_TSA_URL``, and anchoring refuses until one is set. Vaara does not
route anyone's traffic to a third-party provider the operator did not choose.

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
from typing import TYPE_CHECKING, Any, Optional

# The receipt_anchor / timeanchor stack pulls the anchor extra (rfc8785, etc.),
# which is not present in the base install. Import it lazily inside the methods
# that anchor, so importing this module and constructing an Anchorer stay cheap
# and dependency-free - core ServerState must not require the anchor extra.
if TYPE_CHECKING:
    from vaara.audit.receipt_anchor import QualifiedTSA


class AnchorNotConfigured(RuntimeError):
    """Raised when anchoring is attempted with no operator-chosen QTSP set."""


def _issuing_ca_from_probe(tsa_url: str, timeout: float = 20.0) -> bytes:
    """Trust-on-first-use: fetch one token and return its issuing CA (DER).

    For production, pin the CA obtained from the EU trusted list instead of
    from the endpoint itself.
    """
    from asn1crypto import cms

    from vaara.audit.timeanchor import (
        _urllib_transport,
        build_timestamp_request,
        extract_token_from_response,
    )

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
        # No provider is baked in. The operator names their own QTSP, or
        # anchoring refuses (see _ensure). Choosing a provider for everyone is
        # not ours to make.
        self.tsa_url = (
            tsa_url or os.environ.get("VAARA_ANCHOR_TSA_URL", "").strip() or None
        )
        self._ca = ca_cert
        if self._ca is None:
            path = os.environ.get("VAARA_ANCHOR_CA_CERT", "").strip()
            if path:
                self._ca = Path(path).read_bytes()
        self._qtsa: Optional[QualifiedTSA] = None
        self._lock = threading.Lock()

    def _ensure(self) -> "QualifiedTSA":
        if self._qtsa is not None:
            return self._qtsa
        if not self.tsa_url:
            raise AnchorNotConfigured(
                "no qualified TSA configured: set VAARA_ANCHOR_TSA_URL to your "
                "chosen QTSP endpoint. Vaara ships no default provider."
            )
        from vaara.audit.receipt_anchor import QualifiedTSA

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
        from vaara.audit.receipt_anchor import verify_receipt_anchor

        self._ensure()
        dt = verify_receipt_anchor(
            receipt, anchor, trusted_issuer_cert=self._ca
        )
        return dt.isoformat()
