# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public surface: COSE receipts for inclusion in Vaara's transparency log.

``rfc9942_receipt`` emits an RFC 9942 Receipt for Inclusion over the RFC 6962
(RFC9162_SHA256) log: a tagged COSE_Sign1 signed by the log operator, with the
Merkle root as a detached payload. ``verify_rfc9942_receipt`` recomputes the
root from the entry and the proof and checks the signature over it.

    from vaara.attestation.cose_receipt import (
        rfc9942_receipt, verify_rfc9942_receipt,
    )

``cose_inclusion_receipt`` and ``verify_cose_inclusion_receipt`` are the older
Vaara CBOR map. It is not a COSE message, and its check needs an
``expected_root`` the caller holds independently of the receipt.
"""

from __future__ import annotations

from vaara.attestation._receipt_cose import (
    COSE_SIGN1_TAG,
    VDS_RFC9162_SHA256,
    CoseReceiptError,
    cose_inclusion_receipt,
    rfc9942_receipt,
    sign_cose_receipt,
    verify_cose_inclusion_receipt,
    verify_cose_signature,
    verify_rfc9942_receipt,
)

__all__ = [
    "COSE_SIGN1_TAG",
    "VDS_RFC9162_SHA256",
    "CoseReceiptError",
    "cose_inclusion_receipt",
    "verify_cose_inclusion_receipt",
    "rfc9942_receipt",
    "verify_rfc9942_receipt",
    "sign_cose_receipt",
    "verify_cose_signature",
]
