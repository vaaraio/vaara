# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Public surface: SCITT-compatible COSE Receipts over Vaara's transparency log.

Keyless first. A Vaara COSE Receipt is an RFC 6962 inclusion proof serialised in
the draft-ietf-cose-merkle-tree-proofs shape (verifiable-data-structure type 1),
verifiable by recomputation from bytes with no key and no operator to trust. An
operator-signed COSE_Sign1 wrapper is optional, for relying parties that require
the strict SCITT wire form; it never replaces the keyless check.

    from vaara.attestation.cose_receipt import (
        cose_inclusion_receipt, verify_cose_inclusion_receipt,
    )

Emit a receipt from a ``transparency_log`` inclusion proof; verify it anywhere
with only ``cbor2`` and SHA-256 — no Vaara signing key involved.
"""

from __future__ import annotations

from vaara.attestation._receipt_cose import (
    VDS_RFC9162_SHA256,
    CoseReceiptError,
    cose_inclusion_receipt,
    sign_cose_receipt,
    verify_cose_inclusion_receipt,
    verify_cose_signature,
)

__all__ = [
    "VDS_RFC9162_SHA256",
    "CoseReceiptError",
    "cose_inclusion_receipt",
    "verify_cose_inclusion_receipt",
    "sign_cose_receipt",
    "verify_cose_signature",
]
