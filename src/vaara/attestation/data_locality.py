# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Data-locality records: signed evidence of where an agent action's data went.

When personal data leaves for a model endpoint, the accountability question is
where it went, under what rule, and whether the answer can be checked without
trusting the party that sent it. A data-locality record
(``vaara.data-locality/v0``) binds one action's transfer facts — data class,
endpoint, endpoint region, TLS cert fingerprint, payload digest — to the
enforced allow/block decision, and optionally carries a region attestation
signed by a party distinct from the issuer.

The evidence is two-tier, and the corpus checker names which tier a record
reaches:

- Tier A (proof, no trusted party): the issuer signature verifies, the payload
  digest recomputes, and the recorded decision recomputes from the transfer
  facts under the named policy. Reproducible from bytes alone.
- Tier B (carried claim): a ``regionAttestation`` signed by an attester key
  distinct from the issuer, verified against that key and checked to agree with
  the claimed region. Present and valid it attests the observed region; it is
  never an adequacy finding, and its absence is stated (location asserted, not
  attested), not hidden.

This module emits and signature-verifies records that the dependency-free
``data_locality_v0`` conformance corpus grades. Canonicalization is RFC 8785
JCS; signing is via the ``vaara.audit.signer`` ``Signer`` protocol (Ed25519 by
default, ML-DSA-65 optional). Acquiring a genuine region attestation is out of
scope by design.

Install: ``pip install 'vaara[attestation]'``.
"""
from __future__ import annotations

from vaara.attestation._data_locality import (
    SCHEMA,
    TransferFacts,
    emit_data_locality_record,
    emit_from_interception,
    payload_digest,
    region_attestation,
    verify_record_signature,
)

__all__ = [
    "SCHEMA",
    "TransferFacts",
    "emit_data_locality_record",
    "emit_from_interception",
    "payload_digest",
    "region_attestation",
    "verify_record_signature",
]
