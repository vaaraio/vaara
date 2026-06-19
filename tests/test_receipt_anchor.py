"""Self-hosted rfc3161 receipt anchors round-trip and pin to the receipt."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

pytest.importorskip("asn1crypto")  # the 'timeanchor' extra; skip when absent

from vaara.audit.receipt_anchor import (  # noqa: E402
    SelfHostedTSA,
    anchored_digest,
    verify_receipt_anchor,
)
from vaara.audit.timeanchor import TimeAnchorError  # noqa: E402

VECTOR = (Path(__file__).resolve().parents[1]
          / "tests/vectors/x402_settlement_v0/generic/step1/receipt.json")


@pytest.fixture
def receipt() -> dict:
    return json.loads(VECTOR.read_text())


def test_anchor_has_spec_shape_and_verifies(receipt: dict) -> None:
    anchor = SelfHostedTSA.create().anchor_receipt(receipt)
    assert set(anchor) == {"method", "anchoredDigest", "token", "authority"}
    assert anchor["method"] == "rfc3161"
    # anchoredDigest == sha256 of the JCS signed payload (SPEC.md rule 3),
    # recomputed independently of the producer.
    assert anchor["anchoredDigest"] == anchored_digest(receipt)
    attested = verify_receipt_anchor(receipt, anchor)
    assert attested.tzinfo is not None


def test_anchor_rejects_tampered_receipt(receipt: dict) -> None:
    anchor = SelfHostedTSA.create().anchor_receipt(receipt)
    tampered = dict(receipt, version=receipt["version"] + 1)
    with pytest.raises(TimeAnchorError):
        verify_receipt_anchor(tampered, anchor)


def test_anchor_rejects_forged_token(receipt: dict) -> None:
    # A token from a different TSA over a different digest must not pass even if
    # the anchoredDigest field is set correctly: the token's imprint won't match.
    good = SelfHostedTSA.create().anchor_receipt(receipt)
    other = SelfHostedTSA.create()
    forged_token = other.issue_token(bytes(32))
    swapped = dict(good, token=base64.b64encode(forged_token).decode())
    with pytest.raises(TimeAnchorError):
        verify_receipt_anchor(receipt, swapped)


def test_tsa_persists_authority_identity(tmp_path) -> None:
    first = SelfHostedTSA.load_or_create(tmp_path, "vaara-test-tsa")
    second = SelfHostedTSA.load_or_create(tmp_path, "ignored-on-load")
    assert first.authority == second.authority == "vaara-test-tsa"
    # Same persisted key signs verifiable tokens after a reload.
    digest = bytes(range(32))
    assert second.issue_token(digest)
