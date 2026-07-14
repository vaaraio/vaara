"""Static HTML evidence page: content, escaping, anchor status reporting."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# receipt_page renders anchors via receipt_anchor, which needs the
# 'timeanchor' extra; skip cleanly where the extras are absent.
pytest.importorskip("rfc8785")
pytest.importorskip("asn1crypto")

from vaara.audit.receipt_anchor import _signed_payload_digest  # noqa: E402
from vaara.audit.receipt_page import render_receipt_page

VECTOR = (Path(__file__).resolve().parents[1]
          / "tests/vectors/x402_settlement_v0/generic/step1/receipt.json")


@pytest.fixture
def receipt() -> dict:
    return json.loads(VECTOR.read_text())


def test_page_is_standalone_and_carries_the_digest(receipt: dict) -> None:
    page = render_receipt_page(receipt)
    assert page.startswith("<!doctype html>")
    assert _signed_payload_digest(receipt).hex() in page
    assert receipt["issuerAsserted"]["iss"] in page
    assert receipt["decisionDerived"]["decision"] in page
    # Self-contained: no external fetches of any kind.
    for marker in ("src=", "href=", "url(", "@import"):
        assert marker not in page


def test_page_escapes_receipt_content(receipt: dict) -> None:
    receipt["decisionDerived"]["reason"] = '<script>alert("x")</script>'
    page = render_receipt_page(receipt)
    assert "<script>" not in page
    assert "&lt;script&gt;" in page


def test_ots_anchor_status_is_reported(receipt: dict) -> None:
    pytest.importorskip("opentimestamps")
    from tests.test_ots_anchor import _pending_calendar_transport
    from vaara.audit.ots_anchor import ots_anchor_receipt

    anchor = ots_anchor_receipt(
        receipt, calendars=("https://alice.btc.calendar.opentimestamps.org",),
        transport=_pending_calendar_transport)
    receipt["timestampAnchors"] = [anchor]
    page = render_receipt_page(receipt)
    assert "opentimestamps" in page
    assert "pending" in page
    assert "verified against this receipt offline" in page


def test_qualified_anchor_renders_art41_line(receipt: dict) -> None:
    pytest.importorskip("asn1crypto")
    from tests.test_receipt_anchor import _endpoint_for
    from vaara.audit.receipt_anchor import QualifiedTSA, SelfHostedTSA
    from cryptography.hazmat.primitives import serialization

    tsa = SelfHostedTSA.create("EU Test QTSP")
    pin = tsa._cert.public_bytes(serialization.Encoding.PEM)
    anchor = QualifiedTSA("https://qtsp.example/tsr", trusted_signer_cert=pin,
                          transport=_endpoint_for(tsa)).anchor_receipt(receipt)
    receipt["timestampAnchors"] = [anchor]
    page = render_receipt_page(receipt)
    assert "Qualified eIDAS timestamp" in page
    assert "EU Test QTSP" in page
    assert "eIDAS Art. 41" in page
    assert "digest binding re-checked offline" in page
    # The page must not claim the pin: that needs the QTSP certificate
    # held out of band, which a standalone rendering does not have.
    assert "not re-checked here" in page


def test_anchor_detail_is_escaped_exactly_once(receipt: dict) -> None:
    pytest.importorskip("asn1crypto")
    from cryptography.hazmat.primitives import serialization

    from tests.test_receipt_anchor import _endpoint_for
    from vaara.audit.receipt_anchor import QualifiedTSA, SelfHostedTSA

    tsa = SelfHostedTSA.create("R&D <QTSP>")
    pin = tsa._cert.public_bytes(serialization.Encoding.PEM)
    anchor = QualifiedTSA("https://qtsp.example/tsr", trusted_signer_cert=pin,
                          transport=_endpoint_for(tsa)).anchor_receipt(receipt)
    receipt["timestampAnchors"] = [anchor]
    page = render_receipt_page(receipt)
    assert "R&amp;D &lt;QTSP&gt;" in page
    assert "&amp;amp;" not in page  # double escape


def test_tampered_qualified_anchor_shows_invalid(receipt: dict) -> None:
    pytest.importorskip("asn1crypto")
    from tests.test_receipt_anchor import _endpoint_for
    from vaara.audit.receipt_anchor import QualifiedTSA, SelfHostedTSA
    from cryptography.hazmat.primitives import serialization

    tsa = SelfHostedTSA.create("EU Test QTSP")
    pin = tsa._cert.public_bytes(serialization.Encoding.PEM)
    anchor = QualifiedTSA("https://qtsp.example/tsr", trusted_signer_cert=pin,
                          transport=_endpoint_for(tsa)).anchor_receipt(receipt)
    receipt["timestampAnchors"] = [anchor]
    receipt["decisionDerived"]["decision"] = "block"  # tamper after anchoring
    page = render_receipt_page(receipt)
    assert "INVALID" in page


def test_tampered_receipt_shows_invalid_anchor(receipt: dict) -> None:
    pytest.importorskip("opentimestamps")
    from tests.test_ots_anchor import _pending_calendar_transport
    from vaara.audit.ots_anchor import ots_anchor_receipt

    anchor = ots_anchor_receipt(
        receipt, calendars=("https://alice.btc.calendar.opentimestamps.org",),
        transport=_pending_calendar_transport)
    receipt["timestampAnchors"] = [anchor]
    receipt["decisionDerived"]["decision"] = "block"  # tamper after anchoring
    page = render_receipt_page(receipt)
    assert "INVALID" in page
