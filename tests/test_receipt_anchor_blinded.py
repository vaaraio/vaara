# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Blinded anchors: what the TSA sees stops matching what the receipt publishes.

An unblinded anchor sends the TSA exactly the digest the receipt then prints in
`anchoredDigest`. A TSA keeps a log, a log has a customer account behind every
request, and a log that is sold or subpoenaed lets the holder match its entries
against any corpus of published receipts.

Blinding sends `sha256(label || salt || payload_digest)` instead and carries the
salt in the anchor. A party holding the receipt still verifies, because the salt
is right there. A party holding only the log sees a value that appears nowhere
else, so it cannot match the log against receipts it was not given.

That is the whole claim and it is deliberately narrow. Anyone handed the receipt
can still link it to the log entry, which is what verification requires.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("asn1crypto")  # the 'timeanchor' extra; skip when absent

from vaara.audit.receipt_anchor import (  # noqa: E402
    SelfHostedTSA,
    verify_receipt_anchor,
)
from vaara.audit.timeanchor import (  # noqa: E402
    ANCHOR_BLIND_LABEL,
    SALT_LEN,
    TimeAnchorError,
    anchored_digest,
    blinded_anchor_digest,
    new_anchor_salt,
)

VECTOR = (Path(__file__).resolve().parents[1]
          / "tests/vectors/x402_settlement_v0/generic/step1/receipt.json")


@pytest.fixture
def receipt() -> dict:
    return json.loads(VECTOR.read_text())


def _token_imprint(anchor: dict) -> bytes:
    """The digest the TSA actually attested, read straight out of the token."""
    from asn1crypto import cms, core, tsp

    content = cms.ContentInfo.load(base64.b64decode(anchor["token"]))["content"]
    tst_bytes = content["encap_content_info"]["content"].cast(core.OctetString).native
    return tsp.TSTInfo.load(tst_bytes)["message_imprint"]["hashed_message"].native


def _payload_digest(receipt: dict) -> bytes:
    return bytes.fromhex(anchored_digest(receipt).split(":", 1)[1])


# --- the salt ---------------------------------------------------------------


def test_a_salt_is_fresh_every_time():
    assert len({new_anchor_salt() for _ in range(16)}) == 16
    assert all(len(new_anchor_salt()) == SALT_LEN for _ in range(4))


def test_the_blinded_digest_is_domain_separated(receipt: dict):
    salt = new_anchor_salt()
    payload = _payload_digest(receipt)
    assert blinded_anchor_digest(payload, salt) == hashlib.sha256(
        ANCHOR_BLIND_LABEL + salt + payload
    ).digest()


def test_a_wrong_length_salt_is_refused(receipt: dict):
    for bad in (b"", bytes(16), bytes(64)):
        with pytest.raises(TimeAnchorError):
            blinded_anchor_digest(_payload_digest(receipt), bad)


# --- the leak this closes ---------------------------------------------------


def test_an_unblinded_anchor_sends_the_tsa_the_published_digest(receipt: dict):
    """The behaviour being fixed, pinned so it cannot come back unnoticed."""
    anchor = SelfHostedTSA.create().anchor_receipt(receipt)
    assert _token_imprint(anchor) == _payload_digest(receipt)
    assert anchor["anchoredDigest"] == anchored_digest(receipt)


def test_a_blinded_anchor_sends_the_tsa_something_else(receipt: dict):
    anchor = SelfHostedTSA.create().anchor_receipt(receipt, blind=True)
    imprint = _token_imprint(anchor)
    assert imprint != _payload_digest(receipt)
    assert imprint.hex() not in json.dumps(anchor)
    # The receipt binding is unchanged: anchoredDigest still names the payload.
    assert anchor["anchoredDigest"] == anchored_digest(receipt)


def test_two_blinded_anchors_over_one_receipt_do_not_look_alike(receipt: dict):
    tsa = SelfHostedTSA.create()
    first = tsa.anchor_receipt(receipt, blind=True)
    second = tsa.anchor_receipt(receipt, blind=True)
    assert _token_imprint(first) != _token_imprint(second)
    assert first["anchorSalt"] != second["anchorSalt"]


def test_the_holder_can_still_link_the_anchor_to_its_token(receipt: dict):
    """The narrow limit, asserted so the docs cannot drift into overclaiming."""
    anchor = SelfHostedTSA.create().anchor_receipt(receipt, blind=True)
    salt = bytes.fromhex(anchor["anchorSalt"])
    assert _token_imprint(anchor) == blinded_anchor_digest(
        _payload_digest(receipt), salt
    )


# --- shape and verification -------------------------------------------------


def test_blinded_anchor_has_the_spec_shape_and_verifies(receipt: dict):
    anchor = SelfHostedTSA.create().anchor_receipt(receipt, blind=True)
    assert set(anchor) == {
        "method", "anchoredDigest", "anchorSalt", "token", "authority",
    }
    assert anchor["method"] == "rfc3161-blinded"
    assert len(anchor["anchorSalt"]) == 2 * SALT_LEN
    attested = verify_receipt_anchor(receipt, anchor)
    assert attested.tzinfo is not None


def test_a_pinned_salt_reproduces_the_anchor(receipt: dict):
    """A generator needs to pin the salt; production never should."""
    salt = bytes(range(SALT_LEN))
    anchor = SelfHostedTSA.create().anchor_receipt(receipt, blind=True, salt=salt)
    assert anchor["anchorSalt"] == salt.hex()
    assert verify_receipt_anchor(receipt, anchor) is not None


def test_unblinded_anchors_still_verify_unchanged(receipt: dict):
    """Existing anchors keep working; this release adds a method, it moves none."""
    anchor = SelfHostedTSA.create().anchor_receipt(receipt)
    assert anchor["method"] == "rfc3161"
    assert "anchorSalt" not in anchor
    assert verify_receipt_anchor(receipt, anchor) is not None


# --- what must not verify ---------------------------------------------------


def test_a_blinded_anchor_without_its_salt_does_not_verify(receipt: dict):
    anchor = SelfHostedTSA.create().anchor_receipt(receipt, blind=True)
    stripped = {k: v for k, v in anchor.items() if k != "anchorSalt"}
    with pytest.raises(TimeAnchorError):
        verify_receipt_anchor(receipt, stripped)


def test_a_blinded_anchor_with_the_wrong_salt_does_not_verify(receipt: dict):
    anchor = SelfHostedTSA.create().anchor_receipt(receipt, blind=True)
    with pytest.raises(TimeAnchorError):
        verify_receipt_anchor(receipt, dict(anchor, anchorSalt=bytes(SALT_LEN).hex()))


@pytest.mark.parametrize("bad", ["", "zz", "ab", "ff" * 16, "ff" * 64])
def test_a_malformed_salt_does_not_verify(receipt: dict, bad: str):
    anchor = SelfHostedTSA.create().anchor_receipt(receipt, blind=True)
    with pytest.raises(TimeAnchorError):
        verify_receipt_anchor(receipt, dict(anchor, anchorSalt=bad))


def test_an_unblinded_method_carrying_a_salt_is_refused(receipt: dict):
    """Otherwise a verifier that ignores the field reads a blinded anchor as plain."""
    anchor = SelfHostedTSA.create().anchor_receipt(receipt)
    with pytest.raises(TimeAnchorError):
        verify_receipt_anchor(receipt, dict(anchor, anchorSalt=new_anchor_salt().hex()))


def test_a_blinded_anchor_over_a_tampered_receipt_does_not_verify(receipt: dict):
    anchor = SelfHostedTSA.create().anchor_receipt(receipt, blind=True)
    tampered = dict(receipt, version=receipt["version"] + 1)
    with pytest.raises(TimeAnchorError):
        verify_receipt_anchor(tampered, anchor)


def test_a_blinded_anchor_with_a_forged_token_does_not_verify(receipt: dict):
    anchor = SelfHostedTSA.create().anchor_receipt(receipt, blind=True)
    forged = SelfHostedTSA.create().issue_token(bytes(32))
    swapped = dict(anchor, token=base64.b64encode(forged).decode())
    with pytest.raises(TimeAnchorError):
        verify_receipt_anchor(receipt, swapped)


# --- the qualified path -----------------------------------------------------


def _qtsp(tsa):
    """A fake QTSP endpoint: TimeStampReq DER in, TimeStampResp DER out."""
    from asn1crypto import cms, tsp

    def transport(url: str, der_request: bytes, timeout: float) -> bytes:
        req = tsp.TimeStampReq.load(der_request)
        token = tsa.issue_token(req["message_imprint"]["hashed_message"].native)
        return tsp.TimeStampResp({
            "status": tsp.PKIStatusInfo({"status": "granted"}),
            "time_stamp_token": cms.ContentInfo.load(token),
        }).dump()

    return transport


def test_a_blinded_qualified_anchor_keeps_its_pin_and_its_time(receipt, tmp_path):
    from vaara.audit.receipt_anchor import QualifiedTSA

    tsa = SelfHostedTSA.create("EU Test QTSP")
    tsa.save(tmp_path)
    pin = (tmp_path / "tsa_cert.pem").read_bytes()
    qtsa = QualifiedTSA("https://qtsp.example/tsr", trusted_signer_cert=pin,
                        transport=_qtsp(tsa))

    anchor = qtsa.anchor_receipt(receipt, blind=True)
    assert anchor["method"] == "rfc3161-eidas-qualified-blinded"
    assert set(anchor) == {"method", "anchoredDigest", "anchorSalt", "token",
                           "authority", "tsaUrl"}
    # The QTSP was shown the salted value, not the one the receipt publishes.
    assert _token_imprint(anchor) != _payload_digest(receipt)
    attested = verify_receipt_anchor(receipt, anchor, trusted_signer_cert=pin)
    assert attested.tzinfo is not None


def test_a_blinded_qualified_anchor_still_renders_as_qualified(receipt, tmp_path):
    """Otherwise the evidence page silently demotes it to a generic anchor."""
    from vaara.audit.receipt_anchor import QualifiedTSA
    from vaara.audit.receipt_page import render_receipt_page

    tsa = SelfHostedTSA.create("EU Test QTSP")
    tsa.save(tmp_path)
    pin = (tmp_path / "tsa_cert.pem").read_bytes()
    qtsa = QualifiedTSA("https://qtsp.example/tsr", trusted_signer_cert=pin,
                        transport=_qtsp(tsa))

    anchored = dict(receipt)
    anchored["timestampAnchors"] = [qtsa.anchor_receipt(receipt, blind=True)]
    page = render_receipt_page(anchored)
    assert "rfc3161-eidas-qualified-blinded" in page
    assert "status as recorded, not re-checked here" not in page


def test_the_server_anchorer_reads_the_blind_switch(monkeypatch):
    """Construction does no network I/O, so this only reads the switch.

    Guarded because `vaara.server.anchor` imports the app, which needs fastapi.
    Without the guard this fails rather than skips on any checkout that has the
    signing extras and not the server extra, which is exactly what the "Signing
    extras" CI job installs. Main has been red on that job since 2026-08-21,
    when the blind switch shipped in v1.74.0, through four releases, and it did
    not show up locally because a development venv has fastapi.
    """
    try:
        from vaara.server.anchor import Anchorer
    except ImportError:
        pytest.skip("server extra not installed (pip install 'vaara[server]')")

    url = "https://qtsp.example/tsr"
    monkeypatch.delenv("VAARA_ANCHOR_BLIND", raising=False)
    assert Anchorer(tsa_url=url).blind is False
    monkeypatch.setenv("VAARA_ANCHOR_BLIND", "1")
    assert Anchorer(tsa_url=url).blind is True
    # An explicit argument still wins over the environment.
    assert Anchorer(tsa_url=url, blind=False).blind is False


def test_the_unblinded_token_does_not_pass_as_blinded(receipt: dict):
    """Relabelling a plain anchor as blinded must fail: the imprint is the wrong one."""
    tsa = SelfHostedTSA.create()
    plain = tsa.anchor_receipt(receipt)
    relabelled = dict(
        plain, method="rfc3161-blinded", anchorSalt=new_anchor_salt().hex()
    )
    with pytest.raises(TimeAnchorError):
        verify_receipt_anchor(receipt, relabelled)
