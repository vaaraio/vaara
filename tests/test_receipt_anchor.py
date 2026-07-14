"""Self-hosted rfc3161 receipt anchors round-trip and pin to the receipt."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

pytest.importorskip("asn1crypto")  # the 'timeanchor' extra; skip when absent

from vaara.audit.receipt_anchor import (  # noqa: E402
    QualifiedTSA,
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


# ── rfc3161-eidas-qualified: same token from a remote QTSP, cert-pinned ──

def _endpoint_for(tsa: SelfHostedTSA):
    """A fake QTSP HTTP endpoint: TimeStampReq DER in, TimeStampResp DER out."""
    from asn1crypto import cms, tsp

    def transport(url: str, der_request: bytes, timeout: float) -> bytes:
        req = tsp.TimeStampReq.load(der_request)
        digest = req["message_imprint"]["hashed_message"].native
        token = tsa.issue_token(digest)
        return tsp.TimeStampResp({
            "status": tsp.PKIStatusInfo({"status": "granted"}),
            "time_stamp_token": cms.ContentInfo.load(token),
        }).dump()

    return transport


def _cert_pem(tsa: SelfHostedTSA, directory) -> bytes:
    tsa.save(directory)
    return (directory / "tsa_cert.pem").read_bytes()


def test_qualified_anchor_shape_and_pinned_verify(receipt, tmp_path) -> None:
    tsa = SelfHostedTSA.create("EU Test QTSP")
    pin = _cert_pem(tsa, tmp_path)
    qtsa = QualifiedTSA("https://qtsp.example/tsr", trusted_signer_cert=pin,
                        transport=_endpoint_for(tsa))
    anchor = qtsa.anchor_receipt(receipt)
    assert set(anchor) == {"method", "anchoredDigest", "token", "authority",
                           "tsaUrl"}
    assert anchor["method"] == "rfc3161-eidas-qualified"
    assert anchor["anchoredDigest"] == anchored_digest(receipt)
    assert anchor["tsaUrl"] == "https://qtsp.example/tsr"
    # authority derives from the pinned certificate's subject CN.
    assert anchor["authority"] == "EU Test QTSP"
    attested = verify_receipt_anchor(receipt, anchor, trusted_signer_cert=pin)
    assert attested.tzinfo is not None


def test_qualified_anchor_rejects_unpinned_signer(receipt, tmp_path) -> None:
    # The endpoint signs with a key other than the pinned certificate's: the
    # producer must refuse the token, not record it as qualified evidence.
    pinned = SelfHostedTSA.create("EU Test QTSP")
    impostor = SelfHostedTSA.create("Impostor TSA")
    qtsa = QualifiedTSA("https://qtsp.example/tsr",
                        trusted_signer_cert=_cert_pem(pinned, tmp_path),
                        transport=_endpoint_for(impostor))
    with pytest.raises(TimeAnchorError):
        qtsa.anchor_receipt(receipt)


def test_qualified_anchor_requires_pin() -> None:
    # Qualified without a pinned QTSP certificate is a contradiction: the
    # method claim would be self-asserted.
    with pytest.raises(TimeAnchorError):
        QualifiedTSA("https://qtsp.example/tsr", trusted_signer_cert=None)


def test_qualified_anchor_surfaces_tsa_refusal(receipt, tmp_path) -> None:
    from asn1crypto import tsp

    tsa = SelfHostedTSA.create("EU Test QTSP")

    def refusing(url: str, der_request: bytes, timeout: float) -> bytes:
        return tsp.TimeStampResp({
            "status": tsp.PKIStatusInfo({"status": "rejection"}),
        }).dump()

    qtsa = QualifiedTSA("https://qtsp.example/tsr",
                        trusted_signer_cert=_cert_pem(tsa, tmp_path),
                        transport=refusing)
    with pytest.raises(TimeAnchorError):
        qtsa.anchor_receipt(receipt)


def _make_ca_and_leaf():
    """A CA certificate plus a leaf TSA signed by it, as QTSPs deploy."""
    from datetime import datetime, timezone

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    def _name(cn):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])

    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(_name("EU Test QTSP CA")).issuer_name(_name("EU Test QTSP CA"))
        .public_key(ca_key.public_key()).serial_number(1)
        .not_valid_before(datetime(2020, 1, 1, tzinfo=timezone.utc))
        .not_valid_after(datetime(2100, 1, 1, tzinfo=timezone.utc))
        .sign(ca_key, hashes.SHA256())
    )
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf_cert = (
        x509.CertificateBuilder()
        .subject_name(_name("EU Test QTSP Signer #1"))
        .issuer_name(_name("EU Test QTSP CA"))
        .public_key(leaf_key.public_key()).serial_number(2)
        .not_valid_before(datetime(2020, 1, 1, tzinfo=timezone.utc))
        .not_valid_after(datetime(2100, 1, 1, tzinfo=timezone.utc))
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.TIME_STAMPING]),
            critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    ca_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
    return ca_pem, SelfHostedTSA(leaf_key, leaf_cert)


def test_qualified_anchor_pins_issuing_ca(receipt) -> None:
    # QTSP signer certs rotate; the stable trusted-list entry is the issuing
    # CA. Pinning the CA must accept any leaf it actually signed.
    ca_pem, leaf_tsa = _make_ca_and_leaf()
    qtsa = QualifiedTSA("https://qtsp.example/tsr", trusted_issuer_cert=ca_pem,
                        transport=_endpoint_for(leaf_tsa))
    anchor = qtsa.anchor_receipt(receipt)
    assert anchor["method"] == "rfc3161-eidas-qualified"
    assert anchor["authority"] == "EU Test QTSP Signer #1"
    attested = verify_receipt_anchor(receipt, anchor,
                                     trusted_issuer_cert=ca_pem)
    assert attested.tzinfo is not None


def test_issuer_pin_rejects_foreign_signer(receipt) -> None:
    # A token from a TSA the pinned CA never signed must be refused.
    ca_pem, _ = _make_ca_and_leaf()
    impostor = SelfHostedTSA.create("Impostor TSA")
    qtsa = QualifiedTSA("https://qtsp.example/tsr", trusted_issuer_cert=ca_pem,
                        transport=_endpoint_for(impostor))
    with pytest.raises(TimeAnchorError):
        qtsa.anchor_receipt(receipt)


def test_tsa_persists_authority_identity(tmp_path) -> None:
    first = SelfHostedTSA.load_or_create(tmp_path, "vaara-test-tsa")
    second = SelfHostedTSA.load_or_create(tmp_path, "ignored-on-load")
    assert first.authority == second.authority == "vaara-test-tsa"
    # Same persisted key signs verifiable tokens after a reload.
    digest = bytes(range(32))
    assert second.issue_token(digest)
