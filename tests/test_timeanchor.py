"""External time anchor (v0.48): RFC 3161 trusted-timestamp verification.

These tests stand up a self-contained in-process TSA (an EC key plus a
self-signed certificate) and issue real RFC 3161 tokens, so the full
build-request / parse-response / verify-token path is exercised with no
network. The verifier is the security-critical surface; the negative tests
prove it rejects a token over the wrong digest and a tampered token.
"""

from __future__ import annotations

import datetime
import hashlib
import os

import pytest

pytest.importorskip("asn1crypto")
pytest.importorskip("cryptography")

from asn1crypto import algos, cms, tsp  # noqa: E402
from asn1crypto import x509 as asn1_x509  # noqa: E402
from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.x509.oid import NameOID  # noqa: E402

from vaara.audit.timeanchor import (  # noqa: E402
    RFC3161TimeAnchorClient,
    TimeAnchor,
    TimeAnchorError,
    verify_anchor,
    verify_anchor_over_records,
    verify_timestamp_token,
)

_GEN_TIME = datetime.datetime(2026, 5, 31, 9, 0, 0, tzinfo=datetime.timezone.utc)


def _make_tsa():
    """Return (private_key, cert_der) for an in-process test TSA."""
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Vaara Test TSA")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime(2020, 1, 1))
        .not_valid_after(datetime.datetime(2035, 1, 1))
        .sign(key, hashes.SHA256())
    )
    return key, cert.public_bytes(serialization.Encoding.DER)


def _issue_token(digest, key, cert_der, *, hash_algorithm="sha256", gen_time=_GEN_TIME):
    """Build a valid RFC 3161 TimeStampToken over ``digest``."""
    cert_asn1 = asn1_x509.Certificate.load(cert_der)
    tst = tsp.TSTInfo({
        "version": 1,
        "policy": "1.3.6.1.4.1.99999.1",
        "message_imprint": tsp.MessageImprint({
            "hash_algorithm": algos.DigestAlgorithm({"algorithm": hash_algorithm}),
            "hashed_message": digest,
        }),
        "serial_number": 1,
        "gen_time": gen_time,
    })
    tst_der = tst.dump()
    attrs = cms.CMSAttributes([
        cms.CMSAttribute({"type": "content_type", "values": ["tst_info"]}),
        cms.CMSAttribute({
            "type": "message_digest",
            "values": [hashlib.sha256(tst_der).digest()],
        }),
    ])
    signature = key.sign(attrs.dump(), ec.ECDSA(hashes.SHA256()))
    signer_info = cms.SignerInfo({
        "version": "v1",
        "sid": cms.SignerIdentifier({
            "issuer_and_serial_number": cms.IssuerAndSerialNumber({
                "issuer": cert_asn1.issuer,
                "serial_number": cert_asn1.serial_number,
            }),
        }),
        "digest_algorithm": algos.DigestAlgorithm({"algorithm": "sha256"}),
        "signed_attrs": attrs,
        "signature_algorithm": algos.SignedDigestAlgorithm({"algorithm": "sha256_ecdsa"}),
        "signature": signature,
    })
    signed_data = cms.SignedData({
        "version": "v3",
        "digest_algorithms": [algos.DigestAlgorithm({"algorithm": "sha256"})],
        "encap_content_info": cms.EncapsulatedContentInfo({
            "content_type": "tst_info",
            "content": tst,  # pass the object; asn1crypto octet-wraps + encodes
        }),
        "certificates": [cms.CertificateChoices({"certificate": cert_asn1})],
        "signer_infos": [signer_info],
    })
    return cms.ContentInfo({"content_type": "signed_data", "content": signed_data}).dump()


def _make_response(token_der, status="granted"):
    """Wrap a token in an RFC 3161 TimeStampResp (no token on refusal)."""
    body = {"status": tsp.PKIStatusInfo({"status": status})}
    if token_der:
        body["time_stamp_token"] = cms.ContentInfo.load(token_der)
    return tsp.TimeStampResp(body).dump()


def _head_hash():
    """A plausible chain-head record_hash (a real SHA-256 hex digest)."""
    return hashlib.sha256(b"chain head").hexdigest()


def test_verify_token_roundtrip():
    key, cert = _make_tsa()
    digest = bytes.fromhex(_head_hash())
    token = _issue_token(digest, key, cert)
    attested = verify_timestamp_token(token, digest)
    assert attested == _GEN_TIME


def test_verify_rejects_wrong_digest():
    key, cert = _make_tsa()
    token = _issue_token(bytes.fromhex(_head_hash()), key, cert)
    other = hashlib.sha256(b"a different chain head").digest()
    with pytest.raises(TimeAnchorError, match="different value"):
        verify_timestamp_token(token, other)


def test_verify_rejects_tampered_token():
    key, cert = _make_tsa()
    digest = bytes.fromhex(_head_hash())
    token = bytearray(_issue_token(digest, key, cert))
    token[-1] ^= 0xFF  # flip a byte in the signature region
    with pytest.raises(TimeAnchorError):
        verify_timestamp_token(bytes(token), digest)


def test_client_anchor_with_local_tsa():
    """End to end: build request, local TSA issues, client verifies."""
    key, cert = _make_tsa()

    def transport(url, der_request, timeout):
        req = tsp.TimeStampReq.load(der_request)
        digest = req["message_imprint"]["hashed_message"].native
        return _make_response(_issue_token(digest, key, cert))

    client = RFC3161TimeAnchorClient("https://tsa.example/tsr", transport=transport)
    head = _head_hash()
    anchor = client.anchor(chain_position=41, chain_head_hash=head)
    assert isinstance(anchor, TimeAnchor)
    assert anchor.chain_position == 41
    assert anchor.chain_head_hash == head
    assert anchor.backend == "rfc3161"
    # The stored token re-verifies on its own.
    assert verify_anchor(anchor) == _GEN_TIME


def test_client_raises_on_tsa_refusal():
    key, cert = _make_tsa()

    def transport(url, der_request, timeout):
        # A refusal carries a status but the client must not accept the token.
        token = _issue_token(bytes.fromhex(_head_hash()), key, cert)
        return _make_response(token, status="rejection")

    client = RFC3161TimeAnchorClient("https://tsa.example/tsr", transport=transport)
    with pytest.raises(TimeAnchorError, match="refused"):
        client.anchor(0, _head_hash())


def _local_tsa_client():
    key, cert = _make_tsa()

    def transport(url, der_request, timeout):
        req = tsp.TimeStampReq.load(der_request)
        digest = req["message_imprint"]["hashed_message"].native
        return _make_response(_issue_token(digest, key, cert))

    return RFC3161TimeAnchorClient("https://tsa.example/tsr", transport=transport)


def test_anchor_head_binds_live_trail():
    from vaara.audit.trail import AuditTrail
    from vaara.taxonomy.actions import (
        ActionCategory,
        ActionRequest,
        ActionType,
        BlastRadius,
        Reversibility,
    )

    trail = AuditTrail()
    trail.record_action_requested(ActionRequest(
        action_type=ActionType(
            name="t", category=ActionCategory.DATA,
            reversibility=Reversibility.FULLY, blast_radius=BlastRadius.LOCAL,
        ),
        tool_name="t", agent_id="agent", parameters={},
    ))

    anchor = trail.anchor_head(_local_tsa_client())
    assert anchor.chain_position == trail.size - 1
    assert len(trail.anchors) == 1
    # The anchor binds to the actual chain head and re-verifies offline.
    record_hashes = [r.record_hash for r in trail._records]
    assert verify_anchor_over_records(anchor, record_hashes) == _GEN_TIME


def test_anchor_head_rejects_empty_trail():
    from vaara.audit.trail import AuditTrail

    with pytest.raises(ValueError, match="empty trail"):
        AuditTrail().anchor_head(_local_tsa_client())


def test_verify_anchor_over_records_binds_to_chain():
    key, cert = _make_tsa()
    head = _head_hash()
    token = _issue_token(bytes.fromhex(head), key, cert)
    import base64
    anchor = TimeAnchor(
        chain_position=2,
        chain_head_hash=head,
        backend="rfc3161",
        tsa_url="https://tsa.example/tsr",
        hash_algorithm="sha256",
        token_b64=base64.b64encode(token).decode("ascii"),
        anchored_time=_GEN_TIME.isoformat(),
    )
    record_hashes = ["aa" * 32, "bb" * 32, head]  # head sits at position 2
    assert verify_anchor_over_records(anchor, record_hashes) == _GEN_TIME


def test_verify_anchor_over_records_rejects_rewritten_chain():
    key, cert = _make_tsa()
    head = _head_hash()
    token = _issue_token(bytes.fromhex(head), key, cert)
    import base64
    anchor = TimeAnchor(
        chain_position=2,
        chain_head_hash=head,
        backend="rfc3161",
        tsa_url="https://tsa.example/tsr",
        hash_algorithm="sha256",
        token_b64=base64.b64encode(token).decode("ascii"),
        anchored_time=_GEN_TIME.isoformat(),
    )
    # A rewritten chain has a different hash at the anchored position.
    rewritten = ["aa" * 32, "bb" * 32, "cc" * 32]
    with pytest.raises(TimeAnchorError, match="does not match the trail"):
        verify_anchor_over_records(anchor, rewritten)


# Live network test: anchors a real chain head against a real public RFC 3161
# TSA, then verifies the returned token offline. Opt-in only (set VAARA_LIVE_TSA)
# so CI and offline runs never depend on a third party. The in-process tests
# above cover the verifier; this proves the round trip against an authority we
# do not control.
_LIVE_TSAS = [
    "http://timestamp.digicert.com",
    "http://timestamp.sectigo.com",
    "https://freetsa.org/tsr",
]


@pytest.mark.skipif(
    not os.environ.get("VAARA_LIVE_TSA"),
    reason="live TSA round trip; set VAARA_LIVE_TSA=1 to run",
)
def test_anchor_head_against_live_tsa():
    from vaara.audit.trail import AuditTrail
    from vaara.taxonomy.actions import (
        ActionCategory, ActionRequest, ActionType, BlastRadius, Reversibility,
    )

    trail = AuditTrail()
    trail.record_action_requested(ActionRequest(
        action_type=ActionType(
            name="live_anchor", category=ActionCategory.DATA,
            reversibility=Reversibility.FULLY, blast_radius=BlastRadius.LOCAL,
        ),
        tool_name="t", agent_id="agent", parameters={},
    ))

    errors = {}
    for url in _LIVE_TSAS:
        try:
            anchor = trail.anchor_head(RFC3161TimeAnchorClient(url, timeout=20.0))
        except TimeAnchorError as exc:
            errors[url] = str(exc)
            continue
        # Offline verification, bound to the real chain head, no network.
        record_hashes = [r.record_hash for r in trail._records]
        attested = verify_anchor_over_records(anchor, record_hashes)
        assert anchor.backend == "rfc3161"
        assert anchor.chain_position == trail.size - 1
        assert anchor.chain_head_hash == trail._records[-1].record_hash
        assert attested.isoformat() == anchor.anchored_time
        return
    pytest.fail(f"no live TSA granted/verified: {errors}")


# Automatic cadence-based anchoring (v0.49): once a TSA is configured the
# trail anchors its own head every N records, fail-open with a chained gap
# marker so a TSA outage is itself recorded and provable.

def _add_actions(trail, n):
    from vaara.taxonomy.actions import (
        ActionCategory, ActionRequest, ActionType, BlastRadius, Reversibility,
    )
    for i in range(n):
        trail.record_action_requested(ActionRequest(
            action_type=ActionType(
                name=f"a{i}", category=ActionCategory.DATA,
                reversibility=Reversibility.FULLY, blast_radius=BlastRadius.LOCAL,
            ),
            tool_name="t", agent_id="agent", parameters={},
        ))


def _failing_tsa_client():
    def transport(url, der_request, timeout):
        raise ConnectionError("TSA unreachable")
    return RFC3161TimeAnchorClient("https://tsa.example/tsr", transport=transport)


def test_auto_anchor_off_by_default():
    from vaara.audit.trail import AuditTrail, EventType
    trail = AuditTrail()
    _add_actions(trail, 5)
    assert trail.anchors == []
    assert not any(r.event_type == EventType.ANCHOR_GAP for r in trail._records)


def test_enable_auto_anchor_rejects_non_positive_cadence():
    from vaara.audit.trail import AuditTrail
    trail = AuditTrail()
    with pytest.raises(ValueError):
        trail.enable_auto_anchor(_local_tsa_client(), every_records=0)


def test_enable_auto_anchor_has_default_cadence():
    # Turning anchoring on should not require choosing a number: the default
    # cadence (32 records) applies when every_records is omitted.
    from vaara.audit.trail import AuditTrail
    trail = AuditTrail()
    trail.enable_auto_anchor(_local_tsa_client())
    _add_actions(trail, 31)
    assert trail.anchors == []
    _add_actions(trail, 1)
    assert len(trail.anchors) == 1


def test_auto_anchor_fires_on_cadence():
    from vaara.audit.trail import AuditTrail
    trail = AuditTrail()
    trail.enable_auto_anchor(_local_tsa_client(), every_records=3)

    _add_actions(trail, 3)
    assert len(trail.anchors) == 1
    # The anchor binds the head that was current when the cadence tripped.
    record_hashes = [r.record_hash for r in trail._records]
    assert verify_anchor_over_records(trail.anchors[0], record_hashes) == _GEN_TIME

    _add_actions(trail, 3)
    assert len(trail.anchors) == 2


def test_auto_anchor_fail_open_records_chained_gap():
    from vaara.audit.trail import AuditTrail, EventType
    trail = AuditTrail()
    trail.enable_auto_anchor(_failing_tsa_client(), every_records=3)

    # Recording must not raise even though the TSA is unreachable.
    _add_actions(trail, 3)

    gaps = [r for r in trail._records if r.event_type == EventType.ANCHOR_GAP]
    assert len(gaps) == 1
    assert "unreachable" in gaps[0].data["reason"].lower()
    # No successful anchor, but the chain (including the gap marker) is intact.
    assert trail.anchors == []
    assert trail.verify_chain() is None
    # The gap marker sits inside the hash chain as the latest record.
    assert trail._records[-1].event_type == EventType.ANCHOR_GAP
