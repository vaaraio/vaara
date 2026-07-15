"""1.37: qualified existence-in-time proof on the SEP-2828 execution record.

The record already proves *what* an attested call did and links to the
decision it answers. It does not, on its own, prove *when* it existed to a
party who runs none of your software. ``existenceProof`` carries an RFC 3161
trusted timestamp over the record's own bytes, so a relying party can check
the exact record existed no later than an attested instant, offline. When the
token's signer chains to a CA the caller pins from an EU trusted list (an
eIDAS qualified TSA), the time is *qualified*; absent a pin it is only
self-asserted by whoever holds the signing certificate, and the verifier says
so. That distinction is the whole point: an unwitnessed timestamp is not
evidence.

The proof rides outside the signed preimage, exactly like ``pqSignature``: it
is produced after signing and imprints the signed record, so it cannot be a
field the signature covers.

The verifier tests stand up an in-process CA that issues a TSA certificate and
mint real RFC 3161 tokens, so the full recompute / imprint / issuer-pin path
runs with no network. The negatives prove it rejects a token over the wrong
digest and reports a token whose signer is not on the pinned list as
self-asserted rather than qualified.
"""

from __future__ import annotations

import base64
import datetime
import hashlib

import pytest

_HAS_CRYPTO = True
try:
    from asn1crypto import algos, cms, tsp
    from asn1crypto import x509 as asn1_x509
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
except ImportError:  # pragma: no cover - base install
    _HAS_CRYPTO = False

_HAS_CANON = True
try:
    import rfc8785  # noqa: F401 - JCS canonicalization, part of the attestation extra
except ImportError:  # pragma: no cover - base install
    _HAS_CANON = False

requires_canon = pytest.mark.skipif(
    not _HAS_CANON, reason="record digest needs the attestation extra (rfc8785)"
)
requires_crypto = pytest.mark.skipif(
    not (_HAS_CRYPTO and _HAS_CANON),
    reason="qualified existence proof needs the timeanchor extra",
)

_GEN_TIME = datetime.datetime(2026, 5, 31, 9, 0, 0, tzinfo=datetime.timezone.utc)


def _minimal_record() -> dict:
    """A well-formed execution record dict, signature included."""
    return {
        "version": 1,
        "alg": "ES256",
        "backLink": {
            "attestationDigest": "sha256:" + "aa" * 32,
            "attestationNonce": "n0",
        },
        "outcomeDerived": {
            "status": "executed",
            "completedAt": "2026-05-31T09:00:00Z",
        },
        "receiptAsserted": {
            "alg": "ES256",
            "iat": "2026-05-31T09:00:00Z",
            "iss": "did:web:issuer.example",
            "nonce": "n0",
            "secretVersion": "1",
            "sub": "tool:read",
        },
        "signature": "ab12cd34",
    }


@requires_canon
def test_record_digest_excludes_existence_proof_and_is_sha256():
    """The digest the timestamp imprints covers the record but not the proof.

    The proof cannot cover itself, so attaching it must not move the digest a
    verifier recomputes. And the digest is a ``sha256:`` hex string.
    """
    from vaara.attestation._receipt_existence import existence_record_digest

    base = _minimal_record()
    with_proof = dict(base)
    with_proof["existenceProof"] = {
        "backend": "rfc3161-eidas-qualified",
        "hashAlgorithm": "sha256",
        "recordDigest": "sha256:" + "00" * 32,
        "token": "zzz",
    }

    d_base = existence_record_digest(base)
    d_with = existence_record_digest(with_proof)

    assert d_base == d_with
    assert d_base.startswith("sha256:")
    assert len(d_base) == len("sha256:") + 64


# --- crypto fixtures: an in-process CA that issues a TSA certificate ---------


def _self_signed_ca():
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Vaara Test Root CA")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime(2020, 1, 1))
        .not_valid_after(datetime.datetime(2035, 1, 1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return key, cert


def _tsa_issued_by(ca_key, ca_cert, cn="Vaara Test QTSA"):
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime(2020, 1, 1))
        .not_valid_after(datetime.datetime(2035, 1, 1))
        .sign(ca_key, hashes.SHA256())
    )
    return key, cert.public_bytes(serialization.Encoding.DER)


def _issue_token(digest, key, cert_der, *, gen_time=_GEN_TIME):
    """Build a valid RFC 3161 TimeStampToken over ``digest`` (32 bytes)."""
    cert_asn1 = asn1_x509.Certificate.load(cert_der)
    tst = tsp.TSTInfo({
        "version": 1,
        "policy": "1.3.6.1.4.1.99999.1",
        "message_imprint": tsp.MessageImprint({
            "hash_algorithm": algos.DigestAlgorithm({"algorithm": "sha256"}),
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
            "content": tst,
        }),
        "certificates": [cms.CertificateChoices({"certificate": cert_asn1})],
        "signer_infos": [signer_info],
    })
    return cms.ContentInfo({"content_type": "signed_data", "content": signed_data}).dump()


def _make_response(token_der, status="granted"):
    body = {"status": tsp.PKIStatusInfo({"status": status})}
    if token_der:
        body["time_stamp_token"] = cms.ContentInfo.load(token_der)
    return tsp.TimeStampResp(body).dump()


def _ca_der(ca_cert) -> bytes:
    return ca_cert.public_bytes(serialization.Encoding.DER)


def _attach_proof(record: dict, token_der: bytes) -> dict:
    """Staple an existence proof to a record (recordDigest = recomputed)."""
    from vaara.attestation._receipt_existence import existence_record_digest

    out = dict(record)
    out["existenceProof"] = {
        "backend": "rfc3161-eidas-qualified",
        "hashAlgorithm": "sha256",
        "recordDigest": existence_record_digest(record),
        "token": base64.b64encode(token_der).decode("ascii"),
    }
    return out


def _token_over_record(record, signer_key, signer_cert_der):
    from vaara.attestation._receipt_existence import existence_record_digest

    digest_hex = existence_record_digest(record).split(":", 1)[1]
    return _issue_token(bytes.fromhex(digest_hex), signer_key, signer_cert_der)


@requires_crypto
def test_qualified_when_signer_chains_to_pinned_ca():
    from vaara.attestation._receipt_existence import verify_existence_proof

    ca_key, ca_cert = _self_signed_ca()
    tsa_key, tsa_der = _tsa_issued_by(ca_key, ca_cert)
    record = _minimal_record()
    token = _token_over_record(record, tsa_key, tsa_der)
    record = _attach_proof(record, token)

    res = verify_existence_proof(record, trusted_issuer_cert=_ca_der(ca_cert))

    assert res.present
    assert res.ok
    assert res.qualified
    assert res.basis == "qualified"
    assert res.attested_time == _GEN_TIME.isoformat()


@requires_crypto
def test_self_asserted_without_a_pin():
    from vaara.attestation._receipt_existence import verify_existence_proof

    ca_key, ca_cert = _self_signed_ca()
    tsa_key, tsa_der = _tsa_issued_by(ca_key, ca_cert)
    record = _minimal_record()
    record = _attach_proof(record, _token_over_record(record, tsa_key, tsa_der))

    res = verify_existence_proof(record)  # no trusted list pinned

    assert res.present
    assert res.ok  # the token is internally valid
    assert not res.qualified  # but not backed by a pinned authority
    assert res.basis == "self_asserted"
    assert res.attested_time == _GEN_TIME.isoformat()


@requires_crypto
def test_not_qualified_when_pinned_ca_did_not_issue_signer():
    from vaara.attestation._receipt_existence import verify_existence_proof

    ca_key, ca_cert = _self_signed_ca()
    tsa_key, tsa_der = _tsa_issued_by(ca_key, ca_cert)
    record = _minimal_record()
    record = _attach_proof(record, _token_over_record(record, tsa_key, tsa_der))

    other_ca_key, other_ca_cert = _self_signed_ca()
    res = verify_existence_proof(record, trusted_issuer_cert=_ca_der(other_ca_cert))

    # The timestamp is valid, so ok holds; it is simply not from the pinned
    # authority, so it is not qualified. A wrong pin is not a forged proof.
    assert res.ok
    assert not res.qualified
    assert res.basis == "self_asserted"


@requires_crypto
def test_rejects_token_over_a_different_digest():
    from vaara.attestation._receipt_existence import verify_existence_proof

    ca_key, ca_cert = _self_signed_ca()
    tsa_key, tsa_der = _tsa_issued_by(ca_key, ca_cert)
    record = _minimal_record()
    # A token timestamping some other value, then stapled to this record.
    wrong = _issue_token(hashlib.sha256(b"not the record").digest(), tsa_key, tsa_der)
    record = _attach_proof(record, wrong)

    res = verify_existence_proof(record, trusted_issuer_cert=_ca_der(ca_cert))

    assert res.present
    assert not res.ok
    assert not res.qualified


@requires_crypto
def test_rejects_claimed_digest_that_does_not_match_the_record():
    from vaara.attestation._receipt_existence import verify_existence_proof

    ca_key, ca_cert = _self_signed_ca()
    tsa_key, tsa_der = _tsa_issued_by(ca_key, ca_cert)
    record = _minimal_record()
    record = _attach_proof(record, _token_over_record(record, tsa_key, tsa_der))
    # Mutate the record after the proof was attached; the recomputed digest
    # no longer matches the proof's claimed recordDigest.
    record["outcomeDerived"]["status"] = "refused"

    res = verify_existence_proof(record, trusted_issuer_cert=_ca_der(ca_cert))

    assert res.present
    assert not res.ok


@requires_crypto
def test_attach_then_verify_roundtrips_via_local_tsa():
    """Anchoring a record against a TSA yields a proof that verifies."""
    from vaara.attestation._receipt_existence import (
        attach_existence_proof,
        verify_existence_proof,
    )

    ca_key, ca_cert = _self_signed_ca()
    tsa_key, tsa_der = _tsa_issued_by(ca_key, ca_cert)

    def transport(url, der_request, timeout):
        req = tsp.TimeStampReq.load(der_request)
        digest = req["message_imprint"]["hashed_message"].native
        return _make_response(_issue_token(digest, tsa_key, tsa_der))

    record = attach_existence_proof(
        _minimal_record(), tsa_url="https://tsa.example/tsr", transport=transport
    )

    assert record["existenceProof"]["backend"] == "rfc3161-eidas-qualified"
    res = verify_existence_proof(record, trusted_issuer_cert=_ca_der(ca_cert))
    assert res.ok
    assert res.qualified
    assert res.attested_time == _GEN_TIME.isoformat()


def test_absent_proof_reports_not_present():
    from vaara.attestation._receipt_existence import verify_existence_proof

    res = verify_existence_proof(_minimal_record())

    assert not res.present
    assert not res.qualified


def test_receipt_schema_roundtrips_existence_proof():
    """The closed record schema accepts existenceProof and preserves it.

    Adding a field to a closed, signature-covered schema is an explicit
    version bump, not a silently tolerated extra: parsing must accept it and
    ``to_dict`` must round-trip it byte-for-byte.
    """
    from vaara.attestation._receipt_types import receipt_from_dict

    rec = _minimal_record()
    rec["existenceProof"] = {
        "backend": "rfc3161-eidas-qualified",
        "hashAlgorithm": "sha256",
        "recordDigest": "sha256:" + "00" * 32,
        "token": "zzz",
    }

    receipt = receipt_from_dict(rec)

    assert receipt.to_dict()["existenceProof"] == rec["existenceProof"]


@requires_canon
def test_existence_proof_stays_outside_the_signed_preimage():
    """Attaching a proof must not change the bytes the signature covers."""
    from vaara.attestation._receipt_emit import _signing_payload
    from vaara.attestation._receipt_types import receipt_from_dict

    rec = _minimal_record()
    plain = receipt_from_dict(rec)
    with_proof = dict(rec)
    with_proof["existenceProof"] = {
        "backend": "rfc3161-eidas-qualified",
        "hashAlgorithm": "sha256",
        "recordDigest": "sha256:" + "00" * 32,
        "token": "zzz",
    }
    proofed = receipt_from_dict(with_proof)

    def preimage(r):
        return _signing_payload(
            version=r.version, alg=r.alg, back_link=r.back_link,
            outcome_derived=r.outcome_derived, receipt_asserted=r.receipt_asserted,
        )

    assert preimage(plain) == preimage(proofed)
