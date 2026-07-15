#!/usr/bin/env python3
"""Generate the qualified_time_v0 conformance vectors.

Mints an in-process CA that issues a TSA certificate, timestamps execution
records over their own canonical bytes, and writes the cases, the expected
verdicts, and the pinned CA certificate. Imports Vaara only to build and
cross-check the records; the committed vectors are graded by
``_check_independent.py`` with no Vaara import.

The vectors prove the existence-in-time verifier obligation: a relying party
recomputes the record digest, checks the RFC 3161 token imprints exactly that
digest, and checks the token's signer against a CA it pins from a trusted list.
A signer that chains to the pinned CA makes the attested time qualified; one
that does not is a valid but self-asserted timestamp; a token over the wrong
digest, a record mutated after the fact, or a corrupt token is not evidence.

Run: ``python tests/vectors/qualified_time_v0/_generate.py``.
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
from pathlib import Path

from asn1crypto import algos, cms, tsp
from asn1crypto import x509 as asn1_x509
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from vaara.attestation.receipt import (
    attach_existence_proof,
    existence_record_digest,
    verify_existence_proof,
)

HERE = Path(__file__).resolve().parent
GEN_TIME = datetime.datetime(2026, 7, 15, 9, 0, 0, tzinfo=datetime.timezone.utc)


def _ca(cn: str):
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])
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


def _tsa_issued_by(ca_key, ca_cert, cn: str):
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


def _issue_token(digest: bytes, key, cert_der: bytes) -> bytes:
    cert_asn1 = asn1_x509.Certificate.load(cert_der)
    tst = tsp.TSTInfo({
        "version": 1,
        "policy": "1.3.6.1.4.1.99999.1",
        "message_imprint": tsp.MessageImprint({
            "hash_algorithm": algos.DigestAlgorithm({"algorithm": "sha256"}),
            "hashed_message": digest,
        }),
        "serial_number": 1,
        "gen_time": GEN_TIME,
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


def _make_response(token_der: bytes) -> bytes:
    return tsp.TimeStampResp({
        "status": tsp.PKIStatusInfo({"status": "granted"}),
        "time_stamp_token": cms.ContentInfo.load(token_der),
    }).dump()


def _minter(key, cert_der):
    def transport(url, der_request, timeout):
        req = tsp.TimeStampReq.load(der_request)
        digest = req["message_imprint"]["hashed_message"].native
        return _make_response(_issue_token(digest, key, cert_der))
    return transport


def _record(sub: str, status: str = "executed") -> dict:
    return {
        "version": 1,
        "alg": "ES256",
        "backLink": {
            "attestationDigest": "sha256:" + "aa" * 32,
            "attestationNonce": "n-" + sub,
        },
        "outcomeDerived": {"status": status, "completedAt": "2026-07-15T08:59:00Z"},
        "receiptAsserted": {
            "alg": "ES256",
            "iat": "2026-07-15T08:59:00Z",
            "iss": "did:web:issuer.example",
            "nonce": "n-" + sub,
            "secretVersion": "1",
            "sub": sub,
        },
        "signature": "ab12cd34",
    }


def main() -> int:
    trusted_ca_key, trusted_ca_cert = _ca("Vaara Test Trusted-List CA")
    tsa_key, tsa_der = _tsa_issued_by(trusted_ca_key, trusted_ca_cert, "Vaara Test QTSA")
    other_ca_key, other_ca_cert = _ca("Vaara Test Untrusted CA")
    other_tsa_key, other_tsa_der = _tsa_issued_by(other_ca_key, other_ca_cert, "Untrusted TSA")
    trusted_ca_der = trusted_ca_cert.public_bytes(serialization.Encoding.DER)

    cases: list[dict] = []

    # qualified: token signer chains to the pinned trusted-list CA.
    cases.append({
        "name": "qualified_ok",
        "record": attach_existence_proof(
            _record("tool:read"), tsa_url="local", transport=_minter(tsa_key, tsa_der)
        ),
    })

    # valid token, signer not on the pinned list: self-asserted, not qualified.
    cases.append({
        "name": "self_asserted_untrusted_issuer",
        "record": attach_existence_proof(
            _record("tool:write"), tsa_url="local",
            transport=_minter(other_tsa_key, other_tsa_der),
        ),
    })

    # token timestamps a different value, then stapled to this record.
    base = _record("tool:list")
    wrong = _issue_token(hashlib.sha256(b"not the record").digest(), tsa_key, tsa_der)
    stapled = dict(base)
    stapled["existenceProof"] = {
        "backend": "rfc3161-eidas-qualified",
        "hashAlgorithm": "sha256",
        "recordDigest": existence_record_digest(base),
        "token": base64.b64encode(wrong).decode("ascii"),
    }
    cases.append({"name": "neg_wrong_digest", "record": stapled})

    # record mutated after the proof was attached; claimed digest no longer matches.
    tampered = attach_existence_proof(
        _record("tool:delete"), tsa_url="local", transport=_minter(tsa_key, tsa_der)
    )
    tampered["outcomeDerived"]["status"] = "refused"
    cases.append({"name": "neg_tampered_record", "record": tampered})

    # corrupt the token bytes.
    malformed = attach_existence_proof(
        _record("tool:exec"), tsa_url="local", transport=_minter(tsa_key, tsa_der)
    )
    raw = bytearray(base64.b64decode(malformed["existenceProof"]["token"]))
    raw[-1] ^= 0xFF
    malformed["existenceProof"]["token"] = base64.b64encode(bytes(raw)).decode("ascii")
    cases.append({"name": "neg_malformed_token", "record": malformed})

    # Ground truth: the reference verifier, pinning the trusted-list CA. The
    # independent checker must reproduce exactly these verdicts.
    expected: dict[str, dict] = {}
    for case in cases:
        res = verify_existence_proof(
            case["record"], trusted_issuer_cert=trusted_ca_der
        )
        expected[case["name"]] = {"ok": res.ok, "qualified": res.qualified}

    (HERE / "cases.json").write_text(
        json.dumps({"cases": cases}, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "trusted_ca.pem").write_bytes(
        trusted_ca_cert.public_bytes(serialization.Encoding.PEM)
    )
    print(f"wrote {len(cases)} cases; expected = {json.dumps(expected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
