# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""`vaara trail verify-anchor` takes the TSA pin the library already enforces.

Without a pin the time is whatever the certificate inside the package asserts,
and the command says so. With ``--tsa-cert`` (the signer) or ``--tsa-issuer``
(the CA that issued it) a token from any other authority is refused. A self-
signed in-process TSA issues real RFC 3161 tokens, so no network is used.
"""
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

import pytest

pytest.importorskip("asn1crypto")
pytest.importorskip("cryptography")

from asn1crypto import algos, cms, tsp  # noqa: E402
from asn1crypto import x509 as asn1_x509  # noqa: E402
from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
    Ed25519PrivateKey,
)
from cryptography.x509.oid import NameOID  # noqa: E402

from vaara.audit.article12_export import export_article12  # noqa: E402
from vaara.audit.timeanchor import RFC3161TimeAnchorClient  # noqa: E402
from vaara.audit.trail import AuditTrail  # noqa: E402
from vaara.cli import main  # noqa: E402
from vaara.taxonomy.actions import ActionRequest, create_default_registry  # noqa: E402

_GEN_TIME = datetime.datetime(2026, 6, 7, 12, 0, 0, tzinfo=datetime.timezone.utc)
_UNPINNED = "UNPINNED: time as asserted by the certificate in the package"


def _name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _cert(subject, issuer, public_key, signing_key) -> x509.Certificate:
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime(2020, 1, 1))
        .not_valid_after(datetime.datetime(2035, 1, 1))
        .sign(signing_key, hashes.SHA256())
    )


class _Tsa:
    """A TSA key and certificate; ``ca`` (key, cert) makes the CA issue it."""

    def __init__(self, common_name: str, ca=None) -> None:
        self.key = ec.generate_private_key(ec.SECP256R1())
        subject = _name(common_name)
        if ca is None:
            self.cert = _cert(subject, subject, self.key.public_key(), self.key)
        else:
            ca_key, ca_cert = ca
            self.cert = _cert(subject, ca_cert.subject, self.key.public_key(), ca_key)

    def pem(self) -> bytes:
        return self.cert.public_bytes(serialization.Encoding.PEM)

    def issue(self, digest: bytes) -> bytes:
        cert_asn1 = asn1_x509.Certificate.load(
            self.cert.public_bytes(serialization.Encoding.DER)
        )
        tst = tsp.TSTInfo({
            "version": 1,
            "policy": "1.3.6.1.4.1.99999.1",
            "message_imprint": tsp.MessageImprint({
                "hash_algorithm": algos.DigestAlgorithm({"algorithm": "sha256"}),
                "hashed_message": digest,
            }),
            "serial_number": 1,
            "gen_time": _GEN_TIME,
        })
        attrs = cms.CMSAttributes([
            cms.CMSAttribute({"type": "content_type", "values": ["tst_info"]}),
            cms.CMSAttribute({
                "type": "message_digest",
                "values": [hashlib.sha256(tst.dump()).digest()],
            }),
        ])
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
            "signature_algorithm": algos.SignedDigestAlgorithm(
                {"algorithm": "sha256_ecdsa"}),
            "signature": self.key.sign(attrs.dump(), ec.ECDSA(hashes.SHA256())),
        })
        signed = cms.SignedData({
            "version": "v3",
            "digest_algorithms": [algos.DigestAlgorithm({"algorithm": "sha256"})],
            "encap_content_info": cms.EncapsulatedContentInfo({
                "content_type": "tst_info", "content": tst,
            }),
            "certificates": [cms.CertificateChoices({"certificate": cert_asn1})],
            "signer_infos": [signer_info],
        })
        return cms.ContentInfo({"content_type": "signed_data", "content": signed}).dump()

    def client(self) -> RFC3161TimeAnchorClient:
        def transport(url, der_request, timeout):
            req = tsp.TimeStampReq.load(der_request)
            digest = req["message_imprint"]["hashed_message"].native
            return tsp.TimeStampResp({
                "status": tsp.PKIStatusInfo({"status": "granted"}),
                "time_stamp_token": cms.ContentInfo.load(self.issue(digest)),
            }).dump()

        return RFC3161TimeAnchorClient("https://tsa.test/tsr", transport=transport)


def _package(tmp_path: Path, tsa: _Tsa) -> Path:
    """An Article 12 package whose time anchor was issued by ``tsa``."""
    registry = create_default_registry()
    trail = AuditTrail()
    for i in range(2):
        trail.record_action_requested(ActionRequest(
            agent_id=f"agent-{i}",
            tool_name="send_funds",
            action_type=registry.get("tx.transfer"),
            parameters={"amount": i},
        ))
    records = trail._records
    anchor = tsa.client().anchor(len(records) - 1, records[-1].record_hash)
    key = tmp_path / "signer.pem"
    key.write_bytes(Ed25519PrivateKey.generate().private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    out = tmp_path / "art12.zip"
    export_article12(trail, out, signer_key=key, time_anchor=anchor)
    return out


def _write(tmp_path: Path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


def test_unpinned_says_the_time_is_as_asserted_by_the_package(tmp_path, capsys):
    pkg = _package(tmp_path, _Tsa("Package TSA"))

    rc = main(["trail", "verify-anchor", "--zip", str(pkg)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "time anchor OK" in out
    assert _UNPINNED in out
    assert "no later than" in out
    assert _GEN_TIME.isoformat() in out
    assert "pin the TSA certificate" not in out


def test_a_matching_signer_pin_is_reported_as_pinned(tmp_path, capsys):
    tsa = _Tsa("Pinned TSA")
    pkg = _package(tmp_path, tsa)

    rc = main([
        "trail", "verify-anchor", "--zip", str(pkg),
        "--tsa-cert", _write(tmp_path, "tsa.pem", tsa.pem()),
    ])

    out = capsys.readouterr().out
    assert rc == 0
    assert "time anchor OK" in out
    assert "signer certificate equals --tsa-cert" in out
    assert "UNPINNED" not in out


def test_a_signer_pin_for_another_authority_is_refused(tmp_path, capsys):
    pkg = _package(tmp_path, _Tsa("Package TSA"))
    other = _Tsa("Some Other TSA")

    rc = main([
        "trail", "verify-anchor", "--zip", str(pkg),
        "--tsa-cert", _write(tmp_path, "other.pem", other.pem()),
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "time anchor INVALID" in captured.err
    assert "does not match the pinned trusted TSA certificate" in captured.err
    assert "time anchor OK" not in captured.out


def test_a_matching_issuer_pin_is_reported_as_pinned(tmp_path, capsys):
    ca = _Tsa("Trusted CA")
    tsa = _Tsa("Issued TSA", ca=(ca.key, ca.cert))
    pkg = _package(tmp_path, tsa)

    rc = main([
        "trail", "verify-anchor", "--zip", str(pkg),
        "--tsa-issuer", _write(tmp_path, "ca.pem", ca.pem()),
    ])

    out = capsys.readouterr().out
    assert rc == 0
    assert "signer issued by --tsa-issuer" in out
    assert "UNPINNED" not in out


def test_an_issuer_pin_that_did_not_issue_the_signer_is_refused(tmp_path, capsys):
    ca = _Tsa("Trusted CA")
    tsa = _Tsa("Issued TSA", ca=(ca.key, ca.cert))
    pkg = _package(tmp_path, tsa)
    stranger = _Tsa("Stranger CA")

    rc = main([
        "trail", "verify-anchor", "--zip", str(pkg),
        "--tsa-issuer", _write(tmp_path, "stranger.pem", stranger.pem()),
    ])

    captured = capsys.readouterr()
    assert rc == 1
    assert "time anchor INVALID" in captured.err
    assert "was not issued by the pinned trusted CA" in captured.err


def test_a_missing_pin_file_is_a_usage_error(tmp_path, capsys):
    pkg = _package(tmp_path, _Tsa("Package TSA"))

    rc = main([
        "trail", "verify-anchor", "--zip", str(pkg),
        "--tsa-cert", str(tmp_path / "nope.pem"),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "--tsa-cert file not found" in captured.err
    assert "time anchor OK" not in captured.out


def test_a_pin_file_that_is_not_a_certificate_is_a_usage_error(tmp_path, capsys):
    pkg = _package(tmp_path, _Tsa("Package TSA"))

    rc = main([
        "trail", "verify-anchor", "--zip", str(pkg),
        "--tsa-issuer", _write(tmp_path, "junk.pem", b"not a certificate"),
    ])

    captured = capsys.readouterr()
    assert rc == 2
    assert "--tsa-issuer is not a PEM or DER certificate" in captured.err
    assert "time anchor OK" not in captured.out
