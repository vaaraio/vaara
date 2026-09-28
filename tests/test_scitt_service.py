# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SCITT Transparency Service registration and the offline receipt check.

The live half (a real Microsoft SCITT CCF ledger, built and started in CI) is
``.github/workflows/scitt-roundtrip.yml``. Here a stand-in service builds CCF
profile receipts the way the ledger does, so the verifier's every branch runs
on each PR, and a local HTTP stub walks the SCRAPI 303, 302, 200 sequence.
"""
from __future__ import annotations

import hashlib
import http.server
import threading

import pytest

cbor2 = pytest.importorskip("cbor2")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives import hashes, serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec, utils  # noqa: E402

from vaara.audit import scitt_service as ss  # noqa: E402


def _kid(pub) -> bytes:
    der = pub.public_bytes(serialization.Encoding.DER,
                           serialization.PublicFormat.SubjectPublicKeyInfo)
    return hashlib.sha256(der).hexdigest().encode()


def _cose_key(pub) -> dict:
    n = pub.public_numbers()
    size = (pub.curve.key_size + 7) // 8
    crv = {256: 1, 384: 2, 521: 3}[pub.curve.key_size]
    return {1: 2, 2: _kid(pub), -1: crv,
            -2: n.x.to_bytes(size, "big"), -3: n.y.to_bytes(size, "big")}


class FakeLedger:
    """Signs CCF-profile receipts as the SCITT CCF ledger does."""

    def __init__(self, curve=ec.SECP384R1(), alg=-35, hash_=hashes.SHA384()):
        self.key = ec.generate_private_key(curve)
        self.alg, self.hash, self.size = alg, hash_, (curve.key_size + 7) // 8

    def key_set(self) -> bytes:
        return cbor2.dumps([_cose_key(self.key.public_key())])

    def receipt(self, statement: bytes, *, siblings=3, data_hash=None, vds=2) -> bytes:
        internal = hashlib.sha256(b"txn").digest()
        evidence = "ce:2.15:" + "ab" * 32
        leaf_data = data_hash or hashlib.sha256(statement).digest()
        acc = hashlib.sha256(internal + hashlib.sha256(evidence.encode()).digest()
                             + leaf_data).digest()
        path = []
        for i in range(siblings):
            sib = hashlib.sha256(bytes([i])).digest()
            left = i % 2 == 0
            path.append([left, sib])
            acc = hashlib.sha256(sib + acc if left else acc + sib).digest()
        proof = cbor2.dumps({1: [internal, evidence, leaf_data], 2: path})
        protected = cbor2.dumps({1: self.alg, 4: _kid(self.key.public_key()), 395: vds})
        tbs = cbor2.dumps(["Signature1", protected, b"", acc])
        r, s = utils.decode_dss_signature(self.key.sign(tbs, ec.ECDSA(self.hash)))
        sig = r.to_bytes(self.size, "big") + s.to_bytes(self.size, "big")
        return cbor2.dumps(cbor2.CBORTag(18, [protected, {396: {-1: [proof]}}, None, sig]))


@pytest.fixture
def statement():
    return b"\xd2" + b"any registered statement bytes"


@pytest.mark.parametrize("ledger", [
    FakeLedger(),
    FakeLedger(ec.SECP256R1(), -7, hashes.SHA256()),
], ids=["ES384", "ES256"])
def test_a_ledger_receipt_verifies_offline(ledger, statement):
    keys = ss.load_key_set(ledger.key_set())
    check = ss.verify_receipt(ledger.receipt(statement), statement=statement, keys=keys)
    assert check.ok, check.detail
    assert check.kid in keys


def test_other_statement_bytes_fail(statement):
    ledger = FakeLedger()
    keys = ss.load_key_set(ledger.key_set())
    check = ss.verify_receipt(ledger.receipt(statement), statement=statement + b"x", keys=keys)
    assert not check.ok and "different statement" in check.detail


def test_a_forged_proof_fails(statement):
    ledger = FakeLedger()
    keys = ss.load_key_set(ledger.key_set())
    msg = cbor2.loads(ledger.receipt(statement))
    protected, unprotected, _, sig = msg.value
    proof = cbor2.loads(unprotected[396][-1][0])
    proof[2][0][0] = not proof[2][0][0]
    forged = cbor2.dumps(cbor2.CBORTag(18, [protected, {396: {-1: [cbor2.dumps(proof)]}},
                                            None, sig]))
    check = ss.verify_receipt(forged, statement=statement, keys=keys)
    assert not check.ok and "signature" in check.detail


def test_another_services_key_fails(statement):
    keys = ss.load_key_set(FakeLedger().key_set())
    check = ss.verify_receipt(FakeLedger().receipt(statement), statement=statement, keys=keys)
    assert not check.ok and "kid" in check.detail


def test_an_rfc9162_receipt_is_not_taken_for_ccf(statement):
    ledger = FakeLedger()
    check = ss.verify_receipt(ledger.receipt(statement, vds=1), statement=statement,
                              keys=ss.load_key_set(ledger.key_set()))
    assert not check.ok and "vds" in check.detail


def test_garbage_is_refused_not_raised(statement):
    for bad in (b"", b"\x00", cbor2.dumps([1, 2, 3]), cbor2.dumps(cbor2.CBORTag(18, [1]))):
        assert not ss.verify_receipt(bad, statement=statement, keys={}).ok


def _chain():
    import datetime

    from cryptography import x509
    from cryptography.x509.oid import NameOID

    def cert(subject_key, issuer_key, cn, issuer_cn, ca):
        now = datetime.datetime.now(datetime.timezone.utc)
        b = (x509.CertificateBuilder()
             .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)]))
             .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer_cn)]))
             .public_key(subject_key.public_key()).serial_number(x509.random_serial_number())
             .not_valid_before(now).not_valid_after(now + datetime.timedelta(days=1))
             .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True))
        return b.sign(issuer_key, hashes.SHA256()).public_bytes(serialization.Encoding.DER)

    root_key, leaf_key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    root = cert(root_key, root_key, "vaara-test-root", "vaara-test-root", True)
    leaf = cert(leaf_key, root_key, "vaara-test-issuer", "vaara-test-root", False)
    return leaf_key, [leaf, root]


def test_signed_statement_shape():
    from vaara.attestation._receipt_cose import _es256_verify

    key, chain = _chain()
    issuer = ss.did_x509(chain[-1], "vaara-test-issuer")
    assert issuer.startswith("did:x509:0:sha256:") and issuer.endswith("::subject:CN:vaara-test-issuer")
    raw = ss.signed_statement(b'{"a":1}', private_key=key, chain=chain, issuer=issuer,
                              subject="vaara:test", issued_at=1)
    msg = cbor2.loads(raw)
    assert msg.tag == 18
    protected_bytes, unprotected, payload, sig = msg.value
    phdr = cbor2.loads(protected_bytes)
    assert phdr[1] == -7 and phdr[3] == "application/json"
    assert phdr[15] == {1: issuer, 2: "vaara:test", 6: 1}
    assert list(phdr[33]) == chain
    assert payload == b'{"a":1}'
    tbs = cbor2.dumps(["Signature1", protected_bytes, b"", payload], canonical=True)
    assert _es256_verify(key.public_key(), sig, tbs)


class _Scrapi(http.server.BaseHTTPRequestHandler):
    ledger = FakeLedger()
    polls = 0
    registered = b""

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body=b"", headers=None):
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        assert self.path.startswith("/entries?api-version=")
        assert self.headers["Content-Type"] == "application/cose"
        type(self).registered = self.rfile.read(int(self.headers["Content-Length"]))
        self._send(303, headers={"Location": "/entries/2.15"})

    def do_GET(self):
        if self.path.startswith("/.well-known/scitt-keys"):
            return self._send(200, self.ledger.key_set())
        assert self.path.startswith("/entries/2.15?api-version=")
        type(self).polls += 1
        if self.polls < 2:
            return self._send(302, headers={"Location": "/entries/2.15", "Retry-After": "0"})
        self._send(200, self.ledger.receipt(self.registered),
                   {"Content-Type": "application/cose"})


def test_register_walks_303_then_302_then_200():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Scrapi)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        stmt = b"statement-bytes"
        reg = ss.register(url, stmt, timeout=10)
        keys = ss.load_key_set(ss.service_keys(url))
    finally:
        server.shutdown()
    assert reg.entry_id == "2.15"
    assert _Scrapi.polls == 2
    assert ss.verify_receipt(reg.receipt, statement=stmt, keys=keys).ok
