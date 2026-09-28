#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Register a Vaara decision receipt with a SCITT Transparency Service.

Records one decision on a fresh trail, takes the signed receipt the engine
wrote for it, wraps it in a SCITT Signed Statement under a throwaway did:x509
issuer, registers it, and saves what an offline verifier needs:

  statement.cose   the signed statement, exactly as registered
  receipt.cose     the service's COSE receipt
  scitt-keys.cbor  the service's /.well-known/scitt-keys, fetched once

Then checks the receipt once here. The CI job stops the service and checks
it again with `vaara receipt verify-transparency`, so the second check runs
with nothing to call.

    python scripts/scitt_roundtrip.py --url https://localhost:8000 --insecure --out out/
"""
from __future__ import annotations

import argparse
import datetime
import sys
import tempfile
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from vaara.audit import scitt_service as ss
from vaara.audit.sqlite_backend import SQLiteAuditBackend

ISSUER_CN = "vaara-scitt-roundtrip"


def _cert(subject_key, issuer_key, cn: str, issuer_cn: str, ca: bool) -> bytes:
    now = datetime.datetime.now(datetime.timezone.utc)
    return (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)]))
            .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, issuer_cn)]))
            .public_key(subject_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + datetime.timedelta(days=1))
            .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
            # The ledger's did:x509 check validates the chain as X.509 does and
            # refuses a CA certificate without keyUsage.
            .add_extension(x509.KeyUsage(
                digital_signature=True, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=ca,
                crl_sign=ca, encipher_only=False, decipher_only=False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(subject_key.public_key()),
                           critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(
                issuer_key.public_key()), critical=False)
            .sign(issuer_key, hashes.SHA256())
            .public_bytes(serialization.Encoding.DER))


def _decision_receipt() -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "audit.db"
        trail = SQLiteAuditBackend(db).load_trail()
        trail.record_decision(action_id="scitt-roundtrip", agent_id="ci", tool_name="read_file",
                              decision="allow", reason="scitt round-trip", risk_score=0.1)
        files = sorted((Path(tmp) / "receipts").rglob("*.json"))
        if not files:
            raise SystemExit("the engine wrote no decision receipt")
        return files[0].read_bytes()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", required=True, help="Transparency Service base URL")
    ap.add_argument("--out", required=True, help="Directory for the three files")
    ap.add_argument("--cafile", default=None, help="CA bundle for the service's TLS certificate")
    ap.add_argument("--insecure", action="store_true", help="Skip TLS verification (local dev ledger)")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args(argv)

    root_key = ec.generate_private_key(ec.SECP256R1())
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    root = _cert(root_key, root_key, "vaara-scitt-roundtrip-root", "vaara-scitt-roundtrip-root", True)
    leaf = _cert(leaf_key, root_key, ISSUER_CN, "vaara-scitt-roundtrip-root", False)
    issuer = ss.did_x509(root, ISSUER_CN)

    statement = ss.signed_statement(
        _decision_receipt(), private_key=leaf_key, chain=[leaf, root],
        issuer=issuer, subject="vaara.receipt/v1", content_type="application/json")
    reg = ss.register(args.url, statement, cafile=args.cafile, insecure=args.insecure,
                      timeout=args.timeout)
    key_set = ss.service_keys(args.url, cafile=args.cafile, insecure=args.insecure)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "statement.cose").write_bytes(statement)
    (out / "receipt.cose").write_bytes(reg.receipt)
    (out / "scitt-keys.cbor").write_bytes(key_set)

    check = ss.verify_receipt(reg.receipt, statement=statement, keys=ss.load_key_set(key_set))
    print(f"issuer   {issuer}")
    print(f"entry    {reg.entry_id}")
    print(f"receipt  {len(reg.receipt)} bytes, kid {check.kid!r}")
    print(f"verify   {'OK' if check.ok else 'FAIL'}: {check.detail}")
    return 0 if check.ok else 1


if __name__ == "__main__":
    sys.exit(main())
