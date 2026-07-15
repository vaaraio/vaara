#!/usr/bin/env python3
"""Independent checker for the qualified_time_v0 vectors.

A second implementation of the existence-in-time verifier obligation, written
from the schema and RFC 3161 alone with only the standard library plus
``rfc8785``, ``asn1crypto``, and ``cryptography``. It imports no Vaara code.
For each committed record it reproduces the verdict:

1. **Digest.** RFC 8785 JCS over the record with ``existenceProof`` removed,
   SHA-256, as ``sha256:<hex>``. The proof's ``recordDigest`` must equal it, so
   a proof stapled to mutated bytes fails.
2. **Imprint.** Parse the RFC 3161 token, require its message imprint to equal
   that digest, and require the SignedAttributes message-digest to equal the
   hash of the TSTInfo content.
3. **Signature.** Verify the TSA's signature over the SignedAttributes under
   the certificate the token carries.
4. **Qualified.** The attested time is qualified only if the signer certificate
   is directly issued by the CA pinned in ``trusted_ca.pem``; otherwise the
   token is valid but self-asserted.

Verdicts (``ok`` and ``qualified``) are compared against ``expected.json``.
Exit 0 means every case matched. Run:
``python tests/vectors/qualified_time_v0/_check_independent.py``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

# A conformant environment for this suite carries the ``timeanchor`` extra.
# When an optional dependency is absent, exit with the standard skip code (77)
# so the aggregate runner reports SKIP with a reason rather than a false
# failure; the suite still runs and passes where the extra is installed.
_SKIP_EXIT_CODE = 77
try:
    import rfc8785
    from asn1crypto import cms, core, tsp
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509 import (
        load_der_x509_certificate,
        load_pem_x509_certificate,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - base install
    print(
        "SKIP: qualified_time_v0 needs an optional dependency not installed: "
        f"{exc.name} (pip install 'vaara[timeanchor]')",
        file=sys.stderr,
    )
    raise SystemExit(_SKIP_EXIT_CODE) from None

HERE = Path(__file__).resolve().parent
COMPARE = ("ok", "qualified")


def _record_digest(record: dict) -> str:
    covered = {k: v for k, v in record.items() if k != "existenceProof"}
    return "sha256:" + hashlib.sha256(rfc8785.dumps(covered)).hexdigest()


def _signer_cert_after_verify(token_der: bytes, expected_digest: bytes):
    """Verify the token internally and return its signer certificate.

    Raises ``ValueError`` on any inconsistency: not SignedData, wrong imprint,
    a SignedAttributes digest that does not cover the TSTInfo, or a signature
    that does not verify.
    """
    content_info = cms.ContentInfo.load(token_der)
    if content_info["content_type"].native != "signed_data":
        raise ValueError("token is not CMS SignedData")
    signed_data = content_info["content"]

    encap = signed_data["encap_content_info"]
    if encap["content_type"].native != "tst_info":
        raise ValueError("token does not encapsulate a TSTInfo")
    tst_bytes = encap["content"].cast(core.OctetString).native
    tst_info = tsp.TSTInfo.load(tst_bytes)

    imprint = tst_info["message_imprint"]
    if imprint["hash_algorithm"]["algorithm"].native != "sha256":
        raise ValueError("token imprint is not sha256")
    if imprint["hashed_message"].native != expected_digest:
        raise ValueError("token imprint does not equal the record digest")

    signer_info = signed_data["signer_infos"][0]
    md_attr = None
    for attr in signer_info["signed_attrs"]:
        if attr["type"].native == "message_digest":
            md_attr = attr["values"][0].native
            break
    if md_attr != hashlib.sha256(tst_bytes).digest():
        raise ValueError("SignedAttributes digest does not cover the TSTInfo")

    signer_cert = _match_signer_cert(signed_data)
    signed_attrs_der = signer_info["signed_attrs"].untag().dump()
    signature = signer_info["signature"].native
    # The test TSAs sign with ECDSA/SHA-256; a real QTSA may use RSA, but these
    # vectors exercise the EC path.
    signer_cert.public_key().verify(
        signature, signed_attrs_der, ec.ECDSA(hashes.SHA256())
    )
    return signer_cert


def _match_signer_cert(signed_data):
    certs = [c for c in signed_data["certificates"] if c.name == "certificate"]
    if not certs:
        raise ValueError("token carries no signer certificate")
    sid = signed_data["signer_infos"][0]["sid"]
    if sid.name == "issuer_and_serial_number":
        want_serial = sid.chosen["serial_number"].native
        want_issuer = sid.chosen["issuer"]
        for choice in certs:
            tbs = choice.chosen["tbs_certificate"]
            if tbs["serial_number"].native == want_serial and tbs["issuer"] == want_issuer:
                return load_der_x509_certificate(choice.chosen.dump())
        raise ValueError("no embedded certificate matches the signer id")
    if len(certs) == 1:
        return load_der_x509_certificate(certs[0].chosen.dump())
    raise ValueError("cannot disambiguate the signer certificate")


def evaluate(record: dict, trusted_ca) -> dict:
    ep = record.get("existenceProof")
    if not isinstance(ep, dict):
        return {"ok": False, "qualified": False}
    recomputed = _record_digest(record)
    if ep.get("recordDigest") != recomputed:
        return {"ok": False, "qualified": False}
    token_b64 = ep.get("token")
    if not isinstance(token_b64, str):
        return {"ok": False, "qualified": False}
    try:
        token_der = base64.b64decode(token_b64)
    except (ValueError, TypeError):
        return {"ok": False, "qualified": False}

    expected_digest = bytes.fromhex(recomputed.split(":", 1)[1])
    try:
        signer_cert = _signer_cert_after_verify(token_der, expected_digest)
    except Exception:  # noqa: BLE001 - any inconsistency means not evidence
        return {"ok": False, "qualified": False}

    qualified = False
    try:
        signer_cert.verify_directly_issued_by(trusted_ca)
        qualified = True
    except Exception:  # noqa: BLE001 - not issued by the pinned CA
        qualified = False
    return {"ok": True, "qualified": qualified}


def main() -> int:
    cases = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))["cases"]
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    trusted_ca = load_pem_x509_certificate((HERE / "trusted_ca.pem").read_bytes())

    failures = []
    for case in cases:
        name = case["name"]
        got = {k: evaluate(case["record"], trusted_ca)[k] for k in COMPARE}
        want = {k: expected[name][k] for k in COMPARE}
        if got != want:
            failures.append(f"{name}: expected {want}, got {got}")
        else:
            print(f"{name}: OK {got}")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures))
        return 1
    print(f"\nall {len(cases)} qualified_time_v0 cases matched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
