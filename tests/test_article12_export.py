"""Tests for the EU AI Act Article 12 one-command regulator export.

See ``docs/design/article12-export-spec.md``.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")  # skip module when the export extra is absent

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from vaara.audit.article12_export import export_article12
from vaara.audit.trail import AuditTrail
from vaara.audit.verify import verify_signed
from vaara.taxonomy.actions import ActionRequest, create_default_registry

_REGISTRY = create_default_registry()
_TX_TRANSFER = _REGISTRY.get("tx.transfer")


def _make_trail(n: int = 3) -> AuditTrail:
    """A trail with a spread of event types so obligations are evidenced."""
    trail = AuditTrail()
    for i in range(n):
        req = ActionRequest(
            agent_id=f"agent-{i}",
            tool_name="send_funds",
            action_type=_TX_TRANSFER,
            parameters={"to": f"0xabc{i}", "amount": 10 * i},
        )
        action_id = trail.record_action_requested(req)
        decision = "deny" if i % 2 else "allow"
        trail.record_decision(
            action_id, f"agent-{i}", "send_funds", decision, "test", 0.5,
        )
        if decision == "allow":
            trail.record_execution(action_id, f"agent-{i}", "send_funds", {"ok": True})
            trail.record_outcome(action_id, f"agent-{i}", "send_funds", "success")
    return trail


def _key_pem(tmp_path: Path) -> Path:
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    p = tmp_path / "signer.pem"
    p.write_bytes(pem)
    return p


def _read_summary(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read("article12_summary.json"))


def test_package_contains_signed_core_and_article12_files(tmp_path):
    out = tmp_path / "art12.zip"
    export_article12(_make_trail(), out, signer_key=_key_pem(tmp_path))
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert {"trail.jsonl", "manifest.json", "trail.sig", "signer_pubkey.pem"} <= names
    assert "article12_report.md" in names
    assert "article12_summary.json" in names
    assert "verify_instructions.txt" in names


def test_signed_core_still_verifies(tmp_path):
    out = tmp_path / "art12.zip"
    export_article12(_make_trail(), out, signer_key=_key_pem(tmp_path))
    # Folding the report in via append mode must not disturb the signed core.
    assert verify_signed(out).ok


def test_summary_binds_the_signed_trail_sha256(tmp_path):
    out = tmp_path / "art12.zip"
    export_article12(_make_trail(), out, signer_key=_key_pem(tmp_path))
    summary = _read_summary(out)
    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    assert summary["record_keeping_summary"]["trail_sha256"] == manifest["trail_sha256"]
    assert summary["integrity"]["trail_sha256"] == manifest["trail_sha256"]


def test_report_counts_match_the_trail(tmp_path):
    trail = _make_trail(4)
    out = tmp_path / "art12.zip"
    export_article12(trail, out, signer_key=_key_pem(tmp_path))
    summary = _read_summary(out)
    with zipfile.ZipFile(out) as zf:
        lines = [ln for ln in zf.read("trail.jsonl").splitlines() if ln.strip()]
    assert summary["record_keeping_summary"]["records_in_trail"] == len(lines)
    assert sum(e["count"] for e in summary["event_inventory"]) == len(lines)


def test_obligation_mapping_is_evidenced(tmp_path):
    out = tmp_path / "art12.zip"
    export_article12(_make_trail(4), out, signer_key=_key_pem(tmp_path))
    by_id = {o["id"]: o for o in _read_summary(out)["obligation_mapping"]}
    assert by_id["art12_1"]["status"] == "evidenced"
    assert by_id["art12_2_c"]["status"] == "evidenced"
    assert by_id["art12_2_a"]["status"] == "evidenced"


def test_regulatory_tags_are_summarised(tmp_path):
    out = tmp_path / "art12.zip"
    export_article12(_make_trail(4), out, signer_key=_key_pem(tmp_path))
    assert _read_summary(out)["regulatory_tagging"]["records_with_tags"] > 0


def test_system_meta_absent_renders_not_provided(tmp_path):
    out = tmp_path / "art12.zip"
    export_article12(_make_trail(), out, signer_key=_key_pem(tmp_path))
    assert _read_summary(out)["cover"]["system_name"] == "not provided"
    with zipfile.ZipFile(out) as zf:
        assert "not provided" in zf.read("article12_report.md").decode("utf-8")


def test_system_meta_present_renders(tmp_path):
    out = tmp_path / "art12.zip"
    meta = {"system_name": "Loan Triage", "provider": "Acme", "risk_classification": "high"}
    export_article12(_make_trail(), out, signer_key=_key_pem(tmp_path), system_meta=meta)
    summary = _read_summary(out)
    assert summary["cover"]["system_name"] == "Loan Triage"
    assert summary["cover"]["risk_classification"] == "high"


def test_period_narrows_scope_not_the_signed_trail(tmp_path):
    out = tmp_path / "art12.zip"
    # A period entirely in the past excludes every (just-now) record.
    export_article12(
        _make_trail(4), out, signer_key=_key_pem(tmp_path), period=(0.0, 1.0),
    )
    summary = _read_summary(out)
    with zipfile.ZipFile(out) as zf:
        lines = [ln for ln in zf.read("trail.jsonl").splitlines() if ln.strip()]
    assert summary["record_keeping_summary"]["records_in_trail"] == len(lines)
    assert summary["record_keeping_summary"]["records_in_scope"] == 0
    assert summary["record_keeping_summary"]["period_is_report_lens_only"] is True
    assert verify_signed(out).ok


def test_tampering_the_trail_breaks_verification(tmp_path):
    out = tmp_path / "art12.zip"
    export_article12(_make_trail(), out, signer_key=_key_pem(tmp_path))
    tampered = tmp_path / "bad.zip"
    with zipfile.ZipFile(out) as zin, zipfile.ZipFile(tampered, "w") as zout:
        for name in zin.namelist():
            data = zin.read(name)
            if name == "trail.jsonl":
                data = data + b'{"injected":true}\n'
            zout.writestr(name, data)
    assert not verify_signed(tampered).ok


def test_html_format(tmp_path):
    out = tmp_path / "art12.zip"
    export_article12(_make_trail(), out, signer_key=_key_pem(tmp_path), fmt="html")
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        html = zf.read("article12_report.html").decode("utf-8")
    assert "article12_report.html" in names
    assert "article12_report.md" not in names
    assert html.startswith("<!doctype html>")


def test_bad_format_rejected(tmp_path):
    out = tmp_path / "art12.zip"
    with pytest.raises(ValueError):
        export_article12(_make_trail(), out, signer_key=_key_pem(tmp_path), fmt="pdf")


# --- Article 19 external time anchor -------------------------------------


def _anchor_over_trail(trail):
    """Build a real RFC 3161 anchor over the trail head via an in-process TSA.

    Mirrors tests/test_timeanchor.py: a self-signed EC TSA issues a genuine
    TimeStampToken over the chain head, so the full anchor path is exercised
    with no network. Skips when the 'timeanchor' extra is absent.
    """
    pytest.importorskip("asn1crypto")
    import datetime
    import hashlib

    from asn1crypto import algos, cms, tsp
    from asn1crypto import x509 as asn1_x509
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    from vaara.audit.timeanchor import RFC3161TimeAnchorClient

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Vaara Test TSA")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime(2020, 1, 1))
        .not_valid_after(datetime.datetime(2035, 1, 1))
        .sign(key, hashes.SHA256())
    )
    cert_asn1 = asn1_x509.Certificate.load(
        cert.public_bytes(serialization.Encoding.DER)
    )

    def _issue(digest):
        tst = tsp.TSTInfo({
            "version": 1, "policy": "1.3.6.1.4.1.99999.1",
            "message_imprint": tsp.MessageImprint({
                "hash_algorithm": algos.DigestAlgorithm({"algorithm": "sha256"}),
                "hashed_message": digest,
            }),
            "serial_number": 1,
            "gen_time": datetime.datetime(
                2026, 6, 7, 12, 0, 0, tzinfo=datetime.timezone.utc),
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
            "signature": key.sign(attrs.dump(), ec.ECDSA(hashes.SHA256())),
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
        return cms.ContentInfo(
            {"content_type": "signed_data", "content": signed}).dump()

    def transport(url, der_request, timeout):
        req = tsp.TimeStampReq.load(der_request)
        digest = req["message_imprint"]["hashed_message"].native
        return tsp.TimeStampResp({
            "status": tsp.PKIStatusInfo({"status": "granted"}),
            "time_stamp_token": cms.ContentInfo.load(_issue(digest)),
        }).dump()

    client = RFC3161TimeAnchorClient("https://tsa.test/tsr", transport=transport)
    records = trail._records
    return client.anchor(len(records) - 1, records[-1].record_hash)


def test_anchor_folded_into_package(tmp_path):
    trail = _make_trail()
    anchor = _anchor_over_trail(trail)
    out = tmp_path / "art12.zip"
    export_article12(trail, out, signer_key=_key_pem(tmp_path), time_anchor=anchor)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert "time_anchor.json" in names
    ta = _read_summary(out)["time_anchor"]
    assert ta["anchored"] is True
    assert ta["chain_head_hash"] == anchor.chain_head_hash
    # Folding the anchor in via append mode must not disturb the signed core.
    assert verify_signed(out).ok


def test_anchor_absent_marks_not_anchored(tmp_path):
    out = tmp_path / "art12.zip"
    export_article12(_make_trail(), out, signer_key=_key_pem(tmp_path))
    assert _read_summary(out)["time_anchor"]["anchored"] is False
    with zipfile.ZipFile(out) as zf:
        assert "time_anchor.json" not in set(zf.namelist())


def test_anchor_must_bind_the_trail_head(tmp_path):
    from vaara.audit.timeanchor import TimeAnchorError

    # An anchor over a different trail must not be accepted for this one.
    foreign = _anchor_over_trail(_make_trail(2))
    out = tmp_path / "art12.zip"
    with pytest.raises((ValueError, TimeAnchorError)):
        export_article12(
            _make_trail(4), out,
            signer_key=_key_pem(tmp_path), time_anchor=foreign,
        )


def test_packaged_anchor_verifies_offline(tmp_path):
    from vaara.audit.timeanchor import TimeAnchor, verify_anchor_over_records
    from vaara.audit.trail import AuditRecord

    trail = _make_trail()
    anchor = _anchor_over_trail(trail)
    out = tmp_path / "art12.zip"
    export_article12(trail, out, signer_key=_key_pem(tmp_path), time_anchor=anchor)
    # Re-verify from the packaged bytes alone, the way a regulator would.
    with zipfile.ZipFile(out) as zf:
        loaded = TimeAnchor.from_dict(json.loads(zf.read("time_anchor.json")))
        record_hashes = [
            AuditRecord.from_dict(json.loads(ln)).record_hash
            for ln in zf.read("trail.jsonl").splitlines() if ln.strip()
        ]
    attested = verify_anchor_over_records(loaded, record_hashes)
    assert attested.year == 2026
