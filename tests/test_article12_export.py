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
