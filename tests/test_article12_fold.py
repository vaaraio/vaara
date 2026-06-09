"""Tests for the Article 12 regulator-package fold.

``export_article12`` attaches verified SEP-2828 evidence as sidecars under
``evidence/``: cross-org handoff packages (Article 26(6)) and confidential-VM
enforcement bindings. These tests prove the fold lands the right bytes, carries
the honest roll-up, fails closed on a bad attachment, leaves the signed core and
the Article 19 anchor intact, and that every folded verdict reproduces both
against the standalone set checkers and Vaara-free from the zip bytes.

See ``docs/design/article12-export-spec.md`` and the ``article12_fold_v0``
vectors.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("rfc8785")
pytest.importorskip("cryptography")

from vaara.attestation._enforcement import verify_enforcement
from vaara.attestation._handoff import verify_handoff
from vaara.attestation.receipt import check_enforcement_set, check_handoff_set
from vaara.audit.article12_export import export_article12
from vaara.audit.verify import verify_signed

VEC = Path(__file__).resolve().parent / "vectors" / "article12_fold_v0"
EXPECTED = json.loads((VEC / "expected.json").read_text())


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load(VEC / "_generate.py", "_a12_fold_gen")
EXPORT_TEST = _load(
    Path(__file__).parent / "test_article12_export.py", "_a12_export_test")

HCASES = {c["name"]: c
          for c in json.loads((GEN.HANDOFF / "cases.json").read_text())["cases"]}
ECASES = {c["name"]: c
          for c in json.loads((GEN.ENFORCEMENT / "cases.json").read_text())["cases"]}


def _build(scenario: str, tmp_path: Path) -> Path:
    return GEN._build(GEN.SCENARIOS[scenario], HCASES, ECASES, tmp_path)


def _handoff_tuples(cases):
    out = []
    for name in cases:
        case = HCASES[name]
        pkg = case["package"]
        anchored = pkg.get("evidence", {}).get("anchor") is not None
        out.append((name, pkg, case.get("anchoredTime") if anchored else None))
    return out


def _enforcement_tuples(cases):
    out = []
    for name in cases:
        case = ECASES[name]
        out.append((name, case["record"], base64.b64decode(case["report_b64"]),
                    GEN._jwk_to_pem(case["vcek_jwk"])))
    return out


@pytest.mark.parametrize("scenario", ["full", "pinned_handoff", "enforcement_only"])
def test_membership_and_rollup(scenario, tmp_path):
    out = _build(scenario, tmp_path)
    with zipfile.ZipFile(out) as zf:
        members = sorted(m for m in zf.namelist() if m.startswith("evidence/"))
        summary = json.loads(zf.read("evidence/attestations_summary.json"))
    exp = EXPECTED[scenario]
    assert members == exp["evidence_members"]
    if "handoff" in exp:
        rep = summary["handoff"]["report"]
        assert {k: rep[k] for k in GEN.HANDOFF_KEYS} == exp["handoff"]
    if "enforcement" in exp:
        rep = summary["enforcement"]["report"]
        assert {k: rep[k] for k in GEN.ENFORCEMENT_KEYS} == exp["enforcement"]


def test_signed_core_still_verifies(tmp_path):
    # Folding evidence in via append mode must not disturb the signed trail.
    assert verify_signed(_build("full", tmp_path)).ok


def test_report_section_and_honesty_notes(tmp_path):
    out = _build("full", tmp_path)
    with zipfile.ZipFile(out) as zf:
        md = zf.read("article12_report.md").decode("utf-8")
    assert "## Cross-org handoff and enforcement evidence" in md
    assert "Article 26(6)" in md
    assert "Where enforcement ran (confidential VM)" in md
    assert "not attested in this release" in md
    assert "caller_supplied_unverified" in md
    assert "only un-forgeable component" in md
    # The em-dash is the one house no-go in shipped prose.
    assert "—" not in md


def test_rollup_matches_standalone_set_checkers(tmp_path):
    out = _build("full", tmp_path)
    with zipfile.ZipFile(out) as zf:
        summary = json.loads(zf.read("evidence/attestations_summary.json"))
    h = check_handoff_set(
        _handoff_tuples(["clean_no_anchor", "corroborated"]),
        trusted_did_document=None, strict=False).to_dict()
    assert {k: summary["handoff"]["report"][k] for k in GEN.HANDOFF_KEYS} == \
        {k: h[k] for k in GEN.HANDOFF_KEYS}
    e = check_enforcement_set(
        _enforcement_tuples(["clean_bound", "pinned_measurement_match"]),
        expected_measurement=GEN.SCENARIOS["full"]["enforcement"]["expected_measurement"],
        strict=False).to_dict()
    assert {k: summary["enforcement"]["report"][k] for k in GEN.ENFORCEMENT_KEYS} == \
        {k: e[k] for k in GEN.ENFORCEMENT_KEYS}


def test_independent_checker_reproduces_from_zip(tmp_path):
    out = _build("full", tmp_path)
    proc = subprocess.run(
        [sys.executable, str(VEC / "_check_independent.py"), str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_folded_handoff_verifies_from_zip_bytes(tmp_path):
    # The v0.62 trap: re-verify each attachment FROM the folded bytes.
    out = _build("full", tmp_path)
    with zipfile.ZipFile(out) as zf:
        summary = json.loads(zf.read("evidence/attestations_summary.json"))
        for member in [n for n in zf.namelist()
                       if n.startswith("evidence/handoff/") and n.endswith(".json")]:
            base = member[len("evidence/handoff/"):-len(".json")]
            doc = json.loads(zf.read(member))
            attested = summary["handoff"]["anchoredTimes"].get(base)
            assert verify_handoff(doc, anchor_attested_time=attested).ok


def test_folded_enforcement_binds_from_zip_bytes(tmp_path):
    out = _build("full", tmp_path)
    with zipfile.ZipFile(out) as zf:
        summary = json.loads(zf.read("evidence/attestations_summary.json"))
        meas = summary["enforcement"]["expectedMeasurement"]
        for member in [n for n in zf.namelist() if n.endswith(".record.json")]:
            base = member[len("evidence/enforcement/"):-len(".record.json")]
            record = json.loads(zf.read(member))
            report_bytes = zf.read(f"evidence/enforcement/{base}.report.bin")
            vcek_pem = zf.read(f"evidence/enforcement/{base}.vcek.pem")
            v = verify_enforcement(
                record, report_bytes, vcek_pem, expected_measurement=meas)
            assert v.ok and v.bound


def test_fail_closed_bad_handoff(tmp_path):
    spec = GEN.FAIL_CLOSED["bad_handoff"]["handoff"]
    out = tmp_path / "bad.zip"
    with pytest.raises(ValueError, match="signed_after_retirement"):
        export_article12(
            GEN._trail(), out, signer_key=GEN._key(tmp_path),
            handoffs=_handoff_tuples(spec["cases"]))
    assert not out.exists()


def test_fail_closed_bad_enforcement(tmp_path):
    spec = GEN.FAIL_CLOSED["bad_enforcement"]["enforcement"]
    out = tmp_path / "bad.zip"
    with pytest.raises(ValueError, match="bad_signature"):
        export_article12(
            GEN._trail(), out, signer_key=GEN._key(tmp_path),
            enforcements=_enforcement_tuples(spec["cases"]),
            expected_measurement=spec["expected_measurement"])
    assert not out.exists()


def test_anchor_and_fold_coexist(tmp_path):
    # The Article 19 time anchor and the folded SEP-2828 evidence live together.
    pytest.importorskip("asn1crypto")
    trail = EXPORT_TEST._make_trail()
    anchor = EXPORT_TEST._anchor_over_trail(trail)
    out = tmp_path / "both.zip"
    export_article12(
        trail, out, signer_key=GEN._key(tmp_path), time_anchor=anchor,
        handoffs=_handoff_tuples(["clean_no_anchor", "corroborated"]))
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        md = zf.read("article12_report.md").decode("utf-8")
    assert "time_anchor.json" in names
    assert any(n.startswith("evidence/handoff/") for n in names)
    assert "External time anchor (Article 19)" in md
    assert "## Cross-org handoff and enforcement evidence" in md
    assert verify_signed(out).ok


def test_no_attachments_states_none_folded(tmp_path):
    out = tmp_path / "plain.zip"
    export_article12(GEN._trail(), out, signer_key=GEN._key(tmp_path))
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        md = zf.read("article12_report.md").decode("utf-8")
    assert not any(n.startswith("evidence/") for n in names)
    assert "No cross-org handoff or confidential-VM enforcement evidence is folded" \
        in md
