"""SEP-2828 record conformance and the `vaara verify-record` command.

Conformance is keyless: it checks the wire schema and the record's
self-proving binding (projectionDigest over the projection bytes), with
no signing key and no attestation. So unlike the bundle/receipt suites,
this module does NOT skip when the attestation extra is absent: the
whole point is that conformance runs in the base install. Only the
back-link CLI test (which recomputes an attestation digest) needs the
extra, and it skips on its own.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.attestation._receipt_conformance import (
    SCHEMA_NAME,
    check_record_conformance,
)
from vaara.cli import main

VECTORS = Path(__file__).resolve().parent / "vectors" / "record_conformance_v0"
RECORDS = VECTORS / "records"


def _expected() -> dict:
    return json.loads((VECTORS / "expected.json").read_text())


def _cases():
    return sorted(_expected().keys())


def _record(name: str) -> dict:
    return json.loads((RECORDS / f"{name}.json").read_text())


# ── Vectors ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", _cases())
def test_module_reproduces_vector_verdict(name):
    want = _expected()[name]
    report = check_record_conformance(_record(name))
    got = {
        "conforms": report.conforms,
        "requiredFailed": sorted(report.required_failed),
        "advisories": sorted(report.advisories),
    }
    assert got == want


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_at_least_eight_cases_present():
    assert len(_cases()) >= 8


def test_public_reexport_is_wired():
    # check_record_conformance must be reachable from the public surface.
    from vaara.attestation.receipt import check_record_conformance as public
    assert public is check_record_conformance


# ── Unit behaviour ────────────────────────────────────────────────────────────


def test_self_consistency_is_the_keyless_catch():
    # Flipping the projection digest is caught with no key and no attestation.
    doc = _record("conforming_executed_projection")
    doc["outcomeDerived"]["resultCommitment"]["projectionDigest"] = "sha256:" + "0" * 64
    report = check_record_conformance(doc)
    assert not report.conforms
    assert "result_commitment_self_consistent" in report.required_failed


def test_advisory_does_not_gate_conformance():
    report = check_record_conformance(_record("advisory_refused_with_commitment"))
    assert report.conforms
    assert "refused_has_no_result" in report.advisories
    assert report.required_failed == ()


def test_non_object_record_does_not_raise():
    report = check_record_conformance(["not", "an", "object"])
    assert not report.conforms
    assert report.required_failed == ("top_level_object",)


def test_report_to_dict_shape():
    d = check_record_conformance(_record("conforming_refused_no_commitment")).to_dict()
    assert d["schema"] == SCHEMA_NAME
    assert d["conforms"] is True
    assert d["alg"] == "RS256"
    assert d["status"] == "refused"
    assert all({"id", "ok", "severity", "detail"} <= set(c) for c in d["checks"])


# ── CLI (conformance path needs no extra) ─────────────────────────────────────


def test_cli_conforming_exit_0(tmp_path, capsys):
    target = tmp_path / "record.json"
    target.write_text((RECORDS / "conforming_executed_projection.json").read_text())
    rc = main(["verify-record", str(target)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "CONFORMS" in out


def test_cli_non_conforming_exit_1(tmp_path, capsys):
    target = tmp_path / "record.json"
    target.write_text((RECORDS / "neg_digest_mismatch.json").read_text())
    rc = main(["verify-record", str(target)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NON-CONFORMING" in out
    assert "result_commitment_self_consistent" in out


def test_cli_json_output(tmp_path, capsys):
    target = tmp_path / "record.json"
    target.write_text((RECORDS / "conforming_executed_projection.json").read_text())
    rc = main(["verify-record", str(target), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["conformance"]["conforms"] is True
    assert payload["conformance"]["schema"] == SCHEMA_NAME
    assert "backLink" not in payload


def test_cli_missing_file(tmp_path, capsys):
    rc = main(["verify-record", str(tmp_path / "nope.json")])
    assert rc == 2
    assert "not a file" in capsys.readouterr().err


def test_cli_bad_json(tmp_path, capsys):
    target = tmp_path / "record.json"
    target.write_text("{ not json")
    rc = main(["verify-record", str(target)])
    assert rc == 1
    assert "cannot read record JSON" in capsys.readouterr().err


# ── CLI back-link (needs the attestation extra) ───────────────────────────────


def test_cli_back_link_pass(tmp_path, capsys):
    pytest.importorskip("rfc8785")
    pytest.importorskip("cryptography")
    src = Path(__file__).resolve().parent / "vectors" / "execution_receipt_v0"
    case = src / "normative" / "es256_executed_projection"
    rc = main([
        "verify-record", str(case / "receipt.json"),
        "--attestation", str(case / "attestation.json"),
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "back-link: pass" in out


def test_cli_back_link_mismatch_gates(tmp_path, capsys):
    pytest.importorskip("rfc8785")
    pytest.importorskip("cryptography")
    src = Path(__file__).resolve().parent / "vectors" / "execution_receipt_v0"
    case = src / "normative" / "es256_executed_projection"
    # Conforming record, but paired with a tampered attestation whose wire bytes
    # (and so whose digest) no longer match the one the record pins: the
    # back-link fails and the command exits 1 even though conformance passed.
    att = json.loads((case / "attestation.json").read_text())
    att["issuerAsserted"]["nonce"] = "tampered-nonce-deadbeef"
    tampered = tmp_path / "attestation.json"
    tampered.write_text(json.dumps(att))
    rc = main([
        "verify-record", str(case / "receipt.json"),
        "--attestation", str(tampered),
    ])
    out = capsys.readouterr().out
    assert rc == 1
    assert "back-link: FAIL" in out
