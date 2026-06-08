"""SEP-2828 set-level conformance and the `vaara verify-records` command.

The receiving side of the evidence: an auditor points the tool at a
directory of records, possibly from more than one emitter, and gets the
roll-up that `verify-record` cannot give on its own. Like the per-record
check, this is keyless, so the suite does NOT skip when the attestation
extra is absent: the workbench runs in the base install.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vaara.attestation._record_set_conformance import (
    SET_SCHEMA_NAME,
    check_record_set,
)
from vaara.cli import main

VECTORS = Path(__file__).resolve().parent / "vectors" / "record_set_v0"
SETS = VECTORS / "sets"


def _expected() -> dict:
    return json.loads((VECTORS / "expected.json").read_text())


def _cases():
    return sorted(_expected().keys())


def _load_set(name: str):
    files = sorted((SETS / name).glob("*.json"))
    return [(p.name, json.loads(p.read_text())) for p in files]


# ── Vectors ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", _cases())
def test_module_reproduces_vector_verdict(name):
    want = _expected()[name]
    report = check_record_set(_load_set(name))
    got = {
        "conforms": report.conforms,
        "total": report.total,
        "conforming": report.conforming,
        "statusCounts": report.status_counts,
        "findings": [
            {"id": f.id, "severity": f.severity, "records": list(f.records)}
            for f in report.findings
        ],
    }
    assert got == want


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_at_least_four_cases_present():
    assert len(_cases()) >= 4


def test_public_reexport_is_wired():
    from vaara.attestation.receipt import check_record_set as public
    assert public is check_record_set


# ── Unit behaviour ────────────────────────────────────────────────────────────


def test_duplicate_call_is_required_and_gates():
    report = check_record_set(_load_set("duplicate_call"))
    assert not report.conforms
    ids = [f.id for f in report.required_findings]
    assert "duplicate_call" in ids


def test_executed_gap_is_advisory_and_does_not_gate():
    report = check_record_set(_load_set("executed_gap"))
    assert report.conforms
    advisory = [f for f in report.findings if f.severity == "advisory"]
    assert [f.id for f in advisory] == ["executed_without_result_commitment"]
    assert report.required_findings == ()


def test_malformed_record_gates_but_does_not_raise():
    report = check_record_set(_load_set("mixed_nonconforming"))
    assert not report.conforms
    assert report.conforming == 1
    # Cross-record reasoning ran only over the one conforming record, so no
    # set-level finding fired off the malformed one.
    assert report.findings == ()


def test_set_findings_run_only_over_conforming_records():
    # A malformed duplicate of a conforming record must NOT raise a
    # duplicate_call finding: the malformed one is excluded from linkage.
    doc = _load_set("clean")[0][1]
    broken = dict(doc)
    broken["version"] = 99  # malformed: fails per-record conformance
    report = check_record_set([("good.json", doc), ("broken.json", broken)])
    assert not report.conforms  # broken record gates
    assert report.findings == ()  # but no spurious duplicate_call


def test_empty_set_conforms_vacuously():
    report = check_record_set([])
    assert report.conforms
    assert report.total == 0
    assert report.findings == ()


def test_report_to_dict_shape():
    d = check_record_set(_load_set("clean")).to_dict()
    assert d["schema"] == SET_SCHEMA_NAME
    assert d["conforms"] is True
    assert d["total"] == 2
    assert d["statusCounts"] == {"executed": 1, "refused": 1}
    assert d["findings"] == []
    assert all({"name", "conforms"} <= set(e) for e in d["entries"])


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_cli_clean_set_exit_0(capsys):
    rc = main(["verify-records", str(SETS / "clean")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "CONFORMS" in out
    assert "executed: 1" in out


def test_cli_duplicate_set_exit_1(capsys):
    rc = main(["verify-records", str(SETS / "duplicate_call")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NON-CONFORMING" in out
    assert "duplicate_call" in out


def test_cli_json_output(capsys):
    rc = main(["verify-records", str(SETS / "executed_gap"), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["schema"] == SET_SCHEMA_NAME
    assert payload["unreadable"] == []
    assert payload["findings"][0]["id"] == "executed_without_result_commitment"


def test_cli_unreadable_file_gates(tmp_path, capsys):
    (tmp_path / "good.json").write_text(
        (SETS / "clean" / "r1.json").read_text()
    )
    (tmp_path / "bad.json").write_text("{ not json")
    rc = main(["verify-records", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "unreadable" in out


def test_cli_not_a_directory(tmp_path, capsys):
    rc = main(["verify-records", str(tmp_path / "nope")])
    assert rc == 2
    assert "not a directory" in capsys.readouterr().err


def test_cli_empty_directory(tmp_path, capsys):
    rc = main(["verify-records", str(tmp_path)])
    assert rc == 2
    assert "no files matched" in capsys.readouterr().err
