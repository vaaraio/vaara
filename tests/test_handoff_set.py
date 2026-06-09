"""Batch handoff verification and the `vaara verify-handoffs` command.

The batch twin of `verify-handoff`: a regulator points the tool at a directory
of cross-org packages and gets the roll-up a single-file check cannot give.
Covers the `handoff_set_v0` vectors through both Vaara and the standalone
Vaara-free checker, the module behaviour (the verifiable / corroborated tiers,
the producer-pin coverage gap, gating and strict mode), and the CLI end to end.
Runs the record-lens crypto, so the suite skips when the attestation extra is
absent.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from vaara.attestation.receipt import (  # noqa: E402
    HANDOFF_SET_SCHEMA_NAME,
    check_handoff_set,
)
from vaara.cli import main  # noqa: E402

VECTORS = Path(__file__).resolve().parent / "vectors" / "handoff_set_v0"
SIBLING = Path(__file__).resolve().parent / "vectors" / "cross_org_handoff_v0"

SUMMARY_KEYS = (
    "ok", "strict", "total", "loaded", "passed", "verifiable",
    "corroborated", "pinned", "pinningGap",
)


def _cases() -> dict:
    return {c["name"]: c
            for c in json.loads((SIBLING / "cases.json").read_text())["cases"]}


def _sets() -> dict:
    return json.loads((VECTORS / "sets.json").read_text())["sets"]


def _expected() -> dict:
    return json.loads((VECTORS / "expected.json").read_text())


def _packages_for(spec: dict, cases: dict):
    out = []
    for case_name in spec["cases"]:
        case = cases[case_name]
        package = case["package"]
        anchor_present = package.get("evidence", {}).get("anchor") is not None
        anchor_time = case.get("anchoredTime") if anchor_present else None
        out.append((case_name, package, anchor_time))
    return out


def _trusted_for(spec: dict, cases: dict):
    if spec["trusted_from_case"] is None:
        return None
    return cases[spec["trusted_from_case"]]["trustedDidDocument"]


# ── Vectors ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(_expected().keys()))
def test_module_reproduces_vector_rollup(name):
    cases, sets, want = _cases(), _sets(), _expected()[name]
    spec = sets[name]
    report = check_handoff_set(
        _packages_for(spec, cases),
        trusted_did_document=_trusted_for(spec, cases),
        strict=spec["strict"],
    )
    got = {k: report.to_dict()[k] for k in SUMMARY_KEYS}
    assert got == want


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_public_reexport_is_wired():
    from vaara.attestation.receipt import check_handoff_set as public
    assert public is check_handoff_set


# ── Unit behaviour ────────────────────────────────────────────────────────────


def test_verifiable_set_passes_and_flags_the_pinning_gap():
    cases = _cases()
    report = check_handoff_set(_packages_for(_sets()["all_verifiable"], cases))
    assert report.ok
    assert report.verifiable == 2
    assert report.corroborated == 1
    assert report.pinned == 0
    assert report.pinning_gap is True  # advisory, did not gate


def test_a_failing_package_gates_the_set():
    cases = _cases()
    report = check_handoff_set(_packages_for(_sets()["with_failure"], cases))
    assert not report.ok
    assert report.passed == 1
    failing = [e for e in report.entries if not e.ok]
    assert len(failing) == 1


def test_strict_with_pin_passes_and_lifts_the_gap():
    cases, sets = _cases(), _sets()
    spec = sets["strict_pinned"]
    report = check_handoff_set(
        _packages_for(spec, cases),
        trusted_did_document=_trusted_for(spec, cases), strict=True)
    assert report.ok
    assert report.pinned == 1
    assert report.pinning_gap is False


def test_strict_without_pin_fails():
    cases = _cases()
    report = check_handoff_set(_packages_for(_sets()["strict_unmet"], cases),
                               strict=True)
    assert not report.ok  # corroborated but self-asserted identity
    assert report.corroborated == 1


def test_malformed_document_is_a_failing_entry_not_a_raise():
    cases = _cases()
    good = _packages_for(_sets()["all_verifiable"], cases)[0]
    report = check_handoff_set([good, ("broken", {"not": "a package"}, None)])
    assert not report.ok
    broken = next(e for e in report.entries if e.name == "broken")
    assert broken.loaded is False
    assert broken.ok is False
    assert broken.error is not None
    assert report.loaded == 1


def test_empty_set_is_ok_without_a_pinning_gap():
    report = check_handoff_set([])
    assert report.ok and report.total == 0
    assert report.pinning_gap is False


def test_report_to_dict_shape():
    cases = _cases()
    d = check_handoff_set(_packages_for(_sets()["all_verifiable"], cases)).to_dict()
    assert d["schema"] == HANDOFF_SET_SCHEMA_NAME
    assert d["ok"] is True
    assert all({"name", "loaded", "ok", "verifiable"} <= set(e) for e in d["entries"])


# ── CLI ───────────────────────────────────────────────────────────────────────


def _materialize(tmp_path: Path, case_names: list[str]) -> Path:
    """Write each case's package as NAME.json in a directory."""
    cases = _cases()
    for name in case_names:
        (tmp_path / f"{name}.json").write_text(json.dumps(cases[name]["package"]))
    return tmp_path


def test_cli_clean_set_exit_0(tmp_path, capsys):
    # --no-anchor keeps this offline: the enclosed RFC 3161 token is not the
    # subject of this check, the batch roll-up is.
    _materialize(tmp_path, ["clean_no_anchor"])
    rc = main(["verify-handoffs", str(tmp_path), "--no-anchor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out and "1/1 verify" in out


def test_cli_failing_set_exit_1(tmp_path, capsys):
    _materialize(tmp_path, ["clean_no_anchor", "signed_after_retirement"])
    rc = main(["verify-handoffs", str(tmp_path), "--no-anchor"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAILED" in out
    assert "signed_after_retirement.json" in out


def test_cli_json_reports_the_pinning_gap(tmp_path, capsys):
    _materialize(tmp_path, ["clean_no_anchor"])
    rc = main(["verify-handoffs", str(tmp_path), "--no-anchor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["schema"] == HANDOFF_SET_SCHEMA_NAME
    assert payload["pinningGap"] is True
    assert payload["unreadable"] == []


def test_cli_not_a_directory(tmp_path, capsys):
    rc = main(["verify-handoffs", str(tmp_path / "nope")])
    assert rc == 2
    assert "not a directory" in capsys.readouterr().err


def test_cli_no_matching_files(tmp_path, capsys):
    rc = main(["verify-handoffs", str(tmp_path)])
    assert rc == 2
    assert "no files matched" in capsys.readouterr().err


def test_cli_unreadable_file_gates(tmp_path, capsys):
    _materialize(tmp_path, ["clean_no_anchor"])
    (tmp_path / "bad.json").write_text("{ not json")
    rc = main(["verify-handoffs", str(tmp_path), "--no-anchor", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["ok"] is False
    assert [u["name"] for u in payload["unreadable"]] == ["bad.json"]
