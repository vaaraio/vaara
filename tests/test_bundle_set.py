"""Batch full-lens verification and the `vaara verify-bundles` command.

The batch twin of `verify-bundle`: an auditor points the tool at a directory
of evidence bundles and gets the roll-up a single-file check cannot give.
Covers the `bundle_set_v0` vectors through both Vaara and the standalone
Vaara-free checker, the module behaviour (malformed documents, coverage gaps,
gating), and the CLI end to end. Runs the lenses' crypto, so the suite skips
when the attestation extra is absent.
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
    BUNDLE_SET_SCHEMA_NAME,
    check_bundle_set,
)
from vaara.cli import main  # noqa: E402

VECTORS = Path(__file__).resolve().parent / "vectors" / "bundle_set_v0"
SETS = VECTORS / "sets"
DOC_BUNDLES = Path(__file__).resolve().parent / "vectors" / "bundle_doc_v0" / "bundles"


def _expected() -> dict:
    return json.loads((VECTORS / "expected.json").read_text())


def _cases():
    return sorted(_expected().keys())


def _load_set(name: str):
    files = sorted((SETS / name).glob("*.json"))
    return [(p.name, json.loads(p.read_text())) for p in files]


# ── Vectors ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", _cases())
def test_module_reproduces_vector_rollup(name):
    want = _expected()[name]
    report = check_bundle_set(_load_set(name))
    got = {
        "ok": report.ok,
        "total": report.total,
        "loaded": report.loaded,
        "passed": report.passed,
        "authenticated": report.authenticated,
        "lensApplicable": report.lens_applicable,
        "lensPassed": report.lens_passed,
        "lensGaps": list(report.lens_gaps),
    }
    assert got == want


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_public_reexport_is_wired():
    from vaara.attestation.receipt import check_bundle_set as public
    assert public is check_bundle_set


# ── Unit behaviour ────────────────────────────────────────────────────────────


def test_clean_set_is_ok_with_no_gap():
    report = check_bundle_set(_load_set("clean"))
    assert report.ok
    assert report.passed == report.total == 2
    assert report.authenticated == 2
    assert report.lens_gaps == ()


def test_mixed_set_gates_on_a_failing_bundle():
    report = check_bundle_set(_load_set("mixed"))
    assert not report.ok
    assert report.passed == 1
    failing = [e for e in report.entries if not e.ok]
    assert len(failing) == 1
    assert failing[0].lens_states["inclusion"] == "fail"
    assert report.lens_gaps == ("signature",)  # no bundle carried signature material


def test_thin_set_is_ok_but_reports_gaps_in_lens_order():
    report = check_bundle_set(_load_set("thin"))
    assert report.ok  # a thin but valid set still verifies
    # Ordering follows the lens stack, not the alphabet.
    assert report.lens_gaps == (
        "identity", "back_link", "inclusion", "consistency", "revocation",
    )


def test_unauthenticated_bundle_gates_and_lowers_auth_count():
    report = check_bundle_set(_load_set("unauthenticated"))
    assert not report.ok
    assert report.authenticated == 1  # one of two established its signature
    unauth = next(e for e in report.entries if not e.authenticity_established)
    assert not unauth.ok


def test_malformed_document_is_a_failing_entry_not_a_raise():
    good = _load_set("clean")[0]
    report = check_bundle_set([good, ("broken.json", {"not": "a bundle"})])
    assert not report.ok  # a non-loadable document gates the set
    broken = next(e for e in report.entries if e.name == "broken.json")
    assert broken.loaded is False
    assert broken.ok is False
    assert broken.error is not None
    assert report.loaded == 1  # the good one still loaded


def test_empty_set_is_ok_with_all_lenses_gapped():
    report = check_bundle_set([])
    assert report.ok and report.total == 0 and len(report.lens_gaps) == 6


def test_report_to_dict_shape():
    d = check_bundle_set(_load_set("clean")).to_dict()
    assert d["schema"] == BUNDLE_SET_SCHEMA_NAME
    assert d["ok"] is True
    assert d["total"] == 2
    assert d["lensGaps"] == []
    assert all({"name", "loaded", "ok", "lensStates"} <= set(e) for e in d["entries"])


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_cli_clean_set_exit_0(capsys):
    rc = main(["verify-bundles", str(SETS / "clean")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out
    assert "2/2 bundles verify" in out


def test_cli_mixed_set_exit_1(capsys):
    rc = main(["verify-bundles", str(SETS / "mixed")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAILED" in out
    assert "tampered_inclusion.json" in out
    assert "coverage gap" in out


def test_cli_json_output(capsys):
    rc = main(["verify-bundles", str(SETS / "unauthenticated"), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["ok"] is False
    assert payload["schema"] == BUNDLE_SET_SCHEMA_NAME
    assert payload["authenticated"] == 1
    assert payload["lensGaps"] == ["signature"]
    assert payload["unreadable"] == []


def test_cli_not_a_directory(tmp_path, capsys):
    rc = main(["verify-bundles", str(tmp_path / "nope")])
    assert rc == 2
    assert "not a directory" in capsys.readouterr().err


def test_cli_no_matching_files(tmp_path, capsys):
    rc = main(["verify-bundles", str(tmp_path)])
    assert rc == 2
    assert "no files matched" in capsys.readouterr().err


def test_cli_unreadable_file_gates(tmp_path, capsys):
    (tmp_path / "good.json").write_text((DOC_BUNDLES / "all_lenses_pass.json").read_text())
    (tmp_path / "bad.json").write_text("{ not json")
    rc = main(["verify-bundles", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["ok"] is False
    assert [u["name"] for u in payload["unreadable"]] == ["bad.json"]
