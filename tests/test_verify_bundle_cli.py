"""On-disk evidence bundles and the `vaara verify-bundle` command.

Covers the loader `evidence_bundle_from_json` (the on-disk document shape), the
`bundle_doc_v0` conformance vectors through both Vaara and the standalone
Vaara-free checker, and the CLI command end to end: file input, directory
input, JSON output, exit codes, and the error paths.
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
    evidence_bundle_from_json,
    verify_evidence_bundle,
)
from vaara.cli import main  # noqa: E402

VECTORS = Path(__file__).resolve().parent / "vectors" / "bundle_doc_v0"
BUNDLES = VECTORS / "bundles"


def _expected() -> dict:
    return json.loads((VECTORS / "expected.json").read_text())


def _cases():
    return sorted(_expected().keys())


# ── Loader + vectors ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", _cases())
def test_loader_reproduces_vector_verdict(name):
    want = _expected()[name]
    doc = json.loads((BUNDLES / f"{name}.json").read_text())
    verdict = verify_evidence_bundle(evidence_bundle_from_json(doc))
    assert verdict.ok is want["ok"]
    assert verdict.authenticity_established is want["authenticity_established"]
    got = {r.lens: {"applicable": r.applicable, "ok": r.ok} for r in verdict.lenses}
    assert got == want["lenses"]


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_verdict_to_dict_round_trips():
    doc = json.loads((BUNDLES / "all_lenses_pass.json").read_text())
    verdict = verify_evidence_bundle(evidence_bundle_from_json(doc))
    d = verdict.to_dict()
    assert d["ok"] is True
    assert {lens["lens"] for lens in d["lenses"]} == {
        "identity", "signature", "back_link", "inclusion", "consistency", "revocation",
    }
    assert all("reason" in lens for lens in d["lenses"])


# ── Loader error paths ───────────────────────────────────────────────────────


def test_loader_rejects_non_object():
    with pytest.raises(ValueError, match="JSON object"):
        evidence_bundle_from_json([])  # type: ignore[arg-type]


def test_loader_requires_receipt():
    with pytest.raises(ValueError, match="receipt"):
        evidence_bundle_from_json({})


def test_loader_reports_malformed_block():
    doc = json.loads((BUNDLES / "all_lenses_pass.json").read_text())
    doc["inclusion"]["root_hex"] = "nothex"
    with pytest.raises(ValueError, match="inclusion.root_hex"):
        evidence_bundle_from_json(doc)


def test_loader_rejects_bool_for_int_field():
    doc = json.loads((BUNDLES / "all_lenses_pass.json").read_text())
    doc["inclusion"]["log_index"] = True
    with pytest.raises(ValueError, match="inclusion.log_index"):
        evidence_bundle_from_json(doc)


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_file_ok(tmp_path, capsys):
    src = BUNDLES / "all_lenses_pass.json"
    target = tmp_path / "bundle.json"
    target.write_text(src.read_text())
    rc = main(["verify-bundle", str(target)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out
    assert "consistency" in out


def test_cli_file_failed_exit_1(tmp_path, capsys):
    src = BUNDLES / "tampered_inclusion.json"
    target = tmp_path / "bundle.json"
    target.write_text(src.read_text())
    rc = main(["verify-bundle", str(target)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAILED" in out


def test_cli_unauthenticated_in_log_is_not_ok(tmp_path, capsys):
    src = BUNDLES / "unauthenticated_in_log.json"
    target = tmp_path / "b.json"
    target.write_text(src.read_text())
    rc = main(["verify-bundle", str(target)])
    assert rc == 1
    assert "authenticity established: False" in capsys.readouterr().out


def test_cli_directory_mode(tmp_path, capsys):
    (tmp_path / "bundle.json").write_text((BUNDLES / "all_lenses_pass.json").read_text())
    rc = main(["verify-bundle", str(tmp_path)])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_cli_json_output(tmp_path, capsys):
    target = tmp_path / "bundle.json"
    target.write_text((BUNDLES / "all_lenses_pass.json").read_text())
    rc = main(["verify-bundle", str(target), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["ok"] is True
    assert payload["authenticity_established"] is True
    assert len(payload["lenses"]) == 6


def test_cli_missing_path(tmp_path, capsys):
    rc = main(["verify-bundle", str(tmp_path / "nope.json")])
    assert rc == 2
    assert "not a file or directory" in capsys.readouterr().err


def test_cli_empty_directory(tmp_path, capsys):
    rc = main(["verify-bundle", str(tmp_path)])
    assert rc == 2
    assert "no bundle.json" in capsys.readouterr().err


def test_cli_bad_json(tmp_path, capsys):
    target = tmp_path / "bundle.json"
    target.write_text("{ not json")
    rc = main(["verify-bundle", str(target)])
    assert rc == 1
    assert "cannot read bundle JSON" in capsys.readouterr().err


def test_cli_invalid_bundle(tmp_path, capsys):
    target = tmp_path / "bundle.json"
    target.write_text(json.dumps({"not": "a receipt"}))
    rc = main(["verify-bundle", str(target)])
    assert rc == 1
    assert "not a valid evidence bundle" in capsys.readouterr().err
