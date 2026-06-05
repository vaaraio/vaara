"""Assembling evidence bundles and the `vaara build-bundle` command.

Covers the issuer-side builder `build_bundle_document` and the directory
discovery `load_bundle_pieces_from_dir`, the `build_bundle_v0` conformance
vectors through both Vaara and the standalone Vaara-free checker, and the CLI
end to end: directory input, explicit flags, stdout, the round-trip into
`verify-bundle`, and the error paths.

The headline property: a bundle assembled by `build-bundle` is byte-for-byte
the document `verify-bundle` reads, and feeding it straight to `verify-bundle`
reproduces the expected verdict.
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
    build_bundle_document,
    evidence_bundle_from_json,
    load_bundle_pieces_from_dir,
    verify_evidence_bundle,
)
from vaara.cli import main  # noqa: E402

VECTORS = Path(__file__).resolve().parent / "vectors" / "build_bundle_v0"
PIECES = VECTORS / "pieces"
DOCUMENTS = VECTORS / "documents"
BUNDLE_DOC = Path(__file__).resolve().parent / "vectors" / "bundle_doc_v0" / "bundles"


def _expected() -> dict:
    return json.loads((VECTORS / "expected.json").read_text())


def _cases():
    return sorted(_expected().keys())


# ── Builder + vectors ────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", _cases())
def test_assembled_pieces_match_verifier_document(name):
    pieces = load_bundle_pieces_from_dir(PIECES / name)
    doc = build_bundle_document(**pieces)
    rendered = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    assert rendered == (DOCUMENTS / f"{name}.json").read_text()


@pytest.mark.parametrize("name", _cases())
def test_build_then_verify_reproduces_vector_verdict(name):
    want = _expected()[name]
    pieces = load_bundle_pieces_from_dir(PIECES / name)
    doc = build_bundle_document(**pieces)
    verdict = verify_evidence_bundle(evidence_bundle_from_json(doc))
    assert verdict.ok is want["ok"]
    assert verdict.authenticity_established is want["authenticity_established"]


def test_documents_equal_bundle_doc_vectors():
    # The two vector sets stay in lockstep: the issuer assembles exactly the
    # file the verifier reads.
    for name in _cases():
        assert (DOCUMENTS / f"{name}.json").read_text() == (
            BUNDLE_DOC / f"{name}.json"
        ).read_text()


def test_independent_checker_passes():
    proc = subprocess.run(
        [sys.executable, str(VECTORS / "_check_independent.py")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ── Builder behaviour ────────────────────────────────────────────────────────


def test_build_requires_receipt():
    with pytest.raises(TypeError):
        build_bundle_document()  # type: ignore[call-arg]


def test_build_validates_pieces():
    pieces = load_bundle_pieces_from_dir(PIECES / "all_lenses_pass")
    pieces["inclusion"]["root_hex"] = "nothex"
    with pytest.raises(ValueError, match="inclusion.root_hex"):
        build_bundle_document(**pieces)


def test_load_pieces_rejects_non_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        load_bundle_pieces_from_dir(tmp_path / "missing")


def test_load_pieces_reports_bad_json(tmp_path):
    (tmp_path / "receipt.json").write_text("{ not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_bundle_pieces_from_dir(tmp_path)


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_from_dir_round_trips(tmp_path, capsys):
    out = tmp_path / "bundle.json"
    rc = main(
        ["build-bundle", "--from-dir", str(PIECES / "all_lenses_pass"),
         "--out", str(out)]
    )
    assert rc == 0
    assert "verify-bundle verdict OK" in capsys.readouterr().err
    assert out.read_text() == (BUNDLE_DOC / "all_lenses_pass.json").read_text()
    assert main(["verify-bundle", str(out)]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_explicit_flags(tmp_path, capsys):
    src = PIECES / "signature_only"
    out = tmp_path / "bundle.json"
    rc = main([
        "build-bundle",
        "--receipt", str(src / "receipt.json"),
        "--verifying-jwk", str(src / "verifying_jwk.json"),
        "--out", str(out),
    ])
    assert rc == 0
    assert out.read_text() == (BUNDLE_DOC / "signature_only.json").read_text()
    assert main(["verify-bundle", str(out)]) == 0


def test_cli_stdout(capsys):
    rc = main(["build-bundle", "--from-dir", str(PIECES / "all_lenses_pass")])
    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out)["receipt"]["version"] == 1
    assert "assembled bundle to stdout" in captured.err


def test_cli_partial_bundle_is_built_and_reported(tmp_path, capsys):
    out = tmp_path / "bundle.json"
    rc = main(
        ["build-bundle", "--from-dir", str(PIECES / "unauthenticated_in_log"),
         "--out", str(out)]
    )
    assert rc == 0
    assert "verify-bundle verdict not ok" in capsys.readouterr().err
    assert out.is_file()


def test_cli_missing_receipt(tmp_path, capsys):
    (tmp_path / "registry.json").write_text(json.dumps({"entries": []}))
    rc = main(["build-bundle", "--from-dir", str(tmp_path)])
    assert rc == 2
    assert "a receipt is required" in capsys.readouterr().err


def test_cli_from_dir_not_a_directory(tmp_path, capsys):
    rc = main(["build-bundle", "--from-dir", str(tmp_path / "nope")])
    assert rc == 2
    assert "not a directory" in capsys.readouterr().err


def test_cli_unreadable_flag_file(tmp_path, capsys):
    rc = main(["build-bundle", "--receipt", str(tmp_path / "nope.json")])
    assert rc == 1
    assert "cannot read --receipt" in capsys.readouterr().err


def test_cli_malformed_piece(tmp_path, capsys):
    src = PIECES / "all_lenses_pass"
    doc = json.loads((src / "inclusion.json").read_text())
    doc["root_hex"] = "nothex"
    bad = tmp_path / "inclusion.json"
    bad.write_text(json.dumps(doc))
    rc = main([
        "build-bundle",
        "--receipt", str(src / "receipt.json"),
        "--inclusion", str(bad),
    ])
    assert rc == 1
    assert "cannot assemble bundle" in capsys.readouterr().err
