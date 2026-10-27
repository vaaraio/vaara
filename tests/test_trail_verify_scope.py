"""`vaara trail verify` and the standalone verifier say what an OK covers.

A hash chain proves the interior: an edit, an insert or a deleted record in
the middle breaks every hash after it. It cannot tell a shorter log from one
whose tail was cut before export. The OK line must not read as more than that.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from vaara.cli import main
from tests.test_cli_ux import _make_db, _make_key

SCOPE = ("  covers: signatures and chain interior; a dropped tail before export "
         "is not detected without an anchor or seal "
         "(docs/prove-what-an-ai-agent-did.md)")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_vaara_trail.py"


def _export(tmp_path) -> Path:
    out = tmp_path / "trail.zip"
    assert main(["trail", "export", "--db", str(_make_db(tmp_path)),
                 "--out", str(out), "--key", str(_make_key(tmp_path))]) == 0
    return out


def _tamper(src: Path, dst: Path) -> None:
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith(".jsonl"):
                data = data.replace(b"tool0", b"tool9", 1)
            zout.writestr(item, data)


def test_cli_ok_states_scope_right_after_ok(tmp_path, capsys):
    out = _export(tmp_path)
    capsys.readouterr()
    assert main(["trail", "verify", "--zip", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    i = lines.index("Verification: OK")
    assert lines[i + 1] == SCOPE


def test_cli_failed_prints_no_scope(tmp_path, capsys):
    bad = tmp_path / "bad.zip"
    _tamper(_export(tmp_path), bad)
    capsys.readouterr()
    assert main(["trail", "verify", "--zip", str(bad)]) == 1
    captured = capsys.readouterr()
    assert "covers:" not in captured.out + captured.err


def test_standalone_ok_states_same_scope(tmp_path):
    out = _export(tmp_path)
    run = subprocess.run([sys.executable, str(SCRIPT), str(out)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    lines = run.stdout.splitlines()
    i = lines.index("Verification: OK")
    assert lines[i + 1] == SCOPE


def test_scope_line_points_at_a_page_that_says_it(tmp_path):
    doc = Path(__file__).resolve().parents[1] / "docs" / "prove-what-an-ai-agent-did.md"
    text = doc.read_text(encoding="utf-8")
    assert "truncated-tail" in text
