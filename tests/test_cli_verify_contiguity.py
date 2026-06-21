"""Tests for ``vaara verify-contiguity`` — gap detection over authorization receipts.

The CLI reads the ``evidence`` half of each ``*-authz.json`` file and reports
whether the per-boundary completeness sequence is contiguous. These tests drive
the surface with authz-shaped files; the signing path is covered elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

from vaara.cli import main

BOUNDARY = "vaara-mcp-proxy"


def _authz(
    d: Path, seq: int, running_count: int, boundary: str = BOUNDARY
) -> Path:
    payload = {
        "record": {},
        "evidence": {
            "schema": "vaara.authorization/v0",
            "completeness": {
                "boundaryId": boundary,
                "seq": seq,
                "runningCount": running_count,
            },
        },
    }
    path = d / f"{boundary}-{seq:04d}-aaaa-authz.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _stream(d: Path, n: int, boundary: str = BOUNDARY) -> None:
    for i in range(n):
        _authz(d, i, i + 1, boundary)


def test_complete_stream_exits_zero(tmp_path, capsys):
    _stream(tmp_path, 5)
    rc = main(["verify-contiguity", str(tmp_path)])
    assert rc == 0
    assert "contiguous" in capsys.readouterr().out


def test_dropped_middle_exits_one_and_names_gap(tmp_path, capsys):
    _stream(tmp_path, 5)
    _authz(tmp_path, 2, 3).unlink()  # drop seq 2
    rc = main(["verify-contiguity", str(tmp_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "INCOMPLETE" in out
    assert "missing seq: 2" in out


def test_json_output(tmp_path, capsys):
    _stream(tmp_path, 3)
    rc = main(["verify-contiguity", str(tmp_path), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] is True
    assert data["expected"] == 3
    assert data["missingSeqs"] == []


def test_no_receipts_is_usage_error(tmp_path, capsys):
    rc = main(["verify-contiguity", str(tmp_path)])
    assert rc == 2
    assert "no authorization receipts" in capsys.readouterr().err


def test_multiple_boundaries_need_explicit_boundary(tmp_path, capsys):
    _stream(tmp_path, 2, "boundary-a")
    _stream(tmp_path, 3, "boundary-b")
    rc = main(["verify-contiguity", str(tmp_path)])
    assert rc == 2
    assert "multiple boundaries" in capsys.readouterr().err

    rc = main(["verify-contiguity", str(tmp_path), "--boundary", "boundary-b"])
    assert rc == 0
    assert "contiguous" in capsys.readouterr().out
