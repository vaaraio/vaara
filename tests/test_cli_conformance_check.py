"""The unified `vaara conformance` front door.

`conformance check <path>` is one memorable command over the keyless
checks that already ship: a file dispatches to the single-record check
(`verify-record`), a directory to the set check (`verify-records`). No
logic of its own, so its verdicts match those commands exactly.
`conformance statement` self-tests against the published corpus. All
keyless: the conformance path runs in the base install, no extra.
"""

from __future__ import annotations

import json
from pathlib import Path

from vaara.cli import main

REC = Path(__file__).resolve().parent / "vectors" / "record_conformance_v0" / "records"
SETS = Path(__file__).resolve().parent / "vectors" / "record_set_v0" / "sets"


def test_check_file_conforms_exit_0(tmp_path, capsys):
    target = tmp_path / "record.json"
    target.write_text((REC / "conforming_executed_projection.json").read_text())
    rc = main(["conformance", "check", str(target)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "CONFORMS" in out


def test_check_file_non_conforming_exit_1(tmp_path, capsys):
    target = tmp_path / "record.json"
    target.write_text((REC / "neg_digest_mismatch.json").read_text())
    rc = main(["conformance", "check", str(target)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NON-CONFORMING" in out
    assert "result_commitment_self_consistent" in out


def test_check_directory_conforms(capsys):
    rc = main(["conformance", "check", str(SETS / "clean")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "record set: CONFORMS" in out


def test_check_directory_gap_fails(capsys):
    rc = main(["conformance", "check", str(SETS / "duplicate_call")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NON-CONFORMING" in out


def test_check_file_json(tmp_path, capsys):
    target = tmp_path / "record.json"
    target.write_text((REC / "conforming_executed_projection.json").read_text())
    rc = main(["conformance", "check", str(target), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["conformance"]["conforms"] is True


def test_check_directory_json(capsys):
    rc = main(["conformance", "check", str(SETS / "clean"), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["conforms"] is True


def test_check_verdict_matches_verify_record(tmp_path, capsys):
    """The front door must reach the same verdict as the command it wraps."""
    target = tmp_path / "record.json"
    target.write_text((REC / "conforming_executed_projection.json").read_text())
    rc_new = main(["conformance", "check", str(target)])
    capsys.readouterr()
    rc_old = main(["verify-record", str(target)])
    capsys.readouterr()
    assert rc_new == rc_old == 0


def test_check_missing_path_exit_2(tmp_path, capsys):
    rc = main(["conformance", "check", str(tmp_path / "nope")])
    assert rc == 2
    assert "no such path" in capsys.readouterr().err


def test_statement_self_test_exit_0(capsys):
    rc = main(["conformance", "statement"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "corpusDigest" in out or "corpus" in out.lower()
