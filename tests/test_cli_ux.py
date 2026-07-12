"""Tests for CLI usability: did-you-mean suggestions and --db trail export."""
from __future__ import annotations

import pytest

pytest.importorskip("cryptography")

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.cli import build_parser, main
from vaara.pipeline import InterceptionPipeline


def _make_db(tmp_path):
    db = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(db)
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    pipeline = InterceptionPipeline(trail=trail, enforce=False)
    for i in range(3):
        pipeline.intercept(
            agent_id="test-agent",
            tool_name=f"mcp__demo__tool{i}",
            parameters={"x": i},
        )
    return db


def _make_key(tmp_path):
    key = tmp_path / "key"
    assert main(["keygen", "--dev", "--out", str(key)]) == 0
    return key


def test_mistyped_command_suggests_closest(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["verion"])
    err = capsys.readouterr().err
    assert "Did you mean 'version'?" in err


def test_mistyped_trail_subcommand_suggests_closest(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["trail", "exoprt"])
    err = capsys.readouterr().err
    assert "Did you mean 'export'?" in err


def test_mistyped_command_without_close_match_points_at_help(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["zzzzzz"])
    err = capsys.readouterr().err
    assert "Did you mean" not in err
    assert "--help" in err


def test_usage_line_shows_metavar_not_command_wall(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    assert "usage: vaara [-h] COMMAND ..." in out.splitlines()[0]


def test_trail_export_from_sqlite_db(tmp_path, capsys):
    db = _make_db(tmp_path)
    key = _make_key(tmp_path)
    out = tmp_path / "trail.zip"
    assert main([
        "trail", "export", "--db", str(db), "--out", str(out), "--key", str(key),
    ]) == 0
    assert out.is_file()
    capsys.readouterr()
    assert main(["trail", "verify", "--zip", str(out)]) == 0
    assert "Verification: OK" in capsys.readouterr().out


def test_trail_export_article12_from_sqlite_db(tmp_path):
    db = _make_db(tmp_path)
    key = _make_key(tmp_path)
    out = tmp_path / "a12.zip"
    assert main([
        "trail", "export-article12",
        "--db", str(db), "--out", str(out), "--key", str(key),
    ]) == 0
    assert out.is_file()


def test_trail_export_rejects_trail_and_db_together(tmp_path, capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "trail", "export", "--trail", "t.jsonl", "--db", "a.db",
            "--out", "o.zip", "--key", "k",
        ])
    assert "not allowed with" in capsys.readouterr().err


def test_trail_export_requires_a_source(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["trail", "export", "--out", "o.zip", "--key", "k"])
    assert "--trail" in capsys.readouterr().err


def test_trail_export_missing_db_file_errors(tmp_path, capsys):
    key = _make_key(tmp_path)
    capsys.readouterr()
    rc = main([
        "trail", "export", "--db", str(tmp_path / "absent.db"),
        "--out", str(tmp_path / "o.zip"), "--key", str(key),
    ])
    assert rc == 2
    assert "audit DB not found" in capsys.readouterr().err
