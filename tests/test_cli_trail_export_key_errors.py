"""`vaara trail export` and `vaara trail verify` answer a bad key path with
one line, not a traceback.

A missing database and a missing zip already got a plain line; a missing
or malformed signing key, and a missing public key, raised out of the
command.
"""

from __future__ import annotations

import pytest

pytest.importorskip("cryptography", reason="signing extras not installed")

from vaara.audit.sqlite_backend import SQLiteAuditBackend  # noqa: E402
from vaara.cli import main  # noqa: E402
from vaara.pipeline import InterceptionPipeline  # noqa: E402


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(path)
    InterceptionPipeline(trail=backend.load_trail()).intercept(
        agent_id="a", tool_name="file.read", parameters={"path": "/x"})
    backend.close()
    return path


def test_missing_key_is_one_line(db, tmp_path, capsys):
    rc = main(["trail", "export", "--db", str(db),
               "--out", str(tmp_path / "t.zip"),
               "--key", str(tmp_path / "nope.pem")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "signing key not found" in err
    assert "nope.pem" in err
    assert "Traceback" not in err


def test_malformed_key_is_one_line(db, tmp_path, capsys):
    bad = tmp_path / "bad.pem"
    bad.write_text("not a pem\n")
    rc = main(["trail", "export", "--db", str(db),
               "--out", str(tmp_path / "t.zip"), "--key", str(bad)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "--key" in err
    assert "Traceback" not in err


def test_verify_missing_pubkey_is_one_line(db, tmp_path, capsys):
    rc = main(["trail", "verify", "--zip", str(tmp_path / "t.zip"),
               "--pubkey", str(tmp_path / "nope.pub")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "public key not found" in err
    assert "Traceback" not in err
