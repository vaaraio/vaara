"""Publishing a trail head is opt-in and needs a yes, with or without a terminal.

`vaara trail publish-head` writes to a public transparency log, and an entry
there cannot be withdrawn. It asked for confirmation only when stdin was a
terminal, so the same command run from a script, a CI job or an agent
published without asking, although `--yes` is documented as the flag for
non-interactive use. Without a terminal it now refuses unless `--yes` is
given. The trail itself publishes nothing unless a publishing key is set.
"""
from __future__ import annotations

import io
import sys

import pytest

pytest.importorskip("cryptography")

from vaara import cli  # noqa: E402
from vaara.attestation import rekor_log  # noqa: E402
from vaara.audit.trail import AuditTrail  # noqa: E402
from vaara.pipeline import InterceptionPipeline  # noqa: E402


class _NotATerminal(io.StringIO):
    def isatty(self) -> bool:
        return False


@pytest.fixture()
def db(tmp_path):
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    path = tmp_path / "audit.db"
    backend = SQLiteAuditBackend(path)
    trail = AuditTrail(on_record=backend.write_record)
    trail.record_decision("a1", "agent-x", "read_file", "allow", "low risk", 0.05)
    backend.close()
    return path


@pytest.fixture()
def published(monkeypatch):
    calls: list[str] = []

    def fake_publish(head, signer, log_url=None):
        calls.append(head)
        raise rekor_log.RekorError("test double, nothing sent")

    monkeypatch.setattr(rekor_log, "publish_head", fake_publish)
    monkeypatch.setattr(sys, "stdin", _NotATerminal(""))
    return calls


def _run(db, tmp_path, *extra):
    argv = ["trail", "publish-head", "--db", str(db), "--key", str(tmp_path / "k.pem"), *extra]
    try:
        return cli.main(argv)
    except SystemExit as exc:
        return exc.code


def test_without_a_terminal_and_without_yes_nothing_is_published(db, tmp_path, published, capsys):
    rc = _run(db, tmp_path)
    assert published == []
    assert rc == 2
    assert "--yes" in capsys.readouterr().err


def test_yes_publishes_without_a_terminal(db, tmp_path, published):
    _run(db, tmp_path, "--yes")
    assert len(published) == 1


def test_dry_run_publishes_nothing(db, tmp_path, published):
    assert _run(db, tmp_path, "--dry-run") == 0
    assert published == []


def test_a_trail_publishes_nothing_by_default(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(rekor_log, "publish_head", lambda *a, **k: calls.append(a[0]))
    trail = AuditTrail()
    pipe = InterceptionPipeline(trail=trail)
    for _ in range(50):
        pipe.intercept(agent_id="a", tool_name="data.read", parameters={})
    assert calls == []
    assert trail.publications() == []
