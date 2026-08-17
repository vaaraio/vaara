"""The audit trail must not run WAL on filesystems that cannot support it.

WAL coordinates readers and writers through an mmap'd shared-memory segment
(the ``-shm`` file). virtiofs, 9p, NFS, SMB and FUSE mounts do not give
SQLite coherent shared memory or working file locking, and the result is a
corrupt trail rather than a refused write. Four corruption events in twelve
days on one container-mounted home directory produced this file.
"""

import sqlite3
from pathlib import Path

import pytest

from vaara.audit import sqlite_backend
from vaara.audit.sqlite_backend import SQLiteAuditBackend


def _mounts(tmp_path: Path, fstype: str) -> Path:
    """A /proc/mounts whose longest matching entry covers ``tmp_path``."""
    mounts = tmp_path / "mounts"
    mounts.write_text(
        "proc /proc proc rw,relatime 0 0\n"
        "/dev/vda1 / ext4 rw,relatime 0 0\n"
        f"host {tmp_path.resolve()} {fstype} rw,relatime 0 0\n"
    )
    return mounts


@pytest.fixture
def trail(tmp_path):
    return tmp_path / "audit.db"


def test_wal_is_declined_on_virtiofs(tmp_path, trail, monkeypatch):
    monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "virtiofs"))
    monkeypatch.delenv("VAARA_TRAIL_JOURNAL_MODE", raising=False)

    backend = SQLiteAuditBackend(str(trail))

    assert backend.journal_mode == "delete"
    assert not Path(str(trail) + "-shm").exists()


def test_wal_is_used_on_a_local_filesystem(tmp_path, trail, monkeypatch):
    monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "ext4"))
    monkeypatch.delenv("VAARA_TRAIL_JOURNAL_MODE", raising=False)

    backend = SQLiteAuditBackend(str(trail))

    assert backend.journal_mode == "wal"


def test_any_fuse_filesystem_declines_wal(tmp_path, trail, monkeypatch):
    monkeypatch.setattr(
        sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "fuse.gcsfuse")
    )
    monkeypatch.delenv("VAARA_TRAIL_JOURNAL_MODE", raising=False)

    assert SQLiteAuditBackend(str(trail)).journal_mode == "delete"


def test_the_decline_names_the_path_the_fstype_and_the_reason(
    tmp_path, trail, monkeypatch, caplog
):
    monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "nfs4"))
    monkeypatch.delenv("VAARA_TRAIL_JOURNAL_MODE", raising=False)

    with caplog.at_level("WARNING", logger="vaara.audit.sqlite_backend"):
        SQLiteAuditBackend(str(trail))

    warning = "\n".join(r.getMessage() for r in caplog.records)
    assert str(trail.resolve()) in warning
    assert "nfs4" in warning
    assert "DELETE" in warning.upper()


def test_env_override_forces_wal_on_an_unsafe_filesystem(tmp_path, trail, monkeypatch):
    monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "virtiofs"))
    monkeypatch.setenv("VAARA_TRAIL_JOURNAL_MODE", "wal")

    assert SQLiteAuditBackend(str(trail)).journal_mode == "wal"


def test_env_override_forces_delete_on_a_local_filesystem(tmp_path, trail, monkeypatch):
    monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "ext4"))
    monkeypatch.setenv("VAARA_TRAIL_JOURNAL_MODE", "DELETE")

    assert SQLiteAuditBackend(str(trail)).journal_mode == "delete"


def test_an_unreadable_env_override_is_ignored_not_fatal(tmp_path, trail, monkeypatch):
    monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "ext4"))
    monkeypatch.setenv("VAARA_TRAIL_JOURNAL_MODE", "banana")

    assert SQLiteAuditBackend(str(trail)).journal_mode == "wal"


def test_no_proc_mounts_keeps_the_previous_behaviour(tmp_path, trail, monkeypatch):
    """macOS and Windows have no /proc/mounts. Detection is Linux-only and
    says so; everywhere else the trail opens exactly as it did before."""
    monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", tmp_path / "no-such-file")
    monkeypatch.delenv("VAARA_TRAIL_JOURNAL_MODE", raising=False)

    assert SQLiteAuditBackend(str(trail)).journal_mode == "wal"


def test_an_existing_wal_trail_on_an_unsafe_mount_is_converted(
    tmp_path, trail, monkeypatch
):
    """The trail that produced this defect already exists and is in WAL. The
    fix has to move it off WAL, not only spare new files."""
    conn = sqlite3.connect(str(trail))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.close()

    monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "virtiofs"))
    monkeypatch.delenv("VAARA_TRAIL_JOURNAL_MODE", raising=False)

    assert SQLiteAuditBackend(str(trail)).journal_mode == "delete"


def test_in_memory_trails_do_not_consult_the_filesystem(tmp_path, monkeypatch):
    monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "virtiofs"))
    monkeypatch.delenv("VAARA_TRAIL_JOURNAL_MODE", raising=False)

    # ":memory:" is not on any mount; detection must not crash or downgrade it.
    assert SQLiteAuditBackend(":memory:").journal_mode in {"wal", "memory"}


def test_the_longest_matching_mount_wins(tmp_path, trail, monkeypatch):
    """A virtiofs mount deeper than an ext4 one must not be masked by it."""
    mounts = tmp_path / "mounts"
    mounts.write_text(
        "/dev/vda1 / ext4 rw,relatime 0 0\n"
        f"host {tmp_path.resolve().parent} ext4 rw,relatime 0 0\n"
        f"host {tmp_path.resolve()} virtiofs rw,relatime 0 0\n"
    )
    monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", mounts)
    monkeypatch.delenv("VAARA_TRAIL_JOURNAL_MODE", raising=False)

    assert SQLiteAuditBackend(str(trail)).journal_mode == "delete"


def test_mount_points_with_escaped_spaces_are_parsed(tmp_path, monkeypatch):
    """/proc/mounts octal-escapes spaces in mount points as \\040."""
    spaced = tmp_path / "my trail"
    spaced.mkdir()
    mounts = tmp_path / "mounts"
    escaped = str(spaced.resolve()).replace(" ", r"\040")
    mounts.write_text(
        "/dev/vda1 / ext4 rw,relatime 0 0\n"
        f"host {escaped} virtiofs rw,relatime 0 0\n"
    )
    monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", mounts)
    monkeypatch.delenv("VAARA_TRAIL_JOURNAL_MODE", raising=False)

    assert SQLiteAuditBackend(str(spaced / "audit.db")).journal_mode == "delete"
