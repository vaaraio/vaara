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


class TestTheDeclineIsNotSilent:
    """The pragma may refuse without raising, and the log reaches nobody.

    ``_set_journal_mode`` gives up after ten retries and keeps whatever mode
    the file is in, reporting through ``logger``. In a Claude Code hook
    there is no logging configured and the process is gone a moment later,
    so a trail could stay in WAL on a mount that corrupts it with nothing
    anywhere saying so. Two halves: notice the silent refusal, and give the
    operator somewhere to read it.
    """

    def test_a_pragma_that_answers_with_the_old_mode_is_not_success(
        self, tmp_path, trail, monkeypatch
    ):
        """SQLite answers a declined mode change with the current mode.

        Taking the absence of an exception as success is what let a WAL to
        DELETE conversion quietly not happen.
        """
        monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "ext4"))
        monkeypatch.delenv("VAARA_TRAIL_JOURNAL_MODE", raising=False)
        backend = SQLiteAuditBackend(str(trail))

        calls = []

        class StubbornConnection:
            def execute(self, statement):
                calls.append(statement)

                class Cursor:
                    def fetchone(self):
                        return ("wal",)

                return Cursor()

        monkeypatch.setattr(backend, "_conn", StubbornConnection())
        monkeypatch.setattr(SQLiteAuditBackend, "_WAL_RETRY_SLEEP", 0)

        # It must not raise: recording evidence in the wrong mode still
        # beats refusing to open the trail at all.
        backend._set_journal_mode("delete")

        assert len(calls) == SQLiteAuditBackend._WAL_RETRIES, (
            "a declined mode change should be retried like a lost lock"
        )

    def test_a_pragma_that_lands_returns_on_the_first_try(
        self, tmp_path, trail, monkeypatch
    ):
        monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "ext4"))
        monkeypatch.delenv("VAARA_TRAIL_JOURNAL_MODE", raising=False)
        backend = SQLiteAuditBackend(str(trail))

        calls = []

        class AgreeableConnection:
            def execute(self, statement):
                calls.append(statement)

                class Cursor:
                    def fetchone(self):
                        return ("delete",)

                return Cursor()

        monkeypatch.setattr(backend, "_conn", AgreeableConnection())
        backend._set_journal_mode("delete")

        assert calls == ["PRAGMA journal_mode=DELETE"]

    def test_an_in_memory_answer_is_accepted_not_retried(
        self, tmp_path, trail, monkeypatch
    ):
        """``:memory:`` answers "memory" whatever it is asked for.

        Treating that as a refusal would put ten sleeps into the
        construction of every in-memory backend, which is most of the suite.
        """
        monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "ext4"))
        monkeypatch.delenv("VAARA_TRAIL_JOURNAL_MODE", raising=False)
        backend = SQLiteAuditBackend(str(trail))

        calls = []

        class MemoryConnection:
            def execute(self, statement):
                calls.append(statement)

                class Cursor:
                    def fetchone(self):
                        return ("memory",)

                return Cursor()

        monkeypatch.setattr(backend, "_conn", MemoryConnection())
        backend._set_journal_mode("wal")

        assert len(calls) == 1


class TestJournalModeWarning:
    """What the operator reads at session start."""

    def test_wal_on_an_unsafe_mount_is_reported(self, tmp_path, trail, monkeypatch):
        monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "virtiofs"))
        monkeypatch.setenv("VAARA_TRAIL_JOURNAL_MODE", "wal")
        SQLiteAuditBackend(str(trail))

        warning = sqlite_backend.journal_mode_warning(trail)

        assert warning is not None
        assert "WAL" in warning
        assert "virtiofs" in warning

    def test_delete_on_an_unsafe_mount_is_silent(self, tmp_path, trail, monkeypatch):
        monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "virtiofs"))
        monkeypatch.delenv("VAARA_TRAIL_JOURNAL_MODE", raising=False)
        SQLiteAuditBackend(str(trail))

        assert sqlite_backend.journal_mode_warning(trail) is None

    def test_wal_on_a_local_filesystem_is_silent(self, tmp_path, trail, monkeypatch):
        monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "ext4"))
        monkeypatch.delenv("VAARA_TRAIL_JOURNAL_MODE", raising=False)
        SQLiteAuditBackend(str(trail))

        assert sqlite_backend.journal_mode_warning(trail) is None

    def test_it_asks_about_the_file_not_about_the_conversion(
        self, tmp_path, trail, monkeypatch
    ):
        """An operator who forced WAL gets the mode they asked for.

        Nothing failed, so a "the conversion did not happen" check would
        stay quiet, and the trail is in exactly the state that eats the
        evidence. The question is the file's mode, not the switch.
        """
        conn = sqlite3.connect(str(trail))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.close()
        monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "nfs4"))

        assert sqlite_backend.journal_mode_warning(trail) is not None

    @pytest.mark.parametrize("path", [None, "", ":memory:", "file::memory:?cache=shared"])
    def test_nothing_to_warn_about(self, path):
        assert sqlite_backend.journal_mode_warning(path) is None

    def test_a_trail_that_does_not_exist_yet_is_silent(self, tmp_path):
        assert sqlite_backend.journal_mode_warning(tmp_path / "absent.db") is None

    def test_an_unreadable_trail_is_left_to_the_other_check(self, tmp_path, monkeypatch):
        """``quick_check`` owns corruption. This one stays out of its way."""
        broken = tmp_path / "audit.db"
        broken.write_bytes(b"this is not a database" + b"\x00" * 200)
        monkeypatch.setattr(sqlite_backend, "_PROC_MOUNTS", _mounts(tmp_path, "virtiofs"))

        assert sqlite_backend.journal_mode_warning(broken) is None


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
