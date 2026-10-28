# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A cage launch's start and stop, written to the operator's trail.

``vaara cage run``, ``vaara cage stop`` and ``vaara run`` call these. The
trail is the one the egress recorder in :mod:`vaara.oslayer.run` writes to:
``$VAARA_DB``, else ``~/.vaara/trail/audit.db``. A record that cannot be
written is reported on stderr and never stops or keeps a cage: the cage is
the control, the record is its evidence.
"""

from __future__ import annotations

import getpass
import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Optional


def trail_path() -> str:
    return os.environ.get("VAARA_DB") or str(Path.home() / ".vaara" / "trail" / "audit.db")


def open_trail(path: Optional[str] = None) -> Any:
    """The trail at ``path`` (default :func:`trail_path`), creating its
    directory (0700) first: SQLite does not, and a fresh account has no
    ``~/.vaara`` until something writes there."""
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    db = path or trail_path()
    Path(db).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return SQLiteAuditBackend(db).load_trail()


def policy_digest(policy: Optional[Path], fallback: str = "") -> str:
    """``sha256:`` of the policy file as given, else the cage's own digest."""
    if policy is not None:
        try:
            return "sha256:" + hashlib.sha256(Path(policy).read_bytes()).hexdigest()
        except OSError:
            pass
    return fallback


def _actor() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name is an empty actor, not an error
        return ""


def _trail() -> Any:
    return open_trail()


def started(driver: str, name: str, digest: str = "") -> Optional[str]:
    """Record a launch's start. The record id, or None when it was not written."""
    try:
        return str(_trail().record_cage_lifecycle(
            "started", driver, name, policy_digest=digest, actor=_actor()))
    except Exception as exc:  # noqa: BLE001 - a lost record is reported, never fatal
        print(f"vaara cage: lifecycle record not written: {exc}", file=sys.stderr)
        return None


def stopped(driver: str, name: str, *, reason: str, start_id: Optional[str] = None,
            processes_ended: Optional[int] = None,
            exit_code: Optional[int] = None) -> Optional[str]:
    """Record a launch's end, closing ``start_id`` or the open start of that launch."""
    try:
        trail = _trail()
        if start_id is None:
            start_id = trail.open_cage_start(driver, name)
        return str(trail.record_cage_lifecycle(
            "stopped", driver, name, actor=_actor(), started_record_id=start_id or "",
            processes_ended=processes_ended, exit_code=exit_code, reason=reason))
    except Exception as exc:  # noqa: BLE001 - a lost record is reported, never fatal
        print(f"vaara cage: lifecycle record not written: {exc}", file=sys.stderr)
        return None
