# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A trail that stops recording has to say so somewhere durable.

Found 2026-09-04 on the maintainer's own machine: the Claude Code hook had
been unable to write a single audit record since 2026-08-22, thirteen days.
Every attempt raised ``sqlite3.DatabaseError: database disk image is
malformed``, printed a full traceback to stderr, and exited 0. The host read
that as success and carried on. Nothing anywhere said the trail was dead, and
it was found only because someone went looking for an unrelated reason.

Two things made it invisible, and both are addressed here rather than by
making the hook fail closed:

* **The signal was per-record noise.** A traceback on every tool call reads
  as normal output within minutes. Volume is not visibility.
* **The counter could not count.** ``AuditTrail._persistence_failures`` is
  per-process, and the hook is a fresh process per tool call, so it could
  never exceed 1. Nothing accumulated, so nothing crossed a threshold.

So the state lives next to the trail, in ``<db>.write-failure.json``. It
survives the process, it accumulates, and the session-start hook reads it and
says the trail is not recording in those words.

Fail-open stays. A governance hook that blocks a session gets uninstalled,
and an uninstalled hook records nothing at all. The defect was the silence,
not the exit code.

Nothing here raises. A failure to report a failure must not become the
failure, and a read-only directory must not stop the agent from working.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Suffix appended to the database filename to locate its marker.
MARKER_SUFFIX = ".write-failure.json"

#: Minimum gap between loud, operator-facing notifications about the same
#: outage. The first failure always notifies; after that, once a minute is
#: enough to stay visible without becoming the noise it replaced.
NOTIFY_INTERVAL_SECONDS = 60.0

#: ``PRAGMA quick_check`` reads the whole file, so it is a startup check only
#: while the file is small enough for that to be cheap. Above this it is
#: skipped rather than made into a session-start stall.
QUICK_CHECK_MAX_BYTES = 512 * 1024 * 1024

_ERROR_LIMIT = 300


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _describe(error: Any) -> str:
    if isinstance(error, BaseException):
        text = f"{type(error).__name__}: {error}"
    else:
        text = str(error)
    text = " ".join(text.split())
    return text[:_ERROR_LIMIT]


def marker_path(db_path: Any) -> Optional[Path]:
    """Where the marker for ``db_path`` lives, or ``None`` if it cannot have one.

    In-memory databases have no directory to put it in, and there is no
    operator watching one either: they are tests and short-lived tools.
    """
    if db_path is None:
        return None
    text = str(db_path)
    if not text or text == ":memory:" or text.startswith("file::memory:"):
        return None
    path = Path(text).expanduser()
    return path.with_name(path.name + MARKER_SUFFIX)


def read_marker(db_path: Any) -> Optional[dict]:
    """The marker state for ``db_path``, resolved or not, or ``None``."""
    path = marker_path(db_path)
    if path is None:
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def active_failure(db_path: Any) -> Optional[dict]:
    """The marker only while the trail is still failing to record.

    This is the question the session-start hook asks, and it is deliberately
    not "has this trail ever failed": a resolved marker is forensics and
    stays on disk, but nagging about a fixed outage every session is how a
    banner gets tuned out.
    """
    state = read_marker(db_path)
    if state is None or state.get("resolved"):
        return None
    return state


def _write_marker(path: Path, state: dict) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def record_failure(
    db_path: Any,
    error: Any,
    *,
    stage: str = "append",
    notify_interval: float = NOTIFY_INTERVAL_SECONDS,
) -> Optional[dict]:
    """Count one failed write against ``db_path`` and return the running state.

    Returns ``None`` only when there is nowhere to keep state (an in-memory
    trail). The returned dict carries a ``notify`` key the caller uses to
    decide whether to say anything out loud this time; it is derived, not
    stored.
    """
    path = marker_path(db_path)
    if path is None:
        return None

    now = time.time()
    previous = read_marker(db_path) or {}
    resolved = bool(previous.get("resolved"))

    count = previous.get("count")
    # A resolved marker describes a finished outage. The next failure is a new
    # one, so it starts its own count and its own clock rather than inheriting
    # a total that spans a working period in between.
    if resolved or not isinstance(count, int) or isinstance(count, bool) or count < 1:
        count = 1
        first = now
        last_notified = 0.0
    else:
        count += 1
        first = previous.get("first_failure")
        if not isinstance(first, (int, float)) or isinstance(first, bool):
            first = now
        last_notified = previous.get("last_notified")
        if not isinstance(last_notified, (int, float)) or isinstance(last_notified, bool):
            last_notified = 0.0

    notify = count == 1 or (now - last_notified) >= notify_interval
    state = {
        "db": str(db_path),
        "stage": stage,
        "error": _describe(error),
        "count": count,
        "first_failure": first,
        "first_failure_utc": _utc(first),
        "last_failure": now,
        "last_failure_utc": _utc(now),
        "last_notified": now if notify else last_notified,
        "pid": os.getpid(),
        "resolved": False,
    }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_marker(path, state)
    except OSError as exc:
        # The marker is the durable half. Losing it means the caller's own
        # log line is the only signal left, so say that once rather than
        # letting the reporting path fail silently in its turn.
        logger.warning(
            "could not write the audit-trail failure marker at %s (%s); "
            "the trail write failure will only be visible in this process",
            path, exc,
        )

    return dict(state, notify=notify)


def mark_recovered(db_path: Any) -> Optional[dict]:
    """Flip an active marker to resolved after a write succeeds.

    Returns the state that was resolved, or ``None`` when there was no active
    failure. The file is kept and stamped rather than deleted: it is the only
    record of how long the trail was not recording, and on a product whose
    argument is that an absent record must not read as an absent action,
    deleting that would be the same mistake one layer up.
    """
    path = marker_path(db_path)
    if path is None:
        return None
    state = active_failure(db_path)
    if state is None:
        return None

    now = time.time()
    state = dict(
        state,
        resolved=True,
        resolved_at=now,
        resolved_at_utc=_utc(now),
        resolved_by_pid=os.getpid(),
    )
    try:
        _write_marker(path, state)
    except OSError as exc:
        logger.warning("could not stamp the recovered failure marker at %s (%s)", path, exc)
    return state


#: The banner has to fit on a terminal line or two, and some of the errors it
#: quotes are paragraphs (``AuditBackendUnreadable`` explains recovery in full).
#: The marker keeps the longer form for whoever goes looking.
_BANNER_ERROR_LIMIT = 140


def failure_banner(state: dict) -> str:
    """One operator-facing line. Says what is not happening, not what threw."""
    error = str(state.get("error", "unknown error"))
    if len(error) > _BANNER_ERROR_LIMIT:
        error = error[:_BANNER_ERROR_LIMIT].rstrip() + "..."
    return (
        "vaara-governance: THE AUDIT TRAIL IS NOT RECORDING. "
        f"{state.get('count', '?')} write(s) have failed since "
        f"{state.get('first_failure_utc', 'an unknown time')} "
        f"({state.get('stage', 'append')}: {error}). "
        "Tool calls are still running and are NOT being recorded. "
        f"Trail: {state.get('db', 'unknown')}. "
        "Check it with `sqlite3 <db> 'PRAGMA integrity_check'` and recover with "
        "`.recover`. Do not delete the file, it is the evidence."
    )


def quick_check(
    db_path: Any, *, max_bytes: int = QUICK_CHECK_MAX_BYTES
) -> Optional[str]:
    """``None`` when the database reads clean, else a one-line description.

    Read-only, and cheap enough to run once at session start. This catches the
    case the marker cannot: a trail damaged while nothing was writing to it,
    where the first failure would otherwise be the next record nobody watches.
    """
    path = marker_path(db_path)
    if path is None:
        return None
    db = Path(str(db_path)).expanduser()
    try:
        if not db.is_file():
            return None
        if db.stat().st_size > max_bytes:
            logger.debug("skipping quick_check on %s: larger than %d bytes", db, max_bytes)
            return None
        uri = f"{db.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    except (OSError, ValueError, sqlite3.Error) as exc:
        return f"unreadable ({_describe(exc)})"

    try:
        rows = conn.execute("PRAGMA quick_check(1)").fetchall()
    except sqlite3.Error as exc:
        return f"unreadable ({_describe(exc)})"
    finally:
        conn.close()

    results = [str(row[0]) for row in rows if row]
    if not results or results == ["ok"]:
        return None
    return _describe("; ".join(results))
