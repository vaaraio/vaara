# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``~/.vaara/sources.json``: every trail the engine writes, in one place.

The macOS app used to find trails by walking ``~/.vaara`` for SQLite files,
which missed a trail kept anywhere else and picked up leftovers from older
installs. Now the engine says where it writes. The first record a process
appends to a trail registers that trail here, and the app reads this file.

    {"version": 1,
     "sources": [{"trail": "/Users/me/.vaara/trail/audit.db",
                  "receipts": "/Users/me/.vaara/trail/receipts",
                  "first_write": "2026-09-25T01:02:03Z",
                  "last_write": "2026-09-25T09:10:11Z"}]}

``last_write`` is refreshed at most once an hour per trail, so a busy hook
does not rewrite the file on every call. An entry whose trail file is gone is
dropped on the next write. The location follows ``VAARA_HOME`` the way the app
does: the directory that holds ``.vaara``, or ``.vaara`` itself.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

FILENAME = "sources.json"
_REFRESH = timedelta(hours=1)


def vaara_dir() -> Path:
    raw = os.environ.get("VAARA_HOME", "").strip()
    if raw:
        p = Path(raw).expanduser()
        return p if p.name == ".vaara" else p / ".vaara"
    return Path.home() / ".vaara"


def sources_path() -> Path:
    return vaara_dir() / FILENAME


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso(t: datetime) -> str:
    return t.isoformat().replace("+00:00", "Z")


def _parse(s: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def read() -> list[dict[str, Any]]:
    try:
        data = json.loads(sources_path().read_text())
    except (OSError, ValueError):
        return []
    entries = data.get("sources") if isinstance(data, dict) else None
    return [e for e in entries if isinstance(e, dict) and isinstance(e.get("trail"), str)] \
        if isinstance(entries, list) else []


def register(db_path: Any, *, now: Optional[datetime] = None) -> bool:
    """Record that this process writes ``db_path``. Returns True when the file changed.

    Never raises: a trail that cannot be listed still records.
    """
    raw = str(db_path)
    if not raw or raw == ":memory:" or raw.startswith("file::memory:"):
        return False
    try:
        return _register(Path(raw).expanduser().resolve(), now or _now())
    except Exception:
        logger.exception("could not register %s in %s", raw, sources_path())
        return False


def _register(trail: Path, now: datetime) -> bool:
    path = sources_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    with open(lock, "a") as fh:
        try:
            import fcntl
            fcntl.flock(fh, fcntl.LOCK_EX)
        except (ImportError, OSError):
            # No flock (Windows, some network filesystems): the write below is
            # still atomic, and a lost race costs one entry until next write.
            pass
        entries = [e for e in read() if Path(e["trail"]).exists() or e["trail"] == str(trail)]
        changed = len(entries) != len(read())
        mine = next((e for e in entries if e["trail"] == str(trail)), None)
        if mine is None:
            entries.append({
                "trail": str(trail),
                "receipts": str(trail.parent / "receipts"),
                "first_write": _iso(now),
                "last_write": _iso(now),
            })
            changed = True
        else:
            last = _parse(mine.get("last_write"))
            if last is None or now - last >= _REFRESH:
                mine["last_write"] = _iso(now)
                changed = True
        if not changed:
            return False
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".sources.", suffix=".tmp")
        with os.fdopen(fd, "w") as out:
            json.dump({"version": 1, "sources": entries}, out, indent=2)
            out.write("\n")
        os.replace(tmp, path)
        return True
