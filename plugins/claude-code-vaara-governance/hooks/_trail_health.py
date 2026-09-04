# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Say when the audit trail is not recording. The fallback scripts' half.

These scripts are the fallback path: ``hooks.json`` runs the ``vaara`` binary
when it is on PATH and these files otherwise. Both paths had the same defect,
so fixing only the packaged one would leave half the installs silent. The
logic lives in ``vaara.audit.write_failure``; this is the thin wrapper that
lets a script use it without importing the whole hook module.

Never raises, and never blocks a tool call. See the module docstring in
``vaara/audit/write_failure.py`` for what happened on 2026-08-22 and why the
answer is a durable marker rather than a fail-closed hook.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _notify import notify as _raw_notify  # noqa: E402


def _emit(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def note_failure(db_path, exc: BaseException, *, stage: str) -> None:
    """Count a failed trail write and say so, at most once a minute."""
    try:
        from vaara.audit.write_failure import failure_banner, record_failure

        state = record_failure(db_path, exc, stage=stage)
        if state is None:
            _emit(f"vaara-governance: trail write failed ({exc!r}); NOT recording.")
            return
        if state.get("notify"):
            _emit(failure_banner(state))
            _raw_notify(
                "TRAIL NOT RECORDING", "audit trail",
                f"{state.get('count', '?')} failed writes since "
                f"{state.get('first_failure_utc', 'an unknown time')}",
            )
    except Exception:
        # Reporting a failure must never become the failure.
        pass


def report(db_path, existed: bool) -> None:
    """Once per session: a standing outage, else whether the file reads clean."""
    try:
        from vaara.audit.write_failure import (
            active_failure, failure_banner, quick_check,
        )

        state = active_failure(db_path)
        if state is not None:
            _emit(failure_banner(state))
            _raw_notify(
                "TRAIL NOT RECORDING", "audit trail",
                f"{state.get('count', '?')} failed writes since "
                f"{state.get('first_failure_utc', 'an unknown time')}",
            )
            return
        if not existed:
            return
        problem = quick_check(db_path)
        if problem is not None:
            _emit(
                f"vaara-governance: the audit trail at {db_path} does not read "
                f"clean ({problem}). Records may not be persisting. Check it with "
                f"`sqlite3 {db_path} 'PRAGMA integrity_check'` and recover with "
                "`.recover`. Do not delete the file, it is the evidence."
            )
            _raw_notify("TRAIL DAMAGED", "audit trail", problem)
    except Exception:
        pass
