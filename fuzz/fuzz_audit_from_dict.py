"""Atheris fuzz target: AuditRecord.from_dict.

Models the JSONL-replay path: trail records on disk get reloaded by
deserialising untrusted JSON into `AuditRecord.from_dict`. Bad input
must surface as `TypeError`/`ValueError`/`KeyError` — never silently
construct a corrupt record and never crash the process.

After deserialisation the target also exercises `compute_hash()` and
the `narrative` property — both are reached when a regulator replays
a trail, and both have already had defensive fixes (NaN/overflow
timestamps, oversize agent_id) that fuzzing should keep honest.
"""

from __future__ import annotations

import json
import sys

import atheris

with atheris.instrument_imports():
    from vaara.audit.trail import AuditRecord


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    raw = fdp.ConsumeUnicode(sys.maxsize)
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return

    if not isinstance(parsed, dict):
        return

    try:
        record = AuditRecord.from_dict(parsed)
    except (TypeError, ValueError, KeyError):
        return

    try:
        record.compute_hash()
    except (TypeError, ValueError):
        return

    try:
        _ = record.narrative
    except (TypeError, ValueError):
        return


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
