# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validate v037 generated jsonl: schema + required-field + per-cat consistency.

Mirrors validate_v036.py with the v037 glob. Fail-loud on any malformed entry
so the build_split step gets a clean input.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GEN = Path("tests/adversarial/generated")
REQUIRED = ["id", "category", "tool_name", "parameters", "context", "expected", "severity", "notes"]
VALID_SEV = {"low", "medium", "high", "critical"}
EXPECTED_CAT = {"TM": "tool_misuse", "PE": "privilege_escalation", "DE": "data_exfil"}


def main() -> int:
    files = sorted(GEN.glob("*-v037-*.jsonl"))
    print(f"{len(files)} v037 files")
    total_in = total_ok = 0
    for f in files:
        n_in = n_ok = n_miss = n_exp = n_sev = n_cat = 0
        expected = EXPECTED_CAT[f.name[:2]]
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                e = json.loads(line)
            except Exception:
                n_miss += 1
                continue
            if not all(k in e for k in REQUIRED):
                n_miss += 1
                continue
            if "original_task" not in (e.get("context") or {}):
                n_miss += 1
                continue
            if e.get("expected") != "DENY":
                n_exp += 1
                continue
            if e.get("severity") not in VALID_SEV:
                n_sev += 1
                continue
            if e.get("category") != expected:
                n_cat += 1
                continue
            n_ok += 1
        flags = []
        if n_miss:
            flags.append(f"missing={n_miss}")
        if n_exp:
            flags.append(f"bad-expected={n_exp}")
        if n_sev:
            flags.append(f"bad-sev={n_sev}")
        if n_cat:
            flags.append(f"wrong-cat={n_cat}")
        total_in += n_in
        total_ok += n_ok
        print(f"{f.name:40s} in={n_in:4d} ok={n_ok:4d} bad={n_in - n_ok:3d} {' '.join(flags)}")
    print(f"TOTAL in={total_in} ok={total_ok} bad={total_in - total_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
