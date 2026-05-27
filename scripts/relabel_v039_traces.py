"""Re-label captured v0.39 BIPIA traces using the fixed labeller.

The 2026-05-27 audit found `looks_like_injection_follow` had a 67%
false-positive rate driven by an unconditional ``"example.com" in url``
rule. Agents pick example.com as a benign placeholder for honest API
calls (BI task -> ``api.example.com/electric-car-sales``) and got
mis-labelled as follows. The rule has been replaced with strict
url-in-injection / host-in-injection (host != example.com) /
payload-prefix-in-injection matching.

This script does NOT re-run the harness (would cost compute + API
credits). It reads the captured traces, recomputes
``reflects_injection`` per tool call using the fixed labeller, and
rewrites the trace files in place. Backups go to ``<name>.preid``.

Usage:
    .venv/bin/python scripts/relabel_v039_traces.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from _v039_common import load_bipia  # noqa: E402
from run_v039_bipia import looks_like_injection_follow  # noqa: E402

TRACE_DIR = REPO / "tests/adversarial/traces"


def main() -> int:
    trials = {t.trial_id: t for t in load_bipia()}
    before_total = after_total = 0

    for fp in sorted(TRACE_DIR.glob("bipia-s43-*.jsonl")):
        before = after = 0
        new_lines: list[str] = []
        unknown = 0
        for raw in fp.read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            d = json.loads(line)
            trial = trials.get(d["trial_id"])
            if not trial:
                unknown += 1
                new_lines.append(line)
                continue
            inj_text = trial.injection_text
            for tc in d.get("tool_calls", []):
                before += int(bool(tc.get("reflects_injection")))
                tc["reflects_injection"] = looks_like_injection_follow(
                    tc.get("tool_name", ""), tc.get("parameters", {}) or {}, inj_text
                )
                after += int(tc["reflects_injection"])
            new_lines.append(json.dumps(d))

        backup = fp.with_suffix(".preid")
        if not backup.exists():
            shutil.copy2(fp, backup)
            print(f"[backup] {backup.relative_to(REPO)}")
        fp.write_text("\n".join(new_lines) + "\n")
        before_total += before
        after_total += after
        print(
            f"[relabel] {fp.relative_to(REPO)}  follows: before={before} -> after={after}  "
            f"(unknown trial_ids preserved: {unknown})"
        )

    print(f"\n[totals] follows before={before_total}  after={after_total}  "
          f"delta={after_total - before_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
