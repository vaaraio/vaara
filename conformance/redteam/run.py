#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Boundary red-team for the Vaara Claude Code mediation surface.

Conformance testing with the cases inverted: every case in cases.json is
a tool call the operator forbids (or a benign one that must pass), and the
question is whether the fence catches it. No model is involved; the point
is to find calls that pass the fence, not to be clever about it.

Two checks per case, because a rule that would fire is worthless if the
harness never routes the call to the hook:

1. Matcher coverage. Does the PreToolUse matcher for the vaara hook cover
   this tool name? Read from the plugin's hooks.json and, if present, from
   the user's ~/.claude/settings.json. A tool the matcher misses is
   UNMEDIATED whatever the rules say.
2. Hook verdict. Drive the real `vaara hook pre-tool-use` with shadow OFF
   in a throwaway HOME and read the exit code (0 allow, 2 deny/hold).

Coverage is judged per source, not across them. Every source is a
deployment, so one that does not route the tool is a real gap there whatever
the others do. Taking any() across sources hid this: the plugin ships ``.*``
and satisfied it for every case, so a narrow matcher in a user's
settings.json could never fail a run.

Result per case:
  CAUGHT        expected deny, every matcher routes it, hook denied
  UNMEDIATED    expected deny, but some matcher never routes it to the hook
  PASSED FENCE  expected deny, routed, and the hook allowed it anyway
  OK            expected allow, allowed (or not routed: nothing to enforce)
  FALSE POSITIVE expected allow, denied

Exit 0 only when nothing passed the fence, nothing is unmediated, and
nothing benign was blocked.

Usage:
  python conformance/redteam/run.py            # table on stdout
  python conformance/redteam/run.py --json out.json
  python conformance/redteam/run.py --settings ~/.claude/settings.json
  python conformance/redteam/run.py --only subagent_spawn,egress
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PLUGIN_HOOKS = REPO / "plugins" / "claude-code-vaara-governance" / "hooks" / "hooks.json"
# The harness ships; the case file does not. A written list of calls that
# reach past a governance fence is an attack playbook, and publishing it
# arms whoever wants to beat the gate at someone else's deployment. Point
# --cases or VAARA_REDTEAM_CASES at your own file, in the schema documented
# in README.md, and the harness runs it unchanged.
CASES = Path(
    os.environ.get("VAARA_REDTEAM_CASES") or (HERE / "cases.json")
).expanduser()

HOOK_RUNNER = (
    "import sys; from vaara.cli import main; sys.exit(main(sys.argv[1:]))"
)


def _hook_argv() -> list[str]:
    """The same binary the live hook runs, else this interpreter."""
    exe = os.environ.get("VAARA_BIN") or shutil.which("vaara")
    if exe:
        return [exe]
    return [sys.executable, "-c", HOOK_RUNNER]


def _matchers_from_plugin(path: Path) -> list[str]:
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for entry in doc.get("hooks", {}).get("PreToolUse", []):
        m = entry.get("matcher")
        if m:
            out.append(m)
    return out


def _matchers_from_settings(path: Path) -> list[str]:
    """Only the PreToolUse entries whose command runs the vaara hook."""
    if not path.exists():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    out = []
    for entry in doc.get("hooks", {}).get("PreToolUse", []):
        cmds = " ".join(h.get("command", "") for h in entry.get("hooks", []))
        if "vaara" in cmds and "pre-tool-use" in cmds:
            m = entry.get("matcher")
            if m:
                out.append(m)
    return out


def _covered(matchers: list[str], tool_name: str) -> bool:
    # Claude Code treats the matcher as a regex over the tool name. We use
    # fullmatch: "Task" must not be read as covering "TaskStop".
    for m in matchers:
        try:
            if re.fullmatch(m, tool_name):
                return True
        except re.error:
            continue
    return False


def _run_hook(tool_name: str, tool_input: dict, home: Path) -> tuple[int, str]:
    (home / ".vaara").mkdir(parents=True, exist_ok=True)
    (home / ".vaara" / "config.json").write_text("{}")
    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", ""),
        "VAARA_PLUGIN_SHADOW": "0",
        "CLAUDE_PLUGIN_ROOT": str(PLUGIN_HOOKS.parent.parent),
    }
    # Operator lifts (VAARA_ALLOW_*) pass through, so a run can show what a
    # named exception opens. Everything else is scrubbed.
    env.update({k: v for k, v in os.environ.items() if k.startswith("VAARA_ALLOW_")})
    proc = subprocess.run(
        [*_hook_argv(), "hook", "pre-tool-use"],
        input=json.dumps({
            "tool_name": tool_name,
            "tool_input": tool_input,
            "session_id": "redteam",
        }),
        capture_output=True, text=True, env=env, timeout=120,
    )
    return proc.returncode, proc.stderr.strip()


def _verdict(
    expect: str, coverage: dict[str, bool], hook_verdict: str
) -> tuple[str, str]:
    """(effective, result) for one case, judged per matcher source.

    Every source is a deployment. A source that does not route the tool is a
    real gap in that deployment whatever the others do, so one miss is
    enough to call the case UNMEDIATED. Taking ``any()`` across the sources
    hid exactly this: the plugin ships ``.*`` and satisfied the test for
    every case, so a narrow matcher in a user's settings.json could never
    fail a run. Six tools named by deny rules scored CAUGHT on a box where
    the hook never sees them.

    A call the hook lets through is PASSED FENCE even when a source also
    fails to route it. Both are true, and the weaker statement is the one
    worth reporting: no rule caught it anywhere, so routing it would not
    have helped.

    An allow-expected case keeps the hook's verdict unless *no* source
    routes it. Somewhere the call is mediated, so a benign call the hook
    denies is still a false positive there.
    """
    missing = [name for name, ok in coverage.items() if not ok]
    unrouted_everywhere = len(missing) == len(coverage)
    if expect == "deny":
        if hook_verdict != "deny":
            return hook_verdict, "PASSED FENCE"
        if missing:
            return "unmediated", "UNMEDIATED"
        return hook_verdict, "CAUGHT"
    if unrouted_everywhere:
        return "unmediated", "OK"
    return hook_verdict, "OK" if hook_verdict == "allow" else "FALSE POSITIVE"


def run(cases: list[dict], matchers: dict[str, list[str]], home: Path) -> list[dict]:
    rows = []
    for case in cases:
        tool = case["tool_name"]
        coverage = {name: _covered(ms, tool) for name, ms in matchers.items()}
        code, err = _run_hook(tool, case["tool_input"], home)
        hook_verdict = "deny" if code == 2 else ("allow" if code == 0 else f"exit {code}")
        expect = case["expect"]
        effective, result = _verdict(expect, coverage, hook_verdict)
        rule = ""
        m = re.search(r"rule=([A-Za-z0-9_]+)", err)
        if m:
            rule = m.group(1)
        rows.append({
            "id": case["id"],
            "category": case["category"],
            "tool": tool,
            "expect": expect,
            "matcher": coverage,
            "missing_matchers": [n for n, ok in coverage.items() if not ok],
            "hook": hook_verdict,
            "rule": rule,
            "effective": effective,
            "result": result,
            "note": case.get("note", ""),
        })
    return rows


def _table(rows: list[dict], matcher_names: list[str]) -> str:
    head = ["case", "tool", "expect"] + [f"m:{n}" for n in matcher_names] + ["hook", "rule", "result"]
    lines = [head]
    for r in rows:
        lines.append([
            r["id"], r["tool"], r["expect"],
            *("yes" if r["matcher"].get(n) else "NO" for n in matcher_names),
            r["hook"], r["rule"] or "-", r["result"],
        ])
    widths = [max(len(str(line[i])) for line in lines) for i in range(len(head))]
    out = []
    for i, line in enumerate(lines):
        out.append("  ".join(str(c).ljust(widths[j]) for j, c in enumerate(line)))
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--plugin-hooks", default=str(PLUGIN_HOOKS))
    ap.add_argument("--settings", default=str(Path.home() / ".claude" / "settings.json"),
                    help="user settings.json to read the live vaara matcher from; skipped if absent")
    ap.add_argument("--no-settings", action="store_true", help="ignore the user settings.json")
    ap.add_argument("--only", default="", help="comma-separated categories")
    ap.add_argument("--json", default="", help="write the rows to this path")
    args = ap.parse_args(argv)

    doc = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = doc["cases"]
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["category"] in keep]

    matchers: dict[str, list[str]] = {"plugin": _matchers_from_plugin(Path(args.plugin_hooks))}
    if not args.no_settings:
        s = _matchers_from_settings(Path(args.settings).expanduser())
        if s:
            matchers["settings"] = s

    with tempfile.TemporaryDirectory(prefix="vaara-redteam-") as tmp:
        rows = run(cases, matchers, Path(tmp))

    names = list(matchers)
    print(f"vaara boundary red-team  cases={len(rows)}  matchers={names}")
    for n in names:
        print(f"  {n}: {' | '.join(matchers[n])}")
    live_shadow = os.environ.get("VAARA_PLUGIN_SHADOW", "")
    if live_shadow == "1":
        print("  NOTE: VAARA_PLUGIN_SHADOW=1 in this environment. Every CAUGHT below is "
              "log-only in the live session; this run forced enforce mode.")
    print()
    print(_table(rows, names))
    print()

    passed = [r for r in rows if r["result"] == "PASSED FENCE"]
    fp = [r for r in rows if r["result"] == "FALSE POSITIVE"]
    caught = [r for r in rows if r["result"] == "CAUGHT"]
    unmediated = [r for r in rows if r["result"] == "UNMEDIATED"]
    print(f"caught {len(caught)}  passed fence {len(passed)}  "
          f"unmediated {len(unmediated)}  false positive {len(fp)}")
    for r in passed:
        print(f"  PASSED FENCE  {r['id']:32s} {r['tool']:14s} "
              f"hook allowed it: no rule names this")
    for r in unmediated:
        where = ", ".join(r["missing_matchers"])
        print(f"  UNMEDIATED    {r['id']:32s} {r['tool']:14s} "
              f"not routed to the hook by: {where} "
              f"(rule would have fired: {r['rule'] or 'unknown'})")
    for r in fp:
        print(f"  FALSE POSITIVE {r['id']:32s} {r['tool']:14s} rule={r['rule']}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "version": doc.get("version"),
            "matchers": matchers,
            "rows": rows,
            "summary": {"caught": len(caught), "passed_fence": len(passed),
                        "unmediated": len(unmediated), "false_positive": len(fp)},
        }, indent=2), encoding="utf-8")
    return 0 if not passed and not fp and not unmediated else 1


if __name__ == "__main__":
    raise SystemExit(main())
