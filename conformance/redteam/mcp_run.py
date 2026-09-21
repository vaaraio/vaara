#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Classifier red-team for the Vaara Claude Code hook, MCP path.

`run.py` measures layer 1: the deny rules, matched by tool name. On the
Claude Code hook that layer only sees the fixed tool names (Bash, Write,
Agent). A tool named by an MCP server, `mcp__anything__whatever`, is never
in a rule's tool list, so the same payload that a deny rule catches on Bash
reaches the hook with nothing but the classifier in front of it.

This runner takes the same case file, rewrites each case's tool name to an
MCP name (Bash -> mcp__shell__run_command, Write -> mcp__fs__write_file,
and so on), drives the real `vaara hook pre-tool-use` with shadow OFF in a
throwaway HOME, and then reads the trail that HOME holds to report what the
conformal scorer actually output for each call: decision, point estimate,
conformal upper bound, and the thresholds in force.

No model is involved in the sense of an LLM; the scorer is the shipped
classifier bundle. That is the layer the question is about.

Result per case:
  DENIED     expected deny, classifier denied (exit 2)
  HELD       expected deny, classifier escalated (hook waited, no approvals
             dir, so it timed out closed: exit 2)
  PASSED     expected deny, classifier allowed it (exit 0)
  NAME-ONLY  expected deny by a match_any rule (tool-name policy such as
             agent_spawn). By content there is nothing to match, so this
             row reports the classifier's verdict and does not fail the run
  OK         expected allow, allowed
  FALSE POSITIVE expected allow, denied or held

Exit 0 only when nothing expected-deny PASSED and nothing benign was held
or denied. NAME-ONLY rows are informational.

Usage:
  python conformance/redteam/mcp_run.py --cases cases.json
  python conformance/redteam/mcp_run.py --cases cases.json --json out.json
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
PLUGIN_ROOT = REPO / "plugins" / "claude-code-vaara-governance"

HOOK_RUNNER = (
    "import sys; from vaara.cli import main; sys.exit(main(sys.argv[1:]))"
)

# One MCP-shaped name per Claude Code tool. The server prefix is arbitrary;
# what matters is that the name starts with mcp__ and is in no rule's list.
MCP_NAMES = {
    "Bash": "mcp__shell__run_command",
    "Write": "mcp__fs__write_file",
    "Edit": "mcp__fs__edit_file",
    "NotebookEdit": "mcp__fs__edit_notebook",
    "Read": "mcp__fs__read_file",
    "WebFetch": "mcp__web__fetch",
    "Agent": "mcp__agents__spawn",
    "Task": "mcp__agents__task",
    "TaskStop": "mcp__agents__stop",
    "Workflow": "mcp__agents__workflow",
    "Skill": "mcp__agents__skill",
    "SendMessage": "mcp__agents__send",
    "RemoteTrigger": "mcp__remote__trigger",
    "CronCreate": "mcp__schedule__create",
    "CronDelete": "mcp__schedule__delete",
    "CronList": "mcp__schedule__list",
    "ScheduleWakeup": "mcp__schedule__wakeup",
    "ExitWorktree": "mcp__git__exit_worktree",
    "ReadMcpResourceTool": "mcp__res__read",
    "ReadMcpResourceDirTool": "mcp__res__list",
}


def _hook_argv() -> list[str]:
    exe = os.environ.get("VAARA_BIN") or shutil.which("vaara")
    if exe:
        return [exe]
    return [sys.executable, "-c", HOOK_RUNNER]


def _env(home: Path, preset: str) -> dict:
    env = {
        "HOME": str(home),
        "PATH": os.environ.get("PATH", ""),
        "VAARA_PLUGIN_SHADOW": "0",
        "CLAUDE_PLUGIN_ROOT": str(PLUGIN_ROOT),
        # An escalate waits on the approvals handshake; nobody answers in a
        # throwaway HOME, and timeout is fail-closed, so a short wait gives
        # the same verdict as the default 60s without a 40-minute run.
        "VAARA_PLUGIN_APPROVALS_TIMEOUT": os.environ.get("VAARA_PLUGIN_APPROVALS_TIMEOUT", "2"),
    }
    if preset:
        env["VAARA_PROTECTION_PRESET"] = preset
    env.update({k: v for k, v in os.environ.items() if k.startswith("VAARA_ALLOW_")})
    return env


def _run_hook(tool_name: str, tool_input: dict, session: str, home: Path, preset: str) -> tuple[int, str]:
    proc = subprocess.run(
        [*_hook_argv(), "hook", "pre-tool-use"],
        input=json.dumps({
            "tool_name": tool_name,
            "tool_input": tool_input,
            "session_id": session,
        }),
        capture_output=True, text=True, env=_env(home, preset), timeout=180,
    )
    return proc.returncode, proc.stderr.strip()


def _scores_from_trail(home: Path) -> dict[str, dict]:
    """action_id -> {session_id, tool_name, decision, point_estimate, ...}.

    Read straight from the sqlite trail the hook wrote: action_requested
    carries the session id, risk_scored the conformal interval, and
    decision_made the verdict and reason. Joined on action_id.
    """
    import sqlite3

    db = home / ".vaara" / "claude-code" / "audit.db"
    if not db.exists():
        cands = list((home / ".vaara").rglob("*.db"))
        if not cands:
            return {}
        db = cands[0]
    out: dict[str, dict] = {}
    con = sqlite3.connect(db)
    try:
        for aid, et, tool, data in con.execute(
            "SELECT action_id, event_type, tool_name, data FROM audit_records ORDER BY seq"
        ):
            try:
                payload = json.loads(data) if data else {}
            except json.JSONDecodeError:
                payload = {}
            slot = out.setdefault(str(aid), {"tool_name": tool, "events": []})
            slot["events"].append(et)
            if et == "action_requested":
                slot["session_id"] = payload.get("session_id")
                slot["base_risk"] = payload.get("base_risk_score")
            elif et == "risk_scored":
                for k in ("point_estimate", "conformal_lower", "conformal_upper", "signals"):
                    slot[k] = payload.get(k)
            elif et in ("decision_made", "action_blocked", "action_allowed"):
                slot["decision"] = payload.get("decision")
                slot["reason"] = payload.get("reason")
                slot["risk_score"] = payload.get("risk_score")
            elif et in ("escalation_resolved", "human_decision"):
                slot["resolution"] = payload
    finally:
        con.close()
    return out


def _content_rule_for(tool_input: dict) -> str | None:
    """Rule id a by-content match would give this input, lifts ignored."""
    sys.path.insert(0, str(REPO / "src"))
    from vaara.deny_rules import load_deny_rules, match_deny_rule_any_field

    rules = [dict(r, unless_env=None) for r in load_deny_rules(None)]
    m = match_deny_rule_any_field(rules, tool_input)
    return m[0] if m else None


def run(cases: list[dict], home: Path, preset: str) -> list[dict]:
    rows = []
    for i, case in enumerate(cases):
        orig = case["tool_name"]
        tool = MCP_NAMES.get(orig)
        if tool is None:
            continue
        session = f"mcpredteam-{i:03d}-{case['id']}"
        code, err = _run_hook(tool, case["tool_input"], session, home, preset)
        rule_hit = ""
        m = re.search(r"rule=([A-Za-z0-9_]+)", err)
        if m:
            rule_hit = m.group(1)
        rows.append({
            "rule": rule_hit,
            "id": case["id"], "category": case["category"],
            "orig_tool": orig, "tool": tool, "expect": case["expect"],
            "exit": code, "stderr": err[-300:], "session": session,
            "content_rule": _content_rule_for(case["tool_input"]),
        })
    scores = _scores_from_trail(home)
    by_session: dict[str, dict] = {}
    for aid, s in scores.items():
        sid = s.get("session_id")
        if sid:
            by_session[sid] = {**s, "action_id": aid}
    for r in rows:
        s = by_session.get(r["session"], {})
        r["decision"] = s.get("decision")
        r["point"] = s.get("point_estimate", s.get("risk_score"))
        r["upper"] = s.get("conformal_upper")
        r["reason"] = s.get("reason")
        held = "APPROVAL" in r["stderr"] or "escalat" in r["stderr"].lower()
        if r["expect"] == "deny":
            if r["exit"] == 0 and r["content_rule"] is None:
                r["result"] = "NAME-ONLY"
            elif r["exit"] == 0:
                r["result"] = "PASSED"
            elif held or r["decision"] == "escalate":
                r["result"] = "HELD"
            else:
                r["result"] = "DENIED"
        else:
            r["result"] = "OK" if r["exit"] == 0 else "FALSE POSITIVE"
    return rows


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _table(rows: list[dict]) -> str:
    head = ["case", "mcp tool", "expect", "exit", "rule", "decision", "point", "upper", "result"]
    lines = [head]
    for r in rows:
        lines.append([r["id"], r["tool"], r["expect"], str(r["exit"]), r.get("rule") or "-",
                      _fmt(r["decision"]), _fmt(r["point"]), _fmt(r["upper"]), r["result"]])
    widths = [max(len(str(line[i])) for line in lines) for i in range(len(head))]
    out = []
    for i, line in enumerate(lines):
        out.append("  ".join(str(c).ljust(widths[j]) for j, c in enumerate(line)))
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default=os.environ.get("VAARA_REDTEAM_CASES", str(HERE / "cases.json")))
    ap.add_argument("--preset", default="", help="protection preset to apply (balanced, strict, ...)")
    ap.add_argument("--only", default="", help="comma-separated categories")
    ap.add_argument("--json", default="", help="write the rows to this path")
    ap.add_argument("--keep-home", action="store_true", help="print and keep the throwaway HOME")
    args = ap.parse_args(argv)

    doc = json.loads(Path(args.cases).expanduser().read_text(encoding="utf-8"))
    cases = doc["cases"]
    if args.only:
        keep = set(args.only.split(","))
        cases = [c for c in cases if c["category"] in keep]

    tmp = Path(tempfile.mkdtemp(prefix="vaara-mcp-redteam-"))
    (tmp / ".vaara").mkdir(parents=True, exist_ok=True)
    cfg = {}
    if args.preset:
        cfg["protection_preset"] = args.preset
    (tmp / ".vaara" / "config.json").write_text(json.dumps(cfg))
    try:
        rows = run(cases, tmp, args.preset)
    finally:
        if args.keep_home:
            print(f"HOME kept at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"vaara classifier red-team (mcp path)  cases={len(rows)}  preset={args.preset or 'default'}")
    print()
    print(_table(rows))
    print()
    passed = [r for r in rows if r["result"] == "PASSED"]
    held = [r for r in rows if r["result"] == "HELD"]
    denied = [r for r in rows if r["result"] == "DENIED"]
    fp = [r for r in rows if r["result"] == "FALSE POSITIVE"]
    name_only = [r for r in rows if r["result"] == "NAME-ONLY"]
    points = [r["point"] for r in rows if isinstance(r.get("point"), (int, float))]
    uppers = [r["upper"] for r in rows if isinstance(r.get("upper"), (int, float))]
    print(f"denied {len(denied)}  held {len(held)}  passed {len(passed)}  "
          f"false positive {len(fp)}  name-only {len(name_only)}")
    if points:
        print(f"point estimate: min {min(points):.3f} max {max(points):.3f}")
    if uppers:
        print(f"conformal upper: min {min(uppers):.3f} max {max(uppers):.3f}")
    for r in passed:
        print(f"  PASSED  {r['id']:32s} {r['tool']:26s} point={_fmt(r['point'])} upper={_fmt(r['upper'])}")
    for r in fp:
        print(f"  FALSE POSITIVE  {r['id']:32s} {r['tool']:26s} point={_fmt(r['point'])}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "version": doc.get("version"), "preset": args.preset, "rows": rows,
            "summary": {"denied": len(denied), "held": len(held),
                        "passed": len(passed), "false_positive": len(fp),
                        "name_only": len(name_only)},
        }, indent=2), encoding="utf-8")
    return 0 if not passed and not fp else 1


if __name__ == "__main__":
    raise SystemExit(main())
