#!/usr/bin/env python3
"""Governed tool call: gate a proposed action through Vaara before it runs,
record what happened after, to a persistent Article 12 audit trail.

This is the script half of the ``vaara-governed-tool-call`` skill. It wraps the
installed ``vaara`` package (``pip install vaara``) so an agent can put an EU AI
Act Article 14 human-oversight checkpoint and an Article 12 record in front of a
high-risk tool call without standing up the HTTP proxy.

Two subcommands:

  gate     Classify, score, and decide allow / escalate / deny for a proposed
           call. Writes the decision to the audit trail. On ``escalate`` the
           call is enqueued to the review queue and the agent MUST stop and wait
           for a human (see ``vaara review``). Exit code carries the verdict:
           0 allow, 10 escalate (needs a human), 20 deny (blocked), 1 error.

  outcome  After an allowed call ran, record what it did to the same trail.

Both reopen the trail from SQLite, so the hash chain continues across calls and
``vaara trail export-article12`` can package the whole lifecycle for a regulator.
"""
from __future__ import annotations

import argparse
import json
import sys


def _load_trail(audit_db: str):
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    backend = SQLiteAuditBackend(audit_db)
    # load_trail() returns a trail wired to persist new records AND seeded with
    # the last hash, so appends continue the existing chain rather than fork it.
    return backend, backend.load_trail()


def _parse_json_arg(raw: str, flag: str) -> dict:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.exit(f"{flag}: not valid JSON: {exc}")
    if not isinstance(value, dict):
        sys.exit(f"{flag}: must be a JSON object, got {type(value).__name__}")
    return value


_EXIT = {"allow": 0, "escalate": 10, "deny": 20}

# (escalate-boundary, deny-boundary). A risk below the escalate boundary
# allows; above the deny boundary denies; in between escalates for human
# review. These mirror `vaara mode list`.
_MODES = {
    "eco": (0.40, 0.60),
    "balanced": (0.55, 0.85),
    "performance": (0.70, 0.92),
    "strict": (0.30, 0.55),
}


def cmd_gate(args: argparse.Namespace) -> int:
    from vaara import Pipeline
    from vaara.audit.review_queue import ReviewQueue
    from vaara.scorer.adaptive import AdaptiveScorer

    params = _parse_json_arg(args.params_json, "--params-json")
    context = _parse_json_arg(args.context_json, "--context-json")

    if args.require_review:
        # The deployer designates this call for mandatory human oversight
        # (EU AI Act Article 14): never auto-allow, never auto-deny, always
        # route to a human. The risk is still scored and recorded as evidence.
        scorer = AdaptiveScorer(threshold_allow=0.0, threshold_deny=1.01)
    else:
        allow_below, deny_above = _MODES[args.mode]
        scorer = AdaptiveScorer(
            threshold_allow=allow_below, threshold_deny=deny_above
        )

    backend, trail = _load_trail(args.audit_db)
    queue = ReviewQueue(args.queue_db)
    try:
        pipe = Pipeline(
            trail=trail, scorer=scorer, review_queue=queue,
            enforce=not args.shadow,
        )
        result = pipe.intercept(
            agent_id=args.agent_id,
            tool_name=args.tool,
            parameters=params,
            context=context or None,
            session_id=args.session_id,
        )
    finally:
        queue.close()
        backend.close()

    verdict = {
        "decision": result.decision,
        "allowed": result.allowed,
        "action_id": result.action_id,
        "risk_score": round(result.risk_score, 4),
        "risk_interval": [round(x, 4) for x in result.risk_interval],
        "reason": result.reason,
    }
    if result.decision == "escalate":
        verdict["next"] = (
            "STOP. This call needs human sign-off. Do not execute it. It is "
            "queued for review; a human resolves it with `vaara review resolve "
            f"--db {args.queue_db} --queue-id <id> --resolution allow|deny "
            f"--reviewer <who> --audit-db {args.audit_db}`. Re-check the queue "
            "before proceeding."
        )
    elif result.decision == "deny":
        verdict["next"] = "BLOCKED by policy. Do not execute. Tell the user why."
    else:
        verdict["next"] = (
            "Allowed. Execute the call, then record what it did with "
            "`governed_call.py outcome`."
        )
    print(json.dumps(verdict, indent=2))
    return _EXIT.get(result.decision, 1)


def cmd_outcome(args: argparse.Namespace) -> int:
    result = _parse_json_arg(args.result_json, "--result-json")

    backend, trail = _load_trail(args.audit_db)
    try:
        trail.record_execution(
            action_id=args.action_id,
            agent_id=args.agent_id,
            tool_name=args.tool,
            result=result,
        )
        if args.severity is not None:
            trail.record_outcome(
                action_id=args.action_id,
                agent_id=args.agent_id,
                tool_name=args.tool,
                outcome_severity=args.severity,
                description=args.description,
            )
    finally:
        backend.close()
    print(json.dumps({
        "action_id": args.action_id,
        "recorded": "execution" + ("+outcome" if args.severity is not None
                                    else ""),
        "audit_db": args.audit_db,
    }, indent=2))
    return 0


def cmd_export_jsonl(args: argparse.Namespace) -> int:
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    backend = SQLiteAuditBackend(args.audit_db)
    try:
        n = backend.export_jsonl(args.out)
    finally:
        backend.close()
    print(json.dumps({"trail_jsonl": args.out, "records": n}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gate", help="gate a proposed tool call before it runs")
    g.add_argument("--agent-id", required=True)
    g.add_argument("--tool", required=True, help="tool/function name")
    g.add_argument("--params-json", default="",
                   help="JSON object of the call arguments")
    g.add_argument("--context-json", default="",
                   help="JSON object of extra context (optional)")
    g.add_argument("--session-id", default="")
    g.add_argument("--audit-db", required=True,
                   help="SQLite path for the Article 12 audit trail")
    g.add_argument("--queue-db", required=True,
                   help="SQLite path for the Article 14 review queue")
    g.add_argument("--mode", choices=sorted(_MODES), default="strict",
                   help="risk thresholds (default: strict, escalate-on-doubt)")
    g.add_argument("--require-review", action="store_true",
                   help="designate this call for mandatory human oversight: "
                        "always escalate regardless of score (Article 14)")
    g.add_argument("--shadow", action="store_true",
                   help="record the decision but always allow (evidence-only)")
    g.set_defaults(func=cmd_gate)

    o = sub.add_parser("outcome", help="record what an allowed call did")
    o.add_argument("--action-id", required=True,
                   help="action_id returned by `gate`")
    o.add_argument("--agent-id", required=True)
    o.add_argument("--tool", required=True)
    o.add_argument("--result-json", default="",
                   help="JSON object summarizing the result")
    o.add_argument("--severity", type=float, default=None,
                   help="observed harm 0.0 safe .. 1.0 catastrophic (optional)")
    o.add_argument("--description", default="")
    o.add_argument("--audit-db", required=True)
    o.set_defaults(func=cmd_outcome)

    e = sub.add_parser("export-jsonl",
                       help="dump the audit trail to JSONL for export-article12")
    e.add_argument("--audit-db", required=True)
    e.add_argument("--out", required=True, help="path to write trail.jsonl")
    e.set_defaults(func=cmd_export_jsonl)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
