"""``vaara-audit`` command-line interface.

Subcommands:

    vaara-audit verify <trail.zip> [--pubkey PATH] [--json]
        Verify signature + hash-chain integrity of a signed trail zip.
        Wraps ``vaara.audit.verify.verify_signed`` and surfaces every check
        (manifest schema, jsonl hash, manifest hash, signature, chain walk).

    vaara-audit inspect <trail.zip> [--agent ID] [--event-type TYPE] \
                                    [--since TS] [--until TS] \
                                    [--limit N] [--json]
        Stream-decode ``trail.jsonl`` and print matching records. Filters
        compose (AND semantics). Default output is tabular (id, ts, agent,
        event_type, decision, action_type). ``--json`` emits one record
        per line.

    vaara-audit stats <trail.zip> [--group-by agent|event_type|decision|category] [--json]
        Aggregate the trail by a grouping key. Prints counts + percentages
        and (for decisions) mean risk. Intended as a one-shot sanity-check
        for conformance reporting.

    vaara-audit anomalies <trail.zip> [--rules RULES] [--json]
        Run heuristic anomaly detectors against the trail. Rules:
          - missing_completion: action_requested without matching decision_emitted
          - timestamp_regression: record with ts < previous record's ts
          - rate_burst: >N records in <T seconds for one agent
          - unknown_spike: unknown action_type fraction > threshold
        ``--rules all`` (default) enables every detector. Pass a
        comma-separated subset to narrow.

Exit codes:
    0  all checks passed / no findings
    1  verification failed or anomalies detected
    2  usage or IO error (bad path, zip not readable, etc.)

The CLI is deliberately read-only. It never mutates a trail and never
forwards data anywhere. Safe to run against production trails by any
third-party reviewer.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


# ── Trail loading helpers ─────────────────────────────────────────────

def _load_records_from_zip(zip_path: Path) -> tuple[list[dict], dict]:
    """Return (records, manifest) from a signed trail zip.

    Does NOT verify the signature. Callers who need that should call
    ``vaara-audit verify`` first or use ``vaara.audit.verify.verify_signed``.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"trail zip not found: {zip_path}")
    records: list[dict] = []
    manifest: dict = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        if "trail.jsonl" not in zf.namelist():
            raise ValueError(f"{zip_path} does not contain trail.jsonl")
        with zf.open("trail.jsonl") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        if "manifest.json" in zf.namelist():
            with zf.open("manifest.json") as f:
                manifest = json.load(f)
    return records, manifest


def _parse_ts(value) -> datetime:
    """Accept ISO 8601 (str), Unix epoch (float/int), or None; always returns aware UTC."""
    if value is None or value == "":
        return datetime.min.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    value = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _record_ts(record: dict) -> Optional[datetime]:
    ts = record.get("timestamp")
    if ts is None:
        ts = record.get("ts")
    if ts is None or ts == "":
        return None
    try:
        return _parse_ts(ts)
    except (ValueError, TypeError):
        return None


# ── Output helpers ────────────────────────────────────────────────────

def _emit_json(obj: Any, stream=None) -> None:
    stream = stream or sys.stdout
    json.dump(obj, stream, indent=2, default=str)
    stream.write("\n")


def _emit_table(rows: Iterable[list[str]], headers: list[str], stream=None) -> None:
    stream = stream or sys.stdout
    rows = list(rows)
    if not rows:
        stream.write("(no matching records)\n")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    def fmt_row(cells):
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))
    stream.write(fmt_row(headers) + "\n")
    stream.write("  ".join("-" * w for w in widths) + "\n")
    for row in rows:
        stream.write(fmt_row(row) + "\n")


# ── Subcommand: verify ────────────────────────────────────────────────

def cmd_verify(args) -> int:
    try:
        from .audit.verify import verify_signed
    except ImportError as exc:
        print(f"vaara-audit verify requires cryptography: {exc}", file=sys.stderr)
        return 2
    zip_path = Path(args.trail)
    if not zip_path.is_file():
        print(f"ERROR: trail zip not found: {zip_path}", file=sys.stderr)
        return 2
    pubkey = Path(args.pubkey) if args.pubkey else None
    result = verify_signed(zip_path, public_key=pubkey)
    payload = {
        "ok": bool(result.ok),
        "trail": str(zip_path),
        "manifest": result.manifest,
        "errors": list(result.errors),
    }
    if args.json:
        _emit_json(payload)
    else:
        status = "PASS" if payload["ok"] else "FAIL"
        print(f"trail:    {zip_path}")
        print(f"status:   [{status}]")
        if payload["manifest"]:
            mf = payload["manifest"]
            print(f"records:  {mf.get('record_count', '?')}")
            print(f"signer:   {mf.get('signer_pubkey_fingerprint', '?')[:16]}...")
        if payload["errors"]:
            print("errors:")
            for e in payload["errors"]:
                print(f"  - {e}")
    return 0 if payload["ok"] else 1


# ── Subcommand: inspect ───────────────────────────────────────────────

def cmd_inspect(args) -> int:
    try:
        records, _ = _load_records_from_zip(Path(args.trail))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    since = _parse_ts(args.since) if args.since else None
    until = _parse_ts(args.until) if args.until else None

    def matches(r: dict) -> bool:
        if args.agent and r.get("agent_id") != args.agent:
            return False
        if args.event_type and r.get("event_type") != args.event_type:
            return False
        ts = _record_ts(r)
        if since and (ts is None or ts < since):
            return False
        if until and (ts is None or ts > until):
            return False
        return True

    filtered = (r for r in records if matches(r))
    if args.limit:
        filtered = list(filtered)[: args.limit]
    else:
        filtered = list(filtered)

    if args.json:
        _emit_json({"n": len(filtered), "records": filtered})
        return 0
    headers = ["seq", "timestamp", "agent_id", "event_type", "decision", "action_type"]
    rows = []
    for r in filtered:
        payload = _payload(r)
        seq = r.get("sequence_position", payload.get("sequence_position", ""))
        rows.append([
            seq,
            r.get("timestamp") or r.get("ts", ""),
            r.get("agent_id", ""),
            r.get("event_type", ""),
            payload.get("decision", ""),
            payload.get("action_type") or payload.get("action_category") or "",
        ])
    _emit_table(rows, headers)
    print(f"\nmatched {len(filtered)} / {len(records)} records")
    return 0


# ── Subcommand: stats ─────────────────────────────────────────────────

def _payload(r: dict) -> dict:
    """Return the record's payload dict under either ``data`` or ``payload``."""
    return r.get("data") or r.get("payload") or {}


_GROUP_KEYS = {
    "agent": lambda r: r.get("agent_id", "(none)"),
    "event_type": lambda r: r.get("event_type", "(none)"),
    "decision": lambda r: _payload(r).get("decision", "(none)"),
    "category": lambda r: _payload(r).get("action_category") or _payload(r).get("category") or "(none)",
}


def cmd_stats(args) -> int:
    try:
        records, manifest = _load_records_from_zip(Path(args.trail))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    key_fn = _GROUP_KEYS.get(args.group_by)
    if key_fn is None:
        print(f"ERROR: unknown --group-by: {args.group_by}", file=sys.stderr)
        return 2
    counts: Counter[str] = Counter()
    risk_sum: dict[str, float] = defaultdict(float)
    risk_n: dict[str, int] = defaultdict(int)
    for r in records:
        k = key_fn(r)
        counts[k] += 1
        payload = _payload(r)
        risk = payload.get("risk_score") or payload.get("base_risk_score")
        if isinstance(risk, (int, float)):
            risk_sum[k] += float(risk)
            risk_n[k] += 1
    total = sum(counts.values())
    rows = []
    summary = []
    for k, n in counts.most_common():
        pct = (n / total * 100) if total else 0.0
        mean_risk = (risk_sum[k] / risk_n[k]) if risk_n[k] else None
        rows.append([
            k,
            n,
            f"{pct:.1f}%",
            f"{mean_risk:.3f}" if mean_risk is not None else "-",
        ])
        summary.append({
            "key": k,
            "count": n,
            "pct": round(pct, 2),
            "mean_risk": round(mean_risk, 4) if mean_risk is not None else None,
        })
    if args.json:
        _emit_json({
            "group_by": args.group_by,
            "total": total,
            "manifest": manifest,
            "groups": summary,
        })
        return 0
    print(f"trail: {args.trail}")
    print(f"total records: {total}")
    print(f"groups by {args.group_by}:\n")
    _emit_table(rows, [args.group_by, "n", "pct", "mean_risk"])
    return 0


# ── Subcommand: anomalies ─────────────────────────────────────────────

DEFAULT_RATE_BURST_WINDOW_S = 10
DEFAULT_RATE_BURST_N = 20
DEFAULT_UNKNOWN_FRAC = 0.25
DEFAULT_UNKNOWN_WINDOW = 50


def _rule_missing_completion(records: list[dict]) -> list[dict]:
    """Every action_requested should have a matching decision_emitted.

    Hash-chain integrity is already checked by the ``verify`` subcommand
    via the cryptographic path. This rule looks at the semantic lifecycle:
    an action that was requested but never saw a decision is a signal
    that enforcement crashed, the trail was truncated, or the gate was
    bypassed. All three are anomalies worth surfacing.
    """
    findings = []
    event_types = {r.get("event_type") for r in records}
    if "decision_emitted" not in event_types:
        # Trail is request-log only (no decisions anywhere). Not enough
        # semantic context to call missing completions an anomaly.
        return findings
    by_action: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        action_id = r.get("action_id")
        if not action_id:
            continue
        by_action[action_id][r.get("event_type", "")] = r
    for action_id, events in by_action.items():
        if "action_requested" in events and "decision_emitted" not in events:
            req = events["action_requested"]
            findings.append({
                "rule": "missing_completion",
                "agent_id": req.get("agent_id"),
                "action_id": action_id,
                "tool_name": req.get("tool_name"),
                "record_id": req.get("record_id"),
            })
    return findings


def _rule_timestamp_regression(records: list[dict]) -> list[dict]:
    findings = []
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        aid = r.get("agent_id") or "(none)"
        by_agent[aid].append(r)
    for aid, rs in by_agent.items():
        rs = sorted(rs, key=lambda r: _payload(r).get("sequence_position", r.get("sequence_position", -1)))
        prev_ts: Optional[datetime] = None
        for r in rs:
            ts = _record_ts(r)
            if ts and prev_ts and ts < prev_ts:
                findings.append({
                    "rule": "timestamp_regression",
                    "agent_id": aid,
                    "record_id": r.get("record_id"),
                    "observed_ts": str(ts),
                    "previous_ts": str(prev_ts),
                    "delta_seconds": (ts - prev_ts).total_seconds(),
                })
            if ts:
                prev_ts = ts
    return findings


def _rule_rate_burst(
    records: list[dict],
    window_s: float = DEFAULT_RATE_BURST_WINDOW_S,
    threshold: int = DEFAULT_RATE_BURST_N,
) -> list[dict]:
    findings = []
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        aid = r.get("agent_id") or "(none)"
        by_agent[aid].append(r)
    for aid, rs in by_agent.items():
        rs_ts = sorted([
            (_record_ts(r), r)
            for r in rs
            if _record_ts(r) is not None
        ], key=lambda x: x[0])
        # Sliding-window count
        left = 0
        for right in range(len(rs_ts)):
            while rs_ts[right][0] - rs_ts[left][0] > __import__("datetime").timedelta(seconds=window_s):
                left += 1
            count = right - left + 1
            if count >= threshold:
                findings.append({
                    "rule": "rate_burst",
                    "agent_id": aid,
                    "window_seconds": window_s,
                    "count": count,
                    "threshold": threshold,
                    "window_start": str(rs_ts[left][0]),
                    "window_end": str(rs_ts[right][0]),
                })
                # Avoid N^2 flood: jump past the burst window
                left = right + 1
    return findings


def _rule_unknown_spike(
    records: list[dict],
    window: int = DEFAULT_UNKNOWN_WINDOW,
    frac_threshold: float = DEFAULT_UNKNOWN_FRAC,
) -> list[dict]:
    findings = []
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        aid = r.get("agent_id") or "(none)"
        by_agent[aid].append(r)
    for aid, rs in by_agent.items():
        rs = sorted(rs, key=lambda r: _payload(r).get("sequence_position", r.get("sequence_position", -1)))
        if len(rs) < window:
            continue
        for i in range(window, len(rs) + 1):
            slab = rs[i - window : i]
            unknowns = sum(
                1 for r in slab
                if (_payload(r).get("action_type") or _payload(r).get("action_category") or "unknown").lower() == "unknown"
            )
            frac = unknowns / window
            if frac >= frac_threshold:
                findings.append({
                    "rule": "unknown_spike",
                    "agent_id": aid,
                    "window": window,
                    "unknown_fraction": round(frac, 3),
                    "threshold": frac_threshold,
                    "at_sequence": _payload(slab[-1]).get("sequence_position", slab[-1].get("sequence_position")),
                })
                break  # One finding per agent per call
    return findings


RULES = {
    "missing_completion": _rule_missing_completion,
    "timestamp_regression": _rule_timestamp_regression,
    "rate_burst": _rule_rate_burst,
    "unknown_spike": _rule_unknown_spike,
}


def cmd_anomalies(args) -> int:
    try:
        records, _ = _load_records_from_zip(Path(args.trail))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.rules == "all":
        selected = list(RULES.keys())
    else:
        selected = [r.strip() for r in args.rules.split(",") if r.strip()]
        unknown = [r for r in selected if r not in RULES]
        if unknown:
            print(f"ERROR: unknown rules: {unknown}. valid: {list(RULES)}", file=sys.stderr)
            return 2
    findings: list[dict] = []
    for name in selected:
        findings.extend(RULES[name](records))
    if args.json:
        _emit_json({
            "trail": args.trail,
            "rules": selected,
            "n_records": len(records),
            "n_findings": len(findings),
            "findings": findings,
        })
    else:
        print(f"trail: {args.trail}")
        print(f"rules: {selected}")
        print(f"records: {len(records)}")
        print(f"findings: {len(findings)}\n")
        if findings:
            for f in findings:
                rule = f.pop("rule")
                details = ", ".join(f"{k}={v}" for k, v in f.items())
                print(f"[{rule}] {details}")
        else:
            print("(no anomalies)")
    return 0 if not findings else 1


# ── Parser + main ─────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vaara-audit",
        description="Third-party auditor CLI for Vaara signed trails.",
    )
    p.add_argument("--version", action="version", version="vaara-audit 0.1.0")
    sub = p.add_subparsers(dest="command", required=True)

    pv = sub.add_parser("verify", help="Verify signature + hash chain.")
    pv.add_argument("trail", help="Path to trail zip.")
    pv.add_argument("--pubkey", help="Path to Ed25519 public key (PEM).")
    pv.add_argument("--json", action="store_true", help="Emit JSON.")
    pv.set_defaults(func=cmd_verify)

    pi = sub.add_parser("inspect", help="Print filtered records.")
    pi.add_argument("trail", help="Path to trail zip.")
    pi.add_argument("--agent", help="Filter by agent_id.")
    pi.add_argument("--event-type", help="Filter by event_type.")
    pi.add_argument("--since", help="ISO8601 lower bound.")
    pi.add_argument("--until", help="ISO8601 upper bound.")
    pi.add_argument("--limit", type=int, default=0, help="Max records (0 = no limit).")
    pi.add_argument("--json", action="store_true")
    pi.set_defaults(func=cmd_inspect)

    ps = sub.add_parser("stats", help="Aggregate counts + percentages.")
    ps.add_argument("trail", help="Path to trail zip.")
    ps.add_argument("--group-by", default="event_type", choices=list(_GROUP_KEYS))
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=cmd_stats)

    pa = sub.add_parser("anomalies", help="Apply heuristic anomaly rules.")
    pa.add_argument("trail", help="Path to trail zip.")
    pa.add_argument(
        "--rules",
        default="all",
        help="Comma-separated rule names or 'all'. Rules: " + ", ".join(RULES),
    )
    pa.add_argument("--json", action="store_true")
    pa.set_defaults(func=cmd_anomalies)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
