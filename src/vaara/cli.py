"""Vaara command-line interface.

Subcommands:

    vaara keygen --dev --out PATH
        Generate an Ed25519 keypair for local evaluation / demo signing.
        For production, use keys from your KMS/HSM — see docs/signing-keys.md.

    vaara trail export --trail PATH --out PATH --key PATH
        Export a signed, regulator-handoff zip from a saved trail.

    vaara trail verify --zip PATH [--pubkey PATH]
        Verify a signed trail zip.

    vaara trail export-prov --trail PATH --out PATH [--action-id ID] [--no-chain]
        Export the trail (or one action's slice) as W3C PROV-JSON.

    vaara trail export-incident --trail PATH --incident-meta PATH --out PATH \
                                 [--trigger-record-id ID]
        Export an EU AI Act Article 73 serious-incident report (INTERIM
        format pending the Commission template). Operator metadata
        (severity, ai_system, reporter, recipient, ...) is supplied via a
        JSON file; trigger and evidence records come from the trail.

    vaara review list --db PATH [--status {pending,claimed,resolved,expired}]
                       [--limit N] [--agent-id ID]
        Human-in-the-loop review queue. Lists pending escalations by
        default with their conformal intervals (EU AI Act Article 14).

    vaara review show --db PATH --queue-id ID
        Print the full queue item as JSON.

    vaara review claim --db PATH --queue-id ID --reviewer NAME
        Mark a pending item as claimed by ``reviewer`` (optimistic).

    vaara review resolve --db PATH --queue-id ID --reviewer NAME \
                          --resolution {allow,deny,abstain} \
                          [--justification TEXT]
        Mark a pending or claimed item as resolved. The ``allow`` and
        ``deny`` resolutions are operator decisions overriding the
        pipeline's ``escalate`` verdict; ``abstain`` leaves
        ``escalate`` as final.

    vaara review expire --db PATH --timeout-seconds N [--dry-run]
        Mark pending items older than ``timeout-seconds`` as expired.
        Claimed items are left alone.

    vaara policy validate POLICY_PATH [--json]
        Load a YAML/JSON policy and run semantic checks. Exit 0 if no
        errors. Warnings (narrow threshold bands, dangling overrides,
        unreachable escalation routes, missing default route, sequence
        steps not naming a declared action class) print without
        flipping the exit code.

    vaara policy test POLICY_PATH --cases CASES_PATH [--json]
        Run a YAML/JSON cases file against a policy (Conftest analog).
        Each case names an action_class, a risk_score, optional
        matched_sequences, and an expected verdict / route. Exit 0 if
        every case passes.

    vaara policy reload POLICY_PATH [--server URL] [--inline] [--format json|yaml]
        Trigger an atomic policy reload on a running ``vaara serve``
        process. The new thresholds and sequence patterns take effect
        on the next ``evaluate()``; in-flight calls keep the old ones.

    vaara version
        Print the installed Vaara version.

Installing the CLI requires no extra deps. ``keygen``/``export``/``verify``
require ``pip install 'vaara[export]'`` (pulls cryptography).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Optional

from vaara import __version__

_INSTALL_HINT = (
    "This command requires the 'cryptography' package. "
    "Install with: pip install 'vaara[export]'"
)


def _cmd_version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_keygen(args: argparse.Namespace) -> int:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return 2

    out = Path(args.out).expanduser()
    pub_out = out.with_suffix(out.suffix + ".pub")

    if out.exists() and not args.force:
        print(
            f"refusing to overwrite existing key at {out} "
            f"(pass --force to overwrite)",
            file=sys.stderr,
        )
        return 2

    if not args.dev:
        print(
            "Refusing to generate a key without --dev. "
            "vaara keygen is a convenience helper for local evaluation and demo "
            "recording only. For production, use keys from your KMS/HSM/Vault — "
            "see docs/signing-keys.md.\n\n"
            "If you understand and want to proceed anyway, re-run with --dev.",
            file=sys.stderr,
        )
        return 2

    key = Ed25519PrivateKey.generate()
    priv_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(priv_pem)
    try:
        os.chmod(out, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        pass
    pub_out.write_bytes(pub_pem)

    raw_pub = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    fingerprint = hashlib.sha256(raw_pub).hexdigest()[:32]

    print("Generated Ed25519 keypair (evaluation/demo use only)")
    print(f"  private key: {out}        (0600)")
    print(f"  public key:  {pub_out}")
    print(f"  fingerprint: {fingerprint}")
    print()
    print(
        "REMINDER: this key was generated on disk with no password and no HSM. "
        "Do not use it to sign trails you submit to a regulator. For production, "
        "see docs/signing-keys.md."
    )
    return 0


def _cmd_trail_export(args: argparse.Namespace) -> int:
    try:
        from vaara.audit.export import export_signed
        from vaara.audit.trail import AuditTrail
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return 2

    trail_path = Path(args.trail).expanduser()
    if not trail_path.exists():
        print(f"trail JSONL not found: {trail_path}", file=sys.stderr)
        return 2

    trail = AuditTrail()
    from vaara.audit.trail import AuditRecord
    with open(trail_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = AuditRecord.from_dict(json.loads(line))
            trail._records.append(rec)
            trail._by_action[rec.action_id].append(rec)
            if rec.record_hash:
                trail._last_hash = rec.record_hash

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    result = export_signed(
        trail,
        out_path=out,
        signer_key=Path(args.key).expanduser(),
        agent_id=args.agent_id or "",
    )

    print(f"Exported signed trail to {result.path}")
    print(f"  records:      {result.manifest['record_count']}")
    print(f"  chain intact: {result.chain_intact}")
    print(f"  fingerprint:  {result.manifest['signer_pubkey_fingerprint']}")
    return 0 if result.chain_intact else 1


def _cmd_trail_verify(args: argparse.Namespace) -> int:
    try:
        from vaara.audit.verify import verify_signed
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return 2

    pubkey = Path(args.pubkey).expanduser() if args.pubkey else None
    result = verify_signed(Path(args.zip).expanduser(), public_key=pubkey)

    if result.manifest:
        print("Manifest:")
        print(f"  schema:       {result.manifest.get('schema_version')}")
        print(f"  records:      {result.manifest.get('record_count')}")
        print(f"  fingerprint:  {result.manifest.get('signer_pubkey_fingerprint')}")
        print(f"  created:      {result.manifest.get('created_utc')}")
        print(f"  vaara:        {result.manifest.get('vaara_version')}")
    if result.ok:
        print("Verification: OK")
        return 0
    print("Verification: FAILED", file=sys.stderr)
    for e in result.errors:
        print(f"  - {e}", file=sys.stderr)
    return 1


def _cmd_trail_export_prov(args: argparse.Namespace) -> int:
    """Export the trail as W3C PROV-JSON (no signing, no extra deps)."""
    from vaara.audit.prov_export import write_prov_json
    from vaara.audit.trail import AuditRecord

    trail_path = Path(args.trail).expanduser()
    if not trail_path.exists():
        print(f"trail JSONL not found: {trail_path}", file=sys.stderr)
        return 2

    records: list[AuditRecord] = []
    with open(trail_path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(AuditRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                print(
                    f"invalid trail JSONL at line {lineno}: {exc}",
                    file=sys.stderr,
                )
                return 2

    out = Path(args.out).expanduser()
    n = write_prov_json(
        records, out,
        action_id=args.action_id,
        include_chain=not args.no_chain,
    )
    scope = f"action {args.action_id}" if args.action_id else "full trail"
    print(f"Exported {n} records ({scope}) to PROV-JSON: {out}")
    return 0


def _cmd_trail_export_incident(args: argparse.Namespace) -> int:
    """Export an Article 73 serious-incident report from a trail JSONL."""
    from vaara.audit.incident_export import (
        build_from_trail,
        write_incident_report,
    )
    from vaara.audit.trail import AuditRecord

    trail_path = Path(args.trail).expanduser()
    if not trail_path.exists():
        print(f"trail JSONL not found: {trail_path}", file=sys.stderr)
        return 2

    meta_path = Path(args.incident_meta).expanduser()
    if not meta_path.exists():
        print(f"incident-meta JSON not found: {meta_path}", file=sys.stderr)
        return 2
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            incident_meta = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"failed to read incident-meta: {exc}", file=sys.stderr)
        return 2

    records: list[AuditRecord] = []
    with open(trail_path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(AuditRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                print(
                    f"invalid trail JSONL at line {lineno}: {exc}",
                    file=sys.stderr,
                )
                return 2

    try:
        report = build_from_trail(
            records,
            incident_meta=incident_meta,
            trigger_record_id=args.trigger_record_id,
        )
    except (ValueError, KeyError) as exc:
        print(f"failed to build incident report: {exc}", file=sys.stderr)
        return 2

    out = Path(args.out).expanduser()
    write_incident_report(report, out)
    print(
        f"Exported Article 73 incident report ({report['report_status']}, "
        f"deadline {report['regulation']['reporting_deadline_days']} days) "
        f"to {out}"
    )
    return 0


def _cmd_trail_purge(args: argparse.Namespace) -> int:
    """Delete audit records older than --retention-days. EU AI Act Article 12(2)."""
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(f"audit DB not found: {db_path}", file=sys.stderr)
        return 1

    if args.retention_days <= 0:
        print(
            f"--retention-days must be > 0, got {args.retention_days}",
            file=sys.stderr,
        )
        return 2

    retention_seconds = args.retention_days * 86400
    # --all-tenants means tenant_id="" which routes to _tenant_clause() == "1=1",
    # i.e. purge across all records regardless of tenant_id. --tenant TID scopes
    # to that tenant only via parameterised "tenant_id = ?".
    tenant_scope = "" if args.all_tenants else args.tenant
    backend = SQLiteAuditBackend(db_path, tenant_id=tenant_scope)
    try:
        count = backend.purge_older_than(retention_seconds, dry_run=args.dry_run)
    finally:
        backend.close()

    if args.dry_run:
        print(f"Would purge {count} record(s) older than {args.retention_days} day(s).")
        print("Run without --dry-run to apply.")
    else:
        print(f"Purged {count} record(s) older than {args.retention_days} day(s).")
        if count > 0:
            print(
                "Note: hash chain has a seam at the retention boundary. "
                "Run 'vaara trail export' BEFORE future purges to preserve "
                "audit history in a signed zip.",
                file=sys.stderr,
            )
    return 0


def _open_review_queue(db_arg: str):
    from vaara.audit.review_queue import ReviewQueue
    db_path = Path(db_arg).expanduser()
    return ReviewQueue(db_path)


def _format_review_row(item) -> str:
    half = (item.conformal_upper - item.conformal_lower) / 2.0
    age = max(0.0, time.time() - item.enqueued_at)
    return (
        f"{item.queue_id[:8]}  {item.status:8s}  "
        f"score={item.risk_score:.3f}  "
        f"[{item.conformal_lower:.3f},{item.conformal_upper:.3f}] "
        f"±{half:.3f}  age={int(age)}s  "
        f"{item.agent_id}  {item.tool_name}"
    )


def _cmd_review_list(args: argparse.Namespace) -> int:
    status = args.status if args.status != "any" else None
    with _open_review_queue(args.db) as q:
        items = q.list_items(
            status=status, limit=args.limit, agent_id=args.agent_id,
        )
        counts = q.counts()
    if not items:
        print(f"No items match (status={status!r}). counts={counts}")
        return 0
    print(f"queue items (status={status!r}, n={len(items)}, counts={counts}):")
    for item in items:
        print("  " + _format_review_row(item))
    return 0


def _cmd_review_show(args: argparse.Namespace) -> int:
    from vaara.audit.review_queue import ItemNotFoundError
    with _open_review_queue(args.db) as q:
        try:
            item = q.get(args.queue_id)
        except ItemNotFoundError as exc:
            print(f"not found: {exc}", file=sys.stderr)
            return 2
    print(json.dumps(item.to_dict(), indent=2, sort_keys=False))
    return 0


def _cmd_review_claim(args: argparse.Namespace) -> int:
    from vaara.audit.review_queue import (
        InvalidTransitionError, ItemNotFoundError,
    )
    with _open_review_queue(args.db) as q:
        try:
            item = q.claim(args.queue_id, reviewer=args.reviewer)
        except (InvalidTransitionError, ItemNotFoundError) as exc:
            print(f"claim failed: {exc}", file=sys.stderr)
            return 2
    print(f"claimed {item.queue_id} by {item.claimed_by} at "
          f"{int(item.claimed_at or 0)}")
    return 0


def _cmd_review_resolve(args: argparse.Namespace) -> int:
    from vaara.audit.review_queue import (
        InvalidTransitionError, ItemNotFoundError,
    )
    trail = None
    backend = None
    if args.audit_db:
        from vaara.audit.sqlite_backend import SQLiteAuditBackend
        from vaara.audit.trail import AuditTrail
        backend = SQLiteAuditBackend(Path(args.audit_db).expanduser())
        trail = AuditTrail(on_record=backend.write_record)
    try:
        with _open_review_queue(args.db) as q:
            try:
                item = q.resolve(
                    args.queue_id,
                    reviewer=args.reviewer,
                    resolution=args.resolution,
                    justification=args.justification or "",
                    trail=trail,
                )
            except (InvalidTransitionError, ItemNotFoundError) as exc:
                print(f"resolve failed: {exc}", file=sys.stderr)
                return 2
            except ValueError as exc:
                print(f"resolve failed: {exc}", file=sys.stderr)
                return 2
    finally:
        if backend is not None:
            backend.close()
    print(
        f"resolved {item.queue_id} resolution={item.resolution} "
        f"reviewer={item.resolved_by}"
    )
    if trail is not None:
        print("Wrote ESCALATION_RESOLVED to audit trail (Article 14(4)(d)).")
    return 0


def _cmd_review_expire(args: argparse.Namespace) -> int:
    with _open_review_queue(args.db) as q:
        try:
            n = q.expire_stale(
                timeout_seconds=args.timeout_seconds, dry_run=args.dry_run,
            )
        except ValueError as exc:
            print(f"expire failed: {exc}", file=sys.stderr)
            return 2
    verb = "Would expire" if args.dry_run else "Expired"
    print(f"{verb} {n} pending item(s) older than "
          f"{args.timeout_seconds} second(s).")
    return 0


def _render_validation_report(report, *, source_label: str, as_json: bool) -> str:
    if as_json:
        return json.dumps(report.to_dict(), indent=2)
    if not report.issues:
        return f"{source_label}: ok (no issues)"
    lines = [
        f"  [{i.level.value}] {i.code}"
        f"{(' at ' + i.path) if i.path else ''}: {i.message}"
        for i in report.issues
    ]
    header = (
        f"{source_label}: {len(report.errors)} error(s), "
        f"{len(report.warnings)} warning(s)"
    )
    return header + "\n" + "\n".join(lines)


def _cmd_policy_validate(args: argparse.Namespace) -> int:
    from vaara.policy.validate import validate_source

    policy_path = Path(args.policy).expanduser()
    _policy, report = validate_source(policy_path)
    print(_render_validation_report(
        report, source_label=str(policy_path), as_json=args.json,
    ))
    return 0 if report.ok else 1


def _render_test_results(results, *, as_json: bool) -> str:
    if as_json:
        return json.dumps({
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "results": [r.to_dict() for r in results],
        }, indent=2)
    lines = []
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        suffix = "" if r.passed else f" — {r.diagnostic}"
        lines.append(f"  [{mark}] {r.case.name}{suffix}")
    failed = sum(1 for r in results if not r.passed)
    header = f"{len(results)} case(s), {len(results) - failed} passed, {failed} failed"
    return header + "\n" + "\n".join(lines)


def _cmd_policy_test(args: argparse.Namespace) -> int:
    from vaara.policy.test_cases import run_test_cases
    from vaara.policy.test_cases_io import load_test_cases
    from vaara.policy.validate import validate_source

    policy_path = Path(args.policy).expanduser()
    cases_path = Path(args.cases).expanduser()

    policy, report = validate_source(policy_path)
    if policy is None:
        print(_render_validation_report(
            report, source_label=str(policy_path), as_json=args.json,
        ), file=sys.stderr)
        return 2

    try:
        cases = load_test_cases(cases_path)
    except Exception as e:
        print(f"failed to load cases from {cases_path}: {e}", file=sys.stderr)
        return 2

    results = run_test_cases(policy, cases)
    print(_render_test_results(results, as_json=args.json))
    return 0 if all(r.passed for r in results) else 1


def _cmd_trail_receipt(args: argparse.Namespace) -> int:
    from vaara.audit.receipts import extract_receipt, verify_receipt
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(f"audit DB not found: {db_path}", file=sys.stderr)
        return 2

    backend = SQLiteAuditBackend(str(db_path))
    try:
        trail = backend.load_trail()
    except Exception as exc:
        print(f"failed to load audit trail: {exc}", file=sys.stderr)
        return 2

    receipt = extract_receipt(trail, args.action_id)
    if receipt is None:
        print(
            f"no decision record found for action_id {args.action_id!r}",
            file=sys.stderr,
        )
        return 1

    if not verify_receipt(receipt):
        print(
            "receipt verification failed — derived hashes do not match payloads",
            file=sys.stderr,
        )
        return 1

    text = json.dumps(receipt.to_dict(), indent=2, sort_keys=False)
    if args.out:
        Path(args.out).expanduser().write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def _cmd_compliance_dashboard(args: argparse.Namespace) -> int:
    from vaara.audit.sqlite_backend import SQLiteAuditTrail
    from vaara.compliance.dashboard import render_html
    from vaara.compliance.engine import ComplianceEngine

    db_path = Path(args.db).expanduser()
    if not db_path.is_file():
        print(f"vaara compliance dashboard: not a file: {db_path}", file=sys.stderr)
        return 2

    trail = SQLiteAuditTrail(str(db_path))
    engine = ComplianceEngine()
    report = engine.assess(
        trail=trail,
        system_name=args.system_name,
        system_version=args.system_version,
    )
    out = Path(args.out).expanduser()
    if out.is_dir() or args.out.endswith("/"):
        out.mkdir(parents=True, exist_ok=True)
        out = out / "index.html"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(report), encoding="utf-8")
    print(str(out))
    return 0


def _cmd_compliance_report(args: argparse.Namespace) -> int:
    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.compliance.engine import create_default_engine
    from vaara.compliance.render import (
        render_json,
        render_markdown,
        render_narrative,
    )

    db_path = Path(args.db).expanduser()
    if not db_path.exists():
        print(f"audit DB not found: {db_path}", file=sys.stderr)
        return 2

    backend = SQLiteAuditBackend(str(db_path))
    try:
        trail = backend.load_trail()
    except Exception as exc:
        print(f"failed to load audit trail: {exc}", file=sys.stderr)
        return 2

    engine = create_default_engine()
    report = engine.assess(
        trail,
        system_name=args.system_name,
        system_version=args.system_version,
    )

    if args.format == "md":
        text = render_markdown(report)
    elif args.format == "json":
        text = render_json(report)
    elif args.format == "narrative":
        text = render_narrative(report)
    else:
        print(f"unknown format: {args.format}", file=sys.stderr)
        return 2

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def _read_text_input(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        return Path(args.file).expanduser().read_text(encoding="utf-8")
    if args.stdin:
        return sys.stdin.read()
    raise SystemExit(
        "vaara detect: supply --text, --file PATH, or --stdin"
    )


def _cmd_detect_injection(args: argparse.Namespace) -> int:
    from vaara.detect import detect_injection

    text = _read_text_input(args)
    result = detect_injection(text, threshold=args.threshold)
    print(json.dumps(result.to_dict(), indent=2 if args.pretty else None))
    return 1 if result.detected else 0


def _cmd_detect_pii(args: argparse.Namespace) -> int:
    from vaara.detect import detect_pii

    text = _read_text_input(args)
    result = detect_pii(text)
    print(json.dumps(result.to_dict(), indent=2 if args.pretty else None))
    return 1 if result.detected else 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "vaara serve requires the server extra. "
            "Install with: pip install 'vaara[server]'",
            file=sys.stderr,
        )
        return 2

    from vaara.server import create_app

    controller = None
    policy_path = getattr(args, "policy", None)
    if policy_path:
        from vaara.policy.controller import PolicyController
        from vaara.policy.validate import validate_source

        policy_obj, report = validate_source(Path(policy_path).expanduser())
        if policy_obj is None:
            print(
                f"vaara serve: policy {policy_path} failed to parse:",
                file=sys.stderr,
            )
            for issue in report.issues:
                print(f"  {issue.level.value}: {issue.message}", file=sys.stderr)
            return 2
        controller = PolicyController(policy_obj)

    app = create_app(policy_controller=controller)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _cmd_policy_reload(args: argparse.Namespace) -> int:
    import json as _json
    import urllib.error
    import urllib.request

    policy_path = Path(args.policy).expanduser().resolve()
    if not policy_path.is_file():
        print(f"vaara policy reload: not a file: {policy_path}", file=sys.stderr)
        return 2

    fmt: Optional[str] = args.format
    if fmt is None:
        if policy_path.suffix in (".yaml", ".yml"):
            fmt = "yaml"
        elif policy_path.suffix == ".json":
            fmt = "json"

    if args.inline:
        # Send the parsed body so the server doesn't need filesystem access
        # to the same path the operator is reading.
        from vaara.policy.loader import from_dict as _from_dict  # noqa: F401
        if fmt == "yaml":
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError:
                print(
                    "vaara policy reload --inline on a YAML file needs the "
                    "yaml extra. Install with: pip install 'vaara[yaml]'",
                    file=sys.stderr,
                )
                return 2
            body = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        else:
            body = _json.loads(policy_path.read_text(encoding="utf-8"))
        payload = {"body": body}
        if fmt is not None:
            payload["format"] = fmt
    else:
        payload = {"path": str(policy_path)}
        if fmt is not None:
            payload["format"] = fmt

    url = args.server.rstrip("/") + "/v1/policy/reload"
    req = urllib.request.Request(
        url,
        data=_json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body_bytes = resp.read()
    except urllib.error.HTTPError as exc:
        print(f"vaara policy reload: HTTP {exc.code}", file=sys.stderr)
        try:
            print(exc.read().decode("utf-8"), file=sys.stderr)
        except Exception:
            pass
        return 1
    except urllib.error.URLError as exc:
        print(
            f"vaara policy reload: cannot reach {url}: {exc.reason}",
            file=sys.stderr,
        )
        return 1

    print(body_bytes.decode("utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vaara", description="Vaara AI Agent Execution Layer")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="Print Vaara version").set_defaults(func=_cmd_version)

    pk = sub.add_parser(
        "keygen",
        help="Generate an Ed25519 keypair (evaluation/demo use only)",
    )
    pk.add_argument("--out", required=True, help="Path to write the private key (PEM)")
    pk.add_argument(
        "--dev",
        action="store_true",
        help="Required. Acknowledges this key is for local evaluation, not production.",
    )
    pk.add_argument("--force", action="store_true", help="Overwrite an existing key file")
    pk.set_defaults(func=_cmd_keygen)

    pt = sub.add_parser("trail", help="Audit-trail commands")
    tsub = pt.add_subparsers(dest="trail_cmd", required=True)

    pe = tsub.add_parser("export", help="Export a signed, regulator-handoff trail zip")
    pe.add_argument("--trail", required=True, help="Path to trail JSONL file")
    pe.add_argument("--out", required=True, help="Path to write the signed zip")
    pe.add_argument("--key", required=True, help="Path to Ed25519 signing private key (PEM)")
    pe.add_argument("--agent-id", default="", help="Optional agent_id tag for the manifest")
    pe.set_defaults(func=_cmd_trail_export)

    pv = tsub.add_parser("verify", help="Verify a signed trail zip")
    pv.add_argument("--zip", required=True, help="Path to signed trail zip")
    pv.add_argument(
        "--pubkey",
        default=None,
        help="Path to Ed25519 public key (PEM). If omitted, uses signer_pubkey.pem from inside the zip.",
    )
    pv.set_defaults(func=_cmd_trail_verify)

    pep = tsub.add_parser(
        "export-prov",
        help="Export the trail as W3C PROV-JSON (no signing, zero extra deps)",
    )
    pep.add_argument("--trail", required=True, help="Path to trail JSONL file")
    pep.add_argument("--out", required=True, help="Path to write the PROV-JSON output")
    pep.add_argument(
        "--action-id", default=None,
        help="If given, emit only that action's bundle",
    )
    pep.add_argument(
        "--no-chain", action="store_true",
        help="Omit the audit-record chain layer",
    )
    pep.set_defaults(func=_cmd_trail_export_prov)

    pei = tsub.add_parser(
        "export-incident",
        help="Export an EU AI Act Article 73 serious-incident report (INTERIM)",
    )
    pei.add_argument("--trail", required=True, help="Path to trail JSONL file")
    pei.add_argument(
        "--incident-meta",
        required=True,
        help="Path to JSON file with operator-supplied incident metadata "
             "(severity, ai_system, causal_link, reporter, recipient, ...)",
    )
    pei.add_argument("--out", required=True, help="Path to write the incident report JSON")
    pei.add_argument(
        "--trigger-record-id",
        default=None,
        help="record_id of the audit event that triggered the report. "
             "If omitted, the most recent outcome_recorded / action_blocked / "
             "policy_override event in the trail is used.",
    )
    pei.set_defaults(func=_cmd_trail_export_incident)

    prec = tsub.add_parser(
        "receipt",
        help="Extract an Article 12 commit-prove receipt pair for an action",
    )
    prec.add_argument("--db", required=True, help="Path to the audit SQLite DB")
    prec.add_argument("--action-id", required=True, help="action_id to extract")
    prec.add_argument("--out", default=None, help="Write to file (default: stdout)")
    prec.set_defaults(func=_cmd_trail_receipt)

    pp = tsub.add_parser(
        "purge",
        help="Delete audit records older than the retention period (Article 12(2))",
    )
    pp.add_argument("--db", required=True, help="Path to the audit SQLite DB")
    pp.add_argument(
        "--retention-days",
        required=True,
        type=int,
        help="Records older than this many days are deleted",
    )
    pp.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the count without modifying the DB",
    )
    # Tenant scoping is required: a shared multi-tenant audit DB must not be
    # silently purged across all tenants. Operator picks --tenant TID for a
    # single tenant or --all-tenants explicitly.
    scope = pp.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--tenant",
        help="Restrict purge to records with this tenant_id",
    )
    scope.add_argument(
        "--all-tenants",
        action="store_true",
        help="Purge across all tenants in this DB. Use only on single-tenant deployments or after deliberate review.",
    )
    pp.set_defaults(func=_cmd_trail_purge)

    pr = sub.add_parser(
        "review",
        help="Human-in-the-loop review queue (EU AI Act Article 14)",
    )
    rsub = pr.add_subparsers(dest="review_cmd", required=True)

    rl = rsub.add_parser("list", help="List queue items (default: pending)")
    rl.add_argument("--db", required=True, help="Path to the review queue DB")
    rl.add_argument(
        "--status", default="pending",
        choices=["pending", "claimed", "resolved", "expired", "any"],
        help="Filter by status. ``any`` lists across all statuses.",
    )
    rl.add_argument("--limit", type=int, default=50, help="Max rows (cap 1000)")
    rl.add_argument("--agent-id", default=None, help="Restrict to one agent_id")
    rl.set_defaults(func=_cmd_review_list)

    rsh = rsub.add_parser("show", help="Print one queue item as JSON")
    rsh.add_argument("--db", required=True, help="Path to the review queue DB")
    rsh.add_argument("--queue-id", required=True, help="queue_id to show")
    rsh.set_defaults(func=_cmd_review_show)

    rc = rsub.add_parser("claim", help="Claim a pending item")
    rc.add_argument("--db", required=True, help="Path to the review queue DB")
    rc.add_argument("--queue-id", required=True, help="queue_id to claim")
    rc.add_argument("--reviewer", required=True, help="Operator identifier")
    rc.set_defaults(func=_cmd_review_claim)

    rr = rsub.add_parser(
        "resolve", help="Resolve a pending or claimed item",
    )
    rr.add_argument("--db", required=True, help="Path to the review queue DB")
    rr.add_argument("--queue-id", required=True, help="queue_id to resolve")
    rr.add_argument("--reviewer", required=True, help="Operator identifier")
    rr.add_argument(
        "--resolution", required=True,
        choices=["allow", "deny", "abstain"],
        help="``allow`` or ``deny`` override the escalate verdict; "
             "``abstain`` leaves escalate as final.",
    )
    rr.add_argument(
        "--justification", default="",
        help="Free-text justification (capped 8192 chars)",
    )
    rr.add_argument(
        "--audit-db", default=None,
        help="If supplied, also write an ESCALATION_RESOLVED record to "
             "the audit DB at this path (Article 14(4)(d) evidence).",
    )
    rr.set_defaults(func=_cmd_review_resolve)

    re_ = rsub.add_parser(
        "expire", help="Expire stale pending items past a timeout",
    )
    re_.add_argument("--db", required=True, help="Path to the review queue DB")
    re_.add_argument(
        "--timeout-seconds", required=True, type=float,
        help="Pending items older than this are marked expired",
    )
    re_.add_argument(
        "--dry-run", action="store_true",
        help="Report the count without modifying the DB",
    )
    re_.set_defaults(func=_cmd_review_expire)

    pp_policy = sub.add_parser(
        "policy",
        help="Policy artifact commands (validate, test, reload)",
    )
    psub = pp_policy.add_subparsers(dest="policy_cmd", required=True)

    pvalid = psub.add_parser(
        "validate",
        help="Load a policy and report parse errors plus semantic warnings",
    )
    pvalid.add_argument("policy", help="Path to a YAML or JSON policy file")
    pvalid.add_argument(
        "--json", action="store_true",
        help="Emit the report as JSON (stable shape for CI)",
    )
    pvalid.set_defaults(func=_cmd_policy_validate)

    ptest = psub.add_parser(
        "test",
        help="Run a YAML/JSON cases file against a policy (Conftest analog)",
    )
    ptest.add_argument("policy", help="Path to a YAML or JSON policy file")
    ptest.add_argument(
        "--cases", required=True,
        help="Path to a YAML or JSON file containing a 'cases:' list",
    )
    ptest.add_argument(
        "--json", action="store_true",
        help="Emit results as JSON (stable shape for CI)",
    )
    ptest.set_defaults(func=_cmd_policy_test)

    pcr = sub.add_parser(
        "compliance",
        help="Compliance reporting commands",
    )
    csub = pcr.add_subparsers(dest="compliance_cmd", required=True)
    pcrep = csub.add_parser(
        "report",
        help="Assemble and render an article-level evidence report",
    )
    pcrep.add_argument(
        "--db", required=True,
        help="Path to the audit SQLite DB to read evidence from",
    )
    pcrep.add_argument(
        "--format", choices=["md", "json", "narrative"], default="md",
        help="Output format (default: md)",
    )
    pcrep.add_argument(
        "--out", default=None,
        help="Write to file (default: stdout)",
    )
    pcrep.add_argument(
        "--system-name", default="Vaara-governed AI system",
        help="System name to include in the report header",
    )
    pcrep.add_argument(
        "--system-version", default="unspecified",
        help="System version to include in the report header",
    )
    pcrep.set_defaults(func=_cmd_compliance_report)

    pcdash = csub.add_parser(
        "dashboard",
        help=(
            "Render the article-level evidence report as a single "
            "self-contained HTML page (auditor-facing static dashboard)"
        ),
    )
    pcdash.add_argument(
        "--db", required=True,
        help="Path to the audit SQLite DB to read evidence from",
    )
    pcdash.add_argument(
        "--out", required=True,
        help=(
            "Output path. A trailing slash or existing directory writes "
            "index.html inside; otherwise the given path is the file."
        ),
    )
    pcdash.add_argument(
        "--system-name", default="Vaara-governed AI system",
        help="System name to include in the dashboard header",
    )
    pcdash.add_argument(
        "--system-version", default="unspecified",
        help="System version to include in the dashboard header",
    )
    pcdash.set_defaults(func=_cmd_compliance_dashboard)

    pserve = sub.add_parser(
        "serve",
        help="Run the Vaara HTTP API reference server (requires vaara[server])",
    )
    pserve.add_argument("--host", default="127.0.0.1", help="Bind host")
    pserve.add_argument("--port", type=int, default=8000, help="Bind port")
    pserve.add_argument(
        "--policy",
        default=None,
        help=(
            "Path to a YAML or JSON policy file. Enables POST "
            "/v1/policy/reload; the policy's default thresholds and "
            "sequence patterns are applied to the scorer at startup."
        ),
    )
    pserve.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    pserve.set_defaults(func=_cmd_serve)

    pdetect = sub.add_parser(
        "detect",
        help="Named detectors (injection, pii) over Vaara's scoring surface",
    )
    dsub = pdetect.add_subparsers(dest="detect_cmd", required=True)

    def _add_text_input_args(p_):
        g = p_.add_mutually_exclusive_group(required=True)
        g.add_argument("--text", help="Inline text to scan")
        g.add_argument("--file", help="Path to a text file to scan")
        g.add_argument(
            "--stdin", action="store_true",
            help="Read text from standard input",
        )
        p_.add_argument(
            "--pretty", action="store_true",
            help="Pretty-print the JSON output",
        )

    pdinj = dsub.add_parser(
        "injection",
        help=(
            "Score text for prompt-injection likelihood via Vaara's "
            "adversarial scorer (the same model behind vaara-bench-v1)"
        ),
    )
    _add_text_input_args(pdinj)
    pdinj.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold (default 0.55, the bench escalation band)",
    )
    pdinj.set_defaults(func=_cmd_detect_injection)

    pdpii = dsub.add_parser(
        "pii",
        help=(
            "Scan text for PII categories (email, phone, ssn, ipv4, "
            "credit_card, iban)"
        ),
    )
    _add_text_input_args(pdpii)
    pdpii.set_defaults(func=_cmd_detect_pii)

    preload = psub.add_parser(
        "reload",
        help=(
            "Trigger an atomic policy reload on a running Vaara server. "
            "The new thresholds and sequence patterns are applied without "
            "restarting the agent process."
        ),
    )
    preload.add_argument(
        "policy", help="Path to the new YAML or JSON policy file",
    )
    preload.add_argument(
        "--server", default="http://127.0.0.1:8000",
        help="Base URL of the running vaara serve process",
    )
    preload.add_argument(
        "--inline", action="store_true",
        help=(
            "Send the parsed policy body in the HTTP request instead of "
            "asking the server to read the file. Use when the server runs "
            "on a different host than the operator."
        ),
    )
    preload.add_argument(
        "--format", choices=["json", "yaml"], default=None,
        help="Override the policy format detection (default: by file suffix)",
    )
    preload.add_argument(
        "--timeout", type=float, default=10.0,
        help="HTTP request timeout in seconds (default 10)",
    )
    preload.set_defaults(func=_cmd_policy_reload)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
