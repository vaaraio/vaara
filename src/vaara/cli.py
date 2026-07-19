# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
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

    vaara mode list
        List the built-in policy mode presets (eco, balanced, performance,
        strict) with their thresholds and one-line descriptions.

    vaara mode show NAME
        Print thresholds, description, and watt profile for a single mode.

    vaara mode emit NAME [--format json|yaml] [--output PATH]
        Emit a mode as a minimal, valid Vaara policy document. Round-trips
        through ``vaara.policy.from_dict`` / ``from_json`` / ``from_yaml``.

    vaara keygen --attest --out PATH [--force]
        Generate an EC P-256 (ES256) keypair for tool-call attestation
        signing with ``vaara-mcp-proxy --attest-signing-key``. Replaces
        the ``openssl ecparam | pkcs8`` pipe. Does not require --dev.

    vaara attest verify ENVELOPE.json
            (--pubkey-file PUB.pem | --hs256-secret-file SECRET)
            [--enforce-ttl]
        Verify a tool-call attestation envelope. Reports signature
        validity and whether the TTL has expired; TTL is not enforced
        by default (a saved attestation is durable evidence). Exit 0
        iff the signature verifies.

    vaara receipt verify RECEIPT.json --attestation ATT.json
            (--pubkey-file PUB.pem | --hs256-secret-file SECRET)
            [--result RESULT.json]
        Verify an execution receipt against its attestation: receipt
        signature, attestation signature (TTL ignored), and the
        back-link binding the two. When the receipt carries a result
        commitment, --result verifies it against the runtime result.
        Exit 0 iff all checks pass.

    vaara verify-bundle PATH [--json]
        Verify a complete evidence bundle in one command. PATH is a bundle
        JSON file, or a directory holding bundle.json. Runs every lens whose
        evidence is present (identity, signature, back-link, inclusion,
        consistency, revocation) and prints one verdict. Exit 0 iff the
        bundle is ok: the receipt signature was established and every
        applicable lens passed.

    vaara verify-record RECORD.json [--attestation ATT.json] [--json]
        Conformance-check any candidate SEP-2828 execution record against
        the wire schema and the binding it proves about itself (the result
        projectionDigest over the projection bytes). Keyless: needs no
        signing key and no attestation. With --attestation the back-link is
        checked too, still keyless. The neutral check a party runs on a
        record someone else produced. Exit 0 iff the record conforms.

    vaara verify-retained RECORD.json --did-document DOC.json
            [--key-history KH.json] [--revocations REV.json]
            [--anchor ANCHOR.json | --anchored-time ISO] [--keyid KEYID] [--json]
        Verify a record under a signing key that has since rotated out, over
        the Article 12 retention window. Binds the signature to a key the
        archived DID document lists (offline), then checks the claimed time
        falls inside that key's validity window and the key was not revoked
        before issuance. A retired key still verifies a signature it made
        while valid. With a verified time anchor the verdict is corroborated:
        the record provably existed before the key's end of life, so it
        cannot be a later forgery under a stolen retired key. Exit 0 iff the
        record is verifiable.

    vaara verify-enforcement RECORD.json --report REPORT.bin --vcek VCEK.pem
            [--expected-measurement HEX] [--strict] [--json]
        Check whether a SEV-SNP attestation report binds a signed record to a
        confidential VM whose VCEK you supply: REPORT_DATA must carry
        sha512(jcs(record)) and the report signature must verify against the
        VCEK, with an optional pinned launch measurement. It does not validate
        the VCEK chain to AMD's ARK (the KDS chain is deferred) or prove the
        decision logic ran in the enclave, so it does not by itself establish
        genuine AMD hardware. Requires the attestation extra. Exit 0 iff the
        report binds and any pinned measurement matched.

    vaara normalize RECORD.json [--format auto|sep2643|sep2787|sep2817] [--json]
        Map an adjacent MCP record onto the SEP-2828 evidence model. Reads a
        SEP-2643 denial, a tool-call attestation, or a SEP-2817 invocation
        audit context and reports which evidence plane it fills, which
        SEP-2828 fields it populates, and what is still missing for a
        complete signed record. Promotes nothing: an unsigned client claim
        stays advisory. Exit 0 iff the record was recognized.

    vaara verify-records DIR [--glob '*.json'] [--json]
        Conformance-check a whole directory of SEP-2828 records at once: the
        receiving side an auditor works from. Reports how many conform, which
        fail and why, and the cross-record gaps verify-record cannot see (a
        call recorded twice, an executed action that committed no result).
        Keyless. Exit 0 iff every record conforms and no required gap fired.

    vaara verify-handoffs DIR [--glob '*.json'] [--trusted-did-document DOC.json]
            [--strict] [--no-anchor] [--json]
        Verify a whole directory of cross-org handoff packages at once: the
        batch twin of verify-handoff. Reports how many records verify offline
        under their rotated-out keys, how many are anchor-corroborated rather
        than resting on the signature alone, and how many had their producer
        identity pinned. Requires the attestation extra. Exit 0 iff every
        package verifies for the chosen mode.

    vaara verify-enforcements DIR [--glob '*.record.json']
            [--expected-measurement HEX] [--strict] [--json]
        Bind a whole directory of (record, report, VCEK) triples at once: the
        batch twin of verify-enforcement. Triples are discovered by stem
        (NAME.record.json with NAME.report.bin and NAME.vcek.pem). Reports how
        many bind to a confidential VM, the per-tier tally, and whether any
        pinned a vetted launch image. Requires the attestation extra. Exit 0
        iff every triple binds for the chosen mode.

    vaara conformance-statement [--corpus DIR] [--records DIR] [--as-of DATE]
            [--out FILE] [--json]
        Self-test this implementation against the published SEP-2828
        conformance corpus and print one reproducible statement: the corpus
        bytes match their manifest, every recorded verdict was reproduced, and
        (with --records) your own records conform. Names the exact corpus byte
        set (version plus corpusDigest), so the claim pins a fixed suite rather
        than a moving target. The answer to "trust us": prove it against the
        neutral suite. Keyless. Exit 0 iff the statement conforms.

    vaara conformance check PATH [--attestation ATT.json] [--glob '*.json'] [--json]
        One keyless front door over the checks above. PATH is a record JSON
        file (checked like verify-record) or a directory of them (checked
        like verify-records); the kind is auto-detected. Reports every check,
        pass or fail. Exit 0 iff it conforms.

    vaara conformance statement [--corpus DIR] [--records DIR] [--out FILE] [--json]
        Self-test this build against the published corpus and print one
        reproducible statement (same check as conformance-statement). Keyless.

    vaara build-bundle (--receipt RECEIPT.json | --from-dir DIR) [piece flags]
            [--out BUNDLE.json]
        Assemble a complete evidence bundle on disk from the issuer's
        pieces: the receipt plus whatever identity, signature, back-link,
        inclusion, consistency, and revocation material the issuer holds.
        Writes the single document verify-bundle reads. The issuer-side
        mirror of verify-bundle. Round-trip: the file this writes, fed
        straight to verify-bundle, verifies.

    vaara version
        Print the installed Vaara version.

Installing the CLI requires no extra deps. ``keygen``/``export``/``verify``
require ``pip install 'vaara[export]'`` (pulls cryptography). The
``attest``/``receipt`` verifiers and ``keygen --attest`` require
``pip install 'vaara[attestation]'``.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
import re
import stat
import sys
import time
from pathlib import Path
from typing import Any, Optional

from vaara import __version__

_INSTALL_HINT = (
    "This command requires the 'cryptography' package. "
    "Install with: pip install 'vaara[export]'"
)

_ATTESTATION_HINT = (
    "This command requires the 'attestation' extra "
    "(rfc8785 + cryptography). Install with: pip install 'vaara[attestation]'"
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

    if args.attest:
        # EC P-256 (ES256) key for tool-call attestation signing. This is the
        # documented operator path for vaara-mcp-proxy --attest-signing-key, so
        # it does not gate on --dev the way the Ed25519 trail-signing key does.
        return _keygen_attest(out, pub_out)

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


def _keygen_attest(out: Path, pub_out: Path) -> int:
    """Generate an EC P-256 (ES256) keypair for tool-call attestation signing.

    Writes a PKCS8 PEM private key (0600) and a SubjectPublicKeyInfo PEM
    public key. The printed ``secretVersion`` is the first 8 hex of the
    SHA-256 over the public-key DER, matching what ``build_attest_emitter``
    derives, so the operator can correlate a signed envelope with this
    keypair. Replaces the documented ``openssl ecparam | pkcs8`` pipe.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
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
        # Best-effort: filesystems without POSIX permission bits still get the
        # key written; the operator owns the enclosing directory's perms.
        pass
    pub_out.write_bytes(pub_pem)

    pub_der = key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    # Despite the field name "secretVersion", this value is a digest
    # of the PUBLIC key and is safe to print and publish. It is named here for
    # what it is so it is not mistaken (by a reader or a scanner) for a secret.
    pubkey_version = hashlib.sha256(pub_der).hexdigest()[:8]

    print("Generated EC P-256 keypair for tool-call attestation signing (ES256)")
    print(f"  private key:   {out}        (0600)")
    print(f"  public key:    {pub_out}")
    print(f"  secretVersion: {pubkey_version}")
    print()
    print("Sign with the proxy:")
    print(
        f"  vaara-mcp-proxy --attest-signing-key {out} "
        "--attest-receipts-dir DIR --upstream NAME=CMD"
    )
    print("Verifiers need only the public key:")
    print(f"  vaara attest verify ATTESTATION.json --pubkey-file {pub_out}")
    print()
    print(
        "The secretVersion above is the first 8 hex of SHA-256 over the public "
        "key DER. It is carried in every attestation this key signs, so a "
        "signed envelope can be matched to this keypair without the private key."
    )
    print()
    print(
        "REMINDER: this private key is on disk with no passphrase. For higher "
        "assurance, hold the signing key in a KMS/HSM. See docs/signing-keys.md."
    )
    return 0


def _cmd_trail_rotate(args: argparse.Namespace) -> int:
    from vaara.audit.rotate import rotate

    db_path = Path(args.db).expanduser()
    if not db_path.is_file():
        print(f"vaara trail rotate: not a file: {db_path}", file=sys.stderr)
        return 2

    result = rotate(
        db_path=db_path,
        out_path=Path(args.out).expanduser(),
        signer_key=Path(args.key).expanduser(),
        retention_days=args.retention_days,
        tenant_id="" if args.all_tenants else args.tenant,
        dry_run=args.dry_run,
    )
    if not result.ok:
        for err in result.errors:
            print(f"vaara trail rotate: {err}", file=sys.stderr)
        return 1
    print(f"Archived {result.exported_records} record(s) to {result.archive_path} (verified).")
    if args.dry_run:
        print(f"Would purge {result.purged_records} record(s) older than {args.retention_days} day(s).")
        print("Run without --dry-run to apply.")
    else:
        print(f"Purged {result.purged_records} record(s) older than {args.retention_days} day(s).")
        print("Archive the zip externally; the live DB now has a chain seam at the boundary.")
    return 0


def _cmd_trail_shadow_report(args: argparse.Namespace) -> int:
    import json as _json

    from vaara.audit.shadow_report import render_text, shadow_report

    db_path = Path(args.db).expanduser()
    if not db_path.is_file():
        print(f"vaara trail shadow-report: not a file: {db_path}", file=sys.stderr)
        return 2
    report = shadow_report(db_path, days=args.days)
    if args.format == "json":
        print(_json.dumps(report, indent=2))
    else:
        print(render_text(report))
    return 0


def _load_trail_source(args: argparse.Namespace):
    """Load an AuditTrail from --trail (JSONL) or --db (SQLite).

    Returns (trail, None) on success or (None, exit_code) on failure with
    the error already printed. The two flags are mutually exclusive and one
    is required, enforced at the parser level.
    """
    from vaara.audit.trail import AuditRecord, AuditTrail

    db_arg = getattr(args, "db", None)
    if db_arg:
        db_path = Path(db_arg).expanduser()
        if not db_path.is_file():
            print(f"audit DB not found: {db_path}", file=sys.stderr)
            return None, 2
        from vaara.audit.sqlite_backend import SQLiteAuditBackend

        return SQLiteAuditBackend(db_path).load_trail(), None

    trail_path = Path(args.trail).expanduser()
    if not trail_path.exists():
        print(f"trail JSONL not found: {trail_path}", file=sys.stderr)
        return None, 2

    trail = AuditTrail()
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
    return trail, None


def _add_trail_source_args(sub_parser: argparse.ArgumentParser) -> None:
    """--trail JSONL / --db SQLite input group shared by the trail exports."""
    src = sub_parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--trail", help="Path to trail JSONL file")
    src.add_argument(
        "--db",
        help="Path to an audit SQLite DB (e.g. the Claude Code plugin's "
             "~/.vaara/claude-code/audit.db)",
    )


def _cmd_trail_export(args: argparse.Namespace) -> int:
    try:
        from vaara.audit.export import export_signed
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return 2

    trail, err = _load_trail_source(args)
    if trail is None:
        return err

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


def _cmd_trail_export_threshold(args: argparse.Namespace) -> int:
    try:
        from vaara.audit.export import (
            _load_private_key,
            export_signed_threshold,
        )
        from vaara.audit.signer import Ed25519Signer
        from vaara.audit.trail import AuditRecord, AuditTrail
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return 2

    trail_path = Path(args.trail).expanduser()
    if not trail_path.exists():
        print(f"trail JSONL not found: {trail_path}", file=sys.stderr)
        return 2

    trail = AuditTrail()
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

    signers = []
    for key_path in args.key:
        try:
            signers.append(Ed25519Signer(_load_private_key(Path(key_path).expanduser())))
        except (OSError, ValueError) as exc:
            print(f"could not load signing key {key_path}: {exc}", file=sys.stderr)
            return 2

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = export_signed_threshold(
            trail,
            out_path=out,
            signers=signers,
            threshold_k=args.threshold_k,
            agent_id=args.agent_id or "",
        )
    except ValueError as exc:
        print(f"threshold export failed: {exc}", file=sys.stderr)
        return 2

    print(f"Exported {result.manifest['threshold_k']}-of-"
          f"{result.manifest['signers_n']} threshold-signed trail to {result.path}")
    print(f"  records:      {result.manifest['record_count']}")
    print(f"  chain intact: {result.chain_intact}")
    print(f"  signer set:   {result.manifest['signer_pubkey_fingerprint']}")
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
        print(f"  algorithm:    {result.manifest.get('signature_algorithm')}")
        print(f"  fingerprint:  {result.manifest.get('signer_pubkey_fingerprint')}")
        if result.manifest.get("threshold_k") is not None:
            print(f"  threshold:    {result.manifest.get('threshold_k')}-of-"
                  f"{result.manifest.get('signers_n')} "
                  f"({result.manifest.get('member_algorithm')})")
        print(f"  created:      {result.manifest.get('created_utc')}")
        print(f"  vaara:        {result.manifest.get('vaara_version')}")
    if result.ok:
        print("Verification: OK")
        if pubkey is None:
            print(
                "  WARNING: no --pubkey given — the trail was checked against the "
                "key inside the archive (integrity / internal consistency only); "
                "issuer identity is NOT authenticated. Pass --pubkey <trusted.pem> "
                "you hold out of band to authenticate.",
                file=sys.stderr,
            )
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


def _parse_period(spec: Optional[str]) -> Optional[tuple]:
    """Parse a ``YYYY-MM-DD:YYYY-MM-DD`` period into a (start, end) epoch pair.

    Either side may be empty for an open bound (``:2026-06-30`` or
    ``2026-01-01:``). The end date is inclusive of its whole day. Returns
    ``None`` when ``spec`` is falsy.
    """
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError(
            "period must be START:END (YYYY-MM-DD:YYYY-MM-DD); either side may be empty"
        )
    start_s, end_s = spec.split(":", 1)

    def _epoch(day: str, *, end_of_day: bool) -> Optional[float]:
        day = day.strip()
        if not day:
            return None
        t = time.strptime(day, "%Y-%m-%d")
        epoch = calendar.timegm(t)
        return epoch + 86399 if end_of_day else float(epoch)

    return (_epoch(start_s, end_of_day=False), _epoch(end_s, end_of_day=True))


def _cmd_trail_verify_anchor(args: argparse.Namespace) -> int:
    """Verify the external time anchor folded into an Article 12 package."""
    import zipfile

    try:
        from vaara.audit.timeanchor import (
            TimeAnchor,
            TimeAnchorError,
            verify_anchor_over_records,
        )
        from vaara.audit.trail import AuditRecord
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return 2

    zip_path = Path(args.zip).expanduser()
    if not zip_path.exists():
        print(f"package zip not found: {zip_path}", file=sys.stderr)
        return 2

    try:
        with zipfile.ZipFile(zip_path) as zf:
            if "time_anchor.json" not in set(zf.namelist()):
                print(
                    "no time_anchor.json in package (it is not time-anchored)",
                    file=sys.stderr,
                )
                return 2
            anchor = TimeAnchor.from_dict(json.loads(zf.read("time_anchor.json")))
            trail_bytes = zf.read("trail.jsonl")
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, OSError,
            TypeError) as exc:
        print(f"could not read package: {exc}", file=sys.stderr)
        return 2

    record_hashes = [
        AuditRecord.from_dict(json.loads(line)).record_hash
        for line in (ln.strip() for ln in trail_bytes.splitlines())
        if line
    ]

    try:
        attested = verify_anchor_over_records(anchor, record_hashes)
    except TimeAnchorError as exc:
        print(f"time anchor INVALID: {exc}", file=sys.stderr)
        return 1

    print("time anchor OK")
    print(f"  backend:        {anchor.backend}")
    print(f"  TSA:            {anchor.tsa_url}")
    print(f"  anchored head:  {anchor.chain_head_hash}")
    print(f"  attested (UTC): {attested.isoformat()}")
    print("  note: pin the TSA certificate to enforce an eIDAS-qualified authority.")
    return 0


def _obtain_time_anchor(args: argparse.Namespace, trail):
    """Build or load the optional Article 19 time anchor over the trail head.

    Returns ``None`` when neither ``--anchor-tsa`` nor ``--anchor-file`` is
    given. Raises ``ValueError`` on any anchor failure so the CLI reports it
    cleanly. The anchor is taken over the trail's final record (its head); the
    export then re-verifies the binding before folding it in.
    """
    anchor_tsa = getattr(args, "anchor_tsa", None)
    anchor_file = getattr(args, "anchor_file", None)
    if not anchor_tsa and not anchor_file:
        return None

    from vaara.audit.timeanchor import (
        RFC3161TimeAnchorClient,
        TimeAnchor,
        TimeAnchorError,
    )

    records = trail._records
    if not records:
        raise ValueError("cannot time-anchor an empty trail")
    head_position = len(records) - 1
    head_hash = records[-1].record_hash
    if not head_hash:
        raise ValueError("trail head has no record_hash to anchor")

    if anchor_file:
        path = Path(anchor_file).expanduser()
        if not path.exists():
            raise ValueError(f"anchor file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return TimeAnchor.from_dict(json.load(f))
        except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
            raise ValueError(f"failed to read anchor file: {exc}") from exc

    try:
        client = RFC3161TimeAnchorClient(anchor_tsa)
        return client.anchor(head_position, head_hash)
    except TimeAnchorError as exc:
        raise ValueError(f"time anchoring failed: {exc}") from exc


def _cmd_trail_export_article50(args: argparse.Namespace) -> int:
    """Export a signed EU AI Act Article 50 transparency evidence package."""
    try:
        from vaara.audit.article50 import export_article50
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return 2

    system_meta = None
    if args.system_meta:
        meta_path = Path(args.system_meta).expanduser()
        if not meta_path.exists():
            print(f"system-meta JSON not found: {meta_path}", file=sys.stderr)
            return 2
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                system_meta = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"failed to read system-meta: {exc}", file=sys.stderr)
            return 2

    try:
        period = _parse_period(args.period)
    except ValueError as exc:
        print(f"invalid --period: {exc}", file=sys.stderr)
        return 2

    trail, err = _load_trail_source(args)
    if trail is None:
        return err

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = export_article50(
            trail, out, signer_key=Path(args.key).expanduser(),
            system_meta=system_meta, period=period,
        )
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return 2

    print(f"Exported Article 50 transparency package to {out}")
    print(f"  records:      {result.manifest['record_count']}")
    print(f"  chain intact: {result.chain_intact}")
    print("  report:       article50_report.md")
    return 0 if result.chain_intact else 1


def _cmd_trail_export_article12(args: argparse.Namespace) -> int:
    """Export a signed EU AI Act Article 12 regulator package from a trail."""
    try:
        from vaara.audit.article12_export import export_article12
        from vaara.audit.timeanchor import TimeAnchorError
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return 2

    system_meta = None
    if args.system_meta:
        meta_path = Path(args.system_meta).expanduser()
        if not meta_path.exists():
            print(f"system-meta JSON not found: {meta_path}", file=sys.stderr)
            return 2
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                system_meta = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"failed to read system-meta: {exc}", file=sys.stderr)
            return 2

    try:
        period = _parse_period(args.period)
    except ValueError as exc:
        print(f"invalid --period: {exc}", file=sys.stderr)
        return 2

    trail, err = _load_trail_source(args)
    if trail is None:
        return err

    # Folding handoff / enforcement evidence needs the attestation extra (the
    # crypto the record and binding lenses use). Fail fast with the install
    # hint rather than deep inside the export.
    fold_requested = bool(args.handoff or args.handoffs or args.enforcements)
    if fold_requested:
        try:
            import rfc8785  # noqa: F401

            from vaara.attestation.receipt import (  # noqa: F401
                check_enforcement_set,
                check_handoff_set,
            )
        except ImportError:
            print(_ATTESTATION_HINT, file=sys.stderr)
            return 2

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    handoffs = enforcements = None
    try:
        if fold_requested:
            handoffs, enforcements, trusted, expected = _collect_fold_attachments(args)
        else:
            trusted, expected = None, None
        time_anchor = _obtain_time_anchor(args, trail)
        result = export_article12(
            trail,
            out_path=out,
            signer_key=Path(args.key).expanduser(),
            system_meta=system_meta,
            period=period,
            time_anchor=time_anchor,
            handoffs=handoffs,
            enforcements=enforcements,
            trusted_did_document=trusted,
            expected_measurement=expected,
            fmt=args.format,
            agent_id=args.agent_id or "",
        )
    except (ValueError, ImportError, TimeAnchorError) as exc:
        print(f"Article 12 export failed: {exc}", file=sys.stderr)
        return 2

    print(f"Exported Article 12 regulator package to {result.path}")
    print(f"  records:      {result.manifest['record_count']}")
    print(f"  chain intact: {result.chain_intact}")
    print(f"  report:       article12_report.{args.format}")
    if time_anchor is not None:
        print(f"  time anchor:  {time_anchor.anchored_time} ({time_anchor.backend})")
    if handoffs:
        print(f"  handoff:      {len(handoffs)} package(s) folded (Article 26(6))")
    if enforcements:
        print(f"  enforcement:  {len(enforcements)} binding(s) folded")
    return 0 if result.chain_intact else 1


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
        backend = SQLiteAuditBackend(Path(args.audit_db).expanduser())
        # Continue the existing hash chain. A fresh AuditTrail starts at
        # previous_hash='' and would fork the chain when the audit DB already
        # holds the action's lifecycle, so the ESCALATION_RESOLVED record
        # (Article 14(4)(d)) must append to the loaded trail, not a new one.
        trail = backend.load_trail()
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

    with SQLiteAuditBackend(str(db_path)) as backend:
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
    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.compliance.dashboard import render_html
    from vaara.compliance.engine import create_default_engine

    db_path = Path(args.db).expanduser()
    if not db_path.is_file():
        print(f"vaara compliance dashboard: not a file: {db_path}", file=sys.stderr)
        return 2

    with SQLiteAuditBackend(str(db_path)) as backend:
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

    with SQLiteAuditBackend(str(db_path)) as backend:
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

    if args.format == "pdf":
        if not args.out:
            print(
                "--out is required when --format pdf (binary output)",
                file=sys.stderr,
            )
            return 2
        try:
            from vaara.compliance.render import render_pdf
        except ImportError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        try:
            render_pdf(report, Path(args.out).expanduser())
        except ImportError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0

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


_OVERT_ENVELOPE_KEYS = (
    "blinded_identifier",
    "request_commitment",
    "encoder_binary_identity",
    "non_content_metadata",
    "monotonic_counter",
    "nanosecond_timestamp",
    "key_identifier",
    "arbiter_instance_identifier",
    "signature",
)


def _cmd_overt_verify(args: argparse.Namespace) -> int:
    """Verify an OVERT 1.0 Protocol Profile 1.0 Base Envelope.

    Reads a canonical CBOR file produced by any conformant implementation,
    decodes the 9-field structure, and validates the Ed25519 signature
    against a supplied raw 32-byte public key. Implementation-agnostic —
    Vaara is the reference verifier for Annex B.6 envelopes.
    """
    try:
        import cbor2
    except ImportError:
        print(
            "vaara overt verify requires the attestation extra. "
            "Install with: pip install 'vaara[attestation]'",
            file=sys.stderr,
        )
        return 2

    from vaara.attestation.overt import BaseEnvelope, verify_base_envelope

    receipt_path = Path(args.receipt).expanduser()
    if not receipt_path.is_file():
        print(f"vaara overt verify: not a file: {receipt_path}", file=sys.stderr)
        return 2

    pubkey_raw = _load_overt_pubkey(args)
    if pubkey_raw is None:
        return 2
    if len(pubkey_raw) != 32:
        print(
            f"vaara overt verify: public key must be 32 raw bytes, got "
            f"{len(pubkey_raw)}",
            file=sys.stderr,
        )
        return 2

    try:
        decoded = cbor2.loads(receipt_path.read_bytes())
    except Exception as exc:
        print(f"vaara overt verify: CBOR decode failed: {exc}", file=sys.stderr)
        return 1

    if not isinstance(decoded, dict):
        print(
            "vaara overt verify: envelope CBOR must decode to a map; "
            f"got {type(decoded).__name__}",
            file=sys.stderr,
        )
        return 1

    missing = [k for k in _OVERT_ENVELOPE_KEYS if k not in decoded]
    if missing:
        print(
            "vaara overt verify: envelope missing required fields: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 1
    extras = set(decoded.keys()) - set(_OVERT_ENVELOPE_KEYS)
    if extras:
        print(
            "vaara overt verify: envelope carries unknown fields (the OVERT "
            "1.0 schema is closed): " + ", ".join(sorted(extras)),
            file=sys.stderr,
        )
        return 1

    try:
        envelope = BaseEnvelope(
            blinded_identifier=decoded["blinded_identifier"],
            request_commitment=decoded["request_commitment"],
            encoder_binary_identity=decoded["encoder_binary_identity"],
            non_content_metadata=decoded["non_content_metadata"],
            monotonic_counter=int(decoded["monotonic_counter"]),
            nanosecond_timestamp=int(decoded["nanosecond_timestamp"]),
            key_identifier=decoded["key_identifier"],
            arbiter_instance_identifier=decoded["arbiter_instance_identifier"],
            signature=decoded["signature"],
        )
    except (TypeError, ValueError) as exc:
        print(
            f"vaara overt verify: envelope field types are invalid: {exc}",
            file=sys.stderr,
        )
        return 1

    if verify_base_envelope(envelope, pubkey_raw):
        result = {
            "valid": True,
            "key_identifier": envelope.key_identifier.hex(),
            "arbiter_instance_identifier": (
                envelope.arbiter_instance_identifier.hex()
            ),
            "monotonic_counter": envelope.monotonic_counter,
            "nanosecond_timestamp": envelope.nanosecond_timestamp,
        }
        print(json.dumps(result, indent=2))
        return 0

    print(
        "vaara overt verify: signature verification failed "
        "(either the supplied public key does not match the envelope's "
        "key_identifier, or the signature is invalid for the canonical "
        "CBOR of the 8 signable fields)",
        file=sys.stderr,
    )
    return 1


def _cmd_tee_parse(args: argparse.Namespace) -> int:
    """Parse an AMD SEV-SNP attestation report blob and print key fields."""
    try:
        from vaara.attestation.tee import (
            TEEAttestationError,
            parse_sev_snp_report,
        )
    except ImportError:
        print(
            "vaara tee parse requires the attestation extra. "
            "Install with: pip install 'vaara[attestation]'",
            file=sys.stderr,
        )
        return 2

    report_path = Path(args.report).expanduser()
    if not report_path.is_file():
        print(f"vaara tee parse: not a file: {report_path}", file=sys.stderr)
        return 2

    try:
        report = parse_sev_snp_report(report_path.read_bytes())
    except TEEAttestationError as exc:
        print(f"vaara tee parse: {exc}", file=sys.stderr)
        return 1

    out = {
        "version": report.version,
        "guest_svn": report.guest_svn,
        "vmpl": report.vmpl,
        "signature_algo": report.signature_algo,
        "policy": report.policy,
        "report_data": report.report_data.hex(),
        "measurement": report.measurement.hex(),
        "host_data": report.host_data.hex(),
        "chip_id": report.chip_id.hex(),
        "reported_tcb": report.reported_tcb,
        "committed_tcb": report.committed_tcb,
        "launch_tcb": report.launch_tcb,
        "current_build": report.current_build,
        "current_minor": report.current_minor,
        "current_major": report.current_major,
    }
    print(json.dumps(out, indent=2))
    return 0


def _cmd_tee_verify(args: argparse.Namespace) -> int:
    """Verify a SEV-SNP report signature, optionally against an OVERT envelope.

    The VCEK (Versioned Chip Endorsement Key) must be supplied as PEM. AMD
    KDS-based cert-chain validation (VCEK -> ASK -> ARK) is tracked for a
    later release; v0.18.0 only validates the report signature against a
    caller-supplied VCEK.
    """
    try:
        import cbor2

        from vaara.attestation.overt import BaseEnvelope
        from vaara.attestation.tee import (
            TEEAttestationError,
            parse_sev_snp_report,
            verify_envelope_binding,
            verify_sev_snp_report_signature,
        )
    except ImportError:
        print(
            "vaara tee verify requires the attestation extra. "
            "Install with: pip install 'vaara[attestation]'",
            file=sys.stderr,
        )
        return 2

    report_path = Path(args.report).expanduser()
    if not report_path.is_file():
        print(f"vaara tee verify: not a file: {report_path}", file=sys.stderr)
        return 2

    vcek_path = Path(args.vcek).expanduser()
    if not vcek_path.is_file():
        print(
            f"vaara tee verify: VCEK PEM not found: {vcek_path}",
            file=sys.stderr,
        )
        return 2

    try:
        report = parse_sev_snp_report(report_path.read_bytes())
    except TEEAttestationError as exc:
        print(f"vaara tee verify: {exc}", file=sys.stderr)
        return 1

    try:
        signature_ok = verify_sev_snp_report_signature(
            report, vcek_path.read_bytes(),
        )
    except TEEAttestationError as exc:
        print(f"vaara tee verify: {exc}", file=sys.stderr)
        return 1

    result = {
        "signature_valid": signature_ok,
        "report_data": report.report_data.hex(),
        "measurement": report.measurement.hex(),
    }

    if args.overt:
        overt_path = Path(args.overt).expanduser()
        if not overt_path.is_file():
            print(
                f"vaara tee verify: OVERT envelope file not found: "
                f"{overt_path}",
                file=sys.stderr,
            )
            return 2
        decoded = cbor2.loads(overt_path.read_bytes())
        envelope = BaseEnvelope(
            blinded_identifier=decoded["blinded_identifier"],
            request_commitment=decoded["request_commitment"],
            encoder_binary_identity=decoded["encoder_binary_identity"],
            non_content_metadata=decoded["non_content_metadata"],
            monotonic_counter=int(decoded["monotonic_counter"]),
            nanosecond_timestamp=int(decoded["nanosecond_timestamp"]),
            key_identifier=decoded["key_identifier"],
            arbiter_instance_identifier=decoded["arbiter_instance_identifier"],
            signature=decoded["signature"],
        )
        result["envelope_binding_valid"] = verify_envelope_binding(
            report, envelope,
        )
    else:
        result["envelope_binding_valid"] = None

    print(json.dumps(result, indent=2))
    if not signature_ok:
        return 1
    if args.overt and result["envelope_binding_valid"] is False:
        return 1
    return 0


def _load_overt_pubkey(args: argparse.Namespace) -> Optional[bytes]:
    """Resolve --pubkey-file or --pubkey-hex to raw bytes. None on error."""
    if args.pubkey_file:
        pubkey_path = Path(args.pubkey_file).expanduser()
        if not pubkey_path.is_file():
            print(
                f"vaara overt verify: pubkey file not found: {pubkey_path}",
                file=sys.stderr,
            )
            return None
        return pubkey_path.read_bytes()
    try:
        return bytes.fromhex(args.pubkey_hex)
    except ValueError as exc:
        print(
            f"vaara overt verify: --pubkey-hex is not valid hex: {exc}",
            file=sys.stderr,
        )
        return None


def _load_jws_verifying_material(args: argparse.Namespace) -> "tuple[Any, bool]":
    """Resolve ``--pubkey-file`` (PEM public key) or ``--hs256-secret-file``.

    Returns ``(material, ok)``. ``material`` is a public-key object (ES256 /
    RS256) when ``--pubkey-file`` is given, or raw ``bytes`` (HS256) when
    ``--hs256-secret-file`` is given. ``ok`` is False on error, with the
    diagnostic already printed to stderr.
    """
    if args.pubkey_file:
        from cryptography.hazmat.primitives import serialization

        path = Path(args.pubkey_file).expanduser()
        if not path.is_file():
            print(f"pubkey file not found: {path}", file=sys.stderr)
            return None, False
        try:
            return serialization.load_pem_public_key(path.read_bytes()), True
        except Exception as exc:
            print(
                f"--pubkey-file is not a usable PEM public key: {exc}",
                file=sys.stderr,
            )
            return None, False

    path = Path(args.hs256_secret_file).expanduser()
    if not path.is_file():
        print(f"hs256 secret file not found: {path}", file=sys.stderr)
        return None, False
    return path.read_bytes(), True


def _alg_material_mismatch(alg: str, material: Any) -> Optional[str]:
    """Return a diagnostic if the key material does not match the envelope alg.

    HS256 needs a raw shared secret (``--hs256-secret-file``); ES256 / RS256
    need a public key (``--pubkey-file``). Catching the mismatch here yields a
    clear message instead of a low-level signature failure, or a crash inside
    the verifier when the wrong material type reaches the crypto backend.
    """
    is_secret = isinstance(material, (bytes, bytearray))
    if alg == "HS256" and not is_secret:
        return "alg is HS256; verify with --hs256-secret-file, not --pubkey-file"
    if alg in ("ES256", "RS256") and is_secret:
        return f"alg is {alg}; verify with --pubkey-file, not --hs256-secret-file"
    return None


def _attest_isolation_now(envelope: Any) -> float:
    """A `now` that isolates signature verification from the TTL window.

    verify_attestation rejects an attestation whose iat is in the future
    (lower bound) or past exp (upper bound). To check the signature alone we
    evaluate at the envelope's own iat, which sits inside the window so both
    time bounds pass and the returned verdict is the pure signature result.
    Uses the same canonical parser verify_attestation uses internally; an
    unparseable iat yields 0.0, and the envelope then fails verification on
    its malformed iat regardless, which is the correct verdict.
    """
    from vaara.attestation._attest_canonical import iso8601_to_epoch

    epoch = iso8601_to_epoch(envelope.issuer_asserted.iat)
    return epoch if epoch is not None else 0.0


def _cmd_attest_verify(args: argparse.Namespace) -> int:
    """Verify a tool-call attestation envelope's signature (and optionally TTL)."""
    try:
        from vaara.attestation.tool_call_attestation import (
            AttestationError,
            parse_attestation,
            verify_attestation,
        )
    except ImportError:
        print(_ATTESTATION_HINT, file=sys.stderr)
        return 2

    env_path = Path(args.envelope).expanduser()
    if not env_path.is_file():
        print(f"vaara attest verify: not a file: {env_path}", file=sys.stderr)
        return 2
    try:
        data = json.loads(env_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"vaara attest verify: cannot read JSON: {exc}", file=sys.stderr)
        return 1
    try:
        envelope = parse_attestation(data)
    except (AttestationError, KeyError, TypeError, ValueError) as exc:
        print(
            f"vaara attest verify: not a valid tool-call attestation: {exc}",
            file=sys.stderr,
        )
        return 1

    material, ok = _load_jws_verifying_material(args)
    if not ok:
        return 2
    mismatch = _alg_material_mismatch(envelope.alg, material)
    if mismatch is not None:
        print(f"vaara attest verify: {mismatch}", file=sys.stderr)
        return 2

    # Evaluating at the envelope's own iat sits inside the validity window
    # (both the lower and upper time bounds pass), isolating signature
    # validity; a second pass at real time reveals whether the TTL expired.
    # Durable evidence files are routinely checked long after exp, so TTL is
    # reported but not enforced unless --enforce-ttl is set.
    isolation_now = _attest_isolation_now(envelope)
    signature_ok = verify_attestation(
        envelope, verifying_material=material, now=isolation_now
    )
    live_ok = verify_attestation(envelope, verifying_material=material)
    ttl_expired = signature_ok and not live_ok

    result = {
        "valid": signature_ok,
        "alg": envelope.alg,
        "iss": envelope.issuer_asserted.iss,
        "sub": envelope.issuer_asserted.sub,
        "intent": envelope.planner_declared.intent,
        "nonce": envelope.issuer_asserted.nonce,
        "issued_at": envelope.issuer_asserted.iat,
        "exp_seconds": envelope.issuer_asserted.exp_seconds,
        "ttl_expired": ttl_expired,
    }
    print(json.dumps(result, indent=2))

    if not signature_ok:
        print("vaara attest verify: signature verification failed", file=sys.stderr)
        return 1
    if args.enforce_ttl and ttl_expired:
        print("vaara attest verify: attestation TTL has expired", file=sys.stderr)
        return 1
    return 0


def _cmd_receipt_verify(args: argparse.Namespace) -> int:
    """Verify an execution receipt: signature, back-link, and optionally result."""
    try:
        from vaara.attestation.receipt import (
            parse_receipt,
            verify_back_link,
            verify_receipt_signature,
        )
        from vaara.attestation.tool_call_attestation import (
            AttestationError,
            parse_attestation,
            verify_args_commitment,
            verify_attestation,
        )
    except ImportError:
        print(_ATTESTATION_HINT, file=sys.stderr)
        return 2

    receipt_path = Path(args.receipt).expanduser()
    att_path = Path(args.attestation).expanduser()
    for label, p in (("receipt", receipt_path), ("attestation", att_path)):
        if not p.is_file():
            print(f"vaara receipt verify: {label} not a file: {p}", file=sys.stderr)
            return 2
    try:
        receipt = parse_receipt(json.loads(receipt_path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"vaara receipt verify: not a valid receipt: {exc}", file=sys.stderr)
        return 1
    try:
        attestation = parse_attestation(
            json.loads(att_path.read_text(encoding="utf-8"))
        )
    except (
        json.JSONDecodeError, OSError, AttestationError, KeyError, TypeError,
        ValueError,
    ) as exc:
        print(f"vaara receipt verify: not a valid attestation: {exc}", file=sys.stderr)
        return 1

    material, ok = _load_jws_verifying_material(args)
    if not ok:
        return 2
    mismatch = _alg_material_mismatch(receipt.alg, material)
    if mismatch is not None:
        print(f"vaara receipt verify: {mismatch}", file=sys.stderr)
        return 2

    receipt_sig_ok = verify_receipt_signature(receipt, verifying_material=material)
    # Attestation signature checked with TTL ignored: a receipt is a durable
    # record of an outcome, so its attestation is expected to be long expired.
    # Evaluate at the attestation's iat to isolate the signature from both
    # time bounds (see _attest_isolation_now).
    attestation_sig_ok = verify_attestation(
        attestation, verifying_material=material,
        now=_attest_isolation_now(attestation),
    )
    back_link = verify_back_link(receipt, attestation=attestation)

    result: dict[str, Any] = {
        "receipt_signature_valid": receipt_sig_ok,
        "attestation_signature_valid": attestation_sig_ok,
        "back_link_valid": back_link.ok,
        "status": receipt.outcome_derived.status,
        "completed_at": receipt.outcome_derived.completed_at,
    }

    commitment = receipt.outcome_derived.result_commitment
    if commitment is None:
        result["result_commitment_valid"] = None
    elif not args.result:
        result["result_commitment_valid"] = None
        result["result_commitment_note"] = (
            "receipt carries a result commitment; pass --result FILE to verify it"
        )
    else:
        rpath = Path(args.result).expanduser()
        if not rpath.is_file():
            print(f"vaara receipt verify: --result not a file: {rpath}", file=sys.stderr)
            return 2
        try:
            runtime_result = json.loads(rpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"vaara receipt verify: cannot read --result JSON: {exc}",
                file=sys.stderr,
            )
            return 1
        rc = verify_args_commitment(commitment, runtime_arguments=runtime_result)
        result["result_commitment_valid"] = rc.ok

    print(json.dumps(result, indent=2))

    failed = not (receipt_sig_ok and attestation_sig_ok and back_link.ok)
    if result.get("result_commitment_valid") is False:
        failed = True
    return 1 if failed else 0


def _load_receipt_for_ots(path_str: str, verb: str) -> "tuple[Optional[dict], Optional[Path]]":
    path = Path(path_str).expanduser()
    if not path.is_file():
        print(f"vaara receipt {verb}: not a file: {path}", file=sys.stderr)
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except (OSError, json.JSONDecodeError) as exc:
        print(f"vaara receipt {verb}: cannot read receipt JSON: {exc}",
              file=sys.stderr)
        return None, None


def _write_receipt_out(receipt: dict, args: argparse.Namespace,
                       fallback: Path) -> None:
    text = json.dumps(receipt, indent=2)
    out = Path(args.out).expanduser() if args.out else fallback
    out.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {out}")


def _cmd_receipt_anchor_ots(args: argparse.Namespace) -> int:
    """Add an OpenTimestamps witness anchor to a receipt's timestampAnchors."""
    try:
        from vaara.audit.ots_anchor import DEFAULT_CALENDARS, ots_anchor_receipt
        from vaara.audit.timeanchor import TimeAnchorError
    except ImportError:
        print("vaara receipt anchor-ots: requires the 'ots' extra "
              "(pip install \"vaara[ots]\")", file=sys.stderr)
        return 2
    receipt, path = _load_receipt_for_ots(args.receipt, "anchor-ots")
    if receipt is None or path is None:
        return 2
    calendars = tuple(args.calendar) if args.calendar else DEFAULT_CALENDARS
    try:
        anchor = ots_anchor_receipt(receipt, calendars=calendars)
    except TimeAnchorError as exc:
        print(f"vaara receipt anchor-ots: {exc}", file=sys.stderr)
        return 2
    receipt.setdefault("timestampAnchors", []).append(anchor)
    _write_receipt_out(receipt, args, path)
    print(f"opentimestamps anchor added: status={anchor['status']} "
          f"calendars={len(anchor['calendars'])}")
    print("Bitcoin finality typically lands in 1-6 hours; upgrade with: "
          "vaara receipt upgrade-ots")
    return 0


def _cmd_receipt_upgrade_ots(args: argparse.Namespace) -> int:
    """Upgrade a receipt's pending OpenTimestamps anchors to Bitcoin-final."""
    try:
        from vaara.audit.timeanchor import TimeAnchorError
        from vaara.audit.ots_anchor import upgrade_ots_anchor
    except ImportError:
        print("vaara receipt upgrade-ots: requires the 'ots' extra "
              "(pip install \"vaara[ots]\")", file=sys.stderr)
        return 2
    receipt, path = _load_receipt_for_ots(args.receipt, "upgrade-ots")
    if receipt is None or path is None:
        return 2
    anchors = receipt.get("timestampAnchors") or []
    ots_indices = [i for i, a in enumerate(anchors)
                   if isinstance(a, dict) and a.get("method") == "opentimestamps"]
    if not ots_indices:
        print("vaara receipt upgrade-ots: receipt carries no opentimestamps "
              "anchors", file=sys.stderr)
        return 2
    changed = False
    for i in ots_indices:
        try:
            upgraded = upgrade_ots_anchor(anchors[i])
        except TimeAnchorError as exc:
            print(f"vaara receipt upgrade-ots: anchor {i}: {exc}", file=sys.stderr)
            # A bad anchor must not discard upgrades already fetched for
            # earlier anchors: persist those before reporting the failure.
            if changed or args.out:
                _write_receipt_out(receipt, args, path)
            return 2
        if upgraded != anchors[i]:
            anchors[i] = upgraded
            changed = True
        print(f"anchor {i}: {upgraded['status']}")
    if changed or args.out:
        _write_receipt_out(receipt, args, path)
    return 0


def _cmd_receipt_render(args: argparse.Namespace) -> int:
    """Render a receipt to a self-contained static HTML evidence page."""
    from vaara.audit.receipt_page import render_receipt_page
    from vaara.audit.timeanchor import TimeAnchorError
    receipt, path = _load_receipt_for_ots(args.receipt, "render")
    if receipt is None or path is None:
        return 2
    try:
        page = render_receipt_page(receipt, title=args.title)
    except TimeAnchorError as exc:
        print(f"vaara receipt render: {exc}", file=sys.stderr)
        return 2
    out = (Path(args.out).expanduser() if args.out
           else path.with_suffix(".html"))
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _cmd_proxy(args: argparse.Namespace) -> int:
    """Run the model-endpoint proxy in observe mode.

    Fronts an OpenAI-compatible or ollama endpoint, passes every request
    through unchanged, and records each tool call the model requests into
    a hash-chained SQLite trail. This is the first-line capture layer:
    point an agent's base_url here and its model traffic is governed
    without any per-framework wiring. Signing (attestation/receipt pairs)
    is optional via --signing-key/--receipts-dir.
    """
    try:
        from vaara.audit.sqlite_backend import SQLiteAuditBackend
        from vaara.integrations._infer_proxy_app import build_app
        from vaara.integrations.infer_proxy import _parse_listen
        from vaara.pipeline import InterceptionPipeline
        import uvicorn  # noqa: F401  (needed at serve time below)
    except ImportError as exc:
        print(
            f"vaara proxy: missing dependency ({exc.name}). "
            "Install with: pip install 'vaara[proxy]'",
            file=sys.stderr,
        )
        return 2

    if args.enforce and not args.allow and not args.approvals_dir:
        print(
            "vaara proxy: --enforce needs --allow PATTERN (repeatable) "
            "and/or --approvals-dir. Without an allow-list every tool call "
            "is gated and clients appear to lose their tools (nothing is "
            "damaged, but the session is unusable). Start with --allow "
            "'mcp__*' and tighten from there, or run without --enforce to "
            "observe first.",
            file=sys.stderr,
        )
        return 2

    try:
        host, port = _parse_listen(args.listen)
    except argparse.ArgumentTypeError as exc:
        print(f"vaara proxy: {exc}", file=sys.stderr)
        raise SystemExit(2)

    trail_path = Path(args.trail).expanduser()
    trail_path.parent.mkdir(parents=True, exist_ok=True)
    backend = SQLiteAuditBackend(trail_path)
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    pipeline = InterceptionPipeline(trail=trail, enforce=args.enforce)

    emitter = None
    if args.signing_key:
        from vaara.integrations._infer_proxy_emit import InferenceAttestEmitter
        from vaara.integrations._infer_proxy_sign import (
            InferProxyConfigError,
            load_signing_key,
        )
        if not args.receipts_dir:
            print("vaara proxy: --signing-key requires --receipts-dir",
                  file=sys.stderr)
            return 2
        try:
            material, alg, secret_version = load_signing_key(
                Path(args.signing_key), None
            )
            emitter = InferenceAttestEmitter(
                signing_key=material, alg=alg,
                receipts_dir=Path(args.receipts_dir).expanduser(),
                secret_version=secret_version,
            )
        except InferProxyConfigError as exc:
            print(f"vaara proxy: {exc}", file=sys.stderr)
            return 2

    app = build_app(
        emitter=emitter, upstream=args.upstream, pipeline=pipeline,
        approvals_dir=(Path(args.approvals_dir).expanduser()
                       if args.approvals_dir else None),
        approvals_timeout=args.approvals_timeout,
        allow_patterns=args.allow or [],
    )
    mode = "enforce" if args.enforce else "observe"
    print(f"vaara proxy: listening on {host}:{port} -> {args.upstream}, "
          f"trail {trail_path}, mode {mode}")

    import uvicorn

    uvicorn.run(app, host=host, port=port)
    return 0


def _cmd_verify_bundle(args: argparse.Namespace) -> int:
    """Verify a whole evidence bundle from disk in one command.

    Reads one bundle JSON document (or a directory holding ``bundle.json``),
    runs every applicable verification lens through ``verify_evidence_bundle``,
    and prints one verdict. Exit 0 iff the bundle is ``ok``: the receipt
    signature was established and every lens whose evidence was present passed.
    """
    try:
        from vaara.attestation.receipt import (
            evidence_bundle_from_json,
            verify_evidence_bundle,
        )
    except ImportError:
        print(_ATTESTATION_HINT, file=sys.stderr)
        return 2

    path = Path(args.bundle).expanduser()
    if path.is_dir():
        candidates = [path / name for name in ("bundle.json", "evidence_bundle.json")]
        found = next((c for c in candidates if c.is_file()), None)
        if found is None:
            print(
                f"vaara verify-bundle: no bundle.json or evidence_bundle.json "
                f"in directory {path}",
                file=sys.stderr,
            )
            return 2
        path = found
    elif not path.is_file():
        print(
            f"vaara verify-bundle: not a file or directory: {path}",
            file=sys.stderr,
        )
        return 2

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"vaara verify-bundle: cannot read bundle JSON: {exc}", file=sys.stderr)
        return 1
    try:
        bundle = evidence_bundle_from_json(doc)
    except ValueError as exc:
        print(
            f"vaara verify-bundle: not a valid evidence bundle: {exc}",
            file=sys.stderr,
        )
        return 1

    trusted_material = None
    if getattr(args, "pubkey", None):
        try:
            from cryptography.exceptions import UnsupportedAlgorithm
            from cryptography.hazmat.primitives.serialization import (
                load_pem_public_key,
            )

            trusted_material = load_pem_public_key(
                Path(args.pubkey).expanduser().read_bytes()
            )
        except (OSError, ValueError, UnsupportedAlgorithm) as exc:
            print(f"vaara verify-bundle: cannot load --pubkey: {exc}", file=sys.stderr)
            return 1

    verdict = verify_evidence_bundle(bundle, trusted_verifying_material=trusted_material)

    if args.json:
        print(json.dumps(verdict.to_dict(), indent=2))
    else:
        print(f"verdict: {'OK' if verdict.ok else 'FAILED'}")
        print(f"  authenticity established: {verdict.authenticity_established}")
        if verdict.keyid:
            print(f"  identity keyid: {verdict.keyid}")
        print("  lenses:")
        for r in verdict.lenses:
            state = ("pass" if r.ok else "FAIL") if r.applicable else "n/a"
            print(f"    {r.lens:12s} {state:4s}  {r.reason}")
        print(f"  {verdict.reason}")

    return 0 if verdict.ok else 1


def _cmd_verify_record(args: argparse.Namespace) -> int:
    """Conformance-check any candidate SEP-2828 execution record.

    Keyless by design: it checks the wire schema and the binding the
    record proves about itself (``projectionDigest`` over the projection
    bytes), so a party that holds neither the signing key nor the
    attestation can still judge whether a record is well-formed. With
    ``--attestation`` the back-link is verified too, still without a key.
    To also check the signature, use ``vaara receipt verify`` with the
    signer's key.
    """
    path = Path(args.record).expanduser()
    if not path.is_file():
        print(f"vaara verify-record: not a file: {path}", file=sys.stderr)
        return 2
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"vaara verify-record: cannot read record JSON: {exc}", file=sys.stderr)
        return 1

    # Route by record kind. A record carrying decisionDerived is a decision
    # record and is graded by the decision checker, which covers the native
    # rationale, binding, and decisionProof envelope; everything else is an
    # execution record. Without this the decisionProof checks only ran on the
    # directory/set path.
    is_decision = isinstance(doc, dict) and "decisionDerived" in doc
    if is_decision:
        from vaara.attestation._decision_conformance import check_decision_conformance

        report = check_decision_conformance(doc)
        proof = _verify_single_decision_proof(doc)
    else:
        from vaara.attestation.receipt import check_record_conformance

        report = check_record_conformance(doc)
        proof = None
    back_link = _record_back_link(args.attestation, doc) if args.attestation else None
    existence = _verify_single_existence_proof(
        doc, getattr(args, "trusted_issuer_cert", None)
    )

    if args.json:
        out: dict[str, Any] = {"conformance": report.to_dict()}
        if proof is not None:
            out["decisionProof"] = proof
        if back_link is not None:
            out["backLink"] = back_link
        if existence is not None:
            out["existenceProof"] = existence
        print(json.dumps(out, indent=2))
    else:
        print(f"conformance: {'CONFORMS' if report.conforms else 'NON-CONFORMING'}")
        for c in report.checks:
            state = ("pass" if c.ok else "FAIL") if c.severity == "required" else (
                "ok" if c.ok else "warn")
            print(f"  [{state:4s}] {c.severity:8s} {c.id}")
            if not c.ok:
                print(f"           {c.detail}")
        if proof is not None:
            if proof["ran"]:
                pstate = "pass" if proof["ok"] else "FAIL"
            else:
                pstate = "n/a"
            print(f"  decisionProof: {pstate}  {proof['detail']}")
        if back_link is not None:
            bstate = "n/a" if back_link.get("skipped") else (
                "pass" if back_link["ok"] else "FAIL")
            print(f"  back-link: {bstate}  {back_link['detail']}")
        if existence is not None:
            if not existence["ran"]:
                estate = "n/a"
            elif not existence["ok"]:
                estate = "FAIL"
            else:
                estate = "qualified" if existence["qualified"] else "self-asserted"
            at = existence.get("attestedTime")
            suffix = f"  @ {at}" if at else ""
            print(f"  existence: {estate}{suffix}  {existence['detail']}")

    if not report.conforms:
        return 1
    if proof is not None and proof["ran"] and not proof["ok"]:
        return 1
    if back_link is not None and not back_link["ok"] and not back_link.get("skipped"):
        return 1
    if existence is not None and existence["ran"] and not existence["ok"]:
        return 1
    return 0


def _verify_single_decision_proof(doc: Any) -> Optional[dict[str, Any]]:
    """Cryptographically verify a decisionProof if the record carries one.

    Returns ``None`` when there is no proof to check, else
    ``{ran, ok, detail}``. ``ran`` is False when the proof engine is not
    importable, in which case the proof is reported present-but-unverified
    rather than failing the record.
    """
    dd = doc.get("decisionDerived") if isinstance(doc, dict) else None
    if not isinstance(dd, dict) or "decisionProof" not in dd:
        return None
    try:
        from vaara.attestation._decision_proof_verify import verify_decision_proof
    except ImportError:
        return {"ran": False, "ok": None,
                "detail": "proof engine unavailable; proof present but unverified"}
    ok, reason = verify_decision_proof(dd)
    return {"ran": True, "ok": ok, "detail": reason}


def _verify_single_existence_proof(
    doc: Any, trusted_issuer_cert_path: Optional[str]
) -> Optional[dict[str, Any]]:
    """Verify a record's existenceProof if it carries one.

    Returns ``None`` when absent, else
    ``{ran, ok, qualified, basis, attestedTime, detail}``. ``ran`` is False
    when the timeanchor extra is missing, in which case the proof is reported
    present-but-unverified rather than failing the record. A missing or
    non-matching trusted-list pin makes the attested time self-asserted, not a
    failure; only a broken or mis-imprinted token sets ``ok`` False.
    """
    if not (isinstance(doc, dict) and isinstance(doc.get("existenceProof"), dict)):
        return None
    from vaara.attestation.receipt import verify_existence_proof

    pin = None
    if trusted_issuer_cert_path:
        try:
            pin = Path(trusted_issuer_cert_path).expanduser().read_bytes()
        except OSError as exc:
            return {"ran": False, "ok": None, "qualified": False, "basis": None,
                    "attestedTime": None,
                    "detail": f"cannot read trusted issuer cert: {exc}"}
    try:
        res = verify_existence_proof(doc, trusted_issuer_cert=pin)
    except ImportError:
        return {"ran": False, "ok": None, "qualified": False, "basis": None,
                "attestedTime": None,
                "detail": "timeanchor extra unavailable; existence proof present "
                          "but unverified"}
    return {"ran": True, "ok": res.ok, "qualified": res.qualified,
            "basis": res.basis, "attestedTime": res.attested_time,
            "detail": res.reason}


def _record_back_link(attestation_path: str, doc: Any) -> dict[str, Any]:
    """Keyless back-link check: does the record pin this attestation?

    Returns ``{ok, skipped, detail}``. ``skipped`` is True when the check
    could not run (extra missing, attestation unreadable, record not a
    parseable receipt) and so does not gate the verdict; a False ``ok``
    with ``skipped`` False is a real back-link failure.
    """
    try:
        from vaara.attestation.receipt import parse_receipt, verify_back_link
        from vaara.attestation.tool_call_attestation import AttestationError, parse_attestation
    except ImportError:
        return {"ok": False, "skipped": True,
                "detail": "back-link check needs the attestation extra "
                          "(pip install 'vaara[attestation]')"}
    p = Path(attestation_path).expanduser()
    if not p.is_file():
        return {"ok": False, "skipped": True, "detail": f"attestation not a file: {p}"}
    try:
        attestation = parse_attestation(json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, AttestationError, KeyError, TypeError,
            ValueError) as exc:
        return {"ok": False, "skipped": True,
                "detail": f"attestation not parseable: {exc}"}
    try:
        receipt = parse_receipt(doc)
    except (AttestationError, KeyError, TypeError, ValueError) as exc:
        return {"ok": False, "skipped": True,
                "detail": f"record not parseable as a receipt: {exc}"}
    try:
        result = verify_back_link(receipt, attestation=attestation)
    except AttestationError as exc:
        # Computing the attestation digest needs rfc8785 (the attestation
        # extra). Absent it, the back-link cannot be checked: skip, do not gate.
        return {"ok": False, "skipped": True,
                "detail": f"back-link check unavailable: {exc}"}
    return {"ok": result.ok, "skipped": False,
            "detail": ("record pins this attestation" if result.ok
                       else "record does not pin this attestation")}


def _load_json_file(path: str, label: str) -> Any:
    """Read and parse a JSON file, or raise ValueError with a CLI message."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"{label} not a file: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc


def _resolve_anchored_time(args: argparse.Namespace) -> "tuple[Optional[str], Optional[str]]":
    """Return (anchored_time, error). Verifies an --anchor, or echoes --anchored-time."""
    if args.anchored_time is not None:
        return args.anchored_time, None
    if args.anchor is None:
        return None, None
    try:
        from vaara.audit.timeanchor import TimeAnchor, TimeAnchorError, verify_anchor
    except ImportError:
        return None, ("verifying --anchor requires the 'timeanchor' extra "
                      "(pip install 'vaara[timeanchor]'); or pass --anchored-time "
                      "with a time you verified separately")
    try:
        anchor = TimeAnchor.from_dict(_load_json_file(args.anchor, "anchor"))
        attested = verify_anchor(anchor)
    except (ValueError, KeyError, TypeError, TimeAnchorError) as exc:
        return None, f"time anchor did not verify: {exc}"
    return attested.isoformat(), None


def _cmd_verify_retained(args: argparse.Namespace) -> int:
    """Verify a record under a key that has since rotated out or retired.

    The 7-year problem: an Article 12 record outlives the key that signed it.
    Binds the signature to a key the archived DID document lists, then judges
    the bound key's validity window and revocation at the claimed issuance
    time. A verified time anchor upgrades the verdict to corroborated. The key
    history and revocations default to what the document records; pass
    --key-history / --revocations to override with an out-of-band list.
    """
    try:
        from vaara.attestation.receipt import (
            KeyHistory,
            RevocationRegistry,
            parse_receipt,
            verify_receipt_retained,
        )
    except ImportError:
        print(_ATTESTATION_HINT, file=sys.stderr)
        return 2

    try:
        record = _load_json_file(args.record, "record")
        did_document = _load_json_file(args.did_document, "DID document")
        key_history = (
            KeyHistory.from_dict(_load_json_file(args.key_history, "key history"))
            if args.key_history else None
        )
        revocations = (
            RevocationRegistry.from_dict(
                _load_json_file(args.revocations, "revocations"))
            if args.revocations else None
        )
    except ValueError as exc:
        print(f"vaara verify-retained: {exc}", file=sys.stderr)
        return 1

    anchored_time, anchor_error = _resolve_anchored_time(args)
    if anchor_error is not None:
        print(f"vaara verify-retained: {anchor_error}", file=sys.stderr)
        return 1

    try:
        receipt = parse_receipt(record)
    except Exception as exc:  # noqa: BLE001 - any parse failure is a bad record
        print(f"vaara verify-retained: not a parseable record: {exc}",
              file=sys.stderr)
        return 1

    result = verify_receipt_retained(
        receipt, did_document, key_history=key_history, revocations=revocations,
        anchored_time=anchored_time, expected_keyid=args.keyid,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_retained_report(result)

    return 0 if result.verifiable else 1


def _print_retained_report(r: Any) -> None:
    # keyid and the window bounds come from a DID document the auditor may have
    # received from an untrusted producer; escape control characters so a
    # crafted value cannot forge extra lines in the human report.
    keyid = _safe_inline(r.keyid) if r.keyid else None
    print(f"verifiable: {'YES' if r.verifiable else 'NO'}")
    print(f"  bound:          {r.bound}" + (f"  ({keyid})" if keyid else ""))
    if r.window_recorded:
        window = (f"[{_safe_inline(r.not_before) if r.not_before else 'open'}, "
                  f"{_safe_inline(r.not_after) if r.not_after else 'open'})")
    else:
        window = "unbounded (no window recorded)"
    print(f"  within window:  {r.within_window}  {window}")
    print(f"  revoked:        {r.revoked}"
          + (f"  (revoked_at={_safe_inline(r.revoked_at)})" if r.revoked_at else ""))
    print(f"  time basis:     {r.time_basis}")
    if r.anchored_time is not None:
        print(f"  corroborated:   {r.corroborated}  "
              f"(anchor predates retirement={r.anchored_before_retirement}, "
              f"revocation={r.anchored_before_revocation})")
    print(f"  {_safe_inline(r.reason)}")


def _resolve_package_anchor_time(
    doc: Any, *, anchored_time: Optional[str] = None, no_anchor: bool = False,
) -> "tuple[Optional[str], Optional[str]]":
    """Return (anchored_time, note) for a handoff package's enclosed anchor.

    ``anchored_time`` (a time the regulator verified out of band) wins. Else,
    if the package carries an anchor and the timeanchor extra is installed, the
    RFC 3161 token is verified and its attested time returned. A missing extra
    or a token that does not verify yields no time and a note: the record stays
    verifiable, not corroborated, never silently upgraded.
    """
    if anchored_time is not None:
        return anchored_time, None
    if no_anchor:
        return None, None
    evidence = doc.get("evidence") if isinstance(doc, dict) else None
    anchor = evidence.get("anchor") if isinstance(evidence, dict) else None
    if not isinstance(anchor, dict):
        return None, None
    try:
        from vaara.audit.timeanchor import TimeAnchor, TimeAnchorError, verify_anchor
    except ImportError:
        return None, (
            "the package carries a time anchor but verifying it needs the "
            "'timeanchor' extra; the record can be verifiable, not corroborated"
        )
    try:
        attested = verify_anchor(TimeAnchor.from_dict(anchor))
    except (ValueError, KeyError, TypeError, TimeAnchorError) as exc:
        return None, f"the enclosed time anchor did not verify: {exc}"
    return attested.isoformat(), None


def _resolve_handoff_anchor_time(
    args: argparse.Namespace, doc: Any
) -> "tuple[Optional[str], Optional[str]]":
    """Args wrapper over :func:`_resolve_package_anchor_time` for verify-handoffs."""
    return _resolve_package_anchor_time(
        doc, anchored_time=args.anchored_time, no_anchor=args.no_anchor,
    )


def _load_handoff_docs(
    paths: "list[Path]", *,
    anchored_time: Optional[str] = None, no_anchor: bool = False,
    on_note: "Optional[Any]" = None,
) -> "tuple[list[tuple[Path, Any, Optional[str]]], list[tuple[str, str]]]":
    """Load handoff packages from disk and resolve each enclosed anchor.

    Shared by ``verify-handoffs`` and the ``export-article12`` fold: reads every
    path, resolves the anchor time (out-of-band override, else the enclosed
    RFC 3161 token), and returns ``(path, doc, anchor_time)`` for each readable
    package plus an ``unreadable`` list. Callers pick the entry name (file name
    vs stem). ``on_note(name, note)`` is called for any anchor-resolution note.
    """
    loaded: list[tuple[Path, Any, Optional[str]]] = []
    unreadable: list[tuple[str, str]] = []
    for path in paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            unreadable.append((path.name, str(exc)))
            continue
        anchor_time, note = _resolve_package_anchor_time(
            doc, anchored_time=anchored_time, no_anchor=no_anchor,
        )
        if note is not None and on_note is not None:
            on_note(path.name, note)
        loaded.append((path, doc, anchor_time))
    return loaded, unreadable


def _discover_enforcement_triples(
    record_paths: "list[Path]", directory: Path,
) -> "tuple[list[tuple[str, Any, bytes, bytes]], list[tuple[str, str]]]":
    """Discover ``(stem, record, report_bytes, vcek_pem)`` triples by stem.

    Shared by ``verify-enforcements`` and the ``export-article12`` fold:
    ``NAME.record.json`` pairs with ``NAME.report.bin`` and ``NAME.vcek.pem`` in
    the same directory. A record missing either companion, or an unreadable
    file, becomes a failing entry in the returned ``missing`` list, never a
    silent skip.
    """
    triples: list[tuple[str, Any, bytes, bytes]] = []
    missing: list[tuple[str, str]] = []
    for record_path in record_paths:
        if record_path.name.endswith(".record.json"):
            stem = record_path.name[: -len(".record.json")]
        else:
            stem = record_path.stem
        report_path = directory / f"{stem}.report.bin"
        vcek_path = directory / f"{stem}.vcek.pem"
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            missing.append((record_path.name, f"unreadable record ({exc})"))
            continue
        if not report_path.is_file():
            missing.append((record_path.name, f"no companion {report_path.name}"))
            continue
        if not vcek_path.is_file():
            missing.append((record_path.name, f"no companion {vcek_path.name}"))
            continue
        try:
            report_bytes = report_path.read_bytes()
            vcek_pem = vcek_path.read_bytes()
        except OSError as exc:
            missing.append((record_path.name, f"cannot read companion ({exc})"))
            continue
        triples.append((stem, record, report_bytes, vcek_pem))
    return triples, missing


def _collect_fold_attachments(
    args: argparse.Namespace,
) -> "tuple[Optional[list], Optional[list], Optional[Any], Optional[str]]":
    """Load the SEP-2828 attachments for the ``export-article12`` fold.

    Returns ``(handoffs, enforcements, trusted_did_document,
    expected_measurement)``, any of which may be ``None``. Single ``--handoff``
    files and a ``--handoffs`` directory both feed the handoff list (entry name
    = file stem, matching the verify verbs' stem discovery). Raises
    :class:`ValueError` on any load failure (a missing file, an unreadable
    package, an incomplete enforcement triple, a bad measurement hex) so the
    fold fails closed before the package is written.
    """
    handoffs: Optional[list] = None
    enforcements: Optional[list] = None
    trusted: Optional[Any] = None
    expected = args.expected_measurement

    hpaths: list[Path] = []
    for f in (args.handoff or []):
        p = Path(f).expanduser()
        if not p.is_file():
            raise ValueError(f"handoff package not found: {p}")
        hpaths.append(p)
    if args.handoffs:
        d = Path(args.handoffs).expanduser()
        if not d.is_dir():
            raise ValueError(f"--handoffs is not a directory: {d}")
        hpaths += sorted(p for p in d.glob("*.json") if p.is_file())
    if hpaths:
        loaded, unreadable = _load_handoff_docs(
            hpaths,
            on_note=lambda name, note: print(
                f"vaara export-article12: {name}: {note}", file=sys.stderr),
        )
        if unreadable:
            joined = "; ".join(f"{n} ({e})" for n, e in unreadable)
            raise ValueError(f"unreadable handoff package(s): {joined}")
        handoffs = [(p.stem, doc, t) for p, doc, t in loaded]

    if args.enforcements:
        d = Path(args.enforcements).expanduser()
        if not d.is_dir():
            raise ValueError(f"--enforcements is not a directory: {d}")
        recs = sorted(p for p in d.glob("*.record.json") if p.is_file())
        triples, missing = _discover_enforcement_triples(recs, d)
        if missing:
            joined = "; ".join(f"{n} ({e})" for n, e in missing)
            raise ValueError(f"incomplete enforcement triple(s): {joined}")
        enforcements = triples

    if args.trusted_did_document:
        trusted = _load_json_file(args.trusted_did_document, "trusted DID document")

    if expected is not None:
        try:
            raw = bytes.fromhex(expected.strip())
        except ValueError:
            raise ValueError("--expected-measurement is not valid hex") from None
        if len(raw) != 48:
            raise ValueError(
                "--expected-measurement must be 48 bytes (96 hex chars, an "
                "SHA-384 launch measurement)")
        expected = expected.strip()

    return handoffs, enforcements, trusted, expected


def _cmd_verify_handoff(args: argparse.Namespace) -> int:
    """Verify a cross-org handoff package: one org's record, another's regulator.

    The regulator side. Reads one self-contained handoff document, recomputes
    every pinned component digest, routes the record through the retained-record
    lens, confirms an enclosed anchor binds to this record, and prints one
    verdict. Exit 0 iff ``ok`` for the chosen mode. Authenticity rests on the
    producer's signature against an identity the regulator establishes out of
    band; pass --trusted-did-document to pin it.
    """
    try:
        from vaara.attestation.receipt import verify_handoff
    except ImportError:
        print(_ATTESTATION_HINT, file=sys.stderr)
        return 2

    try:
        doc = _load_json_file(args.package, "handoff package")
        trusted = (
            _load_json_file(args.trusted_did_document, "trusted DID document")
            if args.trusted_did_document else None
        )
    except ValueError as exc:
        print(f"vaara verify-handoff: {exc}", file=sys.stderr)
        return 1

    anchored_time, note = _resolve_handoff_anchor_time(args, doc)
    if note is not None:
        print(f"vaara verify-handoff: {note}", file=sys.stderr)

    try:
        verdict = verify_handoff(
            doc, anchor_attested_time=anchored_time,
            trusted_did_document=trusted, strict=args.strict,
        )
    except ValueError as exc:
        print(f"vaara verify-handoff: not a valid handoff package: {exc}",
              file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(verdict.to_dict(), indent=2))
    else:
        _print_handoff_report(verdict)

    return 0 if verdict.ok else 1


def _print_handoff_report(v: Any) -> None:
    # producer, holder, keyid and component digests come from a package an
    # untrusted holder assembled; escape control characters so a crafted value
    # cannot forge extra lines in the human report.
    print(f"verdict: {'OK' if v.ok else 'FAILED'}" + ("  [strict]" if v.strict else ""))
    print(f"  integrity:        {v.integrity_ok}")
    producer = _safe_inline(v.producer) if v.producer else "(none)"
    print(f"  producer:         {producer}  [{v.producer_identity_basis}]")
    if v.holder:
        print(f"  holder:           {_safe_inline(v.holder)}")
    keyid = f"  ({_safe_inline(v.keyid)})" if v.keyid else ""
    print(f"  record:           bound={v.bound} verifiable={v.verifiable} "
          f"corroborated={v.corroborated}{keyid}")
    print(f"  window recorded:  {v.window_recorded}   revocation source: "
          f"{v.revocation_source_present}   revoked: {v.revoked}")
    if v.anchor_present:
        print(f"  anchor:           binds={v.anchor_binds} verified={v.anchor_verified}")
    hk = f"  ({_safe_inline(v.holder_keyid)})" if v.holder_keyid else ""
    print(f"  custody:          {v.custody}{hk}")
    print("  components:")
    for c in v.components:
        if c.present:
            state = "ok" if c.ok else "DRIFT"
        else:
            state = "absent" if c.ok else "unexpectedly pinned"
        print(f"    {c.name:13s} {state}")
    print(f"  {_safe_inline(v.reason)}")


def _cmd_verify_contiguity(args: argparse.Namespace) -> int:
    """Check authorization receipts for completeness gaps under one boundary.

    Reads the ``evidence`` half of each ``*-authz.json`` file and checks that the
    per-boundary sequence is contiguous. Exit 0 when complete, 1 when a gap is
    found, 2 on a usage or input error.
    """
    from vaara.credential import evidence_binding_ok, verify_contiguity

    # Opt-in out-of-band issuer key. Without it the completeness check is
    # keyless/structural (integrity only): a forger can renumber the held
    # evidence to hide a drop. With it, only receipts whose signature verifies
    # AND whose evidence binds to the signed digest are counted, so forged or
    # renumbered completeness cannot pass. Draft requires the signature check.
    verifying_material = None
    if getattr(args, "key", None):
        from cryptography.exceptions import UnsupportedAlgorithm
        from cryptography.hazmat.primitives import serialization

        try:
            verifying_material = serialization.load_pem_public_key(
                Path(args.key).expanduser().read_bytes()
            )
        except (OSError, ValueError, UnsupportedAlgorithm) as exc:
            print(f"vaara verify-contiguity: --key: {exc}", file=sys.stderr)
            return 2

    files: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob("*-authz.json")))
        else:
            files.append(p)
    if not files:
        print(
            "vaara verify-contiguity: no authorization receipts found",
            file=sys.stderr,
        )
        return 2

    evidence: list[dict] = []
    dropped = 0
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"vaara verify-contiguity: {f}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(payload, dict):
            continue
        if verifying_material is not None:
            # Authenticated path: full {"record", "evidence"} payloads only.
            if not evidence_binding_ok(payload) or not _receipt_signature_verifies(
                payload, verifying_material
            ):
                dropped += 1
                continue
            evidence.append(payload["evidence"])
        else:
            # Keyless/structural path: accept a full payload or a bare record.
            ev = payload.get("evidence", payload)
            if isinstance(ev, dict):
                evidence.append(ev)

    if verifying_material is not None and dropped:
        print(
            f"vaara verify-contiguity: dropped {dropped} receipt(s) whose signature "
            "or evidence binding did not verify under --key; not counted",
            file=sys.stderr,
        )
    if verifying_material is None:
        print(
            "vaara verify-contiguity: NOTE structural completeness only — receipts "
            "not signature-verified; pass --key <issuer.pem> to authenticate",
            file=sys.stderr,
        )

    try:
        report = verify_contiguity(evidence, boundary_id=args.boundary_id)
    except ValueError as exc:
        print(f"vaara verify-contiguity: {exc}", file=sys.stderr)
        return 2

    # With --key, any dropped receipt means some records could not be
    # authenticated, so a completeness verdict over only the survivors is not a
    # trustworthy pass. In particular a wrong key drops every receipt, leaving
    # an empty set that verify_contiguity would otherwise report as ok=true
    # (a vacuous pass). Fail closed instead.
    ok = report.ok and not (verifying_material is not None and dropped)

    if args.json:
        print(
            json.dumps(
                {
                    "boundaryId": report.boundary_id,
                    "present": report.present,
                    "expected": report.expected,
                    "missingSeqs": report.missing_seqs,
                    "duplicateSeqs": report.duplicate_seqs,
                    "countMismatches": report.count_mismatches,
                    "dropped": dropped,
                    "ok": ok,
                },
                indent=2,
            )
        )
    else:
        print(report.gap_report())
        if verifying_material is not None and dropped:
            print(
                f"vaara verify-contiguity: FAILED — {dropped} receipt(s) did not "
                "verify under --key; completeness cannot be authenticated",
                file=sys.stderr,
            )
    return 0 if ok else 1


def _receipt_signature_verifies(payload: dict, verifying_material: Any) -> bool:
    """The receipt's ES256 record signature verifies under ``verifying_material``.

    ``payload`` is a full authorization receipt; only the ``record`` half is
    signed. Reuses the production decision-record verifier. Returns False on a
    malformed record rather than raising, so a bad receipt is dropped, not fatal.
    """
    from vaara.attestation.decision import (
        parse_decision_record,
        verify_decision_signature,
    )
    from vaara.attestation.tool_call_attestation import AttestationError

    try:
        record = parse_decision_record(payload["record"])
    except (AttestationError, KeyError, TypeError, ValueError):
        return False
    return verify_decision_signature(record, verifying_material=verifying_material)


def _cmd_enforce_by_class(args: argparse.Namespace) -> int:
    """Gate the next unattended action on a boundary's sealed worst-case class.

    Reads each ``*-authz.json`` as a full authorization receipt and consumes the
    sealed ``maxClass`` ONLY from receipts whose ``evidence`` binds to their
    signed ``decisionDerived.evidenceRef.digest`` (``evidence_binding_ok``). A
    receipt that does not bind, e.g. a ``maxClass`` relabeled after signing or an
    evidence-only file with no record to bind against, is dropped before gating,
    so a relabeled seal fails closed instead of loosening the gate. With ``--key``
    (an ES256 public PEM) each receipt's signature is verified too and any that
    does not verify is dropped, which closes the full forgery (recompute the
    digest AND re-sign needs the key). Permits iff the surviving sealed
    ``maxClass`` is in ``--permit``; fails closed when no class is sealed. Exit 0
    permit, 1 deny, 2 on a usage or input error.
    """
    from vaara.credential import enforce_on_sealed_class, evidence_binding_ok

    files: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.glob("*-authz.json")))
        else:
            files.append(p)
    if not files:
        print(
            "vaara enforce-by-class: no authorization receipts found",
            file=sys.stderr,
        )
        return 2

    verifying_material = None
    if args.key:
        from cryptography.hazmat.primitives import serialization

        key_path = Path(args.key).expanduser()
        try:
            verifying_material = serialization.load_pem_public_key(
                key_path.read_bytes()
            )
        except (OSError, ValueError) as exc:
            print(f"vaara enforce-by-class: --key: {exc}", file=sys.stderr)
            return 2

    evidence: list[dict] = []
    dropped_unbound = 0
    dropped_unsigned = 0
    for f in files:
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"vaara enforce-by-class: {f}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(payload, dict):
            continue
        if not evidence_binding_ok(payload):
            dropped_unbound += 1
            continue
        if verifying_material is not None and not _receipt_signature_verifies(
            payload, verifying_material
        ):
            dropped_unsigned += 1
            continue
        evidence.append(payload["evidence"])

    if dropped_unbound:
        print(
            f"vaara enforce-by-class: dropped {dropped_unbound} receipt(s) whose "
            "evidence does not bind to its signed digest; not trusted",
            file=sys.stderr,
        )
    if dropped_unsigned:
        print(
            f"vaara enforce-by-class: dropped {dropped_unsigned} receipt(s) whose "
            "signature did not verify under --key",
            file=sys.stderr,
        )

    permitted = args.permitted_classes or []
    try:
        decision = enforce_on_sealed_class(
            evidence, permitted, boundary_id=args.boundary_id
        )
    except ValueError as exc:
        print(f"vaara enforce-by-class: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "boundaryId": decision.boundary_id,
                    "permit": decision.permit,
                    "reason": decision.reason,
                    "worstCaseClass": decision.worst_case_class,
                    "permittedClasses": decision.permitted_classes,
                },
                indent=2,
            )
        )
    else:
        print(decision.gate_report())
    return 0 if decision.permit else 1


def _cmd_verify_enforcement(args: argparse.Namespace) -> int:
    """Verify a SEV-SNP attestation report binds a SEP-2828 record to a CVM.

    Reads the record, the binary SEV-SNP report, and the VCEK PEM, and prints one
    verdict. The verdict is honest about its limits: a pass proves a report
    carrying sha512(jcs(record)) verifies against the VCEK you supplied, not that
    the VCEK is genuine AMD silicon (the KDS chain is not validated) or that the
    decision logic ran in the enclave. Pass --expected-measurement to pin which
    image ran. Exit 0 iff ``ok``.
    """
    try:
        # Probe the extra eagerly: canonicalisation and the ECDSA check both
        # import lazily, so a bare receipt import succeeds without the extra and
        # would surface a missing dependency as a confusing record error.
        import rfc8785  # noqa: F401
        from cryptography.hazmat.primitives.asymmetric import ec  # noqa: F401

        from vaara.attestation.receipt import verify_enforcement
        from vaara.attestation.tee import TEEAttestationError
    except ImportError:
        print(_ATTESTATION_HINT, file=sys.stderr)
        return 2

    try:
        record = _load_json_file(args.record, "record")
    except ValueError as exc:
        print(f"vaara verify-enforcement: {exc}", file=sys.stderr)
        return 1

    report_path = Path(args.report).expanduser()
    if not report_path.is_file():
        print(f"vaara verify-enforcement: report not a file: {report_path}",
              file=sys.stderr)
        return 1
    vcek_path = Path(args.vcek).expanduser()
    if not vcek_path.is_file():
        print(f"vaara verify-enforcement: VCEK PEM not a file: {vcek_path}",
              file=sys.stderr)
        return 1

    expected = args.expected_measurement
    if expected is not None:
        try:
            raw = bytes.fromhex(expected.strip())
        except ValueError:
            print("vaara verify-enforcement: --expected-measurement is not valid hex",
                  file=sys.stderr)
            return 1
        if len(raw) != 48:
            print("vaara verify-enforcement: --expected-measurement must be 48 "
                  "bytes (96 hex chars, an SHA-384 launch measurement)",
                  file=sys.stderr)
            return 1

    try:
        report_bytes = report_path.read_bytes()
        vcek_pem = vcek_path.read_bytes()
    except OSError as exc:
        print(f"vaara verify-enforcement: cannot read input: {exc}",
              file=sys.stderr)
        return 1

    try:
        verdict = verify_enforcement(
            record, report_bytes, vcek_pem,
            expected_measurement=expected, strict=args.strict,
        )
    except (TEEAttestationError, ValueError, KeyError, TypeError) as exc:
        print(f"vaara verify-enforcement: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(verdict.to_dict(), indent=2))
    else:
        _print_enforcement_report(verdict)

    return 0 if verdict.ok else 1


def _print_enforcement_report(v: Any) -> None:
    # measurement, report_data and report_context come from a report the caller
    # supplied and Vaara did not produce; escape control characters so a crafted
    # value cannot forge extra lines in the human report.
    print(f"verdict: {'OK' if v.ok else 'FAILED'}  tier={v.tier}"
          + ("  [strict]" if v.strict else ""))
    version = f"  (version {v.report_version})" if v.report_version is not None else ""
    print(f"  parsed:            {v.parsed}{version}")
    print(f"  signature:         algo_ok={v.signature_algo_ok} "
          f"valid={v.signature_valid}  [{v.vcek_chain_basis}]")
    print(f"  record binding:    bound={v.bound}")
    meas = _safe_inline(v.measurement) if v.measurement else "(none)"
    print(f"  measurement:       {meas}  [{v.measurement_basis}]")
    print(f"  enforcement logic: {v.enforcement_logic_basis}")
    ctx = v.report_context
    if ctx:
        print(f"  report context:    vmpl={ctx.get('vmpl')} policy={ctx.get('policy')} "
              f"guest_svn={ctx.get('guest_svn')} reported_tcb={ctx.get('reported_tcb')}")
    print(f"  {_safe_inline(v.reason)}")


def _cmd_verify_tpm_binding(args: argparse.Namespace) -> int:
    """Verify a TPM 2.0 quote + IMA evidence bundle binds a SEP-2828 record.

    Reads one ``vaara.tpm-evidence-bundle/v0`` JSON file and prints one verdict.
    Honest limits: a pass proves the quote was signed by the AK you supplied, its
    extraData carries sha512(jcs(record)), the supplied PCR values recompute the
    signed digest, and the IMA log replays to the quoted PCR 10. It does not prove
    the AK belongs to a genuine TPM (the EK chain is not validated) or that the
    measured software decided anything (IMA measures files, not semantics). Pass
    expectedImaPcr in the bundle to pin which measured state ran. Exit 0 iff ok.
    """
    try:
        import rfc8785  # noqa: F401
        from cryptography.hazmat.primitives.asymmetric import ec  # noqa: F401

        from vaara.attestation._tpm import TPMAttestationError
        from vaara.attestation.receipt import verify_tpm_bundle
    except ImportError:
        print(_ATTESTATION_HINT, file=sys.stderr)
        return 2

    try:
        doc = _load_json_file(args.bundle, "TPM evidence bundle")
    except ValueError as exc:
        print(f"vaara verify-tpm-binding: {exc}", file=sys.stderr)
        return 1

    try:
        verdict = verify_tpm_bundle(doc, strict=args.strict)
    except (ValueError, TPMAttestationError) as exc:
        print(f"vaara verify-tpm-binding: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(verdict.to_dict(), indent=2))
    else:
        _print_tpm_binding_report(verdict)
    return 0 if verdict.ok else 1


def _print_tpm_binding_report(v: Any) -> None:
    # extraData, the PCR values and the IMA log come from a bundle the caller
    # supplied and Vaara did not produce; escape control characters so a crafted
    # value cannot forge extra lines in the human report.
    print(f"verdict: {'OK' if v.ok else 'FAILED'}  tier={v.tier}"
          + ("  [strict]" if v.strict else ""))
    print(f"  parsed:          {v.parsed}  magic_ok={v.magic_ok} "
          f"quote={v.attest_type_ok}")
    print(f"  signature:       algo_ok={v.signature_algo_ok} "
          f"valid={v.signature_valid}  [{v.ak_chain_basis}]")
    print(f"  record binding:  bound={v.bound}")
    print(f"  pcr digest:      recomputed={v.pcr_digest_recomputed}")
    quoted = _safe_inline(v.ima_pcr_quoted) if v.ima_pcr_quoted else "(none)"
    print(f"  IMA PCR {v.ima_pcr_index}:      {quoted}  [{v.pcr_pin_basis}]")
    print(f"  IMA replay:      replayed={v.ima_replayed} "
          f"entries={v.ima_log_entries}")
    print(f"  decision logic:  {v.decision_logic_basis}")
    print(f"  freshness:       {v.freshness_basis}")
    ctx = v.pcr_context
    if ctx:
        print(f"  quote context:   reset={ctx.get('reset_count')} "
              f"restart={ctx.get('restart_count')} "
              f"fw={ctx.get('firmware_version')} "
              f"pcrs={_safe_inline(str(ctx.get('selected_pcrs')))}")
    print(f"  {_safe_inline(v.reason)}")


def _cmd_verify_tpm_chain(args: argparse.Namespace) -> int:
    """Verify a continuous-attestation chain binds a SEP-2828 record over a window.

    Reads one ``vaara.tpm-evidence-chain/v0`` JSON file (an ordered list of TPM
    quotes + IMA logs) and prints one verdict. A ``continuous`` pass proves every
    link verifies and binds to the record in order, the TPM clock strictly advanced
    on one uninterrupted boot (no reboot), and the IMA log grew append-only across
    the window. It carries forward the Phase-0 limits: the AK is trusted as
    supplied (EK chain not validated) and IMA measures files, not decision
    semantics. The chain moves freshness from unestablished to chain-continuity; it
    is not a live verifier challenge. Exit 0 iff ok.
    """
    try:
        import rfc8785  # noqa: F401
        from cryptography.hazmat.primitives.asymmetric import ec  # noqa: F401

        from vaara.attestation._tpm import TPMAttestationError
        from vaara.attestation.receipt import verify_tpm_chain_bundle
    except ImportError:
        print(_ATTESTATION_HINT, file=sys.stderr)
        return 2

    try:
        doc = _load_json_file(args.chain, "TPM evidence chain")
    except ValueError as exc:
        print(f"vaara verify-tpm-chain: {exc}", file=sys.stderr)
        return 1

    try:
        verdict = verify_tpm_chain_bundle(doc, strict=args.strict)
    except (ValueError, TPMAttestationError) as exc:
        print(f"vaara verify-tpm-chain: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(verdict.to_dict(), indent=2))
    else:
        _print_tpm_chain_report(verdict)
    return 0 if verdict.ok else 1


def _print_tpm_chain_report(v: Any) -> None:
    print(f"verdict: {'OK' if v.ok else 'FAILED'}  tier={v.tier}"
          + ("  [strict]" if v.strict else ""))
    print(f"  links:           {v.n_links}  all_bound={v.links_bound}")
    print(f"  clock:           monotonic={v.clock_monotonic} "
          f"reboot_free={v.reboot_free}  [reset={v.reset_count} "
          f"restart={v.restart_count}]")
    print(f"  IMA growth:      append_only={v.ima_append_only}  "
          f"[{v.window.get('ima_entries_first')} -> "
          f"{v.window.get('ima_entries_last')} entries]")
    print(f"  attestation key: stable={v.ak_stable}  [{v.ak_chain_basis}]")
    print(f"  IMA PCR pin:     {v.pcr_pin_basis}")
    print(f"  window clock:    {v.window.get('clock_first')} -> "
          f"{v.window.get('clock_last')}")
    print(f"  decision logic:  {v.decision_logic_basis}")
    print(f"  freshness:       {v.freshness_basis}")
    failing = [
        i for i, lk in enumerate(v.links)
        if lk.get("tier") == "unverified"
    ]
    if failing:
        print(f"  failing links:   {failing}")
    print(f"  {_safe_inline(v.reason)}")


def _cmd_export_attestation_result(args: argparse.Namespace) -> int:
    """Re-express a Vaara attestation verdict as an IETF RATS EAR (Phase 2).

    Reads the JSON a ``verify-tpm-binding``, ``verify-tpm-chain``, or
    ``verify-enforcement`` ``--json`` run produced and emits a
    ``vaara.attestation-result/v0`` document: an EAR (draft-ietf-rats-ear) carrying
    an AR4SI trustworthiness vector, root-agnostic so a Relying Party reads a TPM and
    a SEV-SNP appraisal the same way. The mapping never claims more than the verdict:
    while the hardware root is trusted as supplied, the result tops out at the
    ``warning`` tier and ``affirming`` stays out of reach. The EAR is unsigned (it is
    the verifier's appraisal result; the evidence it appraises carries its own
    signatures). Pure standard library; no attestation extra needed.
    """
    # Imported from the leaf module so the export path stays base-install (it does
    # not parse evidence, only re-shapes a verdict the verify commands produced).
    from vaara import __version__
    from vaara.attestation._attestation_result import build_attestation_result

    try:
        verdict = _load_json_file(args.verdict, "verdict")
    except ValueError as exc:
        print(f"vaara export-attestation-result: {exc}", file=sys.stderr)
        return 1

    issued_at = args.iat if args.iat is not None else int(time.time())
    try:
        ear = build_attestation_result(
            verdict,
            issued_at=issued_at,
            verifier_build=f"vaara {__version__}",
            submod_label=args.submod,
        )
    except ValueError as exc:
        print(f"vaara export-attestation-result: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(ear, indent=2)
    if args.out:
        try:
            Path(args.out).expanduser().write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"vaara export-attestation-result: cannot write output: {exc}",
                  file=sys.stderr)
            return 1
        print(f"vaara export-attestation-result: wrote {args.out} "
              f"(ear_status={ear['ear_status']})", file=sys.stderr)
    else:
        print(rendered)
    return 0


def _cmd_build_handoff(args: argparse.Namespace) -> int:
    """Assemble a cross-org handoff package from the producer's pieces.

    The issuer side of ``verify-handoff``: stitches the record, the archived DID
    document, the key history, revocations, and an optional eIDAS time anchor
    into one self-contained document, pins each by digest, and writes it (sorted,
    two-space indent). A holder custody attestation is a programmatic step
    (``sign_manifest``); this command assembles the evidence. Exit 0 once a
    well-formed package is written, even if the record is not yet verifiable.
    """
    try:
        from vaara.attestation.receipt import build_handoff, verify_handoff
        from vaara.attestation.tool_call_attestation import AttestationError
    except ImportError:
        print(_ATTESTATION_HINT, file=sys.stderr)
        return 2

    try:
        record = _load_json_file(args.record, "record")
        did_document = _load_json_file(args.did_document, "DID document")
        key_history = (
            _load_json_file(args.key_history, "key history")
            if args.key_history else None
        )
        revocations = (
            _load_json_file(args.revocations, "revocations")
            if args.revocations else None
        )
        anchor = _load_json_file(args.anchor, "anchor") if args.anchor else None
        cover = _load_json_file(args.cover, "cover") if args.cover else None
    except ValueError as exc:
        print(f"vaara build-handoff: {exc}", file=sys.stderr)
        return 1

    try:
        doc = build_handoff(
            record=record, did_document=did_document, key_history=key_history,
            revocations=revocations, anchor=anchor, producer=args.producer,
            holder=args.holder, cover=cover,
        )
    except (AttestationError, KeyError, TypeError, ValueError) as exc:
        print(f"vaara build-handoff: cannot assemble handoff: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(doc, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(rendered)
        return 0

    out_path = Path(args.out).expanduser()
    try:
        out_path.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        print(f"vaara build-handoff: cannot write {out_path}: {exc}", file=sys.stderr)
        return 2
    verdict = verify_handoff(doc)
    print(f"Wrote handoff package to {out_path}")
    print(f"  integrity:         {verdict.integrity_ok}")
    print(f"  record verifiable: {verdict.verifiable}")
    print(f"  anchor present:    {verdict.anchor_present}")
    return 0


def _cmd_normalize(args: argparse.Namespace) -> int:
    """Map a foreign MCP record onto the SEP-2828 evidence model.

    Vaara is the receiving side: it reads a SEP-2643 denial, a tool-call
    attestation, or a SEP-2817 invocation audit context and reports which
    SEP-2828 evidence plane it fills, which fields it populates, and what
    is still missing for a complete signed execution record. It promotes
    nothing: an unsigned client claim stays advisory.
    """
    from vaara.attestation.receipt import normalize

    path = Path(args.record).expanduser()
    if not path.is_file():
        print(f"vaara normalize: not a file: {path}", file=sys.stderr)
        return 2
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"vaara normalize: cannot read record JSON: {exc}", file=sys.stderr)
        return 1

    result = normalize(doc, source_format=args.format)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_normalize_report(result)

    return 0 if result.recognized else 1


def _resolve_ingest_signer(args: argparse.Namespace) -> tuple[Any, Any, str]:
    """Resolve the signing material and alg for ``vaara ingest``.

    Returns ``(signing_material, alg, error)``. ``error`` is "" on success.
    HS256 takes a raw shared-secret file; a PEM private key signs ES256 (EC)
    or RS256 (RSA), the alg detected from the key type, mirroring the
    ``--pubkey-file`` / ``--hs256-secret-file`` split on the verify side.
    """
    if args.hs256_secret_file:
        try:
            secret = Path(args.hs256_secret_file).expanduser().read_bytes()
        except OSError as exc:
            return None, None, f"--hs256-secret-file: {exc}"
        return secret, "HS256", ""
    if args.key:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec, rsa
        try:
            priv = serialization.load_pem_private_key(
                Path(args.key).expanduser().read_bytes(), password=None
            )
        except (OSError, ValueError) as exc:
            return None, None, f"--key: {exc}"
        if isinstance(priv, ec.EllipticCurvePrivateKey):
            return priv, "ES256", ""
        if isinstance(priv, rsa.RSAPrivateKey):
            return priv, "RS256", ""
        return None, None, "--key: unsupported key type (need EC P-256 or RSA)"
    return None, None, "one of --key (PEM) or --hs256-secret-file is required"


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Seal one foreign evidence record into a signed vaara.ingest/v0 envelope.

    The universal sink. Reads any record ``normalize`` understands, maps it
    onto the SEP-2828 evidence model, and seals it into one signed,
    content-addressed envelope: ``evidenceRef.digest`` pins the normalized
    evidence, and the honest gap report rides inside that digest under the
    signature. Nothing the source did not establish is asserted; an
    unrecognized record still seals, honestly marked as such.
    """
    from vaara.attestation.receipt import emit_ingest_receipt, normalize
    from vaara.attestation.tool_call_attestation import AttestationError

    path = Path(args.record).expanduser()
    if not path.is_file():
        print(f"vaara ingest: not a file: {path}", file=sys.stderr)
        return 2
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"vaara ingest: cannot read record JSON: {exc}", file=sys.stderr)
        return 1

    signing_material, alg, error = _resolve_ingest_signer(args)
    if error:
        print(f"vaara ingest: {error}", file=sys.stderr)
        return 2

    normalized = normalize(doc, source_format=args.format)
    try:
        receipt = emit_ingest_receipt(
            normalized=normalized,
            iss=args.iss,
            sub=args.sub,
            secret_version=args.secret_version,
            alg=alg,
            signing_material=signing_material,
            evidence_ref=args.evidence_ref,
        )
    except AttestationError as exc:
        print(f"vaara ingest: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(receipt.to_dict(), indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out).expanduser()
        try:
            out_path.write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"vaara ingest: cannot write {out_path}: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {out_path} ({normalized.source_format})", file=sys.stderr)
    else:
        print(text)
    return 0


def _path_value(doc: dict[str, Any], dotted: str) -> Any:
    """Follow a dotted path into a nested dict; None if absent."""
    cur: Any = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _safe_inline(value: Any) -> str:
    """Render a value on one line, escaping control characters.

    Advisory and attestation-derived values come from a record Vaara did not
    produce; escaping C0 controls (newlines, tabs) stops a crafted value from
    forging extra lines in the human report.
    """
    return "".join(
        c if c.isprintable() else f"\\x{ord(c):02x}" for c in str(value)
    )


def _print_normalize_report(result: Any) -> None:
    if not result.recognized:
        print(f"source: unrecognized ({result.source_format})")
        for n in result.notes:
            print(f"  note: {n}")
        return
    print(f"source: {result.source_title}")
    print(f"evidence plane: {result.evidence_plane}")
    if result.populated:
        print("fills SEP-2828:")
        for pth in result.populated:
            val = _path_value(result.sep2828, pth)
            print(f"  {pth}" + (f" = {_safe_inline(val)}" if val is not None else ""))
    else:
        print("fills SEP-2828: nothing required (advisory context only)")
    if result.advisory:
        print("carries (advisory, not proof):")
        for k, v in result.advisory.items():
            print(f"  {k}: {_safe_inline(v)}")
    print("still needed for a complete signed record:")
    for m in result.missing:
        print(f"  {m}")
    for n in result.notes:
        print(f"  note: {n}")


def _cmd_verify_records(args: argparse.Namespace) -> int:
    """Conformance-check a whole directory of SEP-2828 records at once.

    The receiving side of the evidence: an auditor points this at a pile of
    records, possibly from more than one emitter, and gets the roll-up that
    ``verify-record`` cannot give on its own. How many conform, which fail
    and why, and the cross-record gaps: a call recorded twice, an executed
    action that committed no result. Keyless, like ``verify-record``.
    """
    from vaara.attestation.receipt import check_record_set

    directory = Path(args.directory).expanduser()
    if not directory.is_dir():
        print(f"vaara verify-records: not a directory: {directory}", file=sys.stderr)
        return 2

    paths = sorted(p for p in directory.glob(args.glob) if p.is_file())
    if not paths:
        print(f"vaara verify-records: no files matched {args.glob!r} in {directory}",
              file=sys.stderr)
        return 2

    parsed: list[tuple[str, Any]] = []
    unreadable: list[tuple[str, str]] = []
    for path in paths:
        try:
            parsed.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError) as exc:
            unreadable.append((path.name, str(exc)))

    report = check_record_set(parsed)
    ok = report.conforms and not unreadable

    if args.json:
        out: dict[str, Any] = report.to_dict()
        out["ok"] = ok
        out["unreadable"] = [{"name": n, "error": e} for n, e in unreadable]
        print(json.dumps(out, indent=2))
    else:
        verdict = "CONFORMS" if ok else "NON-CONFORMING"
        print(f"record set: {verdict}  ({report.conforming}/{report.total} records conform)")
        if report.verdict_counts:
            tally = ", ".join(f"{v}: {n}" for v, n in report.verdict_counts.items())
            print(f"  decisions: {tally}")
        if report.status_counts:
            tally = ", ".join(f"{s}: {n}" for s, n in report.status_counts.items())
            print(f"  outcomes: {tally}")
        for entry in report.entries:
            if not entry.conforms:
                print(f"  [FAIL] {entry.name}: {', '.join(entry.required_failed)}")
        for finding in report.findings:
            mark = "FAIL" if finding.severity == "required" else "warn"
            print(f"  [{mark}] {finding.id}: {finding.detail}")
            print(f"         {', '.join(finding.records)}")
        for name, exc in unreadable:
            print(f"  [FAIL] {name}: unreadable ({exc})")

    return 0 if ok else 1


def _cmd_audit_summary(args: argparse.Namespace) -> int:
    """Render a directory of records as a one-page audit summary a regulator reads.

    Runs the same keyless set check as ``verify-records`` and renders the
    verdict, the record counts, and the findings as a page of plain Markdown:
    what was checked, how many records conform, where the gaps are, and why the
    answer needs no signing key. Writes to ``--out`` or stdout. Exit 0 iff the
    set conforms and every file was readable, the same gate as ``verify-records``.
    """
    from vaara.attestation.receipt import check_record_set, render_record_set_summary

    directory = Path(args.directory).expanduser()
    if not directory.is_dir():
        print(f"vaara audit-summary: not a directory: {directory}", file=sys.stderr)
        return 2

    paths = sorted(p for p in directory.glob(args.glob) if p.is_file())
    if not paths:
        print(f"vaara audit-summary: no files matched {args.glob!r} in {directory}",
              file=sys.stderr)
        return 2

    parsed: list[tuple[str, Any]] = []
    unreadable: list[tuple[str, str]] = []
    for path in paths:
        try:
            parsed.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError) as exc:
            unreadable.append((path.name, str(exc)))

    report = check_record_set(parsed)
    page = render_record_set_summary(report)
    if unreadable:
        names = ", ".join(n for n, _ in unreadable)
        page += f"\n> Note: {len(unreadable)} file(s) could not be read: {names}\n"

    if args.out:
        Path(args.out).expanduser().write_text(page, encoding="utf-8")
        print(f"wrote audit summary to {args.out}", file=sys.stderr)
    else:
        print(page, end="")

    return 0 if (report.conforms and not unreadable) else 1


def _cmd_conformance_statement(args: argparse.Namespace) -> int:
    """Self-test this implementation against the published corpus and state the result.

    The answer to "trust us": run this implementation's keyless conformance
    check over the published SEP-2828 corpus, confirm the bytes match their
    manifest, optionally run the emitter's own records through the same set
    check, and print one reproducible statement that names the exact corpus
    byte set. Exit 0 iff the statement conforms (corpus verifies, the self-test
    reproduced every verdict, and any supplied records conform).
    """
    from vaara.attestation.receipt import (
        ConformanceCorpusError,
        build_conformance_statement,
        render_conformance_statement,
    )

    corpus = Path(args.corpus).expanduser()
    if not corpus.is_dir():
        print(f"vaara conformance-statement: not a corpus directory: {corpus}",
              file=sys.stderr)
        return 2

    records: list[tuple[str, Any]] | None = None
    unreadable: list[tuple[str, str]] = []
    if args.records is not None:
        records_dir = Path(args.records).expanduser()
        if not records_dir.is_dir():
            print(f"vaara conformance-statement: not a directory: {records_dir}",
                  file=sys.stderr)
            return 2
        paths = sorted(p for p in records_dir.glob(args.glob) if p.is_file())
        if not paths:
            print(f"vaara conformance-statement: no files matched {args.glob!r} in "
                  f"{records_dir}", file=sys.stderr)
            return 2
        records = []
        for path in paths:
            try:
                records.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, OSError) as exc:
                unreadable.append((path.name, str(exc)))

    try:
        statement = build_conformance_statement(
            corpus, records=records, unreadable=unreadable, as_of=args.as_of
        )
    except ConformanceCorpusError as exc:
        print(f"vaara conformance-statement: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(statement.to_dict(), indent=2))
    else:
        page = render_conformance_statement(statement)
        if args.out:
            Path(args.out).expanduser().write_text(page, encoding="utf-8")
            print(f"wrote conformance statement to {args.out}", file=sys.stderr)
        else:
            print(page, end="")

    return 0 if statement.conforms else 1


def _cmd_conformance_check(args: argparse.Namespace) -> int:
    """The one keyless SEP-2828 conformance front door: file or directory.

    A single memorable command over the checks that already ship. When
    ``path`` is a file it runs the single-record check (the same verdict as
    ``verify-record``); when it is a directory it runs the set check (the
    same roll-up and cross-record gaps as ``verify-records``). It owns no
    conformance logic of its own, so the two paths never drift from the
    commands they wrap. Keyless: runs in the base install.
    """
    path = Path(args.path).expanduser()
    if path.is_file():
        args.record = str(path)
        return _cmd_verify_record(args)
    if path.is_dir():
        args.directory = str(path)
        return _cmd_verify_records(args)
    print(f"vaara conformance check: no such path: {path}", file=sys.stderr)
    return 2


def _cmd_verify_bundles(args: argparse.Namespace) -> int:
    """Run the full lens stack over a whole directory of evidence bundles.

    The batch twin of ``verify-bundle``: an auditor points this at a pile of
    bundle documents, possibly from more than one issuer, and gets the
    roll-up a single-file check cannot give. How many verify, how many had
    their signature established, and what the evidence covers, with the
    advisory gap naming the lenses no bundle in the set exercised. Requires
    the attestation extra, like ``verify-bundle``.
    """
    try:
        from vaara.attestation.receipt import check_bundle_set
    except ImportError:
        print(_ATTESTATION_HINT, file=sys.stderr)
        return 2

    directory = Path(args.directory).expanduser()
    if not directory.is_dir():
        print(f"vaara verify-bundles: not a directory: {directory}", file=sys.stderr)
        return 2

    paths = sorted(p for p in directory.glob(args.glob) if p.is_file())
    if not paths:
        print(f"vaara verify-bundles: no files matched {args.glob!r} in {directory}",
              file=sys.stderr)
        return 2

    parsed: list[tuple[str, Any]] = []
    unreadable: list[tuple[str, str]] = []
    for path in paths:
        try:
            parsed.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError) as exc:
            unreadable.append((path.name, str(exc)))

    report = check_bundle_set(parsed)
    ok = report.ok and not unreadable

    if args.json:
        out: dict[str, Any] = report.to_dict()
        out["ok"] = ok
        out["unreadable"] = [{"name": n, "error": e} for n, e in unreadable]
        print(json.dumps(out, indent=2))
    else:
        verdict = "OK" if ok else "FAILED"
        print(f"bundle set: {verdict}  ({report.passed}/{report.total} bundles verify, "
              f"{report.authenticated} authenticated)")
        for name, count in report.lens_applicable.items():
            tally = f"{report.lens_passed[name]}/{count} pass" if count else "not present"
            print(f"  {name:12s} {tally}")
        if report.lens_gaps:
            print(f"  coverage gap: no bundle carried {', '.join(report.lens_gaps)}")
        for entry in report.entries:
            if not entry.loaded:
                print(f"  [FAIL] {entry.name}: not a valid bundle ({entry.error})")
            elif not entry.ok:
                failed = [ln for ln, st in entry.lens_states.items() if st == "fail"]
                reason = ", ".join(failed) if failed else "authenticity not established"
                print(f"  [FAIL] {entry.name}: {reason}")
        for name, exc in unreadable:
            print(f"  [FAIL] {name}: unreadable ({exc})")

    return 0 if ok else 1


def _cmd_verify_handoffs(args: argparse.Namespace) -> int:
    """Verify a whole directory of cross-org handoff packages at once.

    The batch twin of ``verify-handoff``: a regulator points this at a pile of
    packages, from one provider or several, and gets the roll-up a single-file
    check cannot give. How many records verify offline under their rotated-out
    keys, how many are anchor-corroborated rather than resting on the signature
    alone, and how many had their producer identity pinned. Each package's
    enclosed anchor is resolved independently (``--no-anchor`` skips it for all);
    a single out-of-band attested time cannot stand in for a whole set, so this
    has no ``--anchored-time``. Requires the attestation extra.
    """
    try:
        from vaara.attestation.receipt import check_handoff_set
    except ImportError:
        print(_ATTESTATION_HINT, file=sys.stderr)
        return 2

    directory = Path(args.directory).expanduser()
    if not directory.is_dir():
        print(f"vaara verify-handoffs: not a directory: {directory}", file=sys.stderr)
        return 2

    paths = sorted(p for p in directory.glob(args.glob) if p.is_file())
    if not paths:
        print(f"vaara verify-handoffs: no files matched {args.glob!r} in {directory}",
              file=sys.stderr)
        return 2

    try:
        trusted = (
            _load_json_file(args.trusted_did_document, "trusted DID document")
            if args.trusted_did_document else None
        )
    except ValueError as exc:
        print(f"vaara verify-handoffs: {exc}", file=sys.stderr)
        return 1

    loaded, unreadable = _load_handoff_docs(
        paths, anchored_time=args.anchored_time, no_anchor=args.no_anchor,
        on_note=lambda name, note: print(
            f"vaara verify-handoffs: {name}: {note}", file=sys.stderr),
    )
    packages: list[tuple[str, Any, Optional[str]]] = [
        (p.name, doc, t) for p, doc, t in loaded
    ]

    report = check_handoff_set(
        packages, trusted_did_document=trusted, strict=args.strict
    )
    ok = report.ok and not unreadable

    if args.json:
        out: dict[str, Any] = report.to_dict()
        out["ok"] = ok
        out["unreadable"] = [{"name": n, "error": e} for n, e in unreadable]
        print(json.dumps(out, indent=2))
    else:
        verdict = "OK" if ok else "FAILED"
        mode = "  [strict]" if report.strict else ""
        print(f"handoff set: {verdict}{mode}  ({report.passed}/{report.total} verify, "
              f"{report.corroborated} corroborated)")
        print(f"  verifiable: {report.verifiable}   corroborated: "
              f"{report.corroborated}   producer pinned: {report.pinned}")
        if report.pinning_gap:
            print("  coverage gap: no package pinned its producer identity "
                  "(every record's authenticity is self-asserted)")
        for entry in report.entries:
            if not entry.loaded:
                print(f"  [FAIL] {entry.name}: not a valid handoff package "
                      f"({_safe_inline(entry.error or '')})")
            elif not entry.ok:
                tier = "corroborated" if entry.corroborated else (
                    "verifiable" if entry.verifiable else "not verifiable")
                print(f"  [FAIL] {entry.name}: {tier}, "
                      f"identity {entry.producer_identity_basis}")
        for name, exc in unreadable:
            print(f"  [FAIL] {name}: unreadable ({exc})")

    return 0 if ok else 1


def _cmd_verify_enforcements(args: argparse.Namespace) -> int:
    """Bind a whole directory of (record, report, VCEK) triples at once.

    The batch twin of ``verify-enforcement``: an auditor points this at a
    directory of enforced records and gets the roll-up. How many bind to a
    confidential VM, at what tier, and whether any pinned a vetted launch image.
    Triples are discovered by stem: ``NAME.record.json`` pairs with
    ``NAME.report.bin`` and ``NAME.vcek.pem``. A record missing either companion
    is a failing entry, never a silent skip. Requires the attestation extra.
    """
    try:
        import rfc8785  # noqa: F401
        from cryptography.hazmat.primitives.asymmetric import ec  # noqa: F401

        from vaara.attestation.receipt import check_enforcement_set
    except ImportError:
        print(_ATTESTATION_HINT, file=sys.stderr)
        return 2

    directory = Path(args.directory).expanduser()
    if not directory.is_dir():
        print(f"vaara verify-enforcements: not a directory: {directory}",
              file=sys.stderr)
        return 2

    expected = args.expected_measurement
    if expected is not None:
        try:
            raw = bytes.fromhex(expected.strip())
        except ValueError:
            print("vaara verify-enforcements: --expected-measurement is not valid hex",
                  file=sys.stderr)
            return 1
        if len(raw) != 48:
            print("vaara verify-enforcements: --expected-measurement must be 48 "
                  "bytes (96 hex chars, an SHA-384 launch measurement)",
                  file=sys.stderr)
            return 1

    records = sorted(p for p in directory.glob(args.glob) if p.is_file())
    if not records:
        print(f"vaara verify-enforcements: no files matched {args.glob!r} in "
              f"{directory}", file=sys.stderr)
        return 2

    triples, missing = _discover_enforcement_triples(records, directory)

    report = check_enforcement_set(
        triples, expected_measurement=expected, strict=args.strict
    )
    ok = report.ok and not missing

    if args.json:
        out: dict[str, Any] = report.to_dict()
        out["ok"] = ok
        out["incomplete"] = [{"name": n, "error": e} for n, e in missing]
        print(json.dumps(out, indent=2))
    else:
        verdict = "OK" if ok else "FAILED"
        mode = "  [strict]" if report.strict else ""
        print(f"enforcement set: {verdict}{mode}  ({report.passed}/{report.total} bind, "
              f"{report.measurement_pinned} measurement-pinned)")
        tally = ", ".join(f"{t}: {n}" for t, n in report.tier_counts.items() if n)
        if tally:
            print(f"  tiers: {tally}")
        if report.pinning_gap:
            print("  coverage gap: no record pinned a launch measurement "
                  "(the set bound to a CVM, never to a vetted image)")
        for entry in report.entries:
            if not entry.loaded:
                print(f"  [FAIL] {entry.name}: {_safe_inline(entry.error or '')}")
            elif not entry.ok:
                print(f"  [FAIL] {entry.name}: tier {entry.tier}, "
                      f"measurement {entry.measurement_basis}")
        for name, exc in missing:
            print(f"  [FAIL] {name}: {exc}")

    return 0 if ok else 1


def _cmd_build_bundle(args: argparse.Namespace) -> int:
    """Assemble an evidence bundle on disk from the issuer's pieces.

    The issuer-side mirror of ``verify-bundle``: where that command checks a
    bundle, this one produces it. Reads each piece (the receipt, and whatever
    identity, signature, back-link, inclusion, consistency, and revocation
    material the issuer holds), assembles the single document ``verify-bundle``
    reads, and writes it (sorted, two-space indent: the exact bytes the
    ``bundle_doc_v0`` vectors commit). Then loads the written document straight
    back and reports the ``verify-bundle`` verdict, the round-trip the format
    promises. Exit 0 once a well-formed bundle is written; a partial bundle
    that is not yet ``ok`` is still a successful build and is reported, not
    rejected.
    """
    try:
        from vaara.attestation.receipt import (
            build_bundle_document,
            evidence_bundle_from_json,
            load_bundle_pieces_from_dir,
            verify_evidence_bundle,
        )
        from vaara.attestation.tool_call_attestation import AttestationError
    except ImportError:
        print(_ATTESTATION_HINT, file=sys.stderr)
        return 2

    pieces: dict[str, Any] = {}

    # Directory discovery first; explicit flags below override piece by piece.
    if args.from_dir is not None:
        try:
            pieces.update(load_bundle_pieces_from_dir(args.from_dir))
        except NotADirectoryError as exc:
            print(f"vaara build-bundle: {exc}", file=sys.stderr)
            return 2
        except (ValueError, OSError) as exc:
            print(f"vaara build-bundle: {exc}", file=sys.stderr)
            return 1

    flag_files = {
        "receipt": args.receipt,
        "did_document": args.did_document,
        "verifying_jwk": args.verifying_jwk,
        "attestation": args.attestation,
        "inclusion": args.inclusion,
        "consistency": args.consistency,
        "registry": args.registry,
    }
    for key, path_str in flag_files.items():
        if path_str is None:
            continue
        path = Path(path_str).expanduser()
        try:
            pieces[key] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            flag = "--" + key.replace("_", "-")
            print(
                f"vaara build-bundle: cannot read {flag} from {path}: {exc}",
                file=sys.stderr,
            )
            return 1

    if args.expected_keyid is not None:
        pieces["expected_keyid"] = args.expected_keyid
    if args.inclusion_leaf_hex is not None:
        pieces["inclusion_leaf_hex"] = args.inclusion_leaf_hex

    if "receipt" not in pieces:
        print(
            "vaara build-bundle: a receipt is required "
            "(--receipt PATH, or a receipt.json under --from-dir)",
            file=sys.stderr,
        )
        return 2

    try:
        doc = build_bundle_document(**pieces)
    except (AttestationError, KeyError, TypeError, ValueError) as exc:
        print(f"vaara build-bundle: cannot assemble bundle: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(doc, indent=2, sort_keys=True) + "\n"

    out_path = None
    if args.out is not None:
        out_path = Path(args.out).expanduser()
        try:
            out_path.write_text(rendered, encoding="utf-8")
        except OSError as exc:
            print(f"vaara build-bundle: cannot write {out_path}: {exc}", file=sys.stderr)
            return 2
    else:
        sys.stdout.write(rendered)

    # Round-trip self-check: load and verify the bytes just written, the same
    # way verify-bundle would. Reported on stderr so it never pollutes the
    # bundle JSON a caller may be piping from stdout.
    verdict = verify_evidence_bundle(evidence_bundle_from_json(doc))
    where = str(out_path) if out_path is not None else "stdout"
    state = "OK" if verdict.ok else "not ok"
    print(
        f"vaara build-bundle: assembled bundle to {where}; "
        f"verify-bundle verdict {state} ({verdict.reason})",
        file=sys.stderr,
    )
    return 0


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

    policy_path = getattr(args, "policy", None)
    policy_dir = getattr(args, "policy_dir", None)
    if policy_path and policy_dir:
        print(
            "vaara serve: pass either --policy or --policy-dir, not both.",
            file=sys.stderr,
        )
        return 2

    controller = None
    registry = None
    if policy_dir:
        from vaara.policy.registry import PolicyRegistry
        from vaara.policy.schema import PolicyError

        registry = PolicyRegistry()
        try:
            tenants = registry.load_directory(Path(policy_dir).expanduser())
        except PolicyError as exc:
            print(f"vaara serve: --policy-dir failed to load: {exc}", file=sys.stderr)
            return 2
        print(
            f"vaara serve: loaded {len(tenants)} tenant policies "
            f"(tenants={tenants!r})",
            file=sys.stderr,
        )
    elif policy_path:
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

    app = create_app(policy_controller=controller, policy_registry=registry)
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


def _cmd_mode_list(_args: argparse.Namespace) -> int:
    from vaara.policy.modes import _BY_NAME, available_modes

    for name in available_modes():
        m = _BY_NAME[name]
        print(
            f"{name:12s} escalate={m.escalate:.2f} deny={m.deny:.2f}  "
            f"{m.description}"
        )
    return 0


def _cmd_mode_show(args: argparse.Namespace) -> int:
    from vaara.policy.modes import get_mode

    try:
        m = get_mode(args.name)
    except KeyError as e:
        print(f"vaara mode show: {e}", file=sys.stderr)
        return 2
    print(f"mode:        {m.name}")
    print(f"escalate:    {m.escalate}")
    print(f"deny:        {m.deny}")
    print(f"description: {m.description}")
    print(f"watts:       {m.watt_profile}")
    return 0


def _cmd_mode_emit(args: argparse.Namespace) -> int:
    from vaara.policy.modes import emit_json, emit_yaml

    try:
        text = emit_yaml(args.name) if args.format == "yaml" else emit_json(args.name)
    except KeyError as e:
        print(f"vaara mode emit: {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        print(f"vaara mode emit: {e}", file=sys.stderr)
        return 1
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def _help_dispatch(parser: argparse.ArgumentParser):
    def _print(_args: argparse.Namespace) -> int:
        parser.print_help()
        return 0
    return _print


class _SuggestingParser(argparse.ArgumentParser):
    """ArgumentParser that adds a did-you-mean hint on unknown commands.

    argparse only grew ``suggest_on_error`` in Python 3.14; this backports
    the one case that matters for a CLI this wide: a mistyped subcommand.
    """

    def error(self, message: str):
        match = re.search(r"invalid choice: '([^']+)' \(choose from (.+)\)", message)
        if match:
            import difflib

            attempted = match.group(1)
            choices = [c.strip().strip("'\"") for c in match.group(2).split(",")]
            close = difflib.get_close_matches(attempted, choices, n=1)
            if close:
                message = (
                    f"invalid choice: {attempted!r}. Did you mean {close[0]!r}? "
                    f"Run `{self.prog} --help` for the full command list."
                )
            else:
                message = (
                    f"invalid choice: {attempted!r}. "
                    f"Run `{self.prog} --help` for the full command list."
                )
        super().error(message)


def _cmd_init(args: argparse.Namespace) -> int:
    from vaara.integrations import init_governance as ig

    if (args.proxy_enforce or args.proxy_allow) and not args.proxy_service:
        print(
            "--proxy-enforce/--proxy-allow configure the installed proxy "
            "service; add --proxy-service.",
            file=sys.stderr,
        )
        return 2
    trail_db = (
        Path(args.trail_db).expanduser() if args.trail_db else ig.DEFAULT_TRAIL_DB
    )
    report = ig.run_init(
        trail_db=trail_db,
        shadow=args.shadow,
        govern_mcp=not args.no_mcp,
        proxy_service=args.proxy_service,
        proxy_enforce=args.proxy_enforce,
        proxy_allow=args.proxy_allow,
    )

    if report.hooks_changed:
        print(f"Claude Code hooks written to {report.hooks_path}")
    else:
        print(f"Claude Code hooks already current at {report.hooks_path}")
    print(f"Trail: {report.trail_db}")

    for client in report.clients:
        if not client.exists:
            continue
        rewritten = report.mcp_rewritten.get(client.name)
        if rewritten:
            print(f"  {client.name}: governed {rewritten} MCP server(s)")
        elif client.ungoverned:
            print(f"  {client.name}: {client.ungoverned} ungoverned (not rewritten)")
        elif client.governed:
            print(f"  {client.name}: already governed ({client.governed})")

    if report.service_path is not None:
        print(f"Proxy service installed: {report.service_path}")

    for warning in report.warnings:
        print(f"  warning: {warning}", file=sys.stderr)

    print()
    print("Governance active. Reverse with: vaara ungovern")
    return 0


def _cmd_ungovern(args: argparse.Namespace) -> int:
    from vaara.integrations import init_governance as ig

    report = ig.run_ungovern()
    if report.hooks_changed:
        print(f"Removed Vaara hooks from {report.hooks_path}")
    else:
        print(f"No Vaara hooks found in {report.hooks_path}")
    for name in report.mcp_restored:
        print(f"  {name}: MCP config restored from backup")
    if not report.mcp_restored:
        print("  no MCP configs to restore")
    if report.service_removed:
        print("  proxy service removed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = _SuggestingParser(prog="vaara", description="Vaara AI Agent Execution Layer")
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")
    p.set_defaults(func=_help_dispatch(p))

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
    pk.add_argument(
        "--attest",
        action="store_true",
        help="Generate an EC P-256 (ES256) key for tool-call attestation "
             "signing with vaara-mcp-proxy --attest-signing-key, instead of "
             "an Ed25519 trail-signing key. Does not require --dev.",
    )
    pk.set_defaults(func=_cmd_keygen)

    pt = sub.add_parser("trail", help="Audit-trail commands")
    tsub = pt.add_subparsers(dest="trail_cmd", metavar="COMMAND")
    pt.set_defaults(func=_help_dispatch(pt))

    pe = tsub.add_parser("export", help="Export a signed, regulator-handoff trail zip")
    _add_trail_source_args(pe)
    pe.add_argument("--out", required=True, help="Path to write the signed zip")
    pe.add_argument("--key", required=True, help="Path to Ed25519 signing private key (PEM)")
    pe.add_argument("--agent-id", default="", help="Optional agent_id tag for the manifest")
    pe.set_defaults(func=_cmd_trail_export)

    pet = tsub.add_parser(
        "export-threshold",
        help="Export a k-of-n threshold-signed trail zip (no single-key issuance)",
    )
    pet.add_argument("--trail", required=True, help="Path to trail JSONL file")
    pet.add_argument("--out", required=True, help="Path to write the signed zip")
    pet.add_argument(
        "--key",
        required=True,
        action="append",
        help="Path to a custodian Ed25519 signing key (PEM). Repeat for each "
             "of the n custodians.",
    )
    pet.add_argument(
        "--threshold-k",
        required=True,
        type=int,
        help="Quorum: minimum valid custodian signatures required to verify.",
    )
    pet.add_argument("--agent-id", default="", help="Optional agent_id tag for the manifest")
    pet.set_defaults(func=_cmd_trail_export_threshold)

    pv = tsub.add_parser("verify", help="Verify a signed trail zip")
    pv.add_argument("--zip", required=True, help="Path to signed trail zip")
    pv.add_argument(
        "--pubkey",
        default=None,
        help="Path to Ed25519 public key (PEM). If omitted, uses signer_pubkey.pem from inside the zip.",
    )
    pv.set_defaults(func=_cmd_trail_verify)

    pva = tsub.add_parser(
        "verify-anchor",
        help="Verify the external time anchor in an Article 12 package",
    )
    pva.add_argument("--zip", required=True, help="Path to the package zip")
    pva.set_defaults(func=_cmd_trail_verify_anchor)

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

    pa12 = tsub.add_parser(
        "export-article12",
        help="Export a signed EU AI Act Article 12 regulator package",
    )
    _add_trail_source_args(pa12)
    pa12.add_argument(
        "--key", required=True,
        help="Ed25519 private key (PEM) to sign the package",
    )
    pa12.add_argument("--out", required=True, help="Path to write the package zip")
    pa12.add_argument(
        "--system-meta", default=None,
        help="Optional JSON with system identity (system_name, provider, "
             "deployer, intended_purpose, risk_classification). Absent fields "
             "render as 'not provided'.",
    )
    pa12.add_argument(
        "--period", default=None,
        help="Optional report lens START:END (YYYY-MM-DD:YYYY-MM-DD); either "
             "side may be empty. Narrows the summary counts only; the signed "
             "trail stays whole.",
    )
    pa12.add_argument(
        "--format", choices=("md", "html"), default="md",
        help="Human-readable report format (default: md)",
    )
    pa12.add_argument(
        "--agent-id", default="",
        help="Optional scope hint written to the manifest (does not filter records)",
    )
    pa12_anchor = pa12.add_mutually_exclusive_group()
    pa12_anchor.add_argument(
        "--anchor-tsa", default=None, metavar="URL",
        help="Fetch an RFC 3161 time anchor over the trail head from this "
             "Time-Stamp Authority and fold it in as Article 19 "
             "existence-in-time evidence. Needs the 'timeanchor' extra and "
             "network access.",
    )
    pa12_anchor.add_argument(
        "--anchor-file", default=None, metavar="PATH",
        help="Fold in a pre-fetched time anchor (TimeAnchor JSON over the "
             "trail head) instead of fetching one.",
    )
    pa12.add_argument(
        "--handoff", action="append", default=None, metavar="PKG.json",
        help="Fold a verified cross-org handoff package (Article 26(6)) in as a "
             "sidecar. Repeatable. Each is verified at export; one that does not "
             "verify fails the export. Needs the attestation extra.",
    )
    pa12.add_argument(
        "--handoffs", default=None, metavar="DIR",
        help="Fold every handoff package matching *.json in this directory.",
    )
    pa12.add_argument(
        "--enforcements", default=None, metavar="DIR",
        help="Fold confidential-VM enforcement bindings from this directory "
             "(NAME.record.json + NAME.report.bin + NAME.vcek.pem triples). "
             "Each is verified at export; one that does not bind fails the "
             "export.",
    )
    pa12.add_argument(
        "--trusted-did-document", default=None, metavar="DOC.json",
        help="DID document trusted out of band, used to pin handoff producer "
             "identity. Without it, producer identity stays self-asserted.",
    )
    pa12.add_argument(
        "--expected-measurement", default=None, metavar="HEX",
        help="Vetted SHA-384 launch measurement (96 hex chars) to pin folded "
             "enforcement bindings against.",
    )
    pa12.set_defaults(func=_cmd_trail_export_article12)

    pa50 = tsub.add_parser(
        "export-article50",
        help="Export a signed EU AI Act Article 50 transparency evidence package",
    )
    _add_trail_source_args(pa50)
    pa50.add_argument(
        "--key", required=True,
        help="Ed25519 private key (PEM) to sign the package",
    )
    pa50.add_argument("--out", required=True, help="Path to write the package zip")
    pa50.add_argument(
        "--system-meta", default=None,
        help="Optional JSON with system identity (system_name, provider, "
             "deployer). Absent fields are simply omitted from the report.",
    )
    pa50.add_argument(
        "--period", default=None,
        help="Optional report lens START:END (YYYY-MM-DD:YYYY-MM-DD); either "
             "side may be empty. Narrows the summary counts only; the signed "
             "trail stays whole.",
    )
    pa50.set_defaults(func=_cmd_trail_export_article50)

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

    prot = tsub.add_parser(
        "rotate",
        help="Export a signed archive, verify it, then purge old records (fail-closed)",
    )
    prot.add_argument("--db", required=True, help="Path to the audit SQLite DB")
    prot.add_argument("--out", required=True, help="Path to write the signed archive zip")
    prot.add_argument("--key", required=True, help="Path to Ed25519 signing private key (PEM)")
    prot.add_argument(
        "--retention-days", required=True, type=int,
        help="Records older than this many days are purged after the archive verifies",
    )
    prot.add_argument(
        "--dry-run", action="store_true",
        help="Export and verify the archive, report the purge count, delete nothing",
    )
    rot_scope = prot.add_mutually_exclusive_group(required=True)
    rot_scope.add_argument("--tenant", help="Restrict rotation to records with this tenant_id")
    rot_scope.add_argument(
        "--all-tenants", action="store_true",
        help="Rotate across all tenants in this DB",
    )
    prot.set_defaults(func=_cmd_trail_rotate)

    psr = tsub.add_parser(
        "shadow-report",
        help="Summarise what enforcement would have blocked over the trailing window",
    )
    psr.add_argument("--db", required=True, help="Path to the audit SQLite DB")
    psr.add_argument(
        "--days", type=int, default=7,
        help="Trailing window in days (default 7)",
    )
    psr.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default text)",
    )
    psr.set_defaults(func=_cmd_trail_shadow_report)

    pr = sub.add_parser(
        "review",
        help="Human-in-the-loop review queue (EU AI Act Article 14)",
    )
    rsub = pr.add_subparsers(dest="review_cmd", metavar="COMMAND")
    pr.set_defaults(func=_help_dispatch(pr))

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
    psub = pp_policy.add_subparsers(dest="policy_cmd", metavar="COMMAND")
    pp_policy.set_defaults(func=_help_dispatch(pp_policy))

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
    csub = pcr.add_subparsers(dest="compliance_cmd", metavar="COMMAND")
    pcr.set_defaults(func=_help_dispatch(pcr))
    pcrep = csub.add_parser(
        "report",
        help="Assemble and render an article-level evidence report",
    )
    pcrep.add_argument(
        "--db", required=True,
        help="Path to the audit SQLite DB to read evidence from",
    )
    pcrep.add_argument(
        "--format", choices=["md", "json", "narrative", "pdf"], default="md",
        help="Output format (default: md). 'pdf' requires the 'pdf' extra "
             "and writes binary, so --out is required for pdf.",
    )
    pcrep.add_argument(
        "--out", default=None,
        help="Write to file (default: stdout; required for --format pdf)",
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
        "--policy-dir",
        default=None,
        help=(
            "Directory of per-tenant policy files. Each *.yaml/*.yml/*.json "
            "file is loaded; filename stem becomes the tenant_id "
            "(default.yaml -> fallback). Mutually exclusive with --policy."
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
    dsub = pdetect.add_subparsers(dest="detect_cmd", metavar="COMMAND")
    pdetect.set_defaults(func=_help_dispatch(pdetect))

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

    povert = sub.add_parser(
        "overt",
        help=(
            "OVERT 1.0 attestation commands (Protocol Profile 1.0 Annex B.6)"
        ),
    )
    osub = povert.add_subparsers(dest="overt_cmd", metavar="COMMAND")
    povert.set_defaults(func=_help_dispatch(povert))
    pov_verify = osub.add_parser(
        "verify",
        help=(
            "Verify an OVERT 1.0 Base Envelope from any conformant emitter. "
            "Requires the attestation extra."
        ),
    )
    pov_verify.add_argument(
        "receipt",
        help="Path to a canonical CBOR file containing one Base Envelope",
    )
    pov_pk_group = pov_verify.add_mutually_exclusive_group(required=True)
    pov_pk_group.add_argument(
        "--pubkey-file",
        dest="pubkey_file",
        default=None,
        help="Path to a file containing the raw 32-byte Ed25519 public key",
    )
    pov_pk_group.add_argument(
        "--pubkey-hex",
        dest="pubkey_hex",
        default=None,
        help="Raw 32-byte Ed25519 public key encoded as 64 hex characters",
    )
    pov_verify.set_defaults(func=_cmd_overt_verify)

    def _add_jws_key_args(p_):
        g = p_.add_mutually_exclusive_group(required=True)
        g.add_argument(
            "--pubkey-file", dest="pubkey_file", default=None,
            help="Path to a PEM public key (SubjectPublicKeyInfo) for "
                 "ES256 / RS256 envelopes",
        )
        g.add_argument(
            "--hs256-secret-file", dest="hs256_secret_file", default=None,
            help="Path to a file holding the raw HS256 shared secret",
        )

    pattest = sub.add_parser(
        "attest",
        help="Vaara tool-call attestation commands",
    )
    asub = pattest.add_subparsers(dest="attest_cmd", metavar="COMMAND")
    pattest.set_defaults(func=_help_dispatch(pattest))
    pa_verify = asub.add_parser(
        "verify",
        help="Verify a tool-call attestation envelope's signature (and TTL). "
             "Requires the attestation extra.",
    )
    pa_verify.add_argument(
        "envelope", help="Path to a tool-call attestation JSON file",
    )
    _add_jws_key_args(pa_verify)
    pa_verify.add_argument(
        "--enforce-ttl", action="store_true",
        help="Fail if the attestation TTL has expired. Off by default: a saved "
             "attestation is durable evidence and its short TTL is normally "
             "long past at verification time.",
    )
    pa_verify.set_defaults(func=_cmd_attest_verify)

    preceipt = sub.add_parser(
        "receipt",
        help="Execution-receipt commands (post-execution sibling of the tool-call attestation)",
    )
    rcsub = preceipt.add_subparsers(dest="receipt_cmd", metavar="COMMAND")
    preceipt.set_defaults(func=_help_dispatch(preceipt))
    prc_verify = rcsub.add_parser(
        "verify",
        help="Verify an execution receipt: signature, back-link to its "
             "attestation, and optional result commitment. Requires the "
             "attestation extra.",
    )
    prc_verify.add_argument(
        "receipt", help="Path to an execution-receipt JSON file",
    )
    prc_verify.add_argument(
        "--attestation", required=True,
        help="Path to the tool-call attestation JSON the receipt answers "
             "(needed to verify the back-link)",
    )
    _add_jws_key_args(prc_verify)
    prc_verify.add_argument(
        "--result", default=None,
        help="Path to the runtime result JSON. When the receipt carries a "
             "result commitment, the commitment is verified against this.",
    )
    prc_verify.set_defaults(func=_cmd_receipt_verify)

    prc_aots = rcsub.add_parser(
        "anchor-ots",
        help="Add an OpenTimestamps witness anchor: submit the receipt's "
             "signed-payload digest to public OTS calendars (pending "
             "immediately, Bitcoin-final in hours). Stacks with rfc3161 "
             "anchors. Requires the ots extra.",
    )
    prc_aots.add_argument(
        "receipt", help="Path to a receipt JSON file (SPEC.md envelope)",
    )
    prc_aots.add_argument(
        "--calendar", action="append", default=None, metavar="URL",
        help="OTS calendar server URL (repeatable; defaults to the public "
             "alice/bob/finney calendars)",
    )
    prc_aots.add_argument(
        "--out", default=None,
        help="Write the anchored receipt here instead of updating in place",
    )
    prc_aots.set_defaults(func=_cmd_receipt_anchor_ots)

    prc_uots = rcsub.add_parser(
        "upgrade-ots",
        help="Upgrade a receipt's pending OpenTimestamps anchors: fetch the "
             "calendar upgrades and fold the Bitcoin attestation into the "
             "proof in place. Idempotent; a network failure leaves the "
             "anchor pending. Requires the ots extra.",
    )
    prc_uots.add_argument(
        "receipt", help="Path to a receipt JSON file carrying OTS anchors",
    )
    prc_uots.add_argument(
        "--out", default=None,
        help="Write the upgraded receipt here instead of updating in place",
    )
    prc_uots.set_defaults(func=_cmd_receipt_upgrade_ots)

    prc_render = rcsub.add_parser(
        "render",
        help="Render the receipt as a self-contained static HTML evidence "
             "page: decision, anchors, commitment chain, and the commands a "
             "skeptic runs to verify it without trusting the page.",
    )
    prc_render.add_argument(
        "receipt", help="Path to a receipt JSON file (SPEC.md envelope)",
    )
    prc_render.add_argument(
        "--out", default=None,
        help="Output HTML path (default: the receipt path with .html)",
    )
    prc_render.add_argument(
        "--title", default=None, help="Page <title> override",
    )
    prc_render.set_defaults(func=_cmd_receipt_render)

    pproxy = sub.add_parser(
        "proxy",
        help="Model-endpoint proxy: front an OpenAI-compatible or ollama "
             "server, pass traffic through unchanged, and record every tool "
             "call the model requests into a hash-chained audit trail.",
    )
    pproxy.add_argument(
        "--listen", default="127.0.0.1:8788",
        help="HOST:PORT to bind (default 127.0.0.1:8788)",
    )
    pproxy.add_argument(
        "--upstream", default="http://127.0.0.1:11434",
        help="Model server base URL (default http://127.0.0.1:11434, ollama)",
    )
    pproxy.add_argument(
        "--trail", default=str(Path.home() / ".vaara" / "proxy" / "audit.db"),
        help="SQLite audit trail path (default ~/.vaara/proxy/audit.db)",
    )
    pproxy.add_argument(
        "--enforce", action="store_true",
        help="Gate instead of observe: denied tool calls are rewritten out "
             "of the response; escalations block on the approvals handshake",
    )
    pproxy.add_argument(
        "--allow", action="append", default=None, metavar="PATTERN",
        help="Tool-name glob that passes without gating under --enforce "
             "(repeatable, e.g. --allow 'mcp__github__*'). Allowed calls "
             "are still recorded in the trail.",
    )
    pproxy.add_argument(
        "--approvals-dir", default=None,
        help="Approvals handshake directory for --enforce escalations "
             "(default: none, escalations fail closed)",
    )
    pproxy.add_argument(
        "--approvals-timeout", type=float, default=60.0,
        help="Seconds to wait for a human decision (default 60)",
    )
    pproxy.add_argument(
        "--signing-key", default=None,
        help="Optional signing key to also emit attestation/receipt pairs",
    )
    pproxy.add_argument(
        "--receipts-dir", default=None,
        help="Directory for signed pairs (required with --signing-key)",
    )
    pproxy.set_defaults(func=_cmd_proxy)

    pvb = sub.add_parser(
        "verify-bundle",
        help="Verify a complete evidence bundle in one command: one receipt "
             "plus whatever identity, signature, back-link, inclusion, "
             "consistency, and revocation evidence accompanies it, all to one "
             "verdict. Requires the attestation extra.",
    )
    pvb.add_argument(
        "bundle",
        help="Path to an evidence-bundle JSON file, or a directory containing "
             "bundle.json (or evidence_bundle.json)",
    )
    pvb.add_argument(
        "--json", action="store_true",
        help="Emit the full verdict as JSON instead of a human-readable summary",
    )
    pvb.add_argument(
        "--pubkey",
        help="Path to the issuer's public key (PEM) you hold OUT OF BAND. When "
             "given, authenticity means the receipt verifies under THIS key, so a "
             "self-consistent bundle signed with an attacker's own key is rejected. "
             "Without it, the verdict is keyless internal consistency only.",
    )
    pvb.set_defaults(func=_cmd_verify_bundle)

    pvr = sub.add_parser(
        "verify-record",
        help="Conformance-check any candidate SEP-2828 execution record "
             "against the wire schema and its self-proving digest. Keyless: "
             "needs no signing key and no attestation. With --attestation the "
             "back-link is checked too (still keyless). The neutral check a "
             "party runs on a record someone else produced.",
    )
    pvr.add_argument(
        "record",
        help="Path to a JSON file claiming to be a SEP-2828 execution record",
    )
    pvr.add_argument(
        "--attestation", default=None,
        help="Path to the tool-call attestation the record answers; when given, "
             "the back-link is verified (no key needed)",
    )
    pvr.add_argument(
        "--trusted-issuer-cert", default=None,
        help="Path to a CA certificate (PEM or DER) you pin from a trusted list, "
             "e.g. an EU-trusted-list eIDAS QTSA issuer. When given, a record's "
             "existenceProof is graded QUALIFIED only if its timestamp signer "
             "chains to this CA; otherwise the attested time is reported as "
             "self-asserted. Needs the timeanchor extra.",
    )
    pvr.add_argument(
        "--json", action="store_true",
        help="Emit the full conformance report as JSON",
    )
    pvr.set_defaults(func=_cmd_verify_record)

    pvt = sub.add_parser(
        "verify-retained",
        help="Verify a record under a signing key that has since rotated out, "
             "over the Article 12 retention window. Binds the signature to a "
             "key the archived DID document lists, then checks the key was "
             "valid (not yet retired, not revoked) when the record was signed. "
             "A verified time anchor upgrades the verdict to corroborated: the "
             "record provably predates the key's end of life. Requires the "
             "attestation extra.",
    )
    pvt.add_argument(
        "record",
        help="Path to a JSON file claiming to be a SEP-2828 execution record",
    )
    pvt.add_argument(
        "--did-document", required=True, dest="did_document",
        help="Path to the archived DID document that lists the (now retired) "
             "verification key; per-method validFrom/validUntil and revoked "
             "markers are read from it unless overridden",
    )
    pvt.add_argument(
        "--key-history", default=None, dest="key_history",
        help="Optional key-history JSON ({version, keys:[{keyid, not_before, "
             "not_after}]}) overriding the windows the document records",
    )
    pvt.add_argument(
        "--revocations", default=None,
        help="Optional revocation-registry JSON overriding the document's "
             "revoked markers",
    )
    anchor_group = pvt.add_mutually_exclusive_group()
    anchor_group.add_argument(
        "--anchor", default=None,
        help="Path to an RFC 3161 time-anchor JSON; it is verified and its "
             "attested time corroborates existence (needs the timeanchor extra)",
    )
    anchor_group.add_argument(
        "--anchored-time", default=None, dest="anchored_time",
        help="An attested time you verified separately, used directly as the "
             "existence-in-time proof (ISO 8601)",
    )
    pvt.add_argument(
        "--keyid", default=None,
        help="Only try the named verification method (its DID-document id)",
    )
    pvt.add_argument(
        "--json", action="store_true",
        help="Emit the full retention verdict as JSON",
    )
    pvt.set_defaults(func=_cmd_verify_retained)

    pvh = sub.add_parser(
        "verify-handoff",
        help="Verify a cross-org handoff package: one organisation's signed "
             "record, checked by another organisation's regulator, offline, "
             "years later. Recomputes every pinned component digest, routes the "
             "record through the retained-record lens (rotated-key window + "
             "revocation + anchor corroboration), and confirms an enclosed "
             "eIDAS anchor binds to this record. Authenticity rests on the "
             "producer's signature against an identity you establish out of "
             "band; pass --trusted-did-document to pin it. Requires the "
             "attestation extra.",
    )
    pvh.add_argument(
        "package",
        help="Path to a JSON cross-org handoff package",
    )
    pvh.add_argument(
        "--trusted-did-document", default=None, dest="trusted_did_document",
        help="Path to the DID document you independently trust as the "
             "producer's (its retained key archive); pins producer identity",
    )
    pvh.add_argument(
        "--strict", action="store_true",
        help="Regulator-grade: pass only when the record is corroborated, with "
             "a recorded validity window, an affirmative revocation source, and "
             "a pinned producer identity",
    )
    anchor_h = pvh.add_mutually_exclusive_group()
    anchor_h.add_argument(
        "--anchored-time", default=None, dest="anchored_time",
        help="An attested time you verified separately, used directly as the "
             "existence-in-time proof (ISO 8601), instead of verifying the "
             "enclosed RFC 3161 token",
    )
    anchor_h.add_argument(
        "--no-anchor", action="store_true", dest="no_anchor",
        help="Do not verify an enclosed time anchor (report the record without "
             "corroboration)",
    )
    pvh.add_argument(
        "--json", action="store_true",
        help="Emit the full handoff verdict as JSON",
    )
    pvh.set_defaults(func=_cmd_verify_handoff)

    pvhs = sub.add_parser(
        "verify-handoffs",
        help="Verify a whole directory of cross-org handoff packages at once. "
             "The batch twin of verify-handoff: how many records verify offline "
             "under their rotated-out keys, how many are anchor-corroborated "
             "rather than resting on the signature alone, and how many had their "
             "producer identity pinned. Requires the attestation extra.",
    )
    pvhs.add_argument(
        "directory",
        help="Directory of cross-org handoff package JSON documents",
    )
    pvhs.add_argument(
        "--glob", default="*.json",
        help="Glob for handoff files within the directory (default: *.json)",
    )
    pvhs.add_argument(
        "--trusted-did-document", default=None, dest="trusted_did_document",
        help="Path to the DID document you independently trust as the "
             "producer's; pins producer identity across the whole set",
    )
    pvhs.add_argument(
        "--strict", action="store_true",
        help="Regulator-grade: every package must be corroborated, with a "
             "recorded window, an affirmative revocation source, and a pinned "
             "producer identity",
    )
    pvhs.add_argument(
        "--no-anchor", action="store_true", dest="no_anchor",
        help="Do not verify any enclosed time anchor (report each record "
             "without corroboration)",
    )
    pvhs.add_argument(
        "--json", action="store_true",
        help="Emit the full set verdict report as JSON",
    )
    # _resolve_handoff_anchor_time reads args.anchored_time; the batch has no
    # single out-of-band time, so it is always None here.
    pvhs.set_defaults(func=_cmd_verify_handoffs, anchored_time=None)

    pve = sub.add_parser(
        "verify-enforcement",
        help="Check whether a SEV-SNP attestation report binds a signed SEP-2828 "
             "record to a confidential VM whose VCEK you supply: the report's "
             "REPORT_DATA must carry sha512(jcs(record)) and its signature must "
             "verify against the VCEK, with an optional pinned launch "
             "measurement. It does not validate the VCEK chain to AMD's ARK (KDS "
             "deferred) or prove the decision logic ran in the enclave, so it "
             "does not by itself establish genuine AMD hardware. Requires the "
             "attestation extra.",
    )
    pve.add_argument(
        "record",
        help="Path to the SEP-2828 execution record (JSON) the report binds to",
    )
    pve.add_argument(
        "--report", required=True,
        help="Path to the binary AMD SEV-SNP attestation report (1184 bytes)",
    )
    pve.add_argument(
        "--vcek", required=True,
        help="Path to the PEM-encoded VCEK to check the report signature "
             "against. Trusted as supplied; its AMD KDS chain is not validated "
             "in v0.",
    )
    pve.add_argument(
        "--expected-measurement", default=None, dest="expected_measurement",
        help="Pin the report's launch measurement (96 hex chars / 48 bytes) "
             "against an independently vetted value; a mismatch fails the check",
    )
    pve.add_argument(
        "--strict", action="store_true",
        help="Regulator-grade: pass only at the chain-rooted attested tier "
             "(validated VCEK chain plus a pinned measurement), which v0 cannot "
             "yet reach, so a strict pass is honestly unavailable",
    )
    pve.add_argument(
        "--json", action="store_true",
        help="Emit the full enforcement verdict as JSON",
    )
    pve.set_defaults(func=_cmd_verify_enforcement)

    pvc = sub.add_parser(
        "verify-contiguity",
        help=(
            "Check authorization receipts for completeness gaps: the per-boundary "
            "sequence must be contiguous, so a dropped receipt is a provable gap "
            "with no issuer access and no external witness."
        ),
    )
    pvc.add_argument(
        "paths",
        nargs="+",
        help=(
            "Authorization receipt JSON files, or directories scanned for "
            "*-authz.json files"
        ),
    )
    pvc.add_argument(
        "--boundary",
        default=None,
        dest="boundary_id",
        help=(
            "Coverage boundary id to check; inferred when the receipts name "
            "exactly one boundary"
        ),
    )
    pvc.add_argument(
        "--json",
        action="store_true",
        help="Emit the contiguity report as JSON",
    )
    pvc.add_argument(
        "--key",
        default=None,
        help="Path to the issuer's public key (PEM), held OUT OF BAND. When given, "
             "only receipts whose signature and evidence binding verify are counted, "
             "so forged or renumbered completeness cannot pass. Without it the check "
             "is keyless/structural (integrity only).",
    )
    pvc.set_defaults(func=_cmd_verify_contiguity)

    pec = sub.add_parser(
        "enforce-by-class",
        help=(
            "Gate the next unattended action on a boundary's sealed worst-case "
            "class: permit iff the sealed maxClass is in --permit, fail closed "
            "when no class is sealed. The maxClass is consumed only from receipts "
            "whose evidence binds to its signed digest, so a relabeled seal fails "
            "closed; pass --key to also verify signatures. A permitted class "
            "permits even over a gap, since the seal bounds the gap at that class. "
            "Exit 0 permit, 1 deny."
        ),
    )
    pec.add_argument(
        "paths",
        nargs="+",
        help=(
            "Authorization receipt JSON files, or directories scanned for "
            "*-authz.json files"
        ),
    )
    pec.add_argument(
        "--permit",
        action="append",
        default=None,
        dest="permitted_classes",
        metavar="CLASS",
        help=(
            "An action class the consumer will proceed under unattended; repeat "
            "to permit several. The gate is membership, not an ordering."
        ),
    )
    pec.add_argument(
        "--boundary",
        default=None,
        dest="boundary_id",
        help=(
            "Coverage boundary id to gate on; inferred when the receipts name "
            "exactly one boundary"
        ),
    )
    pec.add_argument(
        "--key",
        default=None,
        metavar="PEM",
        help=(
            "ES256 public key (PEM) of the issuer; when given, each receipt's "
            "signature is verified and any that does not verify is dropped before "
            "gating. Without it, the evidence-to-digest binding is still enforced"
        ),
    )
    pec.add_argument(
        "--json",
        action="store_true",
        help="Emit the gate decision as JSON",
    )
    pec.set_defaults(func=_cmd_enforce_by_class)

    pvt = sub.add_parser(
        "verify-tpm-binding",
        help="Check whether a TPM 2.0 quote plus the kernel IMA log binds a "
             "signed SEP-2828 record to measured, un-tampered hardware: the "
             "quote's extraData must carry sha512(jcs(record)), its signature "
             "must verify against the AK you supply, the PCR values must "
             "recompute the signed pcrDigest, and the IMA log must replay to the "
             "quoted PCR 10. It does not validate the AK to a TPM vendor root (EK "
             "chain deferred) or prove the measured software made any decision "
             "(IMA measures files, not semantics), so it does not by itself "
             "establish a genuine TPM. Reads one vaara.tpm-evidence-bundle/v0 "
             "JSON file. Requires the attestation extra.",
    )
    pvt.add_argument(
        "bundle",
        help="Path to the vaara.tpm-evidence-bundle/v0 JSON file (record, quote, "
             "AK, PCR values, and IMA log; produced by scripts/tpm/)",
    )
    pvt.add_argument(
        "--strict", action="store_true",
        help="Regulator-grade: pass only at the EK-rooted attested tier "
             "(validated AK chain plus a pinned IMA PCR), which v0 cannot yet "
             "reach, so a strict pass is honestly unavailable",
    )
    pvt.add_argument(
        "--json", action="store_true",
        help="Emit the full TPM binding verdict as JSON",
    )
    pvt.set_defaults(func=_cmd_verify_tpm_binding)

    pvtc = sub.add_parser(
        "verify-tpm-chain",
        help="Check a continuous-attestation chain: an ordered sequence of TPM "
             "quotes plus IMA logs that each bind the same signed SEP-2828 record. "
             "A continuous pass proves every link verifies and binds in order "
             "(hash-linked so a link cannot be dropped, reordered, or spliced), the "
             "TPM clock strictly advanced on one uninterrupted boot (no reboot), "
             "and the IMA log grew append-only across the window. This is what "
             "moves freshness from unestablished (a lone quote) to chain-continuity; "
             "it is not a live verifier challenge. Same Phase-0 limits hold: the AK "
             "is trusted as supplied (EK chain deferred) and IMA measures files, not "
             "decisions. Reads one vaara.tpm-evidence-chain/v0 JSON file. Requires "
             "the attestation extra.",
    )
    pvtc.add_argument(
        "chain",
        help="Path to the vaara.tpm-evidence-chain/v0 JSON file (one record and an "
             "ordered list of quote + AK + PCR + IMA-log links; produced by "
             "scripts/tpm/)",
    )
    pvtc.add_argument(
        "--strict", action="store_true",
        help="Regulator-grade: pass only at the EK-rooted attested tier (validated "
             "AK chain), which v0 cannot yet reach, so a strict pass is honestly "
             "unavailable",
    )
    pvtc.add_argument(
        "--json", action="store_true",
        help="Emit the full TPM chain verdict as JSON",
    )
    pvtc.set_defaults(func=_cmd_verify_tpm_chain)

    pear = sub.add_parser(
        "export-attestation-result",
        help="Re-express an attestation verdict as an IETF RATS EAR "
             "(draft-ietf-rats-ear) carrying an AR4SI trustworthiness vector "
             "(draft-ietf-rats-ar4si). Reads the JSON a verify-tpm-binding, "
             "verify-tpm-chain, or verify-enforcement --json run produced and emits a "
             "vaara.attestation-result/v0 document, root-agnostic so a Relying Party "
             "reads a TPM and a SEV-SNP appraisal the same way. The mapping never "
             "claims more than the verdict: while the hardware root is trusted as "
             "supplied the result tops out at the warning tier and affirming stays "
             "out of reach. The EAR is unsigned. Pure stdlib; no attestation extra.",
    )
    pear.add_argument(
        "verdict",
        help="Path to a verdict JSON file (the output of verify-tpm-binding, "
             "verify-tpm-chain, or verify-enforcement run with --json)",
    )
    pear.add_argument(
        "--out", default=None,
        help="Write the EAR document here instead of stdout",
    )
    pear.add_argument(
        "--iat", type=int, default=None,
        help="Appraisal time as integer epoch seconds (the EAR iat); defaults to "
             "now. Pin it for reproducible output.",
    )
    pear.add_argument(
        "--submod", default=None,
        help="Override the EAR submodule label (defaults to the root type: tpm or "
             "sev-snp)",
    )
    pear.set_defaults(func=_cmd_export_attestation_result)

    pves = sub.add_parser(
        "verify-enforcements",
        help="Bind a whole directory of (record, report, VCEK) triples at once. "
             "The batch twin of verify-enforcement: how many records bind to a "
             "confidential VM, at what tier, and whether any pinned a vetted "
             "launch image. Triples are discovered by stem: NAME.record.json "
             "pairs with NAME.report.bin and NAME.vcek.pem. Requires the "
             "attestation extra.",
    )
    pves.add_argument(
        "directory",
        help="Directory of NAME.record.json records with NAME.report.bin and "
             "NAME.vcek.pem companions",
    )
    pves.add_argument(
        "--glob", default="*.record.json",
        help="Glob for record files within the directory (default: "
             "*.record.json)",
    )
    pves.add_argument(
        "--expected-measurement", default=None, dest="expected_measurement",
        help="Pin every report's launch measurement (96 hex chars / 48 bytes) "
             "against an independently vetted value; a mismatch fails that "
             "record",
    )
    pves.add_argument(
        "--strict", action="store_true",
        help="Regulator-grade: every record must reach the chain-rooted attested "
             "tier, which v0 cannot yet reach, so a strict pass is honestly "
             "unavailable",
    )
    pves.add_argument(
        "--json", action="store_true",
        help="Emit the full set verdict report as JSON",
    )
    pves.set_defaults(func=_cmd_verify_enforcements)

    pbh = sub.add_parser(
        "build-handoff",
        help="Assemble a cross-org handoff package from the producer's pieces: "
             "the record, the archived DID document, the key history, "
             "revocations, and an optional eIDAS time anchor, each pinned by "
             "digest. The issuer side of verify-handoff. Requires the "
             "attestation extra.",
    )
    pbh.add_argument(
        "--record", required=True,
        help="Path to the SEP-2828 execution record (JSON)",
    )
    pbh.add_argument(
        "--did-document", required=True, dest="did_document",
        help="Path to the archived DID document listing the signing key",
    )
    pbh.add_argument(
        "--key-history", default=None, dest="key_history",
        help="Optional key-history JSON overriding the document's windows",
    )
    pbh.add_argument(
        "--revocations", default=None,
        help="Optional revocation-registry JSON overriding the document's "
             "revoked markers (an empty registry affirmatively states none)",
    )
    pbh.add_argument(
        "--anchor", default=None,
        help="Optional RFC 3161 time-anchor JSON over sha256(jcs(record))",
    )
    pbh.add_argument(
        "--producer", default=None,
        help="Producer DID (defaults to the record issuer; must equal it)",
    )
    pbh.add_argument(
        "--holder", default=None,
        help="Holder DID, the deployer relaying the evidence (informational)",
    )
    pbh.add_argument(
        "--cover", default=None,
        help="Optional plain-language cover JSON (system, action, period, "
             "provider, deployer, the obligation served); carried pinned, "
             "Vaara asserts no legal conclusion about it",
    )
    pbh.add_argument(
        "--out", default=None,
        help="Write the package to this file instead of stdout",
    )
    pbh.set_defaults(func=_cmd_build_handoff)

    pnz = sub.add_parser(
        "normalize",
        help="Map an adjacent MCP record (SEP-2643 denial, tool-call "
             "attestation, or SEP-2817 invocation audit context) onto the "
             "SEP-2828 evidence model: which plane it fills, which fields it "
             "populates, and what is still missing for a complete signed "
             "record. The receiving side that reads every dialect.",
    )
    pnz.add_argument(
        "record",
        help="Path to a JSON file holding a SEP-2643, tool-call attestation, or SEP-2817 record",
    )
    pnz.add_argument(
        "--format", default="auto",
        choices=("auto", "sep2643", "vaara-attest", "sep2787", "sep2817"),
        help="Source format (default: auto-detect)",
    )
    pnz.add_argument(
        "--json", action="store_true",
        help="Emit the normalized evidence mapping as JSON",
    )
    pnz.set_defaults(func=_cmd_normalize)

    ping = sub.add_parser(
        "ingest",
        help="Seal any foreign evidence record into one signed, "
             "content-addressed vaara.ingest/v0 envelope: normalize it onto "
             "the SEP-2828 model, then sign, with the honest gap report bound "
             "under the signature. The universal sink.",
    )
    ping.add_argument(
        "record",
        help="Path to a JSON file holding a SEP-2643, tool-call attestation, or SEP-2817 record",
    )
    ping.add_argument(
        "--format", default="auto",
        choices=("auto", "sep2643", "vaara-attest", "sep2787", "sep2817"),
        help="Source format (default: auto-detect)",
    )
    ping.add_argument(
        "--key",
        help="PEM private key to sign with (EC P-256 -> ES256, RSA -> RS256)",
    )
    ping.add_argument(
        "--hs256-secret-file", dest="hs256_secret_file", default=None,
        help="Raw shared-secret file to sign with HS256 (alternative to --key)",
    )
    ping.add_argument("--iss", default="urn:vaara:ingest", help="Issuer identity")
    ping.add_argument("--sub", default="ingest", help="Subject")
    ping.add_argument(
        "--secret-version", dest="secret_version", default="1",
        help="Signing-key version label (default: 1)",
    )
    ping.add_argument(
        "--evidence-ref", dest="evidence_ref", default=None,
        help="Optional non-authoritative locator (URI/path) for the evidence bytes",
    )
    ping.add_argument(
        "--out", default=None,
        help="Write the {record, evidence} JSON here (default: stdout)",
    )
    ping.set_defaults(func=_cmd_ingest)

    pvrs = sub.add_parser(
        "verify-records",
        help="Conformance-check a whole directory of SEP-2828 records at once. "
             "The receiving side of the evidence: how many conform, which fail, "
             "and the cross-record gaps (a call recorded twice, an executed "
             "action that committed no result). Keyless, like verify-record.",
    )
    pvrs.add_argument(
        "directory",
        help="Directory of JSON files, each claiming to be a SEP-2828 record",
    )
    pvrs.add_argument(
        "--glob", default="*.json",
        help="Glob for record files within the directory (default: *.json)",
    )
    pvrs.add_argument(
        "--json", action="store_true",
        help="Emit the full set conformance report as JSON",
    )
    pvrs.set_defaults(func=_cmd_verify_records)

    pas = sub.add_parser(
        "audit-summary",
        help="Render a directory of SEP-2828 records as a one-page audit "
             "summary a regulator reads: the verdict, the record counts, and "
             "the findings as plain Markdown. The human-readable face of "
             "verify-records. Keyless.",
    )
    pas.add_argument(
        "directory",
        help="Directory of JSON files, each claiming to be a SEP-2828 record",
    )
    pas.add_argument(
        "--glob", default="*.json",
        help="Glob for record files within the directory (default: *.json)",
    )
    pas.add_argument(
        "--out", default=None,
        help="Write the Markdown summary to this file instead of stdout",
    )
    pas.set_defaults(func=_cmd_audit_summary)

    pcs = sub.add_parser(
        "conformance-statement",
        help="Self-test this implementation against the published SEP-2828 "
             "conformance corpus and print one reproducible statement: corpus "
             "bytes match the manifest, every recorded verdict was reproduced, "
             "and (with --records) your own records conform. The answer to "
             "'trust us': prove it against the neutral suite. Keyless.",
    )
    pcs.add_argument(
        "--corpus", default="conformance/sep2828",
        help="Path to the published conformance corpus directory "
             "(default: conformance/sep2828)",
    )
    pcs.add_argument(
        "--records", default=None,
        help="Directory of your own SEP-2828 records to check against the corpus",
    )
    pcs.add_argument(
        "--glob", default="*.json",
        help="Glob for record files within --records (default: *.json)",
    )
    pcs.add_argument(
        "--as-of", default=None, dest="as_of",
        help="A date echoed verbatim into the statement (never read from a clock)",
    )
    pcs.add_argument(
        "--out", default=None,
        help="Write the Markdown statement to this file instead of stdout",
    )
    pcs.add_argument(
        "--json", action="store_true",
        help="Emit the full statement as JSON instead of Markdown",
    )
    pcs.set_defaults(func=_cmd_conformance_statement)

    pconf = sub.add_parser(
        "conformance",
        help="Keyless SEP-2828 conformance: check a record or a whole set with "
             "one command, or self-test this build against the published corpus.",
    )
    cfsub = pconf.add_subparsers(dest="conformance_cmd", metavar="COMMAND")
    pconf.set_defaults(func=_help_dispatch(pconf))

    pcc = cfsub.add_parser(
        "check",
        help="Conformance-check a SEP-2828 record (a JSON file) or a record set "
             "(a directory of them). File or directory is auto-detected; the "
             "verdict matches verify-record / verify-records. Keyless.",
    )
    pcc.add_argument(
        "path",
        help="A record JSON file, or a directory of record JSON files",
    )
    pcc.add_argument(
        "--attestation", default=None,
        help="tool-call attestation the record answers, to also check the "
             "back-link (single-file input only; still keyless)",
    )
    pcc.add_argument(
        "--glob", default="*.json",
        help="Glob for record files when path is a directory (default: *.json)",
    )
    pcc.add_argument(
        "--json", action="store_true",
        help="Emit the full conformance report as JSON",
    )
    pcc.set_defaults(func=_cmd_conformance_check)

    pcst = cfsub.add_parser(
        "statement",
        help="Self-test this implementation against the published SEP-2828 "
             "conformance corpus and print one reproducible statement. The same "
             "check as the conformance-statement command. Keyless.",
    )
    pcst.add_argument(
        "--corpus", default="conformance/sep2828",
        help="Path to the published conformance corpus directory "
             "(default: conformance/sep2828)",
    )
    pcst.add_argument(
        "--records", default=None,
        help="Directory of your own SEP-2828 records to check against the corpus",
    )
    pcst.add_argument(
        "--glob", default="*.json",
        help="Glob for record files within --records (default: *.json)",
    )
    pcst.add_argument(
        "--as-of", default=None, dest="as_of",
        help="A date echoed verbatim into the statement (never read from a clock)",
    )
    pcst.add_argument(
        "--out", default=None,
        help="Write the Markdown statement to this file instead of stdout",
    )
    pcst.add_argument(
        "--json", action="store_true",
        help="Emit the full statement as JSON instead of Markdown",
    )
    pcst.set_defaults(func=_cmd_conformance_statement)

    pvbs = sub.add_parser(
        "verify-bundles",
        help="Run the full lens stack over a whole directory of evidence "
             "bundles at once. The batch twin of verify-bundle: how many "
             "verify, how many authenticate, and what the evidence covers, "
             "with the gap naming the lenses no bundle exercised. Requires the "
             "attestation extra.",
    )
    pvbs.add_argument(
        "directory",
        help="Directory of evidence-bundle JSON documents (the shape "
             "verify-bundle reads)",
    )
    pvbs.add_argument(
        "--glob", default="*.json",
        help="Glob for bundle files within the directory (default: *.json)",
    )
    pvbs.add_argument(
        "--json", action="store_true",
        help="Emit the full set verdict report as JSON",
    )
    pvbs.set_defaults(func=_cmd_verify_bundles)

    pbb = sub.add_parser(
        "build-bundle",
        help="Assemble a complete evidence bundle on disk from the issuer's "
             "pieces (receipt, attestation, identity, inclusion, consistency, "
             "revocation), writing the single file verify-bundle reads. The "
             "issuer-side mirror of verify-bundle. Requires the attestation "
             "extra.",
    )
    pbb.add_argument(
        "--from-dir", default=None,
        help="Directory holding the issuer's pieces by conventional name: "
             "receipt.json, did_document.json, verifying_jwk.json, "
             "attestation.json, inclusion.json, consistency.json, "
             "registry.json, plus scalars expected_keyid.txt and "
             "inclusion_leaf_hex.txt. Explicit flags below override.",
    )
    pbb.add_argument(
        "--receipt", default=None,
        help="Path to the execution-receipt JSON. Required unless --from-dir "
             "supplies receipt.json.",
    )
    pbb.add_argument(
        "--did-document", default=None,
        help="Path to the did:web DID document JSON (identity lens).",
    )
    pbb.add_argument(
        "--expected-keyid", default=None,
        help="Pin the verification-method keyid the receipt must resolve to "
             "(identity lens).",
    )
    pbb.add_argument(
        "--verifying-jwk", default=None,
        help="Path to the verifying public-key JWK JSON (signature lens).",
    )
    pbb.add_argument(
        "--attestation", default=None,
        help="Path to the tool-call request attestation JSON (back-link lens).",
    )
    pbb.add_argument(
        "--inclusion", default=None,
        help="Path to the transparency-log inclusion-proof JSON, carrying "
             "log_index, tree_size, siblings_hex, and root_hex (inclusion lens).",
    )
    pbb.add_argument(
        "--inclusion-leaf-hex", default=None,
        help="Override the inclusion leaf bytes as hex. Defaults to the "
             "canonical receipt bytes.",
    )
    pbb.add_argument(
        "--consistency", default=None,
        help="Path to the append-only consistency-proof JSON, carrying "
             "first_size, second_size, hashes_hex, first_root_hex, and "
             "second_root_hex (consistency lens).",
    )
    pbb.add_argument(
        "--registry", default=None,
        help="Path to the revocation-registry JSON (revocation lens).",
    )
    pbb.add_argument(
        "--out", default=None,
        help="Write the assembled bundle to this path. Default: stdout.",
    )
    pbb.set_defaults(func=_cmd_build_bundle)

    ptee = sub.add_parser(
        "tee",
        help=(
            "Hardware TEE attestation commands (experimental, AMD SEV-SNP)"
        ),
    )
    tsub = ptee.add_subparsers(dest="tee_cmd", metavar="COMMAND")
    ptee.set_defaults(func=_help_dispatch(ptee))

    ptee_parse = tsub.add_parser(
        "parse",
        help=(
            "Parse an AMD SEV-SNP attestation report blob and print key "
            "fields as JSON. Requires the attestation extra."
        ),
    )
    ptee_parse.add_argument(
        "report",
        help="Path to a binary SEV-SNP attestation report (1184 bytes)",
    )
    ptee_parse.set_defaults(func=_cmd_tee_parse)

    ptee_verify = tsub.add_parser(
        "verify",
        help=(
            "Verify a SEV-SNP report signature against a VCEK PEM, "
            "and optionally check that REPORT_DATA binds to an OVERT envelope. "
            "Requires the attestation extra."
        ),
    )
    ptee_verify.add_argument(
        "report",
        help="Path to a binary SEV-SNP attestation report (1184 bytes)",
    )
    ptee_verify.add_argument(
        "--vcek",
        required=True,
        help=(
            "Path to a PEM-encoded VCEK (Versioned Chip Endorsement Key). "
            "Must be supplied out of band. AMD KDS chain validation is "
            "tracked for v0.19+."
        ),
    )
    ptee_verify.add_argument(
        "--overt",
        default=None,
        help=(
            "Optional path to a canonical-CBOR OVERT 1.0 Base Envelope. "
            "If supplied, the report's REPORT_DATA is checked against "
            "SHA-512 of the envelope."
        ),
    )
    ptee_verify.set_defaults(func=_cmd_tee_verify)

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

    phook = sub.add_parser(
        "hook",
        help=(
            "Claude Code hook runner (called by the vaara-governance "
            "plugin; reads the hook event JSON on stdin)"
        ),
    )
    hooksub = phook.add_subparsers(dest="hook_cmd", metavar="COMMAND")
    phook.set_defaults(func=_help_dispatch(phook))

    phpre = hooksub.add_parser(
        "pre-tool-use",
        help="Two-layer gate: regex deny patterns, then the classifier on "
             "mcp__* calls. Exit 2 blocks the call.",
    )
    phpre.add_argument(
        "--deny-patterns", default=None,
        help="Deny-patterns JSON (default: VAARA_PLUGIN_DENY_PATTERNS_FILE, "
             "then $CLAUDE_PLUGIN_ROOT/policies/default_deny.json, then the "
             "copy bundled with the package)",
    )
    phpre.set_defaults(func=lambda args: __import__(
        "vaara.integrations.claude_code_hooks", fromlist=["run_pre_tool_use"]
    ).run_pre_tool_use(args.deny_patterns))

    phpost = hooksub.add_parser(
        "post-tool-use",
        help="Append the outcome record and feed the online learner",
    )
    phpost.set_defaults(func=lambda args: __import__(
        "vaara.integrations.claude_code_hooks", fromlist=["run_post_tool_use"]
    ).run_post_tool_use())

    phss = hooksub.add_parser(
        "session-start",
        help="Print the governance status line and record the configured "
             "Article 50(1) disclosure for the session",
    )
    phss.set_defaults(func=lambda args: __import__(
        "vaara.integrations.claude_code_hooks", fromlist=["run_session_start"]
    ).run_session_start())

    pmode = sub.add_parser(
        "mode",
        help=(
            "Preset policy threshold bundles "
            "(eco / balanced / performance / strict)"
        ),
    )
    msub = pmode.add_subparsers(dest="mode_cmd", metavar="COMMAND")
    pmode.set_defaults(func=_help_dispatch(pmode))

    pml = msub.add_parser(
        "list",
        help="List available modes with thresholds and descriptions",
    )
    pml.set_defaults(func=_cmd_mode_list)

    pms = msub.add_parser(
        "show",
        help="Show a mode's thresholds, description, and watt profile",
    )
    pms.add_argument(
        "name", help="Mode name (eco, balanced, performance, strict)",
    )
    pms.set_defaults(func=_cmd_mode_show)

    pme = msub.add_parser(
        "emit",
        help=(
            "Emit a mode as a valid Vaara policy document "
            "(JSON by default, YAML with --format yaml)"
        ),
    )
    pme.add_argument("name", help="Mode name")
    pme.add_argument(
        "--format", choices=["json", "yaml"], default="json",
        help="Output format (default: json)",
    )
    pme.add_argument(
        "--output", "-o", default=None,
        help="Output path (default: stdout)",
    )
    pme.set_defaults(func=_cmd_mode_emit)

    pinit = sub.add_parser(
        "init",
        help=(
            "Set up local governance in one command: write the Claude Code "
            "hooks, route known MCP clients through the proxy, point everything "
            "at one trail. Idempotent and self-healing (safe to re-run)."
        ),
    )
    pinit.add_argument(
        "--trail-db", default=None,
        help="Path to the single audit trail DB "
             "(default: ~/.vaara/trail/audit.db)",
    )
    pinit.add_argument(
        "--shadow", action="store_true",
        help="Govern in watch-only (shadow) mode: record without blocking.",
    )
    pinit.add_argument(
        "--no-mcp", action="store_true",
        help="Only write the Claude Code hooks; leave MCP client configs alone.",
    )
    pinit.add_argument(
        "--proxy-service", action="store_true",
        help="Also install the model proxy as a user service "
             "(launchd on macOS, systemd --user on Linux) so it survives "
             "logout. Removed by 'vaara ungovern'.",
    )
    pinit.add_argument(
        "--proxy-enforce", action="store_true",
        help="Install the proxy service gating instead of observing. "
             "Without --proxy-allow the approvals directory defaults to "
             "~/.vaara/approvals so escalations reach the approval surface "
             "('vaara approvals' or any watcher).",
    )
    pinit.add_argument(
        "--proxy-allow", action="append", default=None, metavar="PATTERN",
        help="Tool-name glob the enforcing proxy service passes without "
             "gating (repeatable). Implies nothing without --proxy-enforce.",
    )
    pinit.set_defaults(func=_cmd_init)

    pungov = sub.add_parser(
        "ungovern",
        help="Reverse 'vaara init': remove the Vaara hooks and restore each "
             "MCP config from its .vaara-backup.",
    )
    pungov.set_defaults(func=_cmd_ungovern)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
