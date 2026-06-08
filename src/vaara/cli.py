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
        Generate an EC P-256 (ES256) keypair for SEP-2787 attestation
        signing with ``vaara-mcp-proxy --attest-signing-key``. Replaces
        the ``openssl ecparam | pkcs8`` pipe. Does not require --dev.

    vaara attest verify ENVELOPE.json
            (--pubkey-file PUB.pem | --hs256-secret-file SECRET)
            [--enforce-ttl]
        Verify a SEP-2787 attestation envelope. Reports signature
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

    vaara normalize RECORD.json [--format auto|sep2643|sep2787|sep2817] [--json]
        Map an adjacent MCP record onto the SEP-2828 evidence model. Reads a
        SEP-2643 denial, a SEP-2787 attestation, or a SEP-2817 invocation
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

    vaara conformance-statement [--corpus DIR] [--records DIR] [--as-of DATE]
            [--out FILE] [--json]
        Self-test this implementation against the published SEP-2828
        conformance corpus and print one reproducible statement: the corpus
        bytes match their manifest, every recorded verdict was reproduced, and
        (with --records) your own records conform. Names the exact corpus byte
        set (version plus corpusDigest), so the claim pins a fixed suite rather
        than a moving target. The answer to "trust us": prove it against the
        neutral suite. Keyless. Exit 0 iff the statement conforms.

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
        # EC P-256 (ES256) key for SEP-2787 attestation signing. This is the
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
    """Generate an EC P-256 (ES256) keypair for SEP-2787 attestation signing.

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
    # Despite the SEP-2787 field name "secretVersion", this value is a digest
    # of the PUBLIC key and is safe to print and publish. It is named here for
    # what it is so it is not mistaken (by a reader or a scanner) for a secret.
    pubkey_version = hashlib.sha256(pub_der).hexdigest()[:8]

    print("Generated EC P-256 keypair for SEP-2787 attestation signing (ES256)")
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


def _cmd_trail_export_article12(args: argparse.Namespace) -> int:
    """Export a signed EU AI Act Article 12 regulator package from a trail."""
    try:
        from vaara.audit.article12_export import export_article12
        from vaara.audit.timeanchor import TimeAnchorError
        from vaara.audit.trail import AuditRecord, AuditTrail
    except ImportError:
        print(_INSTALL_HINT, file=sys.stderr)
        return 2

    trail_path = Path(args.trail).expanduser()
    if not trail_path.exists():
        print(f"trail JSONL not found: {trail_path}", file=sys.stderr)
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

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        time_anchor = _obtain_time_anchor(args, trail)
        result = export_article12(
            trail,
            out_path=out,
            signer_key=Path(args.key).expanduser(),
            system_meta=system_meta,
            period=period,
            time_anchor=time_anchor,
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
    from vaara.attestation._sep2787_canonical import iso8601_to_epoch

    epoch = iso8601_to_epoch(envelope.issuer_asserted.iat)
    return epoch if epoch is not None else 0.0


def _cmd_attest_verify(args: argparse.Namespace) -> int:
    """Verify a SEP-2787 attestation envelope's signature (and optionally TTL)."""
    try:
        from vaara.attestation.sep2787 import (
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
            f"vaara attest verify: not a valid SEP-2787 attestation: {exc}",
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
        from vaara.attestation.sep2787 import (
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

    verdict = verify_evidence_bundle(bundle)

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
    from vaara.attestation.receipt import check_record_conformance

    path = Path(args.record).expanduser()
    if not path.is_file():
        print(f"vaara verify-record: not a file: {path}", file=sys.stderr)
        return 2
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"vaara verify-record: cannot read record JSON: {exc}", file=sys.stderr)
        return 1

    report = check_record_conformance(doc)
    back_link = _record_back_link(args.attestation, doc) if args.attestation else None

    if args.json:
        out: dict[str, Any] = {"conformance": report.to_dict()}
        if back_link is not None:
            out["backLink"] = back_link
        print(json.dumps(out, indent=2))
    else:
        print(f"conformance: {'CONFORMS' if report.conforms else 'NON-CONFORMING'}")
        for c in report.checks:
            state = ("pass" if c.ok else "FAIL") if c.severity == "required" else (
                "ok" if c.ok else "warn")
            print(f"  [{state:4s}] {c.severity:8s} {c.id}")
            if not c.ok:
                print(f"           {c.detail}")
        if back_link is not None:
            bstate = "n/a" if back_link.get("skipped") else (
                "pass" if back_link["ok"] else "FAIL")
            print(f"  back-link: {bstate}  {back_link['detail']}")

    if not report.conforms:
        return 1
    if back_link is not None and not back_link["ok"] and not back_link.get("skipped"):
        return 1
    return 0


def _record_back_link(attestation_path: str, doc: Any) -> dict[str, Any]:
    """Keyless back-link check: does the record pin this attestation?

    Returns ``{ok, skipped, detail}``. ``skipped`` is True when the check
    could not run (extra missing, attestation unreadable, record not a
    parseable receipt) and so does not gate the verdict; a False ``ok``
    with ``skipped`` False is a real back-link failure.
    """
    try:
        from vaara.attestation.receipt import parse_receipt, verify_back_link
        from vaara.attestation.sep2787 import AttestationError, parse_attestation
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


def _cmd_normalize(args: argparse.Namespace) -> int:
    """Map a foreign MCP record onto the SEP-2828 evidence model.

    Vaara is the receiving side: it reads a SEP-2643 denial, a SEP-2787
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
        from vaara.attestation.sep2787 import AttestationError
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vaara", description="Vaara AI Agent Execution Layer")
    sub = p.add_subparsers(dest="cmd")
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
        help="Generate an EC P-256 (ES256) key for SEP-2787 attestation "
             "signing with vaara-mcp-proxy --attest-signing-key, instead of "
             "an Ed25519 trail-signing key. Does not require --dev.",
    )
    pk.set_defaults(func=_cmd_keygen)

    pt = sub.add_parser("trail", help="Audit-trail commands")
    tsub = pt.add_subparsers(dest="trail_cmd")
    pt.set_defaults(func=_help_dispatch(pt))

    pe = tsub.add_parser("export", help="Export a signed, regulator-handoff trail zip")
    pe.add_argument("--trail", required=True, help="Path to trail JSONL file")
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
    pa12.add_argument("--trail", required=True, help="Path to trail JSONL file")
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
    pa12.set_defaults(func=_cmd_trail_export_article12)

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
    rsub = pr.add_subparsers(dest="review_cmd")
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
    psub = pp_policy.add_subparsers(dest="policy_cmd")
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
    csub = pcr.add_subparsers(dest="compliance_cmd")
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
    dsub = pdetect.add_subparsers(dest="detect_cmd")
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
    osub = povert.add_subparsers(dest="overt_cmd")
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
        help="SEP-2787 tool-call attestation commands",
    )
    asub = pattest.add_subparsers(dest="attest_cmd")
    pattest.set_defaults(func=_help_dispatch(pattest))
    pa_verify = asub.add_parser(
        "verify",
        help="Verify a SEP-2787 attestation envelope's signature (and TTL). "
             "Requires the attestation extra.",
    )
    pa_verify.add_argument(
        "envelope", help="Path to a SEP-2787 attestation JSON file",
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
        help="Execution-receipt commands (post-execution sibling of SEP-2787)",
    )
    rcsub = preceipt.add_subparsers(dest="receipt_cmd")
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
        help="Path to the SEP-2787 attestation JSON the receipt answers "
             "(needed to verify the back-link)",
    )
    _add_jws_key_args(prc_verify)
    prc_verify.add_argument(
        "--result", default=None,
        help="Path to the runtime result JSON. When the receipt carries a "
             "result commitment, the commitment is verified against this.",
    )
    prc_verify.set_defaults(func=_cmd_receipt_verify)

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
        help="Path to the SEP-2787 attestation the record answers; when given, "
             "the back-link is verified (no key needed)",
    )
    pvr.add_argument(
        "--json", action="store_true",
        help="Emit the full conformance report as JSON",
    )
    pvr.set_defaults(func=_cmd_verify_record)

    pnz = sub.add_parser(
        "normalize",
        help="Map an adjacent MCP record (SEP-2643 denial, SEP-2787 "
             "attestation, or SEP-2817 invocation audit context) onto the "
             "SEP-2828 evidence model: which plane it fills, which fields it "
             "populates, and what is still missing for a complete signed "
             "record. The receiving side that reads every dialect.",
    )
    pnz.add_argument(
        "record",
        help="Path to a JSON file holding a SEP-2643, SEP-2787, or SEP-2817 record",
    )
    pnz.add_argument(
        "--format", default="auto",
        choices=("auto", "sep2643", "sep2787", "sep2817"),
        help="Source format (default: auto-detect)",
    )
    pnz.add_argument(
        "--json", action="store_true",
        help="Emit the normalized evidence mapping as JSON",
    )
    pnz.set_defaults(func=_cmd_normalize)

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
        help="Path to the SEP-2787 request attestation JSON (back-link lens).",
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
    tsub = ptee.add_subparsers(dest="tee_cmd")
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

    pmode = sub.add_parser(
        "mode",
        help=(
            "Preset policy threshold bundles "
            "(eco / balanced / performance / strict)"
        ),
    )
    msub = pmode.add_subparsers(dest="mode_cmd")
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

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
