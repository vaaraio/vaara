"""Vaara command-line interface.

Subcommands:

    vaara keygen --dev --out PATH
        Generate an Ed25519 keypair for local evaluation / demo signing.
        For production, use keys from your KMS/HSM — see docs/signing-keys.md.

    vaara trail export --trail PATH --out PATH --key PATH
        Export a signed, regulator-handoff zip from a saved trail.

    vaara trail verify --zip PATH [--pubkey PATH]
        Verify a signed trail zip.

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
from pathlib import Path

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

    print(f"Generated Ed25519 keypair (evaluation/demo use only)")
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
        print(f"Manifest:")
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

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
