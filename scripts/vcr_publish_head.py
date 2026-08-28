#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Publish the current VCR chain head to a public transparency log.

Maintainer side. The counterpart to ``scripts/vcr_chain.py --check-witness``,
which anyone runs and which imports no Vaara code.

Why this exists. Rows on the Vaara Conformance Results page are chained, and the
page used to say that made removal impossible for the maintainer. It does not. A
chain makes a break detectable to someone holding an earlier head, and the
maintainer controls both the file and the published page, so a rewritten chain
republished consistently is invisible to a reader who kept nothing. Iman Schrock
raised exactly this on the SCITT list on 2026-08-21, before filing a row, and he
was right to hold it.

Recording each head in a log the maintainer does not operate closes it. After a
rewrite the chain reaches a head that matches no witnessed entry, and a stranger
sees that against the log rather than against a promise.

Publication is permanent and public. It writes one sha256 digest and a signature
over it. It does not write any row content, any party name, or anything else.

Usage:
    .venv/bin/python scripts/vcr_publish_head.py --dry-run
    .venv/bin/python scripts/vcr_publish_head.py --yes
    .venv/bin/python scripts/vcr_publish_head.py rows.json --key vcr_key.pem --yes
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROWS = ROOT / "conformance" / "reproductions.json"
DEFAULT_KEY = ROOT / ".shared" / "vcr-desk" / "vcr_publish_key.pem"


def _load_or_create_key(path: Path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    if path.exists():
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise SystemExit(f"{path} is not an EC private key")
        return key, False
    key = ec.generate_private_key(ec.SECP256R1())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    path.chmod(0o600)
    return key, True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", nargs="?", default=str(DEFAULT_ROWS))
    parser.add_argument("--key", default=str(DEFAULT_KEY))
    parser.add_argument("--log", default=None,
                        help="Transparency log base URL (default: public Rekor)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the prompt. The disclosure still prints, "
                             "because publication cannot be undone.")
    parser.add_argument("--now", default=None,
                        help="ISO 8601 instant to record as the publication "
                             "time (default: the current time)")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT / "scripts"))
    from vcr_chain import ZERO, check, digest  # same chain maths, one authority

    path = Path(args.rows)
    data = json.loads(path.read_text(encoding="utf-8"))
    problems = check(data)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print("\nchain is broken; refusing to witness a broken chain",
              file=sys.stderr)
        return 1

    rows = data.get("reproductions", [])
    head = digest(rows[-1]) if rows else data.get("genesis", ZERO)
    already = {w.get("head") for w in data.get("witnessed_heads", [])}

    print(f"rows:  {len(rows)}")
    print(f"head:  {head}")
    if head in already:
        print("\nthis head is already witnessed, nothing to publish")
        return 0

    print(
        "\nWHAT LEAVES THIS MACHINE: one sha256 digest and a signature over it.\n"
        "No row content, no party name, no commit, nothing else.\n"
        "WHAT THIS MEANS: publication is permanent and public. The log is "
        "enumerable\nby anyone, not only by whoever is shown a link, and "
        "everything published\nunder one key is groupable. It cannot be erased "
        "afterwards."
    )
    if args.dry_run:
        print("\n--dry-run: nothing published")
        return 0
    if not args.yes:
        if input("\npublish this head? [y/N] ").strip().lower() not in ("y", "yes"):
            print("not published")
            return 0

    from vaara.attestation.rekor_log import DEFAULT_REKOR, RekorError, publish_head

    key, created = _load_or_create_key(Path(args.key))
    if created:
        print(f"generated a new publishing key at {args.key}")
    try:
        publication = publish_head(
            head.split(":", 1)[1], key, log_url=args.log or DEFAULT_REKOR
        )
    except RekorError as exc:
        print(f"publication failed: {exc}", file=sys.stderr)
        return 1

    stamp = args.now or datetime.now(timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")
    data.setdefault("witnessed_heads", []).append({
        "head": head,
        "rows": len(rows),
        "uuid": publication.uuid,
        "logIndex": publication.log_index,
        "integratedTime": publication.integrated_time,
        "logUrl": publication.log_url,
        "published": stamp,
    })
    # ensure_ascii=False, because the default escapes every non-ASCII character
    # in a party's name. Row digests come from rfc8785 over the parsed objects,
    # so the chain survives either way, but witnessing a head once rewrote
    # "Emek Can Dogru" with an escape sequence in the file people download.
    # A file about recording what happened should spell the names right.
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\npublished. logIndex={publication.log_index} uuid={publication.uuid}")
    print(f"recorded in {path}")
    print("verify with: python scripts/vcr_chain.py --check-witness")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
