#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Anchor a Vaara release to a qualified RFC 3161 TSA — dogfoods vaara.audit.timeanchor.

The release is already anchored by three independent registries (PyPI, npm,
GitHub) plus Sigstore/SLSA provenance. This adds the one leg those do not: an
eIDAS-qualified trusted timestamp (RFC 3161), which carries the Article 41 legal
presumption of time accuracy and is verifiable offline against the TSA's own
key. It is the strongest priority proof for a formal EU legal or regulatory
dispute, and it demonstrates our own timestamping primitive on our own releases.

It works by fingerprinting the exact release tree — the sha256 over
``git ls-tree -r --full-tree <tag>``, which anyone with the public tag can
recompute — and obtaining a qualified timestamp over that fingerprint. The
committed anchor file binds the two, so a verifier proves the tag's tree existed
no later than the attested time without trusting us.

Choosing the qualified TSA is deployer policy (a listed QTSP's TSA endpoint);
pass it with --tsa-url. Requires the timeanchor extra:
``pip install 'vaara[timeanchor]'``.

Usage:
  python scripts/anchor_release.py --tag v1.21.0 --tsa-url <qualified-TSA-url>
  python scripts/anchor_release.py --verify release-anchors/v1.21.0.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANCHOR_DIR = REPO / "release-anchors"


def tree_fingerprint(tag: str) -> str:
    """sha256 hex over ``git ls-tree -r --full-tree <tag>``.

    Deterministic and recomputable by anyone with the public tag: it lists every
    path, mode, and blob id in the release tree, so the digest fingerprints the
    exact tree state without depending on archive/timestamp quirks.
    """
    proc = subprocess.run(
        ["git", "-C", str(REPO), "ls-tree", "-r", "--full-tree", tag],
        capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"cannot resolve tag {tag!r} locally. Release tags are created on the "
            f"remote and are not fetched automatically — run:\n"
            f"    git fetch origin tag {tag}\n"
            f"then retry.")
    return hashlib.sha256(proc.stdout).hexdigest()


def _recompute_cmd(tag: str) -> str:
    return f"git ls-tree -r --full-tree {tag} | sha256sum"


def anchor(tag: str, tsa_url: str, out_path: Path) -> int:
    from vaara.audit.timeanchor import RFC3161TimeAnchorClient

    fingerprint = tree_fingerprint(tag)
    client = RFC3161TimeAnchorClient(tsa_url)
    # Not a receipt chain: position 0, and the tree fingerprint stands in as the
    # anchored digest (the same role a chain-head hash plays for the trail).
    anc = client.anchor(0, fingerprint)

    doc = {
        "release": tag,
        "treeFingerprint": {"sha256": fingerprint, "recompute": _recompute_cmd(tag)},
        "anchor": anc.to_dict(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"anchored {tag}")
    print(f"  tree sha256   {fingerprint}")
    print(f"  attested time {anc.anchored_time}")
    print(f"  tsa           {anc.tsa_url}")
    print(f"  written       {out_path.relative_to(REPO)}")
    print("Commit the anchor file. Verify anytime with --verify.")
    return 0


def verify(path: Path) -> int:
    from vaara.audit.timeanchor import TimeAnchor, TimeAnchorError, verify_anchor

    doc = json.loads(path.read_text(encoding="utf-8"))
    tag = doc["release"]
    recorded = doc["treeFingerprint"]["sha256"]
    anc = TimeAnchor.from_dict(doc["anchor"])

    current = tree_fingerprint(tag)
    if current != recorded:
        print(f"FAIL: tree fingerprint for {tag} is {current}, "
              f"anchor file records {recorded} (tree changed or wrong tag)",
              file=sys.stderr)
        return 1
    if anc.chain_head_hash != recorded:
        print(f"FAIL: anchor attests {anc.chain_head_hash}, not the recorded "
              f"tree fingerprint {recorded}", file=sys.stderr)
        return 1
    try:
        attested = verify_anchor(anc)
    except TimeAnchorError as exc:
        print(f"FAIL: token verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {tag} tree {recorded[:16]}… attested at {attested.isoformat()} "
          f"by {anc.tsa_url} (token verified offline)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", help="release tag to anchor, e.g. v1.21.0")
    parser.add_argument("--tsa-url", help="qualified RFC 3161 TSA endpoint")
    parser.add_argument("--out", type=Path,
                        help="anchor file path (default: release-anchors/<tag>.json)")
    parser.add_argument("--verify", type=Path, metavar="PATH",
                        help="verify an existing anchor file and exit")
    args = parser.parse_args(argv)

    if args.verify:
        return verify(args.verify)
    if not args.tag or not args.tsa_url:
        parser.error("--tag and --tsa-url are required to anchor (or use --verify)")
    out_path = args.out or (ANCHOR_DIR / f"{args.tag}.json")
    return anchor(args.tag, args.tsa_url, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
