#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail if any commit introduces private key material.

Four private conformance-vector keys reached this repository's public history
in v0.42.0 and stayed reachable after PR #201 deleted them from HEAD, because
deleting a file does not remove its blob. A check that only reads the working
tree would have called that repository clean the whole time. So the range mode
walks every commit in the range and inspects the blobs each one adds, which is
what a reviewer cannot do by eye and what the delete-from-HEAD reflex misses.

Detection loads the candidate rather than matching on it. A PEM header is not
evidence of a key: ``tests/test_signed_export.py`` passes the literal bytes
``-----BEGIN PRIVATE KEY-----\\nnot pem\\n-----END PRIVATE KEY-----`` to prove
the signer rejects malformed input, and that must keep passing this check.
Anything that parses as a usable private key fails, including an encrypted one,
whose passphrase is not the repository's business.

Usage::

    python scripts/check_no_private_keys.py --tree
    python scripts/check_no_private_keys.py --range origin/main..HEAD
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization

# PEM private-key blocks, whatever the flavour: PKCS#8 ("PRIVATE KEY"),
# PKCS#1 ("RSA PRIVATE KEY"), SEC 1 ("EC PRIVATE KEY"), DSA, encrypted, and
# OpenSSH's own container.
_PEM_BLOCK = re.compile(
    rb"-----BEGIN (?:[A-Z0-9 ]*)PRIVATE KEY-----.*?"
    rb"-----END (?:[A-Z0-9 ]*)PRIVATE KEY-----",
    re.DOTALL,
)

# Files whose name alone says signing material, caught even when the contents
# are a format this script cannot parse (DER, PKCS#12, a raw seed).
_SUSPECT_NAME = re.compile(
    r"(^|/)(id_(rsa|dsa|ecdsa|ed25519)|.*_private\.pem|.*\.p12|.*\.pfx)$"
)


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True
    ).stdout


def _is_real_private_key(block: bytes) -> bool:
    """True when the block parses as a private key that could sign something."""
    try:
        serialization.load_pem_private_key(block, password=None)
        return True
    except TypeError:
        # Loaded far enough to learn it is encrypted. Still a private key.
        return True
    except ValueError:
        pass
    try:
        serialization.load_ssh_private_key(block, password=None)
        return True
    except TypeError:
        return True
    except ValueError:
        return False


def _findings(blob: bytes, path: str) -> list[str]:
    found = []
    for block in _PEM_BLOCK.findall(blob):
        if _is_real_private_key(block):
            head = block.split(b"\n", 1)[0].decode("ascii", "replace")
            found.append(f"parsed a usable private key ({head})")
    if _SUSPECT_NAME.search(path) and not found:
        found.append("filename names private key material")
    return found


def _check_tree() -> list[tuple[str, str, str]]:
    """Scan every tracked file as it stands now, staged content included.

    Reads each path from the working tree. It used to read ``HEAD:<path>``,
    which raised on the first file that ``ls-files`` reports and ``HEAD`` does
    not, so the scan died on any newly staged file. That is precisely when this
    check is run: a fresh vector suite with a key in it is the case it exists to
    catch, and a crash is not a pass but it is just as easy to wave through.
    """
    hits = []
    for path in _git("ls-files", "-z").split(b"\0"):
        if not path:
            continue
        name = path.decode("utf-8", "surrogateescape")
        try:
            blob = Path(name).read_bytes()
        except OSError:
            # Tracked but not on disk (a staged deletion, a sparse checkout).
            # Fall back to what is committed; skip if that is absent too.
            try:
                blob = _git("show", f"HEAD:{name}")
            except subprocess.CalledProcessError:
                continue
        for why in _findings(blob, name):
            hits.append(("tree", name, why))
    return hits


def _check_range(rev_range: str) -> list[tuple[str, str, str]]:
    hits = []
    revs = _git("rev-list", rev_range).split()
    for rev in revs:
        sha = rev.decode()
        # Blobs this commit adds or rewrites. A file deleted by a later commit
        # is still caught here, which is the whole point.
        out = _git(
            "diff-tree", "-r", "--no-commit-id", "--diff-filter=AM",
            "--root", sha,
        )
        for line in out.splitlines():
            if not line.startswith(b":"):
                continue
            meta, _, path = line.partition(b"\t")
            if not path:
                continue
            blob_sha = meta.split()[3].decode()
            name = path.decode("utf-8", "surrogateescape")
            try:
                blob = _git("cat-file", "blob", blob_sha)
            except subprocess.CalledProcessError:
                continue
            for why in _findings(blob, name):
                hits.append((sha[:12], name, why))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--tree", action="store_true",
                       help="scan every tracked file as it stands in the "
                            "working tree, staged content included")
    group.add_argument("--range", dest="rev_range",
                       help="scan blobs added by each commit in a rev range")
    args = ap.parse_args()

    hits = _check_tree() if args.tree else _check_range(args.rev_range)
    if not hits:
        print("no private key material found")
        return 0

    for where, path, why in hits:
        print(f"{where}  {path}: {why}", file=sys.stderr)
    print(
        f"\n{len(hits)} private key finding(s). Rotate the key and regenerate "
        "whatever it signed. Deleting the file in a follow-up commit leaves "
        "the blob reachable and does not help.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
