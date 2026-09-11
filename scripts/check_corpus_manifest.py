# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check tests/adversarial/MANIFEST.sha256 against what is actually on disk.

`tests/adversarial/README.md` tells anyone replicating Vaara's published metrics
to verify their corpus matches this manifest. Nothing checked that the manifest
matched OUR OWN corpus, so it drifted every time a generation run landed and the
instruction quietly became "verify against a file that is out of date".

Three kinds of drift, all reported separately because they mean different
things:

  UNLISTED   on disk, absent from the manifest. New generation output. Benign
             in itself and the reason the count drifts, but it means a
             replicator's check passes while their corpus differs from ours.
  MISSING    in the manifest, absent from disk. A file was deleted or renamed.
             This one is serious: published numbers were measured against it.
  CHANGED    present in both, different hash. An entry was edited in place.
             Never expected. The corpus is append-only by policy.

Exit 0 clean, 1 on drift, 2 on a usage error. Regenerate with --write, which
runs the exact command the README documents.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ADV = REPO / "tests" / "adversarial"
MANIFEST = ADV / "MANIFEST.sha256"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def on_disk() -> dict[str, Path]:
    return {f"./{p.relative_to(ADV).as_posix()}": p
            for p in sorted(ADV.rglob("*.jsonl")) if p.is_file()}


def listed() -> dict[str, str]:
    out: dict[str, str] = {}
    if not MANIFEST.is_file():
        return out
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        if digest and name:
            out[name] = digest
    return out


def write_manifest(disk: dict[str, Path]) -> None:
    lines = [f"{sha256(p)}  {name}" for name, p in sorted(disk.items())]
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"[write] {MANIFEST.relative_to(REPO)}: {len(lines)} file(s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="regenerate the manifest from what is on disk")
    ap.add_argument("--quiet-unlisted", action="store_true",
                    help="do not list every unlisted file, just count them")
    args = ap.parse_args()

    if not ADV.is_dir():
        print(f"[error] no such directory: {ADV}", file=sys.stderr)
        return 2

    disk = on_disk()
    if args.write:
        write_manifest(disk)
        return 0

    man = listed()
    unlisted = sorted(set(disk) - set(man))
    missing = sorted(set(man) - set(disk))
    changed = sorted(n for n in set(man) & set(disk) if sha256(disk[n]) != man[n])

    print(f"[manifest] {len(man)} listed, {len(disk)} on disk")
    for label, names in (("UNLISTED", unlisted), ("MISSING", missing),
                         ("CHANGED", changed)):
        if not names:
            continue
        print(f"\n[{label}] {len(names)}")
        show = names if not (args.quiet_unlisted and label == "UNLISTED") else names[:5]
        for n in show:
            print(f"    {n}")
        if len(show) < len(names):
            print(f"    ... and {len(names) - len(show)} more")

    if not (unlisted or missing or changed):
        print("[manifest] clean")
        return 0
    print("\n[manifest] DRIFT. Regenerate with --write once generation has "
          "finished, never while files are still being written.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
