#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check what has and has not been removed from Vaara Conformance Results.

The table records occurrences. A run happened, on a date, at a commit, and that
does not stop being true afterwards, so rows are never taken down or edited.
Saying so is worth little on its own, because the person who would quietly edit
the table is the person publishing the promise. So each row carries ``prev``,
the digest of the row before it, and this script recomputes the whole chain.

Change or reorder a published row and every digest after it stops matching. The
break names the row.

WHAT A CHAIN DOES NOT DO, and this is the part that matters. It says nothing
about rows removed from the END, because what remains is a perfectly valid
shorter chain. Dropping the last row leaves a file this script would once have
called "chain intact" while exiting 0. Emek Can Dogru put it exactly right on
the SCITT list on 2026-08-21: a reader holding one file has been told something
true and something useless, and telling them which is the whole job.

So the tail is either pinned or it is not, and the output says which:

  * ``--expect-count`` and ``--expect-last-hash`` let a caller who retained an
    earlier head pin it. A mismatch exits the same way a mid-chain break does,
    because a chain that is not the chain you pinned is not the chain you
    pinned.
  * ``--check-witness`` fetches the heads recorded in a public transparency log
    the maintainer does not operate. A head the log carries and this file no
    longer reaches is a removal, visible without having retained anything.
  * With neither, ``tailPinned`` is false, it is said on stderr, and it is in
    the JSON, so nothing downstream can read *verified* as *verified to be
    complete*.

This imports no Vaara code, so it checks the maintainer as readily as anyone
else. It needs ``rfc8785`` and nothing more; the witness check uses the standard
library.

Usage:
    python scripts/vcr_chain.py
    python scripts/vcr_chain.py rows.json --check-witness
    python scripts/vcr_chain.py rows.json --expect-count 7 --expect-last-hash sha256:...
    python scripts/vcr_chain.py rows.json --json
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / "conformance" / "reproductions.json"
ZERO = "sha256:" + "0" * 64
TIMEOUT = 20


def digest(row: dict) -> str:
    import rfc8785

    return "sha256:" + hashlib.sha256(rfc8785.dumps(row)).hexdigest()


def check(data: dict) -> list[str]:
    problems = []
    expected = data.get("genesis", ZERO)
    for position, row in enumerate(data.get("reproductions", [])):
        if row.get("prev") != expected:
            problems.append(
                f"row {row.get('id', '?')} ({row.get('party', 'unknown')}) at "
                f"position {position} links to {row.get('prev')}, but the chain "
                f"reaches it at {expected}. Something before it was removed, "
                f"reordered or rewritten."
            )
        expected = digest(row)
    return problems


def heads(data: dict) -> list[str]:
    """Every head this file passes through, oldest first."""
    return [digest(row) for row in data.get("reproductions", [])]


def head_of(data: dict) -> str:
    rows = data.get("reproductions", [])
    return digest(rows[-1]) if rows else data.get("genesis", ZERO)


def fetch_entry(log_url: str, uuid: str) -> dict | None:
    """The log entry, or None when the log could not be reached or read."""
    url = f"{log_url.rstrip('/')}/api/v1/log/entries/{uuid}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:  # noqa: S310
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    return payload.get(uuid) or next(iter(payload.values()), None)


def check_witness(data: dict) -> tuple[list[str], list[dict], list[str]]:
    """Verify each recorded head against the log it names.

    Returns (problems, confirmed entries, unreachable heads). A log that cannot
    be reached is reported as unchecked and never as a pass, because an
    unreachable log proves nothing in either direction.
    """
    problems: list[str] = []
    confirmed: list[dict] = []
    unreachable: list[str] = []
    reachable = set(heads(data)) | {data.get("genesis", ZERO)}

    for entry in data.get("witnessed_heads", []):
        head = entry.get("head")
        uuid = entry.get("uuid")
        log_url = entry.get("logUrl") or ""
        if not (head and uuid and log_url):
            problems.append(f"witnessed head entry is incomplete: {entry!r}")
            continue
        if head not in reachable:
            problems.append(
                f"witnessed head {head} is not a head this file reaches. A row "
                f"it covered was removed, reordered or rewritten."
            )
        record = fetch_entry(log_url, uuid)
        if record is None:
            unreachable.append(head)
            continue
        try:
            body = json.loads(base64.b64decode(record["body"]))
            logged = body["spec"]["data"]["hash"]["value"]
        except (KeyError, ValueError, TypeError) as exc:
            problems.append(f"log entry {uuid} is not readable: {exc}")
            continue
        if f"sha256:{logged}" != head:
            problems.append(
                f"log entry {uuid} commits to sha256:{logged}, not {head}"
            )
            continue
        confirmed.append({
            "head": head,
            "logIndex": record.get("logIndex"),
            "integratedTime": record.get("integratedTime"),
            "logUrl": log_url,
        })
    return problems, confirmed, unreachable


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute the VCR row chain and say whether the tail is pinned.")
    parser.add_argument("path", nargs="?", default=str(DEFAULT))
    parser.add_argument("--check-witness", action="store_true",
                        help="Verify recorded heads against the public log")
    parser.add_argument("--expect-count", type=int, default=None,
                        help="Row count you retained earlier. A mismatch fails.")
    parser.add_argument("--expect-last-hash", default=None,
                        help="Head you retained earlier. A mismatch fails.")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Machine-readable result, including tailPinned")
    args = parser.parse_args(argv)

    path = Path(args.path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("reproductions", [])
    head = head_of(data)
    problems = check(data)

    # A retained head is the only thing that turns a valid shorter chain into a
    # detectable truncation for a reader holding nothing else. A mismatch is the
    # same failure as a break in the middle, deliberately: a chain that is not
    # the chain you pinned is not the chain you pinned.
    pinned_by_caller = False
    if args.expect_count is not None:
        if len(rows) != args.expect_count:
            problems.append(
                f"expected {args.expect_count} row(s), found {len(rows)}. Rows "
                f"were removed from the end, which leaves a valid shorter chain."
            )
        else:
            pinned_by_caller = True
    if args.expect_last_hash is not None:
        if head != args.expect_last_hash:
            problems.append(
                f"expected head {args.expect_last_hash}, found {head}."
            )
        else:
            pinned_by_caller = True

    witness_problems: list[str] = []
    confirmed: list[dict] = []
    unreachable: list[str] = []
    if args.check_witness:
        witness_problems, confirmed, unreachable = check_witness(data)
        if not data.get("witnessed_heads"):
            witness_problems.append(
                "no witnessed heads recorded, so nothing here pins the tail")
    problems += witness_problems

    pinned_by_witness = any(c["head"] == head for c in confirmed)
    tail_pinned = pinned_by_caller or pinned_by_witness

    result = {
        "rows": len(rows),
        "head": head,
        "chainIntact": not check(data),
        "tailPinned": tail_pinned,
        "tailPinnedBy": (
            ["caller"] * pinned_by_caller + ["witness"] * pinned_by_witness
        ),
        "witnessedHeadsRecorded": len(data.get("witnessed_heads", [])),
        "witnessedHeadsConfirmed": confirmed,
        "witnessUnreachable": unreachable,
        "problems": problems,
    }

    if args.as_json:
        print(json.dumps(result, indent=2))
        return 1 if problems else 0

    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} problem(s).", file=sys.stderr)
        return 1

    print(f"{len(rows)} row(s), chain intact.")
    print(f"head: {head}")
    for c in confirmed:
        print(f"  WITNESSED  {c['head']}  logIndex={c['logIndex']} "
              f"integratedTime={c['integratedTime']}")
    for h in unreachable:
        print(f"  UNCHECKED  {h}  log unreachable", file=sys.stderr)

    if tail_pinned:
        print(f"tail pinned by: {', '.join(result['tailPinnedBy'])}")
    else:
        # Said on stderr rather than buried in a document nobody reads at
        # verification time. Chain intact and tail unpinned are different
        # findings and a reader is entitled to both.
        print(
            "\ntail NOT pinned. The chain proves nothing about rows removed "
            "from the end,\nbecause what remains is a valid shorter chain. To "
            "pin it, either:\n"
            "  --expect-count N --expect-last-hash sha256:...  (a head you kept)\n"
            "  --check-witness                                 (the public log)",
            file=sys.stderr,
        )

    if not rows and path == DEFAULT:
        print(
            "\nThis is the committed baseline on main, which ships with no rows.\n"
            "Published rows live on the `vcr` branch. To check those:\n"
            "  git fetch origin vcr && git show origin/vcr:vcr/reproductions.json > rows.json\n"
            "  python scripts/vcr_chain.py rows.json --check-witness\n"
            "Each row is also served standalone at https://vaara.io/badge/<slug>.json"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
