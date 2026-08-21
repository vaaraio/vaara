#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check that nothing has been removed from Vaara Conformance Results.

The table records occurrences. A run happened, on a date, at a commit, and that
does not stop being true afterwards, so rows are never taken down or edited.
Saying so is worth little on its own, because the person who would quietly edit
the table is the person publishing the promise. So each row carries ``prev``,
the digest of the row before it, and this script recomputes the whole chain.

Drop a row, reorder two, or change a single character of a published one, and
every digest after the change stops matching. The break names the row.

WHAT THE CHAIN ALONE GIVES, stated exactly. A break is detectable to someone
holding an earlier head. It is not detectable to someone who does not, because
the maintainer controls both the file and the published page and could rewrite
the chain and republish it consistently. The earlier wording said the maintainer
"cannot" remove a row on the strength of the chain, which was an overclaim. Iman
Schrock raised it on the SCITT list on 2026-08-21 and was right.

So each published head is recorded in a public transparency log the maintainer
does not operate. ``--check-witness`` fetches those entries and confirms each
witnessed head is the head this file actually reaches, so a rewritten chain is
caught by anyone, not only by someone who kept an old copy. The residual is
narrow: a row added after the last witnessing carries only the chain until the
next head is published.

This imports no Vaara code, so it checks the maintainer as readily as anyone
else. It needs ``rfc8785`` and nothing more; the witness check uses the standard
library.

Usage:
    python scripts/vcr_chain.py                       # check the committed file
    python scripts/vcr_chain.py path/to/rows.json     # or one you downloaded
    python scripts/vcr_chain.py --check-witness       # also verify the log
"""
from __future__ import annotations

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
    """Every head this file passes through, oldest first, ending at the current."""
    out, current = [], data.get("genesis", ZERO)
    for row in data.get("reproductions", []):
        current = digest(row)
        out.append(current)
    return out


def check_witness(data: dict) -> list[str]:
    """Confirm each recorded head really is in the public log it names.

    Fetches the log entry by uuid and reads back the digest it committed to. A
    head the log does not carry, or carries with a different digest, is the
    failure this exists to surface. Network trouble is reported as unchecked
    rather than as a pass, because an unreachable log proves nothing either way.
    """
    problems: list[str] = []
    witnessed = data.get("witnessed_heads", [])
    if not witnessed:
        problems.append(
            "no witnessed heads recorded, so the chain is only checkable by "
            "someone holding an earlier head"
        )
        return problems

    reachable = set(heads(data)) | {data.get("genesis", ZERO)}
    for entry in witnessed:
        head = entry.get("head")
        uuid = entry.get("uuid")
        log_url = (entry.get("logUrl") or "").rstrip("/")
        if not (head and uuid and log_url):
            problems.append(f"witnessed head entry is incomplete: {entry!r}")
            continue
        if head not in reachable:
            problems.append(
                f"witnessed head {head} is not a head this file reaches. A row "
                f"before it was removed, reordered or rewritten."
            )
        url = f"{log_url}/api/v1/log/entries/{uuid}"
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as response:  # noqa: S310
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            print(f"  UNCHECKED  {head[:23]}...  log unreachable: {exc}",
                  file=sys.stderr)
            continue
        record = payload.get(uuid) or next(iter(payload.values()), {})
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
        print(f"  WITNESSED  {head}  logIndex={record.get('logIndex')} "
              f"integratedTime={record.get('integratedTime')}")
    return problems


def main(argv: list[str]) -> int:
    argv = list(argv)
    want_witness = "--check-witness" in argv
    argv = [a for a in argv if a != "--check-witness"]
    path = Path(argv[0]) if argv else DEFAULT
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("reproductions", [])
    problems = check(data)

    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} break(s) in the chain.", file=sys.stderr)
        return 1

    head = digest(rows[-1]) if rows else data.get("genesis", ZERO)
    print(f"{len(rows)} row(s), chain intact.")
    print(f"head: {head}")

    witnessed = data.get("witnessed_heads", [])
    if want_witness:
        print(f"\nchecking {len(witnessed)} witnessed head(s) against the log:")
        witness_problems = check_witness(data)
        for problem in witness_problems:
            print(f"  {problem}", file=sys.stderr)
        if witness_problems:
            print(f"\n{len(witness_problems)} witness problem(s).", file=sys.stderr)
            return 1
        unwitnessed = [h for h in heads(data)
                       if h not in {w.get("head") for w in witnessed}]
        if unwitnessed:
            print(f"\n{len(unwitnessed)} row(s) added since the last witnessing "
                  f"carry only the chain until the next head is published.")
    elif witnessed:
        print(f"{len(witnessed)} witnessed head(s) recorded. Run with "
              f"--check-witness to verify them against the public log.")

    # A checkout of main ships this file EMPTY on purpose: rows arrive by issue
    # form and land on the unprotected `vcr` branch, which is what lets a
    # submitter appear on the table without write access, a fork or a PR. The
    # published page therefore shows rows that this file does not have.
    #
    # Without the note below, a stranger who follows the page's own instruction
    # to verify with this script reads "0 row(s), chain intact" against a page
    # listing rows, and the honest conclusion available to them is that the
    # tool is lying. Saying where the rows actually live costs four lines and
    # removes the only reading that makes this table look dishonest.
    if not rows and path == DEFAULT:
        print(
            "\nThis is the committed baseline on main, which ships with no rows.\n"
            "Published rows live on the `vcr` branch. To check those:\n"
            "  git fetch origin vcr && git show origin/vcr:vcr/reproductions.json > rows.json\n"
            "  python scripts/vcr_chain.py rows.json\n"
            "Each row is also served standalone at https://vaara.io/badge/<slug>.json"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
