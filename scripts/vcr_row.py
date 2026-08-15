#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Turn a conformance-row issue into a validated row, with no human in the loop.

The results table is only worth reading if getting onto it costs the same for
everyone, so the checks that decide a row are the ones a machine can actually
enforce: the consent boxes are ticked, the commit exists in this repository,
the suites exist, and the record is a public https link. Nothing here judges
whether a claim is true. It does not have to. A row says what that party
reported, quoted as they wrote it, next to a link to where they wrote it, and
every verdict on the page recomputes from committed bytes, so a reader who
doubts a row can go and disprove it without asking anyone's permission.

What a machine cannot check is flagged for a human instead of guessed at.

Usage:
    python scripts/vcr_row.py --body issue.md --author octocat
    python scripts/vcr_row.py --body issue.md --author octocat --write
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPRODUCTIONS = REPO / "conformance" / "reproductions.json"
VECTORS = REPO / "tests" / "vectors"

# The maintainer cannot be an independent reproduction of the maintainer.
MAINTAINER = {"vaaraio", "vaara", "henri sirkkavaara", "hello@vaara.io", "henri"}

LIMITS = {
    "party": 80,
    "affiliation": 80,
    "suites": 200,
    "at_commit": 40,
    "result": 300,
    "their_scoping": 600,
    "record": 400,
}

FIELDS = {
    "Name to publish": "party",
    "Affiliation": "affiliation",
    "Which suites you ran": "suites",
    "Commit you ran at": "at_commit",
    "Result": "result",
    "Your own scoping": "their_scoping",
    "Link to where you reported it": "record",
}


class Rejected(Exception):
    """A row that cannot go up, carrying the reason the submitter will read."""


def parse_issue_form(body: str) -> dict:
    """Split a rendered issue form into its fields.

    GitHub renders each field as an ``### Label`` heading followed by the
    value, and writes ``_No response_`` where an optional field was left empty.
    """
    out: dict[str, object] = {}
    chunks = re.split(r"^### +", body.replace("\r\n", "\n"), flags=re.MULTILINE)
    for chunk in chunks[1:]:
        label, _, value = chunk.partition("\n")
        label = label.strip()
        value = value.strip()
        if value == "_No response_":
            value = ""
        if label == "Consent":
            out["consent"] = [
                line.strip().lower().startswith(("- [x]", "* [x]"))
                for line in value.splitlines()
                if line.strip().startswith(("- [", "* ["))
            ]
        elif label in FIELDS:
            out[FIELDS[label]] = value
    return out


def slugify(party: str, taken: set[str]) -> str:
    """A stable, ASCII, filesystem-safe name for this party's badge."""
    ascii_name = (
        unicodedata.normalize("NFKD", party).encode("ascii", "ignore").decode("ascii")
    )
    base = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-") or "party"
    slug, n = base, 2
    while slug in taken:
        slug, n = f"{base}-{n}", n + 1
    return slug


def commit_exists(sha: str) -> bool:
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=REPO,
            check=True,
            capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def known_suites() -> set[str]:
    return {p.name for p in VECTORS.iterdir() if p.is_dir()} if VECTORS.exists() else set()


def validate(fields: dict, author: str) -> dict:
    """Return the row to publish, or raise Rejected with a readable reason."""
    consent = fields.get("consent") or []
    if len(consent) < 3 or not all(consent):
        raise Rejected(
            "All three consent boxes have to be ticked, including the one "
            "about the row being permanent. A row that cannot come down is "
            "only fair if you knew that before you asked for it, so this one "
            "stops here."
        )

    for name, limit in LIMITS.items():
        value = str(fields.get(name, ""))
        if len(value) > limit:
            raise Rejected(f"`{name}` is longer than {limit} characters.")
        if any(ord(ch) < 32 and ch != "\n" for ch in value):
            raise Rejected(f"`{name}` contains control characters.")

    party = str(fields.get("party", "")).strip()
    if not party:
        raise Rejected("A name to publish is required.")
    if party.lower() in MAINTAINER or author.lower() in MAINTAINER:
        raise Rejected(
            "The maintainer is not an independent reproduction. This table is "
            "for parties other than Vaara."
        )

    record = str(fields.get("record", "")).strip()
    if not record.startswith("https://"):
        raise Rejected(
            "The record has to be a public https link. Private mail and "
            "screenshots do not qualify, because a reader cannot check them."
        )

    sha = str(fields.get("at_commit", "")).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise Rejected(
            "The commit has to be a full 40-character SHA, so anyone can rerun "
            f"exactly what you ran. Got: `{sha}`"
        )
    if not commit_exists(sha):
        raise Rejected(f"Commit `{sha}` is not in this repository.")

    raw_suites = [s.strip() for s in str(fields.get("suites", "")).replace("\n", ",").split(",")]
    suites = [s for s in raw_suites if s]
    if not suites:
        raise Rejected("Name at least one suite you ran.")
    unknown = sorted(set(suites) - known_suites())
    if unknown:
        raise Rejected(f"These are not suites in this repository: {', '.join(unknown)}")

    result = str(fields.get("result", "")).strip()
    if not result:
        raise Rejected("The result is required.")

    return {
        "party": party,
        "affiliation": str(fields.get("affiliation", "")).strip(),
        "suites": suites,
        "result": result,
        "at_commit": sha,
        "their_scoping": str(fields.get("their_scoping", "")).strip(),
        "record": record,
        "submitted_by": author,
    }


def row_digest(row: dict) -> str:
    """sha256 over the JCS-canonical row, the same bytes that get served."""
    import hashlib

    import rfc8785

    return "sha256:" + hashlib.sha256(rfc8785.dumps(row)).hexdigest()


def add_row(row: dict, date: str, data: dict) -> dict:
    """Append the row, chaining it to the one before it.

    The table records occurrences rather than memberships. A run happened, on a
    date, at a commit, and that does not stop being true later, so nothing here
    ever removes a row. ``prev`` carries the digest of the previous entry, which
    turns that from a promise into something a stranger can check: drop a row or
    reorder two and every digest after the change stops matching.

    The maintainer is bound by this as much as anyone. That is the useful part.
    A table its owner can quietly edit is the owner's opinion, and the answer to
    "why is that name still up" becomes a fact rather than a decision.

    Numbers are never reused. The row also records which version of the terms it
    was listed under, so a later change to the terms reaches later rows only.
    """
    rows = data.setdefault("reproductions", [])
    if any(r.get("record") == row["record"] for r in rows):
        raise Rejected("That record is already listed.")
    highest = max((int(r.get("id", 0)) for r in rows), default=0)
    next_id = max(highest, int(data.get("issued", highest))) + 1
    genesis = data.get("genesis", "sha256:" + "0" * 64)
    row = {
        "id": next_id,
        "slug": slugify(row["party"], {r["slug"] for r in rows}),
        "date": date,
        "terms_version": data.get("terms_version", "unversioned"),
        "prev": row_digest(rows[-1]) if rows else genesis,
        **row,
    }
    rows.append(row)
    data["issued"] = next_id
    return row


def verify_chain(data: dict) -> list[str]:
    """Every place the chain says a row was removed, reordered or rewritten."""
    genesis = data.get("genesis", "sha256:" + "0" * 64)
    problems = []
    expected = genesis
    for position, row in enumerate(data.get("reproductions", [])):
        if row.get("prev") != expected:
            problems.append(
                f"row {row.get('id', '?')} at position {position} links to "
                f"{row.get('prev')!r}, but the chain reaches it at {expected!r}"
            )
        expected = row_digest(row)
    return problems


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--body", required=True, help="file holding the issue body")
    ap.add_argument("--author", required=True, help="GitHub login that filed it")
    ap.add_argument("--date", required=True, help="date to record, YYYY-MM-DD")
    ap.add_argument("--write", action="store_true", help="write reproductions.json")
    args = ap.parse_args(argv)

    body = Path(args.body).read_text(encoding="utf-8")
    data = json.loads(REPRODUCTIONS.read_text(encoding="utf-8"))
    try:
        row = add_row(validate(parse_issue_form(body), args.author), args.date, data)
    except Rejected as why:
        print(str(why), file=sys.stderr)
        return 1

    if args.write:
        REPRODUCTIONS.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
