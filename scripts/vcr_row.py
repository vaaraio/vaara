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
    "prior_art": 400,
}

FIELDS = {
    "Name to publish": "party",
    "Affiliation": "affiliation",
    "Which suites you ran": "suites",
    "Commit you ran at": "at_commit",
    "Result": "result",
    "Kind of run": "kind",
    "Your own scoping": "their_scoping",
    "Link to where you reported it": "record",
    "Prior art": "prior_art",
}

#: What a run establishes is a property of who wrote the verifier and who wrote
#: the vectors, never of how well the run went. The three are not degrees of one
#: another: only the second says anything about the specification text, and only
#: the third says anything about both the text and the cases.
#:
#: Joel Hillier proposed this on the SCITT list on 2026-08-21, after Iman
#: Schrock scoped his own row that way unprompted and Emek Can Dogru filed one
#: carrying it. The loss it prevents happens at the point a result is filed: a
#: row outlives the message that explains it, and a register that cannot tell a
#: reproduction from an independent implementation will be cited for the second
#: while holding the first, by someone who is not lying.
#: The fourth kind arrived from Tiago Pinto on the SCITT list on 2026-08-25.
#: He reproduced a published class digest from the prose that describes it,
#: without running the author's verifier and without writing a second
#: implementation, and said plainly that it fitted none of the three. He is
#: right, and it is not a rung on the same ladder. A reproduction asks whether
#: the artefact runs somewhere else. This asks whether the description was
#: sufficient to derive the published value at all, which is a question about
#: the text, answerable with no code on either side.
KINDS = {
    "reproduction": "the author's checkers over the author's vectors",
    "construction_reproduction": (
        "a published construction derived from its prose alone, without running "
        "the author's verifier"
    ),
    "independent_implementation": (
        "an independent implementation from the text, run against the author's vectors"
    ),
    "independent_implementation_and_vectors": (
        "an independent implementation run against independently constructed vectors"
    ),
}

#: An unnamed kind reads as the weakest claim available, never the strongest.
#: Rows listed before the field existed carry no kind, and nothing may be added
#: to them after the fact, so the page states the rule instead of editing them.
DEFAULT_KIND = "reproduction"


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


#: Path to the vector directory as git spells it, for `git ls-tree`.
VECTORS_IN_TREE = "tests/vectors"


def known_suites(at_commit: str | None = None) -> set[str]:
    """The suite names, at a commit if one is given, else in the working tree.

    Resolving against the working tree was the only behaviour until 2026-08-25,
    and it made an honest historical row impossible to file. A row is scoped to
    the commit it ran at, which is the whole reason the field exists, so a run
    over the 44 suites that existed at 62a7080 was being measured against the 46
    on the branch today and refused. The rejection then told the submitter to
    rerun at the commit they were listing and use the link it prints, which is
    exactly what they had done: the runner at that commit prints `all 44
    suites`, and this function refused it. The instruction contradicted itself.

    Naming the suites individually is not a way round it either. FIELD_LIMITS
    caps `suites` at 200 characters and the 44 names come to 802, which the
    comment on WHOLE_CORPUS below already says.

    So the only path through was to relabel a 44-suite run as 46, and Iman
    Schrock refused to do that and said so on issue #618. He was right. A
    register built to record what people actually got must never be the reason
    somebody restates their run.
    """
    if at_commit:
        try:
            out = subprocess.run(
                ["git", "ls-tree", "--name-only", f"{at_commit}:{VECTORS_IN_TREE}"],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            names = {line.strip().rstrip("/") for line in out.splitlines() if line.strip()}
            if names:
                return names
            # An empty listing means the path did not exist at that commit.
            # Fall through rather than accept a row against nothing.
        except (subprocess.CalledProcessError, FileNotFoundError):
            # A shallow clone cannot see the tree at an old commit. Better to
            # fall back to the working tree than to refuse a row for a reason
            # that is about our checkout depth and not about their run.
            pass
    return {p.name for p in VECTORS.iterdir() if p.is_dir()} if VECTORS.exists() else set()


#: What scripts/conformance_runner.py writes into the prefilled issue link after
#: a whole-corpus run: `all 43 suites`. It says so rather than listing every
#: name, both because the row reads better and because the names do not fit the
#: 200-character field.
WHOLE_CORPUS = re.compile(r"^all\s+(\d+)\s+suites$", re.IGNORECASE)


def parse_suites(raw: str, at_commit: str | None = None) -> list[str]:
    """The suites a row claims, from either the tool's words or the party's.

    Two accepted forms, and nothing else:

    * ``all <N> suites``, exactly as the runner prints it, where N has to match
      the corpus AT THE COMMIT THE ROW IS SCOPED TO. A whole-corpus claim that
      names the wrong size for that commit is a claim about some other run.
    * a comma or newline separated list of suite directory names.

    The first form existed on the printing side from 2026-08-17 and was never
    taught to this validator, so the single path we tell people to walk was
    refused here. Issue #587, the first real application, hit it. Kept narrow on
    purpose: it accepts what our own tool emits and still refuses a hand-written
    paragraph, because a row is quoted onto a public page and free prose in a
    field with a fixed meaning is how that page stops meaning one thing.
    """
    known = known_suites(at_commit)
    text = raw.strip()
    whole = WHOLE_CORPUS.match(text)
    if whole:
        claimed = int(whole.group(1))
        if claimed != len(known):
            where = f"at commit {at_commit[:9]}" if at_commit else "in this repository"
            raise Rejected(
                f"The corpus {where} has {len(known)} suites, and the row claims "
                f"a run over {claimed}. Rerun `python scripts/conformance_runner.py` "
                "at the commit you are listing and use the link it prints."
            )
        return sorted(known)

    names = [s.strip() for s in text.replace("\n", ",").split(",") if s.strip()]
    if not names:
        raise Rejected("Name at least one suite you ran.")
    unknown = sorted(set(names) - known)
    if unknown:
        where = f"suites at commit {at_commit[:9]}" if at_commit else "suites in this repository"
        raise Rejected(f"These are not {where}: {', '.join(unknown)}")
    return names


def parse_kind(raw: str) -> str:
    """Map the dropdown's text to a stable key, defaulting to the weakest kind.

    The form's options are prose, and prose gets reworded. The row stores a key
    so a later edit to the wording cannot silently restate what a published row
    claimed. An unrecognised or empty value is never promoted: it becomes the
    weakest kind, which is the same rule the page states for rows listed before
    the field existed.
    """
    text = raw.strip().lower()
    if not text:
        return DEFAULT_KIND
    if text in KINDS:
        return text
    # Checked before the bare "reproduction" prefix, because the fourth kind's
    # wording carries that word too and the order is what keeps them apart.
    if text.startswith("construction"):
        return "construction_reproduction"
    if text.startswith("reproduction"):
        return "reproduction"
    if "independently constructed" in text:
        return "independent_implementation_and_vectors"
    if text.startswith("independent implementation"):
        return "independent_implementation"
    return DEFAULT_KIND


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

    # `sha` is validated above, so the suite check can be scoped to the commit
    # the row is actually about rather than to whatever is on the branch today.
    suites = parse_suites(str(fields.get("suites", "")), sha)

    result = str(fields.get("result", "")).strip()
    if not result:
        raise Rejected("The result is required.")

    row = {
        "party": party,
        "affiliation": str(fields.get("affiliation", "")).strip(),
        "suites": suites,
        "result": result,
        "at_commit": sha,
        "kind": parse_kind(str(fields.get("kind", ""))),
        "their_scoping": str(fields.get("their_scoping", "")).strip(),
        "record": record,
        "submitted_by": author,
    }
    # Only carried when the submitter gave one. Adding an empty key to every
    # row would change the JCS bytes of rows that say nothing new, and the
    # digest is supposed to move only when the content does.
    prior_art = str(fields.get("prior_art", "")).strip()
    if prior_art:
        row["prior_art"] = prior_art
    return row


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
