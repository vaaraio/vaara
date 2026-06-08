"""The shipped docs must contain no em-dashes.

The em-dash is the one AI-writing tell that does not belong in Vaara's
public prose, and it keeps creeping back in on edits. This test locks the
doc surface: every git-tracked Markdown file, plus ``llms.txt``, is scanned
for U+2014 and the build fails if one is present.

Only tracked files are checked, so local-only working notes are never
considered. The test vectors and adversarial corpora are excluded too, since
a sample or attack string may legitimately contain an em-dash.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

EM_DASH = "\u2014"  # escape, so this file carries no literal em-dash
ROOT = Path(__file__).resolve().parent.parent

# Tracked paths under these prefixes are corpora or fixtures, not shipped prose.
EXCLUDE_PREFIXES = ("tests/vectors/", "tests/adversarial/", "research/")


def _tracked_doc_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "*.md", "llms.txt"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    files: list[Path] = []
    for rel in result.stdout.split("\0"):
        if not rel or rel.startswith(EXCLUDE_PREFIXES):
            continue
        files.append(ROOT / rel)
    return files


def test_shipped_docs_have_no_em_dash() -> None:
    offenders: list[str] = []
    for path in _tracked_doc_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if EM_DASH in line:
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "em-dash (U+2014) found in a shipped doc; use a colon, comma, or "
        "period instead:\n" + "\n".join(offenders)
    )
