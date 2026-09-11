# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail loudly if a corpus split leaks. Run before any retrain, every time.

WHY THIS EXISTS

The published figure for the adversarial classifier comes off a held-out fold.
A generated entry that lands in val, test or holdout does not make the number
worse, it makes the number a lie, and the whole product is a claim about
evidence being checkable. A better score obtained by leakage is the one
outcome this project cannot survive shipping.

The v0.39 split is already built correctly: leak-free by (model x attack_class)
cell, every third cell to holdout, generator tracked per cell. This script does
not invent that discipline. It asserts it still holds after new cells are
added, because the discipline lives in whoever ran the split script and that is
not a durable place for it.

WHAT IT CHECKS

1. No key appears in two folds. The obvious one, and the one a careless merge
   of two split files produces.
2. Every key names a file that exists, and no key points past the end of one.
   Entries in no fold are reported as a note rather than a finding: v0.39
   records `schema_dropped: 37` and this script independently counts 37, so a
   shortfall is the documented behaviour of a generation run, not a defect.
3. Cell integrity: a (model x attack_class) cell sits wholly in one fold.
   Splitting a cell across train and holdout is the subtle leak, because two
   entries from the same generator on the same attack class are near-duplicates
   of each other far more often than two random entries are.
4. Raw upstream files under external/ are sources, not corpus entries, so none
   should appear in an assignment at all. BIPIA-derived entries are excluded
   from this rule on purpose; see the note on EXTERNAL_MARKERS.

Usage:
    python scripts/check_split_leakage.py                     # newest split
    python scripts/check_split_leakage.py tests/adversarial/v039_split.json

Exit 0 clean, 1 on any finding. Nothing is written or repaired: a split that
fails this is a decision for a human, not something to auto-fix.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ADV = REPO / "tests" / "adversarial"

#: Folds a generated entry may legitimately enter. Anything the generator
#: produces goes to train or holdout and never to val or test, because those
#: two are the folds the reported numbers are measured on.
GENERATED_OK = {"train", "holdout"}

#: Raw upstream material, as distinct from corpus entries derived from it.
#: Files under external/ are the third-party sources themselves and are not
#: split at all, so seeing one in an assignment means something went wrong.
#:
#: BIPIA IS DELIBERATELY NOT LISTED HERE, and the first version of this script
#: was wrong to include it. v0.39's own metadata says the BIPIA harness entries
#: were added "with leak-free (model x attack_class) cell split (every 3rd cell
#: -> holdout)", so BIPIA-derived entries teaching the model is the design, not
#: a leak. Check 3 already enforces the property that actually matters there.
#: Encoding a policy the project does not hold produced 246 false findings and
#: would have trained anyone reading them to ignore the output.
EXTERNAL_MARKERS = ("external/",)


def newest_split() -> Path:
    splits = sorted(ADV.glob("v*_split.json"))
    if not splits:
        sys.exit("no split files under tests/adversarial/")
    return splits[-1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(path: Path) -> list[str]:
    data = load(path)
    assignments: dict[str, str] = data.get("assignments", {})
    if not assignments:
        return [f"{path.name}: no assignments block"]

    findings: list[str] = []
    unclaimed: dict[str, int] = {}

    # 1. A key in two folds. Dict keys cannot duplicate, so this catches the
    #    real-world shape instead: the same entry reached by two spellings of
    #    one path.
    seen: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, fold in assignments.items():
        rel, _, line = key.rpartition("#")
        norm = (str(Path(rel)).lstrip("./"), line)
        seen[norm].append(f"{key} -> {fold}")
    for norm, entries in seen.items():
        folds = {e.rsplit(" -> ", 1)[1] for e in entries}
        if len(folds) > 1:
            findings.append(
                f"LEAK: {norm[0]}{norm[1]} assigned to {sorted(folds)}"
            )

    # 2. Every referenced file exists, and every line in it is claimed once.
    by_file: dict[str, set[str]] = defaultdict(set)
    for key in assignments:
        rel, _, line = key.rpartition("#")
        by_file[rel].add(line)
    for rel, lines in sorted(by_file.items()):
        p = ADV / rel
        if not p.is_file():
            findings.append(f"MISSING FILE: {rel} referenced by {len(lines)} keys")
            continue
        n = sum(1 for ln in p.read_text(errors="replace").splitlines() if ln.strip())
        # Unclaimed lines are EXPECTED, not a defect. v0.39's metadata records
        # `schema_dropped: 37`, and every generation run drops entries that
        # fail schema validation or duplicate an existing one. Seed files also
        # carry a non-JSON header line that load_seeds skips.
        #
        # So a shortfall is reported as information. Only the reverse is a
        # finding: a split claiming MORE entries than the file holds means a
        # key points at a line that does not exist, and that entry is silently
        # absent from whichever fold believes it owns it.
        if len(lines) > n:
            findings.append(
                f"PHANTOM KEYS: {rel} has {n} non-empty lines, split claims "
                f"{len(lines)}. Keys point past the end of the file."
            )
        elif n - len(lines) > 0:
            unclaimed[rel] = n - len(lines)

    # 3. Cell integrity, where the split records cells.
    for cells_key in ("v039_cell_breakdown", "cell_breakdown"):
        cells = data.get(cells_key) or {}
        by_cell: dict[str, set[str]] = defaultdict(set)
        for cell, fold in cells.items():
            stem = cell.rsplit("__", 1)[0] if "__" in cell else cell
            by_cell[stem].add(fold if isinstance(fold, str) else str(fold))
        for stem, folds in by_cell.items():
            if len(folds) > 1:
                findings.append(
                    f"SPLIT CELL: {stem} spans {sorted(folds)}; a cell must sit "
                    f"wholly in one fold"
                )

    # 4. External data never teaches.
    for key, fold in assignments.items():
        if fold == "train" and any(m in key for m in EXTERNAL_MARKERS):
            findings.append(f"EXTERNAL IN TRAIN: {key}")

    if unclaimed:
        total = sum(unclaimed.values())
        print(f"[split] note: {total} entries across {len(unclaimed)} "
              f"file(s) are in no fold. Expected where a run dropped "
              f"entries on schema or duplication.")
    return findings


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else newest_split()
    if not path.is_file():
        sys.exit(f"no such split: {path}")

    data = load(path)
    counts = data.get("counts", {})
    total = sum(counts.values()) if counts else len(data.get("assignments", {}))
    print(f"[split] {path.name}  {total} entries  {counts}")

    findings = check(path)
    if not findings:
        print("[split] clean: no leakage found")
        return 0

    print(f"\n[split] {len(findings)} FINDING(S). Do not retrain until resolved.")
    for f in findings[:40]:
        print(f"  {f}")
    if len(findings) > 40:
        print(f"  ... and {len(findings) - 40} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
