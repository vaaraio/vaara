# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build tests/adversarial/v040_split.json.

Composition:
  - Inherit every v039_split.json assignment unchanged.
  - Add the v0.40 entries: four adversarial categories that the generator had
    never been pointed at, plus their matched benign counterparts.

WHY THIS SPLITS DIFFERENTLY FROM v0.39, and the departure is deliberate

v0.39 sent every third (model x attack_class) cell to holdout and everything
else to train. That is right for its purpose, which is measuring
generalisation to an attacker model the classifier never trained on.

It is wrong here, for two reasons.

FIRST, there is only one generator. Every v0.40 entry comes from
Llama-3.3-70B, so a (model x attack_class) cell IS a whole category. Splitting
at that grain would put entire categories wholly in train or wholly in holdout,
and a category that lands in holdout contributes nothing to learning while a
category that lands in train cannot be measured at all.

SECOND, and this is the reason the run would otherwise be unmeasurable: these
four categories currently hold n=36, 37, 37 and 46 in the TEST fold, which is
the fold the published figure is measured on. At n=36 a five-point recall
change sits inside the noise. Adding to train only would improve the model and
leave nobody able to show it.

So v0.40 splits on (category x batch), and feeds test and val as well as train
and holdout.

WHY BATCH IS THE RIGHT CELL

The generator writes in batches of 20 from one prompt with one set of seeds,
so entries within a batch are far more alike than entries across batches.
Batch is therefore the natural near-duplicate cluster, and holding out whole
batches is what stops a near-twin of a test entry sitting in train. Splitting
inside a batch would leak in exactly the way the cell discipline exists to
prevent.

Allocation is a fixed rotation over six batches: three train, one val, one
test, one holdout. That is roughly 50/17/17/17, and it takes the per-category
test denominator from about 36 to about 150.

BENIGN COUNTERPARTS

The matched benigns get the same rotation over their own batches. They are not
paired entry-to-entry with a specific attack, unlike v0.39's BIPIA follows, so
proportional allocation per category is the honest equivalent of v0.39's
"benigns move with their cell" rather than a weaker version of it.

Usage:  python scripts/build_v040_split.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ADV = REPO / "tests" / "adversarial"
V039_SPLIT = ADV / "v039_split.json"
OUT = ADV / "v040_split.json"

GEN_DIR = ADV / "generated"
BENIGN_DIR = ADV / "benign_generated"

#: The four categories this release adds, by file prefix.
PREFIXES = {"PI": "prompt_injection", "SR": "ssrf_via_tools",
            "DA": "destructive_actions", "CE": "credential_exfil"}

#: Fixed rotation over batches. Three train, then one each of val, test and
#: holdout. Written out rather than computed so the shape is legible and a
#: future reader can see the proportions without running it.
ROTATION = ("train", "train", "train", "val", "test", "holdout")

#: `PI-v037-001-020` -> batch 001. The generator numbers batches in the id, so
#: the cluster boundary is recoverable from the entry itself and does not have
#: to be recorded separately.
_ID_BATCH = re.compile(r"^[A-Z]{2}-[^-]+-(\d+)-\d+$")


def batch_of(entry: dict, line_index: int) -> str:
    """Which generation batch an entry came from.

    Falls back to a synthetic batch of 20 by line position when an id does not
    carry one, so a file written by an older generator still clusters rather
    than splitting per entry.
    """
    m = _ID_BATCH.match(str(entry.get("id", "")))
    if m:
        return m.group(1)
    return f"L{line_index // 20:03d}"


def collect(directory: Path, prefixes: dict[str, str]) -> list[tuple[str, str, str]]:
    """Return (split-key, category, batch) for every new entry on disk."""
    out: list[tuple[str, str, str]] = []
    if not directory.is_dir():
        return out
    for fp in sorted(directory.glob("*.jsonl")):
        stem_prefix = fp.name.split("-")[0]
        # Benign files are BT-v035-<PREFIX>-... so look one field further in.
        if stem_prefix == "BT":
            parts = fp.name.split("-")
            stem_prefix = parts[2] if len(parts) > 2 else ""
        if stem_prefix not in prefixes:
            continue
        # Only files this release produced. Anything already carried by v039 is
        # inherited untouched and must not be re-assigned here.
        if "v037" not in fp.name and "v040" not in fp.name and "v035" not in fp.name:
            continue
        rel = fp.relative_to(ADV).as_posix()
        with fp.open() as fh:
            for li, raw in enumerate(fh):
                line = raw.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append((f"{rel}#L{li}", prefixes[stem_prefix], batch_of(e, li)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not V039_SPLIT.is_file():
        print(f"[error] missing {V039_SPLIT}")
        return 2
    v039 = json.loads(V039_SPLIT.read_text())
    assignments: dict[str, str] = dict(v039["assignments"])
    inherited = len(assignments)

    new = collect(GEN_DIR, PREFIXES) + collect(BENIGN_DIR, PREFIXES)
    new = [(k, c, b) for k, c, b in new if k not in assignments]
    if not new:
        print("[error] no new v0.40 entries found; has generation finished?")
        return 2

    # Rank batches within each category so the rotation is deterministic and
    # does not depend on filesystem order.
    per_cat_batches: dict[str, set[str]] = defaultdict(set)
    for _, cat, batch in new:
        per_cat_batches[cat].add(batch)
    cell_to_fold: dict[tuple[str, str], str] = {}
    for cat, batches in per_cat_batches.items():
        for idx, batch in enumerate(sorted(batches)):
            cell_to_fold[(cat, batch)] = ROTATION[idx % len(ROTATION)]

    added: dict[str, int] = defaultdict(int)
    per_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for key, cat, batch in new:
        fold = cell_to_fold[(cat, batch)]
        assignments[key] = fold
        added[fold] += 1
        per_cat[cat][fold] += 1

    counts: dict[str, int] = defaultdict(int)
    for fold in assignments.values():
        counts[fold] += 1

    doc = {
        "metadata": {
            "version": "v0.40",
            "purpose": (
                "v039_split inherited verbatim; v0.40 adds prompt_injection, "
                "ssrf_via_tools, destructive_actions and credential_exfil with "
                "their matched benigns, split leak-free by (category x batch) "
                "cell on a fixed train/train/train/val/test/holdout rotation."
            ),
            "departure_from_v039": (
                "v0.39 split by (model x attack_class) with every 3rd cell to "
                "holdout. Every v0.40 entry comes from one generator, so that "
                "grain would place whole categories wholly in one fold. It also "
                "feeds test and val, because these four categories held only "
                "36 to 46 entries in test and the published figure is measured "
                "there, so a train-only addition would be unmeasurable."
            ),
            "cell": "(category, generation batch)",
            "rotation": list(ROTATION),
            "key_format": "<relative_path>#L<line_index>",
            "inherits_from": "adversarial/v039_split.json",
            "v039_inherited_total": inherited,
            "v040_additions": dict(added),
            "v040_per_category": {c: dict(f) for c, f in sorted(per_cat.items())},
        },
        "counts": dict(counts),
        "v040_cell_breakdown": {
            f"llama33__{cat}__{batch}": fold
            for (cat, batch), fold in sorted(cell_to_fold.items())
        },
        "assignments": assignments,
    }

    print(f"[v040] inherited {inherited}, added {sum(added.values())}")
    for cat, folds in sorted(per_cat.items()):
        row = "  ".join(f"{f}={folds[f]}" for f in ("train", "val", "test", "holdout"))
        print(f"  {cat:22s} {row}")
    print(f"[v040] totals: {dict(counts)}")

    if args.dry_run:
        print("[v040] dry run, nothing written")
        return 0

    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"[v040] wrote {OUT}")
    print("[v040] now run: python scripts/check_split_leakage.py "
          "tests/adversarial/v040_split.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
