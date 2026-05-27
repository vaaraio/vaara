"""Build tests/adversarial/v039_split.json.

Composition:
  - Inherit all v037_split.json train/holdout assignments unchanged.
  - Add new v0.39 BIPIA entries (under tests/adversarial/v039_bipia/) with
    a leak-free (model x attack_class) cell split:
      * Cells are ranked by attack_class then model name.
      * Every 3rd cell goes to "holdout" so train and holdout never share
        the same (model, attack_class) cell.
      * Both follows and benign-under-pressure in a chosen holdout cell
        move together, so FPR holdout measures the same distribution as
        recall holdout.

The split is keyed by "<relative_path>#L<line_index>" so the existing
load_corpus_keyed/save_classifier_bundle pipeline picks it up without
code changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ADV = REPO / "tests/adversarial"
V037_SPLIT = ADV / "v037_split.json"
V039_DIR = ADV / "v039_bipia"
OUT = ADV / "v039_split.json"


def cell_key(model: str, attack_class: str) -> str:
    return f"{attack_class}__{model}"


def load_v039_entries() -> list[tuple[str, dict]]:
    """Return (split-key, entry) for every v039 BIPIA entry on disk."""
    out: list[tuple[str, dict]] = []
    for fp in sorted(V039_DIR.glob("*.jsonl")):
        rel = fp.relative_to(ADV).as_posix()
        with fp.open() as fh:
            for li, raw in enumerate(fh):
                line = raw.strip()
                if not line:
                    continue
                e = json.loads(line)
                out.append((f"{rel}#L{li}", e))
    return out


def main() -> int:
    v037 = json.loads(V037_SPLIT.read_text())
    base_assignments: dict[str, str] = dict(v037["assignments"])

    keyed = load_v039_entries()
    if not keyed:
        print("[error] no v039 entries found under tests/adversarial/v039_bipia/")
        return 2

    # Enumerate cells; assign every-3rd to holdout (sorted for determinism).
    cells: set[tuple[str, str]] = set()
    for _, e in keyed:
        meta = e.get("_v039_bipia", {})
        cells.add((meta["model"], meta["attack_class"]))
    ordered_cells = sorted(cells, key=lambda mc: (mc[1], mc[0]))
    cell_to_fold: dict[tuple[str, str], str] = {}
    for idx, mc in enumerate(ordered_cells):
        cell_to_fold[mc] = "holdout" if idx % 3 == 0 else "train"

    # Apply assignments to v039 entries.
    n_train = 0
    n_holdout = 0
    per_cell_counts: dict[tuple[str, str, str], dict[str, int]] = {}
    for key, e in keyed:
        meta = e["_v039_bipia"]
        cell = (meta["model"], meta["attack_class"])
        fold = cell_to_fold[cell]
        base_assignments[key] = fold
        kind = "follow" if meta["reflects_injection"] else "benign"
        bk = (meta["model"], meta["attack_class"], fold)
        per_cell_counts.setdefault(bk, {"follow": 0, "benign": 0})
        per_cell_counts[bk][kind] += 1
        if fold == "train":
            n_train += 1
        else:
            n_holdout += 1

    # Roll up fold totals across the full assignments dict.
    fold_counts: dict[str, int] = {}
    for fold in base_assignments.values():
        fold_counts[fold] = fold_counts.get(fold, 0) + 1

    # Cell-level summary (model, attack_class, fold) -> (follow, benign)
    cell_summary = {
        f"{m}__{ac}__{fold}": v
        for (m, ac, fold), v in sorted(per_cell_counts.items())
    }

    # Stats per fold for v039 additions only.
    v039_follow_train = sum(
        v["follow"] for (m, ac, fold), v in per_cell_counts.items() if fold == "train"
    )
    v039_follow_holdout = sum(
        v["follow"] for (m, ac, fold), v in per_cell_counts.items() if fold == "holdout"
    )
    v039_benign_train = sum(
        v["benign"] for (m, ac, fold), v in per_cell_counts.items() if fold == "train"
    )
    v039_benign_holdout = sum(
        v["benign"] for (m, ac, fold), v in per_cell_counts.items() if fold == "holdout"
    )

    metadata = {
        "version": "v0.39",
        "purpose": (
            "v037_split inherited verbatim; v0.39 BIPIA harness entries added with "
            "leak-free (model x attack_class) cell split (every 3rd cell -> holdout)."
        ),
        "key_format": "<relative_path>#L<line_index>",
        "inherits_from": str(V037_SPLIT.relative_to(ADV.parent)),
        "v037_inherited_total": sum(1 for v in v037["assignments"].values()),
        "v039_additions": {
            "n_train_entries": n_train,
            "n_holdout_entries": n_holdout,
            "follows_train": v039_follow_train,
            "follows_holdout": v039_follow_holdout,
            "benign_under_pressure_train": v039_benign_train,
            "benign_under_pressure_holdout": v039_benign_holdout,
            "n_cells_total": len(ordered_cells),
            "n_cells_holdout": sum(1 for f in cell_to_fold.values() if f == "holdout"),
        },
        "v039_cell_fold": {
            cell_key(m, ac): f for (m, ac), f in sorted(cell_to_fold.items(), key=lambda x: (x[0][1], x[0][0]))
        },
    }
    out = {
        "metadata": metadata,
        "counts": fold_counts,
        "v039_cell_breakdown": cell_summary,
        "assignments": dict(sorted(base_assignments.items())),
    }
    payload = json.dumps(out, indent=2) + "\n"
    OUT.write_text(payload)
    sha = hashlib.sha256(payload.encode()).hexdigest()

    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"sha256 = {sha}")
    print(f"v037 inherited assignments: {metadata['v037_inherited_total']}")
    print(
        f"v039 additions: +{n_train} train, +{n_holdout} holdout "
        f"(follows train/holdout = {v039_follow_train}/{v039_follow_holdout}, "
        f"benign train/holdout = {v039_benign_train}/{v039_benign_holdout})"
    )
    print(f"cells holdout/total = {metadata['v039_additions']['n_cells_holdout']}/{metadata['v039_additions']['n_cells_total']}")
    print(f"fold counts (whole manifest): {fold_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
