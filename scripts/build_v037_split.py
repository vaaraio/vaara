"""Build tests/adversarial/v037_split.json.

Composition per the v0.37 scope in STATE.md:
  - Inherit all v035_split.json train/val/test assignments unchanged.
  - Fold v036 Mixtral TM/PE + v036 Claude TM/PE entries into "train".
  - Keep v036 Mixtral DE + v036 Claude DE as "holdout" (carries v036 cross-model
    held-out signal forward into the v037 eval).
  - Add v037 Llama-3.3-70B TM/PE/DE entries as "holdout" (the new third-family
    cross-model leg).

Drops schema-invalid entries the same way build_v036_split_v2.py does. Dedupes
by fingerprint across the new additions so repeat fingerprints don't show up
in both train and holdout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ADV = Path("tests/adversarial")
GEN = ADV / "generated"
V035_SPLIT = ADV / "v035_split.json"
OUT = ADV / "v037_split.json"

REQUIRED = ["id", "category", "tool_name", "parameters", "context", "expected", "severity", "notes"]
VALID_SEV = {"low", "medium", "high", "critical"}
EXPECTED_CAT = {"TM": "tool_misuse", "PE": "privilege_escalation", "DE": "data_exfil"}


def fingerprint(entry: dict) -> str:
    canonical = json.dumps(
        {
            "t": entry.get("tool_name", ""),
            "p": entry.get("parameters", {}),
            "c": entry.get("context", {}),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(canonical.encode()).hexdigest()


def well_formed(e: dict, expected_cat: str) -> bool:
    if not all(k in e for k in REQUIRED):
        return False
    if "original_task" not in (e.get("context") or {}):
        return False
    if e.get("expected") != "DENY":
        return False
    if e.get("severity") not in VALID_SEV:
        return False
    if e.get("category") != expected_cat:
        return False
    return True


def main() -> int:
    v035 = json.loads(V035_SPLIT.read_text())
    base_assignments: dict[str, str] = dict(v035["assignments"])

    seen_fp: set[str] = set()
    new_train = 0
    new_holdout = 0
    schema_dropped = 0
    semantic_dups = 0
    per_bucket: dict[tuple[str, str, str], int] = {}

    def assign(rel_glob: str, fold_for_de: str | None, fold_for_tmpe: str | None):
        nonlocal new_train, new_holdout, schema_dropped, semantic_dups
        for f in sorted(GEN.glob(rel_glob)):
            relpath = str(f.relative_to(ADV))
            cat_prefix = f.name[:2]
            expected_cat = EXPECTED_CAT[cat_prefix]
            fold = fold_for_de if cat_prefix == "DE" else fold_for_tmpe
            if fold is None:
                continue
            if "mixtral" in f.name:
                leg = "mixtral"
            elif "claude" in f.name:
                leg = "claude"
            elif "llama33" in f.name:
                leg = "llama33"
            else:
                leg = "other"
            for li, raw in enumerate(f.read_text().splitlines()):
                line = raw.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    schema_dropped += 1
                    continue
                if not well_formed(e, expected_cat):
                    schema_dropped += 1
                    continue
                fp = fingerprint(e)
                if fp in seen_fp:
                    semantic_dups += 1
                    continue
                seen_fp.add(fp)
                key = f"{relpath}#L{li}"
                if key in base_assignments:
                    continue
                base_assignments[key] = fold
                bk = (e["category"], leg, fold)
                per_bucket[bk] = per_bucket.get(bk, 0) + 1
                if fold == "train":
                    new_train += 1
                else:
                    new_holdout += 1

    assign("*-v036-mixtral.jsonl", fold_for_de="holdout", fold_for_tmpe="train")
    assign("*-v036-claude.jsonl", fold_for_de="holdout", fold_for_tmpe="train")
    assign("*-v037-llama33.jsonl", fold_for_de="holdout", fold_for_tmpe="holdout")

    fold_counts: dict[str, int] = {}
    for fold in base_assignments.values():
        fold_counts[fold] = fold_counts.get(fold, 0) + 1

    metadata = {
        "version": "v0.37",
        "purpose": (
            "v035_split inherited verbatim; v036 TM/PE both legs folded into train; "
            "v036 DE both legs + v037 Llama-3.3 TM/PE/DE held out for cross-model eval."
        ),
        "key_format": "<relative_path>#L<line_index>",
        "inherits_from": str(V035_SPLIT.relative_to(ADV.parent)),
        "v035_inherited_total": sum(1 for v in v035["assignments"].values()),
        "v037_additions": {
            "new_train_entries": new_train,
            "new_holdout_entries": new_holdout,
            "schema_dropped": schema_dropped,
            "semantic_duplicates_dropped": semantic_dups,
        },
    }
    out = {
        "metadata": metadata,
        "counts": fold_counts,
        "per_bucket": {f"{c}__{leg}__{fold}": n for (c, leg, fold), n in sorted(per_bucket.items())},
        "assignments": dict(sorted(base_assignments.items())),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"wrote {OUT}")
    print(f"v035 inherited: {metadata['v035_inherited_total']}")
    print(f"v037 additions: +{new_train} train, +{new_holdout} holdout")
    print(f"schema dropped: {schema_dropped}  semantic_dups: {semantic_dups}")
    print(f"fold counts: {fold_counts}")
    print("per bucket (cat__leg__fold):")
    for k in sorted(per_bucket):
        print(f"  {'__'.join(k):42s} {per_bucket[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
