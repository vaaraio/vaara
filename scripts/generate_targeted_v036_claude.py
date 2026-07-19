# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""v0.36 cross-model held-out generation via Claude (Anthropic SDK).

Closed-weight RLHF-heavy attacker leg. Same prompt structure as the Mixtral
leg (generate_targeted_v036.py) so the two legs differ only in the model
fingerprint. Schema, fingerprint, and per-category instructions are shared
via scripts/_v036_common.py.

Reads ANTHROPIC_API_KEY from env. Fails loudly if missing.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from _v036_common import (
    CATEGORY_INSTRUCTIONS,
    CATEGORY_PREFIX,
    GEN_DIR,
    SEEDS_DIR,
    SYSTEM_PROMPT,
    build_user_message,
    fingerprint,
    load_prior_fingerprints,
    load_seeds,
    parse_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--model-tag", default="claude")
    ap.add_argument("--category", choices=list(CATEGORY_INSTRUCTIONS), required=True)
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seeds-per-batch", type=int, default=5)
    ap.add_argument("--max-batches", type=int, default=120)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--dedupe-prior", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.stderr.write("FATAL: ANTHROPIC_API_KEY not set in environment.\n")
        return 3

    try:
        import anthropic
    except ImportError:
        sys.stderr.write("FATAL: anthropic SDK not installed. pip install anthropic\n")
        return 3

    client = anthropic.Anthropic()

    random.seed(args.random_seed)
    prefix = CATEGORY_PREFIX[args.category]
    seeds_path = SEEDS_DIR / f"{args.category}.jsonl"
    out_path = Path(args.out) if args.out else GEN_DIR / f"{prefix}-v036-{args.model_tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = load_prior_fingerprints("v036-claude") if args.dedupe_prior else set()
    initial_seen = len(seen)
    written, batch_idx = 0, 0
    with out_path.open("a", encoding="utf-8") as fh:
        while written < args.n and batch_idx < args.max_batches:
            batch_idx += 1
            seeds = load_seeds(seeds_path, args.seeds_per_batch)
            user_msg = build_user_message(args.category, args.batch_size, seeds, batch_idx)
            t0 = time.time()
            try:
                resp = client.messages.create(
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            except Exception as exc:
                sys.stderr.write(f"batch {batch_idx} call failed: {exc}\n")
                time.sleep(2)
                continue
            new = 0
            for entry in parse_jsonl(text, args.category):
                fp = fingerprint(entry)
                if fp in seen:
                    continue
                seen.add(fp)
                fh.write(json.dumps(entry) + "\n")
                fh.flush()
                written += 1
                new += 1
                if written >= args.n:
                    break
            sys.stderr.write(
                f"[{args.category}] batch {batch_idx:>3}  new={new:>3}  "
                f"total={written:>4}/{args.n}  prior_seen={initial_seen}  "
                f"{time.time()-t0:.1f}s\n"
            )

    sys.stderr.write(f"[{args.category}] done: {written} entries -> {out_path}\n")
    return 0 if written else 2


if __name__ == "__main__":
    sys.exit(main())
