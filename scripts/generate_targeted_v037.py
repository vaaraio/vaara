# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""v0.37 cross-family held-out generation via Llama-3.3-70B-Instruct / vLLM.

Mirrors generate_targeted_v036.py with v037 IDs and Llama-family defaults.
Default attacker model is RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic
(FP8 dynamic quantization, fits one MI300X at ~70 GB, vLLM-ROCm native).
Override with --model and --model-tag to swap.

Held-out sizing default is 300/category (900 total) so Wilson CI on a
sub-cell stays within +/- 6 pp at p ~ 0.7. Override with --n.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

from _v037_common import (
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


def call_chat(
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    body = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    ).encode("utf-8")
    req = urllib.request.Request(
        url=base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic")
    ap.add_argument("--model-tag", default="llama33")
    ap.add_argument("--category", choices=list(CATEGORY_INSTRUCTIONS), required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seeds-per-batch", type=int, default=5)
    ap.add_argument("--max-batches", type=int, default=80)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--dedupe-prior", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    random.seed(args.random_seed)
    prefix = CATEGORY_PREFIX[args.category]
    seeds_path = SEEDS_DIR / f"{args.category}.jsonl"
    out_path = Path(args.out) if args.out else GEN_DIR / f"{prefix}-v037-{args.model_tag}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = load_prior_fingerprints(f"v037-{args.model_tag}") if args.dedupe_prior else set()
    initial_seen = len(seen)
    written, batch_idx = 0, 0
    with out_path.open("a", encoding="utf-8") as fh:
        while written < args.n and batch_idx < args.max_batches:
            batch_idx += 1
            seeds = load_seeds(seeds_path, args.seeds_per_batch)
            user_msg = build_user_message(args.category, args.batch_size, seeds, batch_idx)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            t0 = time.time()
            try:
                text = call_chat(
                    args.base_url, args.model, messages, args.temperature, args.max_tokens
                )
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
