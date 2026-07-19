# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-entry attribution of Pipeline.intercept + AdversarialClassifier.

Runs every corpus entry through a fresh Pipeline (clean state) and the
classifier in parallel, captures signals + decision + classifier score
per entry, and writes a single JSON with provenance.

Deliverable for v0.31 item 5. Feeds item 6's three-way variant compare.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from train_adversarial_classifier import load_corpus_keyed  # noqa: E402


def file_sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p and p.exists() else None


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def source_of(rel):
    return "hand_curated" if "/" not in rel else "llm_generated"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="tests/adversarial/v031_split.json")
    ap.add_argument("--fold", default="all",
                    choices=("all", "train", "val", "test"))
    ap.add_argument("--classifier-bundle",
                    default="src/vaara/data/adversarial_classifier_v9.joblib")
    ap.add_argument("--out",
                    default="tests/adversarial/v031/pipeline_rule_attribution_v031.json")
    ap.add_argument("--manifest", default="tests/adversarial/MANIFEST.sha256")
    args = ap.parse_args()

    split_path = Path(args.split)
    bundle_path = Path(args.classifier_bundle)
    out_path = Path(args.out)
    manifest_path = Path(args.manifest)

    assignments = json.loads(split_path.read_text())["assignments"]

    from vaara import Pipeline
    from vaara.adversarial_classifier import AdversarialClassifier

    clf = AdversarialClassifier(bundle_path=str(bundle_path))
    print(f"[classifier] {bundle_path}  v={clf.bundle_version}  th={clf.threshold}")

    keyed = load_corpus_keyed()
    if args.fold != "all":
        keyed = [(k, e) for k, e in keyed if assignments.get(k) == args.fold]
    print(f"[corpus] {len(keyed)} entries (fold={args.fold})")

    rows, t0 = [], time.monotonic()
    top_signal = Counter(); decisions = Counter()
    clf_above = 0; n_int_err = n_clf_err = 0

    for i, (key, e) in enumerate(keyed):
        rel = key.split("#L", 1)[0]
        pipe = Pipeline()
        try:
            r = pipe.intercept(
                agent_id=f"attr-{i:06d}",
                tool_name=e["tool_name"],
                parameters=e.get("parameters", {}),
                context=e.get("context", {}),
            )
            sig = {k: float(v) for k, v in (r.signals or {}).items()}
            ib = {
                "action_type": getattr(r.action_type, "name", str(r.action_type)),
                "decision": r.decision,
                "point_estimate": float(r.risk_score),
                "conformal_interval": [float(r.risk_interval[0]),
                                       float(r.risk_interval[1])],
                "signals": sig,
                "reason": r.reason,
                "evaluation_ms": float(r.evaluation_ms),
            }
            decisions[r.decision] += 1
            if sig:
                top_signal[max(sig.items(), key=lambda kv: kv[1])[0]] += 1
        except Exception as exc:
            n_int_err += 1
            ib = {"error": f"{type(exc).__name__}: {exc}"}

        try:
            s = clf.score(tool_name=e["tool_name"],
                          parameters=e.get("parameters", {}),
                          context=e.get("context", {}))
            cb = {"score": float(s), "threshold": float(clf.threshold),
                  "above_threshold": bool(s >= clf.threshold)}
            if cb["above_threshold"]:
                clf_above += 1
        except Exception as exc:
            n_clf_err += 1
            cb = {"error": f"{type(exc).__name__}: {exc}"}

        rows.append({
            "key": key, "id": e.get("id", ""),
            "category": e.get("category", "UNKNOWN"),
            "source": source_of(rel),
            "expected": e.get("expected", ""),
            "fold": assignments.get(key, "unassigned"),
            "intercept": ib, "classifier": cb,
        })

        if (i + 1) % 500 == 0:
            el = time.monotonic() - t0
            print(f"  [progress] {i+1}/{len(keyed)}  el={el:.1f}s  "
                  f"rate={(i+1)/el:.1f}/s")

    el = time.monotonic() - t0
    print(f"[done] {len(rows)} entries in {el:.1f}s")
    print(f"[decisions] {dict(decisions)}")
    print(f"[top_signal] {dict(top_signal)}")
    print(f"[clf] above={clf_above}/{len(rows)}  "
          f"errors int={n_int_err} clf={n_clf_err}")

    out = {
        "metadata": {
            "created_utc": dt.datetime.now(dt.UTC).isoformat(),
            "vaara_commit": git_commit(),
            "fold": args.fold,
            "n_entries": len(rows),
            "elapsed_seconds": round(el, 2),
            "classifier_bundle_path":
                str(bundle_path.resolve().relative_to(REPO))
                if bundle_path.exists() else None,
            "classifier_bundle_sha256": file_sha256(bundle_path),
            "classifier_bundle_version": clf.bundle_version,
            "classifier_default_threshold": float(clf.threshold),
            "split_manifest_path":
                str(split_path.resolve().relative_to(REPO))
                if split_path.exists() else None,
            "split_manifest_sha256": file_sha256(split_path),
            "corpus_manifest_sha256": file_sha256(manifest_path),
        },
        "summary": {
            "decisions": dict(decisions),
            "top_signal_counts": dict(top_signal),
            "classifier_above_threshold": clf_above,
            "intercept_errors": n_int_err,
            "classifier_errors": n_clf_err,
        },
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[saved] {out_path}  size={out_path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
