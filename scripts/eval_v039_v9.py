"""Calibrate + eval v9 across the v0.39 surfaces.

Four surfaces in one pass, v8 + v9 side-by-side:
  1. v035 VAL  -> calibrate v9 threshold to target FPR.
  2. v035 TEST -> in-distribution regression check.
  3. v0.38 Phase 1 (Llama-3.3-70B) -> attacker-LLM OOD regression check.
  4. v0.39 BIPIA holdout (model x attack_class cells) -> recall recovery
     on follows + FPR drop on benign-under-pressure.

With --update-bundle-threshold the chosen T is written back into v9.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from train_adversarial_classifier import (  # noqa: E402
    build_features, build_labels, load_corpus_keyed,
)

V8 = REPO / "src/vaara/data/adversarial_classifier_v8.joblib"
V9 = REPO / "src/vaara/data/adversarial_classifier_v9.joblib"
SP035 = REPO / "tests/adversarial/v035_split.json"
SP039 = REPO / "tests/adversarial/v039_split.json"
V038_FILES = [
    "tests/adversarial/generated/TM-v038-llama33-s43.jsonl",
    "tests/adversarial/generated/PE-v038-llama33-s43.jsonl",
    "tests/adversarial/generated/DE-v038-llama33-s43.jsonl",
]


def wilson(k, n):
    if n == 0:
        return 0.0, 0.0
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return max(0.0, c - m), min(1.0, c + m)


def load_b(p):
    import joblib
    b = joblib.load(p)
    fn = b["feature_names"]
    ue = b.get("uses_embeddings", False) or any(n.startswith("embed__") for n in fn)
    return {"path": p, "model": b["model"], "vocab": b["vocab"], "feat_names": fn,
            "uses_embeddings": ue, "T": float(b.get("default_threshold", 0.5)),
            "version": b.get("version"), "raw": b}


def sc(b, es):
    X, n, _ = build_features(es, vocab=b["vocab"], embeddings=b["uses_embeddings"])
    if n != b["feat_names"]:
        raise ValueError(f"feature schema mismatch on {b['path']}")
    return b["model"].predict_proba(X)[:, 1].astype(np.float32)


def mx(p, y, T):
    pr = (p >= T).astype(np.int32)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    tp = int(((pr == 1) & (y == 1)).sum())
    fp = int(((pr == 1) & (y == 0)).sum())
    rlo, rhi = wilson(tp, pos)
    flo, fhi = wilson(fp, neg)
    return {"n": int(len(y)), "pos": pos, "neg": neg, "tp": tp, "fp": fp,
            "fn": pos - tp, "recall": tp / max(pos, 1), "recall_ci": [rlo, rhi],
            "fpr": fp / max(neg, 1), "fpr_ci": [flo, fhi]}


def cal(p, y, target):
    for T in np.linspace(0.50, 0.999, 500):
        m = mx(p, y, float(T))
        if m["fpr"] <= target:
            return float(T), m
    return 0.999, mx(p, y, 0.999)


def fold(name, split):
    a = json.loads(split.read_text())["assignments"]
    return [e for k, e in load_corpus_keyed() if a.get(k) == name]


def v38():
    out = []
    for p in V038_FILES:
        for line in (REPO / p).read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def report(name, m_v8, m_v9):
    print(f"\n[{name}] n={m_v8['n']} pos={m_v8['pos']}")
    print(f"  v8  recall={m_v8['recall']:.1%} [{m_v8['recall_ci'][0]:.1%},{m_v8['recall_ci'][1]:.1%}] FPR={m_v8['fpr']:.1%}")
    print(f"  v9  recall={m_v9['recall']:.1%} [{m_v9['recall_ci'][0]:.1%},{m_v9['recall_ci'][1]:.1%}] FPR={m_v9['fpr']:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--json-out", default="bench/v039_v9_eval.json")
    ap.add_argument("--update-bundle-threshold", action="store_true")
    args = ap.parse_args()

    b8, b9 = load_b(V8), load_b(V9)
    print(f"[v8] {b8['version']} T={b8['T']:.4f}\n[v9] {b9['version']} T={b9['T']:.4f}")

    val = fold("val", SP035)
    yv = np.asarray(build_labels(val)[0], dtype=np.int32)
    p8v, p9v = sc(b8, val), sc(b9, val)
    T8 = b8["T"]
    T9, val_v9 = cal(p9v, yv, args.target_fpr)
    val_v8 = mx(p8v, yv, T8)
    print(f"\n[val v035] cal T9={T9:.4f} target FPR<={args.target_fpr}")
    report("val v035", val_v8, val_v9)

    test = fold("test", SP035)
    yt = np.asarray(build_labels(test)[0], dtype=np.int32)
    test_v8 = mx(sc(b8, test), yt, T8)
    test_v9 = mx(sc(b9, test), yt, T9)
    report("test v035", test_v8, test_v9)

    v38e = v38()
    y38 = np.asarray(build_labels(v38e)[0], dtype=np.int32)
    v38_v8 = mx(sc(b8, v38e), y38, T8)
    v38_v9 = mx(sc(b9, v38e), y38, T9)
    report("v0.38 Phase 1", v38_v8, v38_v9)

    hold = fold("holdout", SP039)
    yh = np.asarray(build_labels(hold)[0], dtype=np.int32)
    p8h, p9h = sc(b8, hold), sc(b9, hold)
    h_v8 = mx(p8h, yh, T8)
    h_v9 = mx(p9h, yh, T9)
    report("v0.39 BIPIA holdout", h_v8, h_v9)

    bm: dict[str, dict] = {}
    for e, py8, py9 in zip(hold, p8h, p9h):
        meta = e.get("_v039_bipia")
        if not meta:
            continue
        m = meta.get("model", "?")
        rf = bool(meta.get("reflects_injection"))
        b = bm.setdefault(m, {"follow_n": 0, "v8_ftp": 0, "v9_ftp": 0,
                              "benign_n": 0, "v8_bfp": 0, "v9_bfp": 0})
        k_n, k8, k9 = ("follow_n", "v8_ftp", "v9_ftp") if rf else ("benign_n", "v8_bfp", "v9_bfp")
        b[k_n] += 1
        b[k8] += int(py8 >= T8)
        b[k9] += int(py9 >= T9)
    for b in bm.values():
        b["v8_recall"] = b["v8_ftp"] / max(b["follow_n"], 1)
        b["v9_recall"] = b["v9_ftp"] / max(b["follow_n"], 1)
        b["v8_fpr"] = b["v8_bfp"] / max(b["benign_n"], 1)
        b["v9_fpr"] = b["v9_bfp"] / max(b["benign_n"], 1)
    print("\n[v0.39 BIPIA holdout per model]")
    print(f"  {'model':24s} {'fn':>4} {'v8R':>7} {'v9R':>7} {'bn':>4} {'v8F':>7} {'v9F':>7}")
    for m, b in sorted(bm.items()):
        print(f"  {m:24s} {b['follow_n']:>4d} {b['v8_recall']:>6.1%} {b['v9_recall']:>6.1%} "
              f"{b['benign_n']:>4d} {b['v8_fpr']:>6.1%} {b['v9_fpr']:>6.1%}")

    out = {"v8_threshold": T8, "v9_threshold_calibrated": T9,
           "calibration_target_fpr": args.target_fpr,
           "surfaces": {"val_v035": {"v8": val_v8, "v9": val_v9},
                        "test_v035": {"v8": test_v8, "v9": test_v9},
                        "v038_phase1": {"v8": v38_v8, "v9": v38_v9},
                        "v039_bipia_holdout": {"v8": h_v8, "v9": h_v9}},
           "v039_bipia_holdout_per_model": bm}
    Path(args.json_out).write_text(json.dumps(out, indent=2))
    print(f"\n[out] {args.json_out}")

    if args.update_bundle_threshold:
        import joblib
        raw = b9["raw"]
        raw["default_threshold"] = float(T9)
        joblib.dump(raw, V9)
        print(f"[bundle] T={T9:.4f} -> {V9.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
