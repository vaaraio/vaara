#!/usr/bin/env python3
"""classifier_vs_heuristic.py — honest held-out test.
Train on 200 original seeds + 50 benign. Test on 1945 unseen Qwen variants.
Force labels from category-origin (not Qwen's `expected` field)."""
import sys, json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')

import numpy as np
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb
import train_adversarial_classifier as T
from vaara import Pipeline

SEEDS_DIR = Path("tests/adversarial")
VARIANTS_DIR = Path("tests/adversarial/generated_by_category")

def load_dir(d, force_malicious=None):
    out = []
    for fp in sorted(d.glob("*.jsonl")):
        cat = fp.stem.lower()
        is_benign_file = cat == "benign_control"
        for line in fp.read_text().splitlines():
            if not line.strip(): continue
            try:
                e = json.loads(line)
                e["category"] = cat
                if not isinstance(e.get("context"), dict):
                    e["context"] = {"original_task": str(e.get("context", ""))}
                if not isinstance(e.get("parameters"), dict):
                    e["parameters"] = {"_raw": str(e.get("parameters", ""))}
                if force_malicious is not None:
                    e["expected"] = "ALLOW" if is_benign_file else "DENY"
                out.append(e)
            except json.JSONDecodeError:
                pass
    return out

def heuristic_preds(entries, pipe):
    p = []
    for e in entries:
        r = pipe.intercept(
            agent_id=e.get("agent_id", "adv"),
            tool_name=e["tool_name"],
            parameters=e.get("parameters", {}),
            context=e.get("context", {}),
        )
        p.append(0 if r.decision == "allow" else 1)
    return np.array(p)

def main():
    import random, re as _re
    rng = random.Random(42)
    all_seeds = load_dir(SEEDS_DIR)
    benign = [e for e in all_seeds if e.get("category") == "benign_control"]
    adv_seeds = [e for e in all_seeds if e.get("category") != "benign_control"]
    adv_variants = load_dir(VARIANTS_DIR, force_malicious=True)
    benign_variants = load_dir(Path("tests/adversarial/benign_generated"))
    for e in benign_variants: e["category"] = "benign_control"; e["expected"] = "ALLOW"

    # By-seed holdout: 50 seed IDs go to test, their variants too; 150 seeds to train, theirs too.
    seed_ids = sorted({e["id"] for e in adv_seeds})
    rng.shuffle(seed_ids)
    test_seed_ids = set(seed_ids[:50])
    def parent_id(vid): 
        m = _re.match(r"^(.*?)-v\d+$", vid)
        return m.group(1) if m else vid
    adv_train = [e for e in adv_seeds if e["id"] not in test_seed_ids]
    adv_test_seeds = [e for e in adv_seeds if e["id"] in test_seed_ids]
    adv_variants_train = [v for v in adv_variants if parent_id(v["id"]) not in test_seed_ids]
    adv_variants_test = [v for v in adv_variants if parent_id(v["id"]) in test_seed_ids]

    rng.shuffle(benign); rng.shuffle(benign_variants)
    benign_test = benign[:10] + benign_variants[:50]
    benign_train = benign[10:] + benign_variants[50:]
    for e in benign_train: e["expected"] = "ALLOW"
    for e in benign_test: e["expected"] = "ALLOW"

    train = adv_train + adv_variants_train + benign_train
    test = adv_test_seeds + adv_variants_test + benign_test
    print(f"[train] {len(train)} (adv_seeds={len(adv_train)}, adv_variants={len(adv_variants_train)}, benign={len(benign_train)}) [test] {len(test)} (adv_seeds={len(adv_test_seeds)}, adv_variants={len(adv_variants_test)}, benign={len(benign_test)})")

    vocab = T.fit_vocabulary(train)
    X_train, _, _ = T.build_features(train, vocab)
    y_train, _ = T.build_labels(train)
    X_test, _, _ = T.build_features(test, vocab)
    y_test, _ = T.build_labels(test)

    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", n_jobs=-1, random_state=42,
    )
    clf.fit(X_train, y_train)

    clf_proba = clf.predict_proba(X_test)[:, 1]
    clf_pred = (clf_proba >= 0.5).astype(int)
    heur_pred = heuristic_preds(test, Pipeline())

    cats = defaultdict(lambda: {"n": 0, "clf": 0, "heur": 0})
    for e, y, cp, hp in zip(test, y_test, clf_pred, heur_pred):
        c = e["category"]
        cats[c]["n"] += 1
        cats[c]["clf"] += int(cp == y)
        cats[c]["heur"] += int(hp == y)

    print(f"\n{'category':<24} {'n':>4} {'heur':>7} {'clf':>7} {'delta':>8}")
    print("-" * 55)
    for c in sorted(cats):
        n = cats[c]["n"]
        h, cl = cats[c]["heur"]/n, cats[c]["clf"]/n
        print(f"{c:<24} {n:>4} {h*100:>6.1f}% {cl*100:>6.1f}% {(cl-h)*100:>+7.1f}")

    # Threshold sweep
    from sklearn.metrics import balanced_accuracy_score as _ba
    print(f"\n{'threshold':<10} {'attack_recall':>14} {'benign_FPR':>11} {'balanced_acc':>13}")
    for th in [0.3, 0.5, 0.7, 0.8, 0.9]:
        pred = (clf_proba >= th).astype(int)
        y_adv = y_test == 1; y_ben = y_test == 0
        recall = (pred[y_adv] == 1).mean()
        fpr = (pred[y_ben] == 1).mean()
        print(f"{th:<10} {recall*100:>13.1f}% {fpr*100:>10.1f}% {_ba(y_test, pred)*100:>12.1f}%")
    ba_c = balanced_accuracy_score(y_test, clf_pred)
    ba_h = balanced_accuracy_score(y_test, heur_pred)
    print(f"\nOVERALL balanced_acc: heur={ba_h*100:.1f}% clf={ba_c*100:.1f}% delta={(ba_c-ba_h)*100:+.1f}")

    # Save bundle for v0.5.0 ship
    import joblib
    feature_names = T.build_features(train[:1], vocab)[1]
    bundle = {
        "version": "1.0",
        "model": clf,
        "vocab": vocab,
        "feature_names": feature_names,
        "default_threshold": 0.8,
        "train_stats": {"n_train": len(train), "n_test": len(test)},
    }
    out = Path("src/vaara/data/adversarial_classifier_v1.joblib")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    print(f"\n[saved] {out} ({out.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
