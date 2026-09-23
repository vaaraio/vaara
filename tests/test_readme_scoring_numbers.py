"""Every number in the README's "How it scores" section comes from an artifact.

Until 2026-09-23 no test read that section. A pass over it found the hand
feature count still at the v0.32 figure (236; the shipped bundle has 254), a
BIPIA line whose v9 and v8 comparisons did not reproduce on the committed
entries, and v0.31 PAIR results listed among the v11 figures without a date.
Each figure is now formatted from the artifact and looked up in the README.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = " ".join((ROOT / "README.md").read_text().split())
BENCH = ROOT / "bench"


def _load(name: str) -> dict:
    return json.loads((BENCH / name).read_text())


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def interval(ci) -> str:
    return f"[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]"


def test_headline_test_split_numbers():
    # In v040_shipped_eval.json the keys "v8"/"v9" hold the baseline (v9
    # bundle) and the candidate (v11 bundle); the bundle paths say which.
    d = _load("v040_shipped_eval.json")
    assert d["candidate_bundle_path"].endswith("adversarial_classifier_v11.joblib")
    new = d["test_by_origin"]["inherited"]["v9"]
    old = d["test_by_origin"]["inherited"]["v8"]
    assert f"n={new['n']:,}" in README
    assert f"recall {pct(new['recall'])} {interval(new['recall_ci'])}" in README
    assert f"FPR {pct(new['fpr'])} {interval(new['fpr_ci'])}" in README
    assert f"Recall is up {(new['recall'] - old['recall']) * 100:.1f} points" in README
    assert f"false-positive rate is up {(new['fpr'] - old['fpr']) * 100:.1f}" in README


def test_cross_model_numbers():
    d = _load("v041_crossmodel_qwen25.json")["surfaces"]["ALL"]
    assert f"recall {pct(d['models']['v11']['recall'])} over n={d['n']:,}" in README
    assert f"v9 reads {pct(d['models']['v9']['recall'])}" in README


def test_four_category_confirmation_numbers():
    s = _load("v041_confirmation_full.json")["surfaces"]
    a, c = s["ALL"]["models"], s["credential_exfil"]["models"]
    assert f"{a['v9']['recall']:.1%} to {a['v11']['recall']:.1%}" in README
    assert f"{a['v9']['fpr']:.1%} to {a['v11']['fpr']:.1%}" in README
    assert f"{c['v9']['recall']:.1%} to {c['v11']['recall']:.1%}" in README
    assert f"{c['v9']['fpr']:.1%} to {c['v11']['fpr']:.1%} FPR" in README


def test_bipia_benign_numbers():
    d = _load("v041_bipia_benign_entries.json")
    v11, v9, v8 = (d[k]["pooled"] for k in ("v11", "v9", "v8"))
    assert f"{pct(v11['fpr'])} {interval(v11['fpr_ci'])}" in README
    assert f"none of {v11['n_benign']} calls" in README
    assert v11["fp"] == 0
    assert f"v9 flags {pct(v9['fpr'])}" in README
    assert f"v8 {pct(v8['fpr'])}" in README


def test_split_counts():
    counts = json.loads((ROOT / "tests" / "adversarial" / "v040_split.json").read_text())["counts"]
    total = sum(counts.values())
    assert f"assigns {total:,} of them" in README
    for fold in ("train", "val", "test", "holdout"):
        assert f"{counts[fold]:,} {fold}" in README


def test_classifier_shape_and_threshold():
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("xgboost")
    b = joblib.load(ROOT / "src" / "vaara" / "data" / "adversarial_classifier_v11.joblib")
    names = b["feature_names"]
    embed = sum(n.startswith("embed__") for n in names)
    assert f"{len(names) - embed} hand-features + {embed}-dim" in README
    assert f"threshold {b['default_threshold']:.4f}" in README


def test_pair_results_are_dated():
    assert "measured on v0.31 and not re-run on v11" in README
    assert "0/25" in (BENCH / "vaara-bench-v0.31.md").read_text()
