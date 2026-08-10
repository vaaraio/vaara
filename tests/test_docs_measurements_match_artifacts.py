# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every measured number COMPLIANCE.md quotes has to be in the artefact.

The "Current limits" section is the honest-numbers section: attack
recall, benign FPR, layer ablation, and the PAIR adaptive-attacker run.
It names the JSON files it draws from, so a reader can check it, and
that is exactly what makes a wrong digit expensive.

Two were wrong. The hand-curated benign FPR was quoted as 70.0% when the
artefact says 69.57%, which also inflated the stated distribution-shift
gap to 18pp. And the claim that "most benign escalations come from the
heuristic ESCALATE branch" held only for the LLM-generated source: on
hand-curated benigns the heuristic contributes 26.1pp of a 69.6pp
full-stack FPR, so classifier upgrades are the larger half.

These tests fail the build when a re-run moves the artefacts and the doc
does not move with them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "COMPLIANCE.md"
DIST = ROOT / "tests" / "adversarial" / "distribution_shift_v0_5_3.json"
ABLATION = ROOT / "tests" / "adversarial" / "stack_ablation_v0_5_3.json"
PAIR = ROOT / "tests" / "adversarial" / "pair_v0_5_3.json"


def _text() -> str:
    return DOC.read_text(encoding="utf-8")


def _dist(key: str) -> dict:
    buckets = json.loads(DIST.read_text())["buckets"]
    return next(b for b in buckets if b["key"] == key)


def _ablation(config: str, key: str) -> dict:
    rows = json.loads(ABLATION.read_text())["rows"]
    return next(r for r in rows if r["config"] == config and r["key"] == key)


def _table_row(label: str) -> tuple[float, float]:
    """The (recall, FPR) percentages COMPLIANCE.md prints for one source."""
    match = re.search(
        rf"\|\s*{re.escape(label)}[^|]*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|", _text()
    )
    assert match, f"no distribution-shift table row for {label!r}"
    return float(match.group(1)), float(match.group(2))


@pytest.mark.parametrize("label,source,entries", [
    ("Hand-curated", "hand_curated", 250),
    ("LLM-generated", "llm_generated", 5705),
])
def test_distribution_shift_table_matches_the_artifact(label, source, entries):
    documented_recall, documented_fpr = _table_row(label)
    attack = _dist(f"{source}/attack")
    benign = _dist(f"{source}/benign")

    assert round(attack["value"] * 100, 1) == documented_recall
    assert round(benign["value"] * 100, 1) == documented_fpr
    assert attack["n"] + benign["n"] == entries, (
        "the entry count in the table is the corpus the artefact measured"
    )


def test_the_stated_gaps_follow_from_the_stated_numbers():
    text = _text()
    hand_recall, hand_fpr = _table_row("Hand-curated")
    llm_recall, llm_fpr = _table_row("LLM-generated")

    recall_gap = re.search(r"The ([\d.]+)pp recall gap", text)
    fpr_gap = re.search(r"The ([\d.]+)pp benign-FPR gap", text)
    assert recall_gap and fpr_gap, "COMPLIANCE.md no longer states the gaps"
    assert float(recall_gap.group(1)) == round(hand_recall - llm_recall, 1)
    assert float(fpr_gap.group(1)) == round(llm_fpr - hand_fpr, 1)


def test_layer_ablation_recalls_match_the_artifact():
    text = _text()
    for config in ("heuristic_only", "classifier_only"):
        match = re.search(rf"`{config}` recall\s*\n?\s*is (\d+)% / (\d+)%", text)
        assert match, f"COMPLIANCE.md no longer states {config} recall"
        hand, llm = (int(g) for g in match.groups())
        assert hand == round(_ablation(config, "hand_curated/attack")["value"] * 100)
        assert llm == round(_ablation(config, "llm_generated/attack")["value"] * 100)


def test_the_fpr_attribution_matches_the_ablation():
    """Which layer dominates the benign FPR is a per-source claim."""
    text = _text()
    match = re.search(
        r"heuristic `ESCALATE`\s*\n?\s*branch does \(([\d.]+)pp of a ([\d.]+)pp "
        r"full-stack FPR\), on hand-curated\s*\n?\s*benigns it is the other way "
        r"round \(heuristic ([\d.]+)pp of ([\d.]+)pp",
        text,
    )
    assert match, "COMPLIANCE.md no longer attributes the benign FPR per source"
    llm_heuristic, llm_full, hand_heuristic, hand_full = (
        float(g) for g in match.groups()
    )

    real = {
        "llm_heuristic": _ablation("heuristic_only", "llm_generated/benign")["value"],
        "llm_full": _ablation("full_stack", "llm_generated/benign")["value"],
        "hand_heuristic": _ablation("heuristic_only", "hand_curated/benign")["value"],
        "hand_full": _ablation("full_stack", "hand_curated/benign")["value"],
    }
    assert llm_heuristic == round(real["llm_heuristic"] * 100, 1)
    assert llm_full == round(real["llm_full"] * 100, 1)
    assert hand_heuristic == round(real["hand_heuristic"] * 100, 1)
    assert hand_full == round(real["hand_full"] * 100, 1)

    # The direction claimed in prose has to be the direction in the data.
    assert real["llm_heuristic"] > real["llm_full"] - real["llm_heuristic"]
    assert real["hand_heuristic"] < real["hand_full"] - real["hand_heuristic"]


def test_the_two_artifacts_diverge_only_where_the_doc_says_they_do():
    """The doc explains the LLM-generated divergence as ERROR handling."""
    text = _text()
    match = re.search(
        r"against ([\d.]+)% / ([\d.]+)% in the ablation file", text
    )
    assert match, "COMPLIANCE.md no longer reconciles the two artefacts"
    recall, fpr = (float(g) for g in match.groups())
    assert recall == round(_ablation("full_stack", "llm_generated/attack")["value"] * 100, 1)
    assert fpr == round(_ablation("full_stack", "llm_generated/benign")["value"] * 100, 1)

    errors = re.search(r"\((\d+) attack, (\d+) benign\)", text)
    assert errors, "COMPLIANCE.md no longer states the ERROR counts"
    assert int(errors.group(1)) == _dist("llm_generated/attack")["error"]
    assert int(errors.group(2)) == _dist("llm_generated/benign")["error"]


def test_pair_calibration_numbers_match_the_artifact():
    text = _text()
    pair = json.loads(PAIR.read_text())
    verdicts = [h["vaara"] for r in pair["results"] for h in r["history"]]

    asr = re.search(r"\*\*ASR: ([\d.]+)% \((\d+)/(\d+)\)\*\*", text)
    assert asr, "COMPLIANCE.md no longer states the PAIR ASR"
    assert float(asr.group(1)) == pair["asr"] * 100
    assert int(asr.group(2)) == pair["n_successes"]
    assert int(asr.group(3)) == pair["n_seeds"]

    counts = re.search(
        r"Across (\d+) candidate prompts, Vaara\s*\n?\s*escalated (\d+) and "
        r"allowed (\d+)", text,
    )
    assert counts, "COMPLIANCE.md no longer states the PAIR outcome counts"
    total, escalated, allowed = (int(g) for g in counts.groups())
    assert total == len(verdicts)
    assert escalated == verdicts.count("ESCALATE")
    assert allowed == verdicts.count("ALLOW")

    iters = re.search(r"Max iterations per seed: (\d+)", text)
    assert iters and int(iters.group(1)) == pair["max_iters"]
    assert pair["attacker_model"] in text
    assert pair["judge_model"] in text
