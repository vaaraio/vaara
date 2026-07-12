"""Per-action-class policy thresholds are enforced, not just parsed.

A policy can tighten one action class ("tx.sign: {escalate: 0.40}") without
restating the default. Before, threshold_for existed but nothing called it,
so per-class overrides were silently ignored. The scorer now resolves the
override by the call's tool name at decision time.
"""

from __future__ import annotations

from vaara.policy import from_dict
from vaara.scorer.adaptive import AdaptiveScorer

_POLICY = {
    "version": "0.1",
    "domains": ["eu_ai_act"],
    "action_classes": {
        "tx.sign": {
            "category": "financial",
            "reversibility": "irreversible",
            "blast_radius": "shared",
            "urgency": "irrevocable",
        },
    },
    "thresholds": {
        "default": {"escalate": 0.55, "deny": 0.85},
        "tx.sign": {"escalate": 0.10, "deny": 0.20},
    },
}


def _ctx(tool_name, tenant_id=""):
    return {
        "tool_name": tool_name,
        "agent_id": "agent-x",
        "base_risk_score": 0.30,
        "tenant_id": tenant_id,
        "parameters": {},
    }


def test_per_class_override_tightens_named_class():
    scorer = AdaptiveScorer()
    scorer.apply_policy(from_dict(_POLICY))
    decision = scorer.evaluate(_ctx("tx.sign"))
    # deny threshold is 0.20 for tx.sign; a mid-risk call must not slip to allow.
    assert decision["threshold_deny"] == 0.20
    assert decision["threshold_allow"] == 0.10


def test_unnamed_class_keeps_default_thresholds():
    scorer = AdaptiveScorer()
    scorer.apply_policy(from_dict(_POLICY))
    decision = scorer.evaluate(_ctx("read_file"))
    assert decision["threshold_deny"] == 0.85
    assert decision["threshold_allow"] == 0.55


def test_partial_override_inherits_default_deny():
    policy = dict(_POLICY)
    policy["thresholds"] = {
        "default": {"escalate": 0.55, "deny": 0.85},
        "tx.sign": {"escalate": 0.10},  # deny omitted
    }
    scorer = AdaptiveScorer()
    scorer.apply_policy(from_dict(policy))
    decision = scorer.evaluate(_ctx("tx.sign"))
    assert decision["threshold_allow"] == 0.10
    assert decision["threshold_deny"] == 0.85  # inherited


def test_no_policy_uses_constructor_defaults():
    scorer = AdaptiveScorer(threshold_allow=0.4, threshold_deny=0.7)
    decision = scorer.evaluate(_ctx("tx.sign"))
    assert decision["threshold_allow"] == 0.4
    assert decision["threshold_deny"] == 0.7
