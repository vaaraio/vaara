"""PolicyController hot-reload tests.

Cover the contract the v0.13.0 operator surface promises:

- A reload swaps the live policy and the scorer thresholds atomically.
- A malformed reload leaves the previous policy in place.
- The version counter monotonically increases on every accepted reload.
- Concurrent intercepts during a reload never observe a torn (allow, deny)
  pair.
"""

from __future__ import annotations

import threading
import time

import pytest

from vaara.policy.controller import PolicyController
from vaara.policy.loader import from_dict
from vaara.policy.schema import PolicyError
from vaara.scorer.adaptive import AdaptiveScorer


def _policy_dict(*, escalate, deny):
    return {
        "version": "0.1",
        "domains": ["eu_ai_act"],
        "action_classes": {
            "tx.transfer": {
                "category": "financial",
                "reversibility": "irreversible",
                "blast_radius": "local",
                "urgency": "timely",
                "regulatory": ["article_14"],
            }
        },
        "thresholds": {"default": {"escalate": escalate, "deny": deny}},
        "sequences": {
            "data_exfil": {
                "pattern": ["data.read", "data.export"],
                "risk_boost": 0.2,
                "window_seconds": 60,
                "regulatory": [],
            }
        },
        "escalation": {
            "routes": [{"operator_group": "on_call", "if": []}]
        },
    }


@pytest.fixture
def base_policy():
    return from_dict(_policy_dict(escalate=0.55, deny=0.85))


def _mutate(_p, *, escalate, deny):
    """Build a fresh policy dict with the supplied thresholds.

    Tests only exercise the threshold + sequence + escalation surfaces,
    so the rest of the document can be a fixed minimum that does not
    depend on YAML loading.
    """
    return _policy_dict(escalate=escalate, deny=deny)


def test_listener_applies_on_register(base_policy):
    scorer = AdaptiveScorer(threshold_allow=0.4, threshold_deny=0.7)
    controller = PolicyController(base_policy)
    controller.add_listener(scorer.apply_policy)

    assert scorer._threshold_allow == base_policy.thresholds_default.escalate
    assert scorer._threshold_deny == base_policy.thresholds_default.deny
    assert controller.version == 1


def test_reload_translates_sequences_for_matching(base_policy):
    """A reload must rebind sequence patterns in the scorer's runtime form.

    The policy schema carries `.pattern` / `.window_seconds`; the matcher
    reads `.actions` / `.window_size`. Storing the policy form verbatim
    raised AttributeError on the first sequence match after a hot reload.
    """
    scorer = AdaptiveScorer(threshold_allow=0.4, threshold_deny=0.7)
    controller = PolicyController(base_policy)
    controller.add_listener(scorer.apply_policy)

    assert scorer._sequences
    pat = scorer._sequences[0]
    assert hasattr(pat, "actions") and hasattr(pat, "window_size")
    assert pat.actions == ("data.read", "data.export")

    # the data_exfil sequence must actually fire after the reload
    scorer.evaluate(
        {"tool_name": "data.read", "agent_id": "a", "base_risk_score": 0.1}
    )
    result = scorer.evaluate(
        {"tool_name": "data.export", "agent_id": "a", "base_risk_score": 0.1}
    )
    assert result.get("raw_result", {}).get("sequence_risk", 0) > 0


def test_reload_swaps_thresholds(base_policy):
    scorer = AdaptiveScorer()
    controller = PolicyController(base_policy)
    controller.add_listener(scorer.apply_policy)

    result = controller.reload(_mutate(base_policy, escalate=0.30, deny=0.95))

    assert result.version == 2
    assert result.thresholds_default_escalate == 0.30
    assert result.thresholds_default_deny == 0.95
    assert scorer._threshold_allow == 0.30
    assert scorer._threshold_deny == 0.95
    assert controller.policy.thresholds_default.escalate == 0.30


def test_invalid_reload_keeps_previous_policy(base_policy):
    scorer = AdaptiveScorer()
    controller = PolicyController(base_policy)
    controller.add_listener(scorer.apply_policy)

    before_allow = scorer._threshold_allow
    before_deny = scorer._threshold_deny
    before_version = controller.version

    broken = _mutate(base_policy, escalate=0.9, deny=0.4)  # inverted
    with pytest.raises(PolicyError):
        controller.reload(broken)

    assert scorer._threshold_allow == before_allow
    assert scorer._threshold_deny == before_deny
    assert controller.version == before_version


def test_reload_clears_sequence_match_cache(base_policy):
    scorer = AdaptiveScorer()
    controller = PolicyController(base_policy)
    controller.add_listener(scorer.apply_policy)

    scorer._seq_match_state[("agent-1", "stale_pattern")] = True

    controller.reload(_mutate(base_policy, escalate=0.6, deny=0.8))
    assert scorer._seq_match_state == {}


def test_concurrent_evaluate_during_reload(base_policy):
    """No torn-pair: every observed (allow, deny) must be one of the
    two known threshold pairs."""
    scorer = AdaptiveScorer()
    controller = PolicyController(base_policy)
    controller.add_listener(scorer.apply_policy)

    other = _mutate(base_policy, escalate=0.25, deny=0.95)
    pairs_seen: set[tuple[float, float]] = set()
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            with scorer._lock:
                pairs_seen.add(
                    (scorer._threshold_allow, scorer._threshold_deny)
                )

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()

    for _ in range(50):
        controller.reload(other)
        controller.reload(_mutate(base_policy, escalate=0.55, deny=0.85))
        time.sleep(0.001)

    stop.set()
    for t in threads:
        t.join()

    allowed_pairs = {(0.25, 0.95), (0.55, 0.85)}
    assert pairs_seen.issubset(allowed_pairs)


def test_dict_source_bypasses_parser(base_policy):
    controller = PolicyController(base_policy)
    src = _mutate(base_policy, escalate=0.4, deny=0.8)
    result = controller.reload(src)
    assert result.version == 2


def test_apply_policy_rejects_non_policy():
    scorer = AdaptiveScorer()
    with pytest.raises(TypeError):
        scorer.apply_policy({"not": "a policy"})  # type: ignore[arg-type]
