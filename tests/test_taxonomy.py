"""Tests for the action taxonomy module."""

import pytest

from vaara.taxonomy.actions import (
    BUILTIN_ACTIONS,
    UNKNOWN_ACTION,
    ActionCategory,
    ActionRegistry,
    ActionRequest,
    ActionType,
    BlastRadius,
    Reversibility,
    UrgencyClass,
    create_default_registry,
)


class TestActionType:
    def test_base_risk_score_fully_reversible_self(self):
        at = ActionType("test", ActionCategory.DATA, Reversibility.FULLY,
                        BlastRadius.SELF)
        # (0.1 + 0.0 + 0.0) / 1.6 = 0.0625
        assert 0.05 < at.base_risk_score < 0.10

    def test_base_risk_score_irreversible_global_irrevocable(self):
        at = ActionType("danger", ActionCategory.FINANCIAL,
                        Reversibility.IRREVERSIBLE, BlastRadius.GLOBAL,
                        UrgencyClass.IRREVOCABLE)
        # (0.8 + 0.5 + 0.3) / 1.6 = 1.0
        assert at.base_risk_score == 1.0

    def test_base_risk_score_capped_at_1(self):
        at = ActionType("max", ActionCategory.PHYSICAL,
                        Reversibility.IRREVERSIBLE, BlastRadius.GLOBAL,
                        UrgencyClass.IRREVOCABLE)
        assert at.base_risk_score <= 1.0

    def test_base_risk_score_minimum(self):
        at = ActionType("min", ActionCategory.DATA, Reversibility.FULLY,
                        BlastRadius.SELF, UrgencyClass.DEFERRABLE)
        assert at.base_risk_score >= 0.0

    def test_frozen_dataclass(self):
        at = ActionType("test", ActionCategory.DATA, Reversibility.FULLY,
                        BlastRadius.SELF)
        with pytest.raises(AttributeError):
            at.name = "changed"


class TestActionRegistry:
    def test_register_and_get(self):
        reg = ActionRegistry()
        at = ActionType("test.action", ActionCategory.DATA, Reversibility.FULLY,
                        BlastRadius.SELF)
        reg.register(at)
        assert reg.get("test.action") is at

    def test_map_tool(self):
        reg = ActionRegistry()
        at = ActionType("test.action", ActionCategory.DATA, Reversibility.FULLY,
                        BlastRadius.SELF)
        reg.register(at)
        reg.map_tool("my_tool", "test.action")
        classified = reg.classify("my_tool")
        assert classified is at

    def test_map_tool_unknown_type_raises(self):
        reg = ActionRegistry()
        with pytest.raises(KeyError):
            reg.map_tool("tool", "nonexistent")

    def test_classify_unknown_returns_sentinel(self):
        reg = ActionRegistry()
        result = reg.classify("unknown_tool")
        assert result is UNKNOWN_ACTION

    def test_prefix_matching(self):
        reg = ActionRegistry()
        at = ActionType("defi", ActionCategory.FINANCIAL,
                        Reversibility.IRREVERSIBLE, BlastRadius.SHARED)
        reg.register(at)
        reg.map_tool("defi", "defi")
        assert reg.classify("defi.swap") is at
        assert reg.classify("defi.deposit") is at

    def test_default_registry_has_builtins(self):
        reg = create_default_registry()
        assert len(reg.all_types) == len(BUILTIN_ACTIONS)
        assert reg.get("tx.sign") is not None
        assert reg.get("data.read") is not None
        assert reg.get("gov.vote") is not None


class TestActionRequest:
    def test_to_policy_context(self):
        at = ActionType("tx.sign", ActionCategory.FINANCIAL,
                        Reversibility.IRREVERSIBLE, BlastRadius.SHARED,
                        UrgencyClass.IRREVOCABLE)
        req = ActionRequest(
            agent_id="agent-1",
            tool_name="tx.sign",
            action_type=at,
            parameters={"to": "0xabc"},
            confidence=0.9,
        )
        ctx = req.to_policy_context()
        assert ctx["agent_id"] == "agent-1"
        assert ctx["tool_name"] == "tx.sign"
        assert ctx["action_type"] == "financial"
        assert ctx["reversibility"] == "irreversible"
        assert ctx["blast_radius"] == "shared"
        assert ctx["agent_confidence"] == 0.9
        assert isinstance(ctx["base_risk_score"], float)


class TestBuiltinActions:
    def test_all_have_unique_names(self):
        names = [a.name for a in BUILTIN_ACTIONS]
        assert len(names) == len(set(names))

    def test_all_categories_represented(self):
        categories = {a.category for a in BUILTIN_ACTIONS}
        # All 7 domain categories should be present
        expected = {
            ActionCategory.FINANCIAL, ActionCategory.DATA,
            ActionCategory.COMMUNICATION, ActionCategory.INFRASTRUCTURE,
            ActionCategory.IDENTITY, ActionCategory.GOVERNANCE,
            ActionCategory.PHYSICAL,
        }
        assert categories == expected

    def test_risk_scores_are_valid(self):
        for action in BUILTIN_ACTIONS:
            assert 0.0 <= action.base_risk_score <= 1.0, (
                f"{action.name} has invalid risk score: {action.base_risk_score}"
            )

    def test_financial_actions_are_high_risk(self):
        financial = [a for a in BUILTIN_ACTIONS
                     if a.category == ActionCategory.FINANCIAL]
        for action in financial:
            assert action.base_risk_score > 0.3, (
                f"Financial action {action.name} should be high risk"
            )
