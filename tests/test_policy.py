"""Tests for vaara.policy — JSON-native loader, YAML extra, validation paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaara.policy import (
    SCHEMA_VERSION,
    PolicyError,
    from_dict,
    from_json,
    from_yaml,
)
from vaara.taxonomy.actions import (
    ActionCategory,
    RegulatoryDomain,
    Reversibility,
    UrgencyClass,
)


MINIMAL_POLICY: dict = {
    "version": SCHEMA_VERSION,
    "domains": ["eu_ai_act"],
    "action_classes": {},
    "thresholds": {"default": {"escalate": 0.55, "deny": 0.85}},
}

FULL_POLICY: dict = {
    "version": SCHEMA_VERSION,
    "domains": ["eu_ai_act", "dora"],
    "action_classes": {
        "fs.write_file": {
            "category": "data", "reversibility": "partially_reversible",
            "blast_radius": "local", "urgency": "timely",
            "regulatory": ["aiact:9", "aiact:12"],
        },
        "tx.sign": {
            "category": "financial", "reversibility": "irreversible",
            "blast_radius": "shared", "urgency": "irrevocable",
            "regulatory": ["aiact:14", "dora:10"],
        },
    },
    "thresholds": {
        "default": {"escalate": 0.55, "deny": 0.85},
        "fs.write_file": {"deny": 0.75},
        "tx.sign": {"escalate": 0.40, "deny": 0.65},
    },
    "sequences": {
        "data_exfiltration": {
            "pattern": ["read_data", "export_data", "delete_data"],
            "risk_boost": 0.4, "window_seconds": 300,
            "regulatory": ["aiact:9(2)(a)"],
        },
    },
    "escalation": {
        "routes": [
            {"if": ["aiact:14"], "operator_group": "ai_oversight_team"},
            {"if": ["dora:10"], "operator_group": "ict_risk_team"},
            {"default": "on_call"},
        ],
    },
}


# ── Loading ──────────────────────────────────────────────────────────────────

def test_minimal_policy_loads() -> None:
    p = from_dict(MINIMAL_POLICY)
    assert p.version == SCHEMA_VERSION
    assert p.domains == (RegulatoryDomain.EU_AI_ACT,)
    assert p.thresholds_default.escalate == 0.55


def test_full_policy_loads() -> None:
    p = from_dict(FULL_POLICY)
    assert set(p.domains) == {RegulatoryDomain.EU_AI_ACT, RegulatoryDomain.DORA}
    assert p.action_classes["fs.write_file"].category == ActionCategory.DATA
    assert p.action_classes["fs.write_file"].reversibility == Reversibility.PARTIALLY
    assert p.action_classes["tx.sign"].urgency == UrgencyClass.IRREVOCABLE
    assert p.action_classes["fs.write_file"].regulatory == ("aiact:9", "aiact:12")
    assert p.sequences[0].pattern == ("read_data", "export_data", "delete_data")
    assert p.sequences[0].risk_boost == 0.4


# ── Threshold resolution ─────────────────────────────────────────────────────

def test_threshold_partial_override_inherits_default() -> None:
    p = from_dict(FULL_POLICY)
    t = p.threshold_for("fs.write_file")
    assert t.escalate == 0.55  # inherited
    assert t.deny == 0.75      # overridden


def test_threshold_full_override() -> None:
    p = from_dict(FULL_POLICY)
    t = p.threshold_for("tx.sign")
    assert (t.escalate, t.deny) == (0.40, 0.65)


def test_threshold_unknown_action_class_returns_default() -> None:
    p = from_dict(FULL_POLICY)
    t = p.threshold_for("totally.unknown")
    assert (t.escalate, t.deny) == (0.55, 0.85)


# ── Escalation routing ───────────────────────────────────────────────────────

def test_escalation_route_matches_article() -> None:
    p = from_dict(FULL_POLICY)
    assert p.escalation_route_for({"aiact:14"}) == "ai_oversight_team"
    assert p.escalation_route_for({"dora:10"}) == "ict_risk_team"


def test_escalation_route_falls_back_to_default() -> None:
    p = from_dict(FULL_POLICY)
    assert p.escalation_route_for({"aiact:99"}) == "on_call"


def test_escalation_route_no_routes_returns_on_call() -> None:
    p = from_dict({**MINIMAL_POLICY, "escalation": {"routes": []}})
    assert p.escalation_route_for({"aiact:14"}) == "on_call"


# ── Validation errors ────────────────────────────────────────────────────────

def test_invalid_version_rejected() -> None:
    with pytest.raises(PolicyError, match="not supported"):
        from_dict({**MINIMAL_POLICY, "version": "9.9"})


def test_invalid_category_gives_field_path() -> None:
    bad = {**MINIMAL_POLICY, "action_classes": {"x": {
        "category": "bogus", "reversibility": "fully_reversible",
        "blast_radius": "self", "urgency": "deferrable",
    }}}
    with pytest.raises(PolicyError, match="action_classes.x.category"):
        from_dict(bad)


def test_threshold_escalate_must_be_below_deny() -> None:
    bad = {**MINIMAL_POLICY, "thresholds": {"default": {"escalate": 0.9, "deny": 0.5}}}
    with pytest.raises(PolicyError, match="must be < deny"):
        from_dict(bad)


def test_threshold_out_of_unit_interval_rejected() -> None:
    bad = {**MINIMAL_POLICY, "thresholds": {"default": {"escalate": 1.5, "deny": 2.0}}}
    with pytest.raises(PolicyError, match=r"must be in \[0,1\]"):
        from_dict(bad)


def test_empty_sequence_pattern_rejected() -> None:
    bad = {**MINIMAL_POLICY, "sequences": {
        "empty": {"pattern": [], "risk_boost": 0.1, "window_seconds": 60},
    }}
    with pytest.raises(PolicyError, match="pattern must be non-empty"):
        from_dict(bad)


# ── Strict-shape validation (CodeRabbit #2 / #3) ─────────────────────────────

def test_action_classes_must_be_mapping_not_list() -> None:
    bad = {**MINIMAL_POLICY, "action_classes": []}
    with pytest.raises(PolicyError, match=r"action_classes: must be a mapping"):
        from_dict(bad)


def test_thresholds_must_be_mapping_not_list() -> None:
    bad = {**MINIMAL_POLICY, "thresholds": []}
    with pytest.raises(PolicyError, match=r"thresholds: must be a mapping"):
        from_dict(bad)


def test_escalation_must_be_mapping_not_list() -> None:
    bad = {**MINIMAL_POLICY, "escalation": []}
    with pytest.raises(PolicyError, match=r"escalation: must be a mapping"):
        from_dict(bad)


def test_sequence_pattern_string_rejected_not_silently_split() -> None:
    """`pattern: "abc"` must raise, not silently become ('a','b','c')."""
    bad = {**MINIMAL_POLICY, "sequences": {
        "x": {"pattern": "abc", "risk_boost": 0.1, "window_seconds": 60},
    }}
    with pytest.raises(PolicyError, match=r"sequences.x.pattern: must be a list"):
        from_dict(bad)


def test_per_action_threshold_override_validated_at_load_time() -> None:
    """Override `{escalate: 0.9, deny: 0.2}` must fail at load, not on query."""
    bad = {**MINIMAL_POLICY, "thresholds": {
        "default": {"escalate": 0.55, "deny": 0.85},
        "fs.write_file": {"escalate": 0.9, "deny": 0.2},
    }}
    with pytest.raises(PolicyError, match=r"thresholds.fs.write_file"):
        from_dict(bad)


def test_per_action_partial_override_validated_against_default() -> None:
    """Override `{deny: 0.1}` is invalid because default escalate=0.55 > 0.1."""
    bad = {**MINIMAL_POLICY, "thresholds": {
        "default": {"escalate": 0.55, "deny": 0.85},
        "fs.write_file": {"deny": 0.1},
    }}
    with pytest.raises(PolicyError, match=r"thresholds.fs.write_file"):
        from_dict(bad)


# ── Policy is a frozen value object (CodeRabbit #6) ──────────────────────────

def test_policy_thresholds_overrides_are_immutable() -> None:
    p = from_dict(FULL_POLICY)
    with pytest.raises(TypeError):
        p.thresholds_overrides["fs.write_file"]["escalate"] = 0.99  # type: ignore[index]


def test_policy_action_classes_are_immutable() -> None:
    p = from_dict(FULL_POLICY)
    with pytest.raises(TypeError):
        del p.action_classes["fs.write_file"]  # type: ignore[attr-defined]


# ── JSON loading ─────────────────────────────────────────────────────────────

def test_from_json_string() -> None:
    p = from_json(json.dumps(MINIMAL_POLICY))
    assert p.version == SCHEMA_VERSION


def test_from_json_path(tmp_path: Path) -> None:
    f = tmp_path / "p.json"
    f.write_text(json.dumps(FULL_POLICY))
    p = from_json(f)
    assert "fs.write_file" in p.action_classes


def test_from_json_malformed_object_raises_policy_error() -> None:
    with pytest.raises(PolicyError, match="invalid JSON"):
        from_json("{not_valid_json}")


def test_from_json_unreadable_input_raises_policy_error() -> None:
    with pytest.raises(PolicyError, match="neither a JSON object nor a readable file"):
        from_json("not json at all")


# ── YAML loading (skipif pyyaml not installed) ──────────────────────────────

yaml = pytest.importorskip("yaml")


def test_from_yaml_string() -> None:
    p = from_yaml(yaml.safe_dump(FULL_POLICY))
    assert "fs.write_file" in p.action_classes


def test_from_yaml_path(tmp_path: Path) -> None:
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump(MINIMAL_POLICY))
    p = from_yaml(f)
    assert p.version == SCHEMA_VERSION
