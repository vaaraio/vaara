"""Tests for the Claude Code plugin's config helpers (hooks/_config.py).

The hooks live outside the package, so they are imported by path. This
is the first test coverage for them; keep it dependency-free.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_HOOKS = Path(__file__).parent.parent / "plugins" / "claude-code-vaara-governance" / "hooks"


@pytest.fixture()
def config_mod(monkeypatch):
    monkeypatch.syspath_prepend(str(_HOOKS))
    spec = importlib.util.spec_from_file_location("_config", _HOOKS / "_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # A stray env var must not leak into assertions below.
    for var in ("VAARA_PLUGIN_PROTECTION", "VAARA_PLUGIN_SHADOW"):
        monkeypatch.delenv(var, raising=False)
    yield mod
    sys.modules.pop("_config", None)


def test_custom_thresholds_valid(config_mod):
    cfg = {"thresholds": {"escalate": 0.5, "deny": 0.8}}
    assert config_mod.custom_thresholds(cfg) == (0.5, 0.8)


def test_custom_thresholds_int_values_accepted(config_mod):
    cfg = {"thresholds": {"escalate": 0, "deny": 1}}
    assert config_mod.custom_thresholds(cfg) == (0.0, 1.0)


def test_custom_thresholds_absent(config_mod):
    assert config_mod.custom_thresholds({}) is None


@pytest.mark.parametrize("bad", [
    {"escalate": 0.9, "deny": 0.5},          # escalate above deny
    {"escalate": -0.1, "deny": 0.5},         # below range
    {"escalate": 0.5, "deny": 1.5},          # above range
    {"escalate": "0.5", "deny": 0.8},        # wrong type
    {"escalate": True, "deny": 0.8},         # bool is not a threshold
    {"escalate": 0.5},                       # missing key
    "0.5,0.8",                               # not a dict
    None,
])
def test_custom_thresholds_rejects_malformed(config_mod, bad):
    assert config_mod.custom_thresholds({"thresholds": bad}) is None


def test_preset_still_reads_protection_key(config_mod):
    assert config_mod.protection_preset({"protection": "strict"}) == "strict"
    assert config_mod.protection_preset({}) is None


def test_custom_thresholds_compose_into_policy(config_mod):
    """The pre_tool_use composition: custom overrides the preset default."""
    pytest.importorskip("vaara")
    from vaara.policy import from_dict
    from vaara.policy.modes import get_mode, to_policy_dict

    custom = config_mod.custom_thresholds(
        {"thresholds": {"escalate": 0.33, "deny": 0.66}})
    policy = to_policy_dict(get_mode("balanced"))
    escalate, deny = custom
    policy["thresholds"]["default"] = {"escalate": escalate, "deny": deny}
    parsed = from_dict(policy)  # must validate
    assert parsed is not None
