"""Custom thresholds the config accepts are thresholds the policy accepts.

The macOS app, the plugin README and both hook implementations allowed
escalate == deny. The policy schema requires escalate < deny, so an equal pair
raised inside apply_policy, the hook caught it, and the whole policy was
dropped: an operator who picked `strict` and set equal custom values ran on
the balanced defaults, with the only trace a stderr line from a hook. An
equal pair is now malformed and ignored, and the preset still applies.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from vaara.integrations import claude_code_hooks as engine
from vaara.policy import from_dict
from vaara.policy.modes import get_mode, to_policy_dict

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "plugin_config", ROOT / "plugins/claude-code-vaara-governance/hooks/_config.py")
plugin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plugin)  # type: ignore[union-attr]


@pytest.mark.parametrize("impl", [engine, plugin], ids=["engine", "plugin"])
@pytest.mark.parametrize("pair", [(0.5, 0.5), (0.0, 0.0), (1.0, 1.0), (0.8, 0.3)])
def test_a_pair_the_policy_rejects_is_not_accepted(impl, pair):
    assert impl.custom_thresholds({"thresholds": {"escalate": pair[0], "deny": pair[1]}}) is None


@pytest.mark.parametrize("impl", [engine, plugin], ids=["engine", "plugin"])
@pytest.mark.parametrize("pair", [(0.3, 0.6), (0.0, 1.0), (0.99, 1.0)])
def test_every_accepted_pair_applies_to_a_policy(impl, pair):
    got = impl.custom_thresholds({"thresholds": {"escalate": pair[0], "deny": pair[1]}})
    assert got == pair
    policy = to_policy_dict(get_mode("strict"))
    policy["thresholds"]["default"] = {"escalate": got[0], "deny": got[1]}
    from_dict(policy)  # raises if the schema rejects it


def test_the_app_never_writes_an_equal_pair():
    swift = (ROOT / "clients/macos/Sources/VaaraMenuBar/Model.swift").read_text()
    assert "d = max(e + 0.01, min(deny, 1))" in swift
    assert "min(escalate, 0.99)" in swift
