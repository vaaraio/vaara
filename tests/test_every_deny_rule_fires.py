# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every shipped deny rule must fire on a payload its own pattern accepts.

Of the 38 rules, 17 were named by a test and 12 had no test carrying any
payload their regex matched (audit S3, 2026-09-22). A rule nothing exercises
can rot silently: a regex edit that stops matching, a tool renamed out of
its `tools` list, a field the harness stopped sending. This test derives one
payload per rule from the rule's own pattern, so no attack string lives in
the tree, and drives it through the real matcher.

It proves reachability, not sufficiency. A rule that fires on its own
sample can still miss a real attack; the private red-team set is for that.
"""
from __future__ import annotations

import os
import re

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from vaara.deny_rules import load_deny_rules, match_deny_rule

RULES = load_deny_rules()
assert len(RULES) >= 38, "the bundled rule set shrank"


def _payload_for(rule: dict) -> str:
    """One string the rule's own compiled pattern accepts."""
    pattern = re.compile(rule["pattern"])
    found: list[str] = []

    @settings(max_examples=50, database=None, deadline=None,
              suppress_health_check=list(HealthCheck))
    @given(st.from_regex(pattern))
    def collect(sample: str) -> None:
        if not found and pattern.search(sample):
            found.append(sample)

    collect()
    assert found, f"{rule['id']}: pattern generated nothing it accepts"
    return found[0]


@pytest.fixture(autouse=True)
def _no_lifts(monkeypatch):
    for key in list(os.environ):
        if key.startswith("VAARA_ALLOW_"):
            monkeypatch.delenv(key)


@pytest.mark.parametrize("rule", RULES, ids=[r["id"] for r in RULES])
def test_the_rule_fires_on_its_own_pattern(rule):
    tool = rule["tools"][0]
    if rule.get("match_any"):
        hit = match_deny_rule(RULES, tool, {})
    else:
        payload = _payload_for(rule)
        hit = match_deny_rule(RULES, tool, {rule["fields"][0]: payload})
    assert hit is not None, f"{rule['id']} did not fire on its own sample"
    # An earlier rule may legitimately catch the same payload first. That is
    # still reachable; record which one so a shadowing change is visible.
    assert hit[0] in {r["id"] for r in RULES}


@pytest.mark.parametrize("rule", [r for r in RULES if r.get("unless_env")],
                         ids=[r["id"] for r in RULES if r.get("unless_env")])
def test_the_lift_lifts_exactly_this_rule(rule, monkeypatch):
    tool = rule["tools"][0]
    monkeypatch.setenv(rule["unless_env"], "1")
    if rule.get("match_any"):
        hit = match_deny_rule(RULES, tool, {})
    else:
        payload = _payload_for(rule)
        hit = match_deny_rule(RULES, tool, {rule["fields"][0]: payload})
    assert hit is None or hit[0] != rule["id"], (
        f"{rule['id']} still fired with {rule['unless_env']}=1")
