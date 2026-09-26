# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The dashboard's settings change what they say they change.

The dashboard writes its settings into the hook's config.json. notify_on,
alert_window_minutes, user_level and approval_style were written there and read
by nothing, so choosing a value changed nothing. These tests hold each one to
its help text.
"""
from __future__ import annotations

import time

import pytest

from vaara.integrations import claude_code_hooks as hooks


@pytest.fixture
def sent(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.delenv("VAARA_PLUGIN_NOTIFY", raising=False)
    monkeypatch.setattr(hooks.sys, "platform", "linux")
    monkeypatch.setattr(hooks.shutil, "which", lambda name: "/usr/bin/notify-send")
    monkeypatch.setattr(hooks.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd))
    return calls


def _verdicts(sent: list[list[str]]) -> list[str]:
    return [cmd[2].split(" ", 2)[1] for cmd in sent]


VERDICTS = ["BLOCKED", "DENIED", "SHADOW deny", "ESCALATE", "APPROVAL NEEDED"]


@pytest.mark.parametrize("notify_on, expected", [
    (None, ["BLOCKED", "DENIED", "SHADOW", "ESCALATE", "APPROVAL"]),
    ("all", ["BLOCKED", "DENIED", "SHADOW", "ESCALATE", "APPROVAL"]),
    ("deny", ["BLOCKED", "DENIED", "SHADOW"]),
    ("escalate", ["ESCALATE", "APPROVAL"]),
    ("off", []),
])
def test_notify_on_picks_which_decisions_notify(sent, notify_on, expected):
    cfg = {} if notify_on is None else {"notify_on": notify_on}
    for verdict in VERDICTS:
        hooks.notify(cfg, verdict, "Bash", "detail")
    assert _verdicts(sent) == expected


def test_a_trail_outage_is_not_a_decision_and_still_notifies(sent):
    hooks.notify({"notify_on": "off"}, "TRAIL NOT RECORDING", "audit trail", "3 failed")
    assert _verdicts(sent) == ["TRAIL"]


def test_notifications_off_still_silences_everything(sent):
    hooks.notify({"notifications": False, "notify_on": "all"}, "BLOCKED", "Bash", "d")
    assert sent == []


class _Rec:
    def __init__(self, decision: str, age_s: float):
        self.data = {"decision": decision}
        self.timestamp = time.time() - age_s
        self.tool_name, self.agent_id = "Bash", "a"
        self.event_type = None


class _Trail:
    def __init__(self, records):
        self._records = records


def test_the_summary_counts_interventions_inside_the_alert_window():
    from vaara.dashboard import _summarize

    trail = _Trail([
        _Rec("deny", 3 * 3600), _Rec("escalate", 3 * 3600),   # outside 60 min
        _Rec("deny", 30 * 60), _Rec("escalate", 10 * 60),     # inside 60 min
        _Rec("deny", 60), _Rec("allow", 60),                  # inside 5 min
    ])
    assert _summarize(trail, window_minutes=60)["recent"] == {
        "minutes": 60, "deny": 2, "escalate": 1}
    assert _summarize(trail, window_minutes=5)["recent"] == {
        "minutes": 5, "deny": 1, "escalate": 0}


@pytest.mark.parametrize("stored, minutes", [
    ({"alert_window_minutes": "15"}, 15), ({"alert_window_minutes": 60}, 60),
    ({}, 5), ({"alert_window_minutes": "soon"}, 5), ({"alert_window_minutes": -3}, 5),
])
def test_the_window_comes_from_the_setting(stored, minutes):
    from vaara.dashboard import _alert_window

    assert _alert_window(stored) == minutes


def _bodies(sent: list[list[str]]) -> list[str]:
    return [cmd[3] for cmd in sent]


@pytest.mark.parametrize("user_level, body", [
    ("basic", ""),
    ("professional", "risk 0.91: rm outside the project"),
    (None, "risk 0.91: rm outside the project"),
    ("enterprise", "risk 0.91: rm outside the project (action_id=a-17)"),
])
def test_user_level_sets_how_much_a_notification_explains(sent, user_level, body):
    cfg = {} if user_level is None else {"user_level": user_level}
    hooks.notify(cfg, "BLOCKED", "Bash", "risk 0.91: rm outside the project",
                 action_id="a-17")
    assert _bodies(sent) == [body]
    assert _verdicts(sent) == ["BLOCKED"]


def test_basic_still_explains_a_trail_outage(sent):
    hooks.notify({"user_level": "basic"}, "TRAIL NOT RECORDING", "audit trail", "3 failed")
    assert _bodies(sent) == ["3 failed"]


def test_enterprise_without_an_action_id_adds_nothing(sent):
    hooks.notify({"user_level": "enterprise"}, "BLOCKED", "Bash", "rule rm_rf_root")
    assert _bodies(sent) == ["rule rm_rf_root"]


@pytest.fixture
def no_env(monkeypatch):
    for name in ("VAARA_PLUGIN_APPROVALS_TIMEOUT", "VAARA_HOOK_DEADLINE"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("cfg, seconds", [
    ({"approval_style": "timeout", "approvals_timeout": 20}, 20.0),
    ({"approvals_timeout": 20}, 20.0),
    ({"approval_style": "blocking", "approvals_timeout": 20}, 75.0),
    ({"approval_style": "blocking"}, 75.0),
])
def test_approval_style_decides_how_long_an_escalation_waits(no_env, cfg, seconds):
    assert hooks.approvals_timeout(cfg) == seconds


def test_blocking_waits_to_the_gate_deadline_it_is_given(no_env, monkeypatch):
    monkeypatch.setenv("VAARA_HOOK_DEADLINE", "30")
    assert hooks.approvals_timeout({"approval_style": "blocking"}) == 25.0


def test_enterprise_keeps_the_action_id_when_the_reason_is_long(sent):
    hooks.notify({"user_level": "enterprise"}, "BLOCKED", "Bash", "x" * 400,
                 action_id="a-17")
    assert _bodies(sent)[0].endswith("(action_id=a-17)")
