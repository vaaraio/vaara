"""The `vaara.govern` docstring's claims about fresh-install behaviour hold.

It said to expect `Blocked` from early calls on `tx.transfer`. Since 1.94.0
the default is balanced (0.55 / 0.85) and a first `tx.transfer` allows; it
took a clean PyPI install on 2026-09-23 to notice. The docstring now says a
first call allows at balanced, a burst escalates, and `strict` escalates the
first call. Each of those is checked here.
"""
from __future__ import annotations

import importlib
import json
import re
import sqlite3
from pathlib import Path

import pytest

import vaara
from vaara.pipeline import InterceptionPipeline
from vaara.policy.modes import get_mode
from vaara.scorer.adaptive import AdaptiveScorer

DOC = vaara.govern.__doc__


@pytest.fixture(autouse=True)
def _fresh_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(importlib.import_module("vaara.govern"),
                        "_default_pipeline", None)


def _transfer(pipeline=None):
    @vaara.govern(tool_name="tx.transfer", pipeline=pipeline)
    def transfer(to: str, amount: float) -> str:
        return "sent"
    return transfer


def test_first_call_allows_at_balanced(tmp_path):
    assert "balanced" in DOC
    assert _transfer()("x", 1.0) == "sent"
    db = tmp_path / ".vaara" / "trail" / "audit.db"
    (data,) = sqlite3.connect(db).execute(
        "select data from audit_records where event_type='decision_made'"
    ).fetchone()
    m = re.search(r"risk=([0-9.]+) (\[[0-9.]+, [0-9.]+\])",
                  json.loads(data)["reason"])
    assert m, data
    doc = " ".join(DOC.split())
    assert f"scores {m.group(1)} {m.group(2)}" in doc


def test_a_burst_escalates_at_balanced():
    assert "burst" in DOC
    transfer = _transfer()
    with pytest.raises(vaara.Blocked, match="escalate"):
        for _ in range(30):
            transfer("x", 1.0)


def test_strict_escalates_the_first_call():
    assert "strict" in DOC
    m = get_mode("strict")
    pipe = InterceptionPipeline(
        scorer=AdaptiveScorer(threshold_allow=m.escalate, threshold_deny=m.deny))
    with pytest.raises(vaara.Blocked, match="escalate"):
        _transfer(pipe)("x", 1.0)
