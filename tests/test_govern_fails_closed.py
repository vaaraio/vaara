"""`@vaara.govern` fails closed on the real engine, as the I-D says it does.

draft-sirkkavaara-vaara-receipt s6.2: the decorator "classifies the call,
decides allow, deny, or escalate, records the decision, and fails closed,
raising before the governed body runs on any non-allow verdict". The contract
tests in test_govern.py drive a fake pipeline. These drive the real one, with
real scorers and the real SQLite trail, and add the case the draft implies but
never names: an engine that cannot decide or cannot record must also raise
before the body runs. If any of these ever lets a body through, the draft is
false and this file says where.
"""
from __future__ import annotations

import asyncio
import importlib
import sqlite3
from pathlib import Path

import pytest

import vaara
from vaara.audit.trail import EventType
from vaara.pipeline import InterceptionPipeline
from vaara.scorer.adaptive import AdaptiveScorer

GOVERN = importlib.import_module("vaara.govern")


@pytest.fixture(autouse=True)
def _fresh_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(GOVERN, "_default_pipeline", None)
    monkeypatch.setattr(GOVERN, "_shadow_singleton", None)


def _pipe(allow: float, deny: float) -> InterceptionPipeline:
    return InterceptionPipeline(
        scorer=AdaptiveScorer(threshold_allow=allow, threshold_deny=deny))


def _decisions(db: Path) -> list[tuple[str, str]]:
    return list(sqlite3.connect(db).execute(
        "select event_type, tool_name from audit_records "
        "where event_type='decision_made' order by rowid"))


class _Body:
    def __init__(self) -> None:
        self.ran = 0

    def sync(self, pipeline=None):
        @vaara.govern(tool_name="tx.transfer", pipeline=pipeline)
        def transfer(to: str, amount: float) -> str:
            self.ran += 1
            return "sent"
        return transfer

    def coro(self, pipeline=None):
        @vaara.govern(tool_name="tx.transfer", pipeline=pipeline)
        async def transfer(to: str, amount: float) -> str:
            self.ran += 1
            return "sent"
        return transfer


def test_allow_runs_the_body_and_records_the_decision(tmp_path):
    body = _Body()
    assert body.sync()("x", 1.0) == "sent"
    assert body.ran == 1
    assert _decisions(tmp_path / ".vaara" / "trail" / "audit.db") == [
        ("decision_made", "tx.transfer")]


@pytest.mark.parametrize("verdict, allow, deny", [
    ("deny", 0.0, 0.01),
    ("escalate", 0.01, 0.99),
])
def test_non_allow_raises_before_the_body(verdict, allow, deny):
    body = _Body()
    pipe = _pipe(allow, deny)
    with pytest.raises(vaara.Blocked) as exc:
        body.sync(pipe)("x", 1.0)
    assert exc.value.decision == verdict
    assert body.ran == 0
    # A deny lands as action_blocked, anything else as decision_made.
    kind = EventType.ACTION_BLOCKED if verdict == "deny" else EventType.DECISION_MADE
    decided = pipe.trail.get_records_by_type(kind)
    assert [r.data["decision"] for r in decided] == [verdict]
    assert decided[0].action_id == exc.value.action_id


def test_a_scorer_that_crashes_raises_and_records_a_deny():
    class Crashes(AdaptiveScorer):
        def evaluate(self, ctx):
            raise RuntimeError("scorer exploded")

    body = _Body()
    pipe = InterceptionPipeline(scorer=Crashes())
    with pytest.raises(RuntimeError, match="scorer exploded"):
        body.sync(pipe)("x", 1.0)
    assert body.ran == 0
    decided = pipe.trail.get_records_by_type(EventType.ACTION_BLOCKED)
    assert decided and decided[-1].data["decision"] == "deny"
    assert "scorer failure" in decided[-1].data["reason"]


def test_a_trail_that_cannot_be_opened_raises_before_the_body(tmp_path):
    (tmp_path / ".vaara").write_text("a file where the trail directory goes")
    body = _Body()
    with pytest.raises(OSError):
        body.sync()("x", 1.0)
    assert body.ran == 0


@pytest.mark.parametrize("step", ["record_action_requested", "record_decision"])
def test_a_trail_that_cannot_record_raises_before_the_body(step, monkeypatch):
    pipe = _pipe(0.99, 0.999)  # would allow

    def refuse(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(pipe.trail, step, refuse)
    body = _Body()
    with pytest.raises(OSError, match="disk full"):
        body.sync(pipe)("x", 1.0)
    assert body.ran == 0


@pytest.mark.parametrize("allow, deny", [(0.0, 0.01), (0.01, 0.99)])
def test_async_non_allow_never_starts_the_body(allow, deny):
    body = _Body()
    with pytest.raises(vaara.Blocked):
        asyncio.run(body.coro(_pipe(allow, deny))("x", 1.0))
    assert body.ran == 0


def test_async_allow_reports_the_outcome_after_the_body(monkeypatch):
    pipe = _pipe(0.99, 0.999)
    reported: list = []
    monkeypatch.setattr(pipe, "report_outcome",
                        lambda aid, sev, description=None: reported.append(sev))

    @vaara.govern(tool_name="data.read", pipeline=pipe)
    async def fails():
        await asyncio.sleep(0)
        raise ValueError("async body failed")

    coro = fails()
    assert reported == []  # nothing is known until the body has run
    with pytest.raises(ValueError):
        asyncio.run(coro)
    assert reported == [1.0]

    @vaara.govern(tool_name="data.read", pipeline=pipe)
    async def works():
        return 7

    assert asyncio.run(works()) == 7
    assert reported == [1.0, 0.0]
