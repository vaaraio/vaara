"""Tests for the CrewAI completeness adapter (``VaaraGovernance``).

Uses ``SimpleNamespace`` fakes for CrewAI's ``ToolCallHookContext`` so the suite
runs without CrewAI installed. The adapter only reads ``tool_name``,
``tool_input``, ``agent``, ``crew``, and ``raw_tool_result`` off the context.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from vaara.audit.receipts import verify_receipt
from vaara.credential._contiguity import verify_contiguity
from vaara.integrations.crewai import VaaraGovernance, register


def _ctx(
    *,
    tool_name="search",
    tool_input=None,
    crew_id="crew-1",
    agent_id="a1",
    agent_role="Researcher",
    raw_tool_result=None,
):
    return SimpleNamespace(
        tool_name=tool_name,
        tool_input={"q": "x"} if tool_input is None else tool_input,
        agent=SimpleNamespace(id=agent_id, role=agent_role),
        crew=SimpleNamespace(id=crew_id),
        task=SimpleNamespace(id="t1"),
        tool_result=None,
        raw_tool_result=raw_tool_result,
    )


def test_run_is_contiguous_and_verifies():
    gov = VaaraGovernance()
    for i in range(10):
        gov.before_tool_call(_ctx(tool_input={"i": i}))
        gov.after_tool_call(_ctx(tool_input={"i": i}))

    report = gov.verify_run("crew-1")
    assert report.ok
    assert report.present == 10
    assert report.expected == 10
    assert report.missing_seqs == []

    decisions = gov.decisions("crew-1")
    assert [d["seq"] for d in decisions] == list(range(10))
    for d in decisions:
        assert d["running_count"] == d["seq"] + 1
        comp = d["extensions"]["vaara"]["completeness"]
        assert comp["seq"] == d["seq"]
        assert comp["runningCount"] == d["running_count"]
        assert comp["boundaryId"] == "crew-1"


def test_dropped_record_is_a_provable_gap():
    """The PR #6030 contract-test fixture: a dropped tool-call record is a
    provable gap from the held records alone, no issuer and no witness."""
    gov = VaaraGovernance()
    for i in range(10):
        gov.before_tool_call(_ctx(tool_input={"i": i}))

    held = gov.decisions("crew-1")
    # A verifier is handed nine of the ten records; seq 4 was silently dropped.
    kept = [d for d in held if d["seq"] != 4]
    evidence = [
        {"completeness": d["extensions"]["vaara"]["completeness"]} for d in kept
    ]

    report = verify_contiguity(evidence, "crew-1")
    assert not report.ok
    assert report.missing_seqs == [4]
    assert report.present == 9
    assert report.expected == 10


def test_leading_drop_is_a_provable_gap():
    """A dropped first record is caught: a surviving record's running_count
    establishes the expected length, so the absent seq 0 shows as a gap."""
    gov = VaaraGovernance()
    for i in range(6):
        gov.before_tool_call(_ctx(tool_input={"i": i}))

    held = gov.decisions("crew-1")
    # Drop seq 0. The held records still carry running_count up to 6.
    kept = [d for d in held if d["seq"] != 0]
    evidence = [
        {"completeness": d["extensions"]["vaara"]["completeness"]} for d in kept
    ]

    report = verify_contiguity(evidence, "crew-1")
    assert not report.ok
    assert report.expected == 6
    assert report.present == 5
    assert report.missing_seqs == [0]


def test_tail_drop_is_caught_with_a_sealing_record():
    """A finalized run pins its total, so a dropped tail shows as missing even
    though the removed records took their own ``seq`` with them. This is the
    upgrade over per-record ``running_count`` (which alone cannot see a
    truncation). Interior and leading drops are caught without a seal (above)."""
    gov = VaaraGovernance()
    for i in range(6):
        gov.before_tool_call(_ctx(tool_input={"i": i}))
    seal = gov.finalize_run("crew-1")
    assert seal == {"boundaryId": "crew-1", "sealed": True, "total": 6}

    held = gov.decisions("crew-1")
    kept = [d for d in held if d["seq"] < 4]  # drop the last two
    evidence = [
        {"completeness": d["extensions"]["vaara"]["completeness"]} for d in kept
    ]
    evidence.append({"completeness": seal})  # the seal survives

    report = verify_contiguity(evidence, "crew-1")
    assert not report.ok
    assert report.present == 4
    assert report.expected == 6
    assert report.missing_seqs == [4, 5]


def test_finalized_run_verifies_whole():
    """A complete, sealed run verifies; the seal does not perturb a full set."""
    gov = VaaraGovernance()
    for i in range(6):
        gov.before_tool_call(_ctx(tool_input={"i": i}))
    gov.finalize_run("crew-1")

    report = gov.verify_run("crew-1")
    assert report.ok
    assert report.present == 6
    assert report.expected == 6
    assert report.missing_seqs == []


def test_tail_drop_with_seal_also_removed_is_the_residual():
    """The irreducible limit: a suffix drop that also suppresses the sealing
    record stays invisible from the held set alone. An external anchor (an
    rfc3161 timestamp over the run) is what closes this; the held set cannot."""
    gov = VaaraGovernance()
    for i in range(6):
        gov.before_tool_call(_ctx(tool_input={"i": i}))
    gov.finalize_run("crew-1")

    held = gov.decisions("crew-1")
    kept = [d for d in held if d["seq"] < 4]  # drop last two AND withhold seal
    evidence = [
        {"completeness": d["extensions"]["vaara"]["completeness"]} for d in kept
    ]

    report = verify_contiguity(evidence, "crew-1")
    assert report.ok  # documents the residual, not an endorsement
    assert report.present == 4
    assert report.expected == 4


def test_finalize_run_can_pin_the_boundary_max_class():
    """The seal optionally carries the boundary's highest action class, so a
    gap's worst case is bounded from the held set alone. A dropped tail then
    both shows as missing and reports the most it could have hidden."""
    gov = VaaraGovernance()
    for i in range(6):
        gov.before_tool_call(_ctx(tool_input={"i": i}))
    seal = gov.finalize_run("crew-1", max_class="transfer")
    assert seal == {
        "boundaryId": "crew-1",
        "sealed": True,
        "total": 6,
        "maxClass": "transfer",
    }

    held = gov.decisions("crew-1")
    kept = [d for d in held if d["seq"] < 4]  # drop the last two
    evidence = [
        {"completeness": d["extensions"]["vaara"]["completeness"]} for d in kept
    ]
    evidence.append({"completeness": seal})

    report = verify_contiguity(evidence, "crew-1")
    assert not report.ok
    assert report.missing_seqs == [4, 5]
    assert report.worst_case_class == "transfer"
    assert "gap worst-case: action class up to 'transfer'" in report.gap_report()


def test_finalize_run_without_max_class_is_unchanged():
    """The class is opt-in: omit it and the seal is byte-for-byte as before."""
    gov = VaaraGovernance()
    for i in range(3):
        gov.before_tool_call(_ctx(tool_input={"i": i}))
    seal = gov.finalize_run("crew-1")
    assert seal == {"boundaryId": "crew-1", "sealed": True, "total": 3}
    assert "maxClass" not in seal
    assert gov.verify_run("crew-1").worst_case_class is None


def test_seal_alone_flags_a_fully_dropped_run():
    """A seal asserting N over zero held records is a fully-dropped run."""
    evidence = [
        {"completeness": {"boundaryId": "crew-1", "sealed": True, "total": 3}}
    ]
    report = verify_contiguity(evidence, "crew-1")
    assert not report.ok
    assert report.present == 0
    assert report.expected == 3
    assert report.missing_seqs == [0, 1, 2]


def test_receipts_recompute():
    gov = VaaraGovernance()
    gov.before_tool_call(_ctx(tool_input={"a": 1}))
    gov.after_tool_call(_ctx(tool_input={"a": 1}, raw_tool_result="ok"))

    receipts = gov.receipts("crew-1")
    assert len(receipts) == 1
    assert receipts[0].outcome is not None
    assert verify_receipt(receipts[0])


def test_outcome_references_decision_seq():
    gov = VaaraGovernance()
    gov.before_tool_call(_ctx(tool_input={"a": 1}))
    gov.after_tool_call(_ctx(tool_input={"a": 1}))

    outcomes = gov.outcomes("crew-1")
    decision = gov.decisions("crew-1")[0]
    assert len(outcomes) == 1
    assert outcomes[0]["seq"] == 0
    assert outcomes[0]["outcome"] == "executed"
    assert outcomes[0]["decision_id"] == decision["decision_id"]


def test_errored_outcome_is_recorded():
    gov = VaaraGovernance()
    gov.before_tool_call(_ctx(tool_input={"a": 1}))
    gov.after_tool_call(
        _ctx(tool_input={"a": 1}, raw_tool_result=RuntimeError("boom"))
    )
    assert gov.outcomes("crew-1")[0]["outcome"] == "error"


def test_boundaries_are_independent():
    gov = VaaraGovernance()
    for crew_id in ("crew-A", "crew-B"):
        for i in range(3):
            gov.before_tool_call(_ctx(tool_input={"i": i}, crew_id=crew_id))

    assert [d["seq"] for d in gov.decisions("crew-A")] == [0, 1, 2]
    assert [d["seq"] for d in gov.decisions("crew-B")] == [0, 1, 2]
    assert gov.verify_run("crew-A").ok
    assert gov.verify_run("crew-B").ok


def test_verify_run_without_boundary_raises_when_records_span_many():
    gov = VaaraGovernance()
    gov.before_tool_call(_ctx(crew_id="crew-A"))
    gov.before_tool_call(_ctx(crew_id="crew-B"))
    with pytest.raises(ValueError):
        gov.verify_run()


def test_params_hash_is_order_independent():
    gov = VaaraGovernance()
    gov.before_tool_call(_ctx(tool_input={"a": 1, "b": 2}))
    gov.before_tool_call(_ctx(tool_input={"b": 2, "a": 1}))
    decisions = gov.decisions("crew-1")
    assert decisions[0]["params_hash"] == decisions[1]["params_hash"]
    assert decisions[0]["params_hash"].startswith("sha256:")


def test_default_boundary_when_no_crew():
    gov = VaaraGovernance()
    ctx = SimpleNamespace(
        tool_name="t", tool_input={"a": 1}, agent=None, crew=None, raw_tool_result=None
    )
    gov.before_tool_call(ctx)
    comp = gov.decisions()[0]["extensions"]["vaara"]["completeness"]
    assert comp["boundaryId"] == "crew-run"


def test_custom_boundary_callable():
    gov = VaaraGovernance(boundary_id_for=lambda _context: "run-42")
    gov.before_tool_call(_ctx())
    assert gov.decisions("run-42")[0]["seq"] == 0


def test_non_serializable_tool_input_does_not_raise():
    gov = VaaraGovernance()
    gov.before_tool_call(_ctx(tool_input={"obj": object()}))
    assert gov.decisions("crew-1")[0]["params_hash"].startswith("sha256:")


def test_hooks_return_none():
    gov = VaaraGovernance()
    assert gov.before_tool_call(_ctx()) is None
    assert gov.after_tool_call(_ctx()) is None


def test_register_wires_both_hooks(monkeypatch):
    calls: dict[str, object] = {}
    fake_hooks = types.ModuleType("crewai.hooks")
    fake_hooks.register_before_tool_call_hook = lambda fn: calls.__setitem__("before", fn)
    fake_hooks.register_after_tool_call_hook = lambda fn: calls.__setitem__("after", fn)
    fake_pkg = types.ModuleType("crewai")
    fake_pkg.hooks = fake_hooks
    monkeypatch.setitem(sys.modules, "crewai", fake_pkg)
    monkeypatch.setitem(sys.modules, "crewai.hooks", fake_hooks)

    gov = VaaraGovernance()
    register(gov)
    assert calls["before"] == gov.before_tool_call
    assert calls["after"] == gov.after_tool_call


def test_register_without_crewai_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "crewai", None)
    monkeypatch.setitem(sys.modules, "crewai.hooks", None)
    with pytest.raises(ImportError, match=r"vaara\[crewai\]"):
        register(VaaraGovernance())
