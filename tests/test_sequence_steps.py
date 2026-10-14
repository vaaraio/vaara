"""A decision completing a sequence names the earlier actions by action id."""

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.pipeline import InterceptionPipeline, _sequence_steps

READ = ("read_file", {"path": "/data/customers/acme.csv"})
SUMMARISE = ("summarize", {"text": "..."})
SEND = ("send_email", {"to": "someone@external.example", "body": "summary"})


def _decisions(trail):
    return [r for r in trail._records
            if r.event_type.value in ("decision_made", "action_blocked")]


def _requested_id(trail, tool):
    return next(r.action_id for r in trail._records
                if r.event_type.value == "action_requested" and r.tool_name == tool)


def test_live_pipeline_names_the_earlier_step():
    trail = SQLiteAuditBackend(":memory:").load_trail()
    pipe = InterceptionPipeline(trail=trail)
    for tool, params in (READ, SUMMARISE, SEND):
        pipe.intercept(agent_id="a1", tool_name=tool, parameters=params,
                       session_id="run-1")
    record = _decisions(trail)[-1].data
    assert record["sequence_steps"] == {
        "read_then_outbound": [_requested_id(trail, "read_file")],
    }
    assert "; sequence read_then_outbound matched" in record["reason"]


def test_fresh_process_per_call_names_the_step_from_the_trail(tmp_path):
    # A host hook is one process per call: each call gets a new pipeline and
    # scorer, and the session's earlier actions come back from the trail.
    db = str(tmp_path / "trail.db")
    for tool, params in (READ, SEND):
        trail = SQLiteAuditBackend(db).load_trail()
        InterceptionPipeline(trail=trail).intercept(
            agent_id="a1", tool_name=tool, parameters=params, session_id="run-1")
    trail = SQLiteAuditBackend(db).load_trail()
    record = _decisions(trail)[-1].data
    assert record["sequence_steps"] == {
        "read_then_outbound": [_requested_id(trail, "read_file")],
    }


def test_no_match_adds_no_key():
    trail = SQLiteAuditBackend(":memory:").load_trail()
    InterceptionPipeline(trail=trail).intercept(
        agent_id="a1", tool_name=READ[0], parameters=READ[1], session_id="run-1")
    assert "sequence_steps" not in _decisions(trail)[-1].data


def test_scorer_output_outside_the_shape_is_dropped():
    assert _sequence_steps({}) is None
    assert _sequence_steps("x") is None
    assert _sequence_steps({"sequence_steps": ["a"]}) is None
    assert _sequence_steps({"sequence_steps": {"p": "abc"}}) is None
    assert _sequence_steps({"sequence_steps": {"p": [1, None, ""]}}) is None
    assert _sequence_steps({"sequence_steps": {"p": ["id-1", 2], "q": []}}) == {
        "p": ["id-1"],
    }
