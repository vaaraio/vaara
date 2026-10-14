"""A matched sequence is named in the decision reason, held or not."""

from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.pipeline import InterceptionPipeline
from vaara.scorer.adaptive import (
    BUILTIN_SEQUENCES,
    OUTBOUND_LABEL,
    AdaptiveScorer,
    SequencePattern,
)

READ = ("read_file", {"path": "/data/customers/acme.csv"})
SEND = ("send_email", {"to": "someone@external.example", "body": "summary"})


def _run(steps, scorer=None):
    trail = SQLiteAuditBackend(":memory:").load_trail()
    pipe = InterceptionPipeline(trail=trail, scorer=scorer)
    results = [pipe.intercept(agent_id="a1", tool_name=tool, parameters=params,
                              session_id="run-1")
               for tool, params in steps]
    decisions = [r for r in trail._records
                 if r.event_type.value in ("decision_made", "action_blocked")]
    return results, decisions


def test_raised_but_allowed_sequence_is_named():
    results, decisions = _run([READ, SEND])
    assert results[-1].decision == "allow"
    assert "; sequence read_then_outbound matched" in decisions[-1].data["reason"]


def test_held_sequence_is_named_once():
    seqs = [s for s in BUILTIN_SEQUENCES if s.name != "read_then_outbound"] + [
        SequencePattern("read_then_outbound", ("data.read", OUTBOUND_LABEL),
                        risk_boost=0.4, window_size=5, escalate=True)]
    results, decisions = _run([READ, SEND],
                              AdaptiveScorer(sequence_patterns=seqs))
    reason = decisions[-1].data["reason"]
    assert results[-1].decision == "escalate"
    assert "sequence read_then_outbound, held for a human" in reason
    assert "read_then_outbound matched" not in reason


def test_no_match_names_no_sequence():
    _, decisions = _run([READ])
    assert "sequence" not in decisions[-1].data["reason"]
