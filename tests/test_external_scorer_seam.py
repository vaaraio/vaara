# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a remote /v1/score answer becomes on the trail and in the receipt.

The external scorer speaks the published /v1/score contract inbound. The
pipeline reads ``reason``, ``threshold_allow`` and ``threshold_deny`` off
the scorer result and ``signals``, ``effective_alpha``, ``bucket_category``
and ``calibration_size`` off its ``raw_result``; the receipt reads the
thresholds off the risk record and, when they are absent, states the
operating point of records older than v0.40.0. A remote answer that lost
its ``explanation``, ``thresholds``, ``signals`` and ``alpha`` on the way
in therefore recorded an empty reason and a receipt stating 0.40 / 0.70
for a decision made at some other point. These tests run the whole seam:
a real HTTP peer, the pipeline, the trail, the receipt.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vaara.audit.receipts import extract_receipt
from vaara.audit.sqlite_backend import SQLiteAuditBackend
from vaara.scorer.composite import CompositeScorer
from vaara.scorer.composition import ExternalScorer

ANSWER = {
    "action_id": "remote-1",
    "decision": "escalate",
    "risk": {"point": 0.62, "lower": 0.55, "upper": 0.71, "alpha": 0.08,
             "bucket": "filesystem"},
    "signals": {"heuristic": 0.4, "classifier": 0.7},
    "thresholds": {"allow": 0.3, "deny": 0.9},
    "calibration_size": 212,
    "evaluation_ms": 1.5,
    "explanation": "classifier saw a credential path",
}


class _Peer:
    """A /v1/score peer answering from a script, or misbehaving on request."""

    def __init__(self):
        self.requests: list[dict] = []
        peer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                peer.requests.append(json.loads(self.rfile.read(length) or b"{}"))
                mode = peer.mode
                if mode == "reset":
                    # Close without a status line: the client's read raises.
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                body = json.dumps(peer.answer).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                if mode == "truncate":
                    # Half the JSON, then the connection goes away.
                    self.send_header("content-length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body[: len(body) // 2])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        self.answer = dict(ANSWER)
        self.mode = "ok"
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/score"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def peer():
    p = _Peer()
    yield p
    p.close()


def _pipeline(scorer):
    from vaara.pipeline import InterceptionPipeline
    backend = SQLiteAuditBackend(":memory:")
    trail = backend.load_trail()
    trail._on_record = backend.write_record
    return InterceptionPipeline(trail=trail, scorer=scorer)


def _records(pipeline, action_id):
    return [r for r in pipeline.trail._records if r.action_id == action_id]


def _risk_and_decision(pipeline, action_id):
    risk = decision = None
    for r in _records(pipeline, action_id):
        if r.event_type.value == "risk_scored":
            risk = r
        if "decision" in (r.data or {}):
            decision = r
    assert risk is not None and decision is not None
    return risk, decision


# ── What the remote said is what the trail records ─────────────────────

def test_the_remote_explanation_is_the_recorded_reason(peer):
    pipeline = _pipeline(ExternalScorer(peer.url, name="peer"))
    result = pipeline.intercept(agent_id="a", tool_name="fs.read")
    _, decision = _risk_and_decision(pipeline, result.action_id)
    assert result.decision == "escalate"
    assert decision.data["reason"] == "classifier saw a credential path"
    assert result.reason == "classifier saw a credential path"


def test_the_remote_thresholds_signals_alpha_and_bucket_reach_the_risk_record(peer):
    pipeline = _pipeline(ExternalScorer(peer.url, name="peer"))
    result = pipeline.intercept(agent_id="a", tool_name="fs.read")
    risk, _ = _risk_and_decision(pipeline, result.action_id)
    assert risk.data["threshold_allow"] == 0.3
    assert risk.data["threshold_deny"] == 0.9
    assert risk.data["signals"] == {"heuristic": 0.4, "classifier": 0.7}
    assert risk.data["effective_alpha"] == 0.08
    assert risk.data["bucket_category"] == "filesystem"
    assert risk.data["calibration_size"] == 212
    assert risk.data["point_estimate"] == 0.62
    assert (risk.data["conformal_lower"], risk.data["conformal_upper"]) == (0.55, 0.71)
    assert result.signals == {"heuristic": 0.4, "classifier": 0.7}


def test_the_receipt_states_the_operating_point_the_remote_decided_at(peer):
    """Not the 0.40 / 0.70 the receipt assumes for records older than v0.40.0."""
    pipeline = _pipeline(ExternalScorer(peer.url, name="peer"))
    result = pipeline.intercept(agent_id="a", tool_name="fs.read")
    receipt = extract_receipt(pipeline.trail, result.action_id)
    assert receipt is not None
    assert (receipt.commit.threshold_allow, receipt.commit.threshold_deny) == (0.3, 0.9)
    assert receipt.commit.decision == "escalate"
    assert receipt.commit.risk_score == 0.62


def test_the_request_is_the_policy_context(peer):
    pipeline = _pipeline(ExternalScorer(peer.url, name="peer"))
    pipeline.intercept(agent_id="agent-7", tool_name="fs.read", parameters={"path": "/etc"})
    assert peer.requests and peer.requests[0]["tool_name"] == "fs.read"
    assert peer.requests[0]["agent_id"] == "agent-7"


# ── A score with no decision is decided here, at a stated point ─────────

def test_a_score_only_answer_is_decided_here_and_the_point_is_stated(peer):
    peer.answer = {"risk": {"point": 0.5, "lower": 0.4, "upper": 0.6}}
    pipeline = _pipeline(ExternalScorer(peer.url, name="peer"))
    result = pipeline.intercept(agent_id="a", tool_name="fs.read")
    risk, decision = _risk_and_decision(pipeline, result.action_id)
    assert result.decision == "escalate"
    assert (risk.data["threshold_allow"], risk.data["threshold_deny"]) == (0.4, 0.7)
    receipt = extract_receipt(pipeline.trail, result.action_id)
    assert (receipt.commit.threshold_allow, receipt.commit.threshold_deny) == (0.4, 0.7)
    assert "decided here" in decision.data["reason"]


# ── A decision without the point it was made at is not evidence ─────────

def test_a_remote_decision_without_thresholds_is_a_deny_that_names_the_gap(peer):
    """The contract requires ``thresholds``. A remote verdict without them
    would put an operating point on the receipt that nobody stated, so the
    answer is refused as malformed, on the ground that the check did not
    complete, naming the input that was missing."""
    peer.answer = {"decision": "allow", "risk": {"point": 0.1, "lower": 0.05, "upper": 0.2}}
    pipeline = _pipeline(ExternalScorer(peer.url, name="peer"))
    result = pipeline.intercept(agent_id="a", tool_name="fs.read")
    _, decision = _risk_and_decision(pipeline, result.action_id)
    assert result.decision == "deny"
    assert decision.data["violation_type"] == "scorer_unavailable"
    assert decision.data["unavailable_input"] == "scorer:peer.thresholds"
    assert "thresholds" in decision.data["reason"]


# ── Every transport failure is a deny on the same ground ────────────────

@pytest.mark.parametrize("mode", ["reset", "truncate"])
def test_a_connection_lost_mid_answer_is_a_scorer_unavailable_deny(peer, mode):
    """``URLError``, ``HTTPError`` and ``TimeoutError`` were caught; a reset
    while reading, or a body shorter than its content-length, raised out of
    ``evaluate`` and was recorded as ``scorer_failure`` with a traceback,
    which is the ground for a scorer that broke, not one that was
    unreachable."""
    peer.mode = mode
    pipeline = _pipeline(ExternalScorer(peer.url, name="peer", timeout=2.0))
    result = pipeline.intercept(agent_id="a", tool_name="fs.read")
    _, decision = _risk_and_decision(pipeline, result.action_id)
    assert result.decision == "deny"
    assert decision.data["violation_type"] == "scorer_unavailable"
    assert decision.data["unavailable_input"] == "scorer:peer"


def test_the_recorded_reason_says_the_scorer_never_answered(peer):
    peer.mode = "reset"
    pipeline = _pipeline(ExternalScorer(peer.url, name="peer", timeout=2.0))
    result = pipeline.intercept(agent_id="a", tool_name="fs.read")
    _, decision = _risk_and_decision(pipeline, result.action_id)
    assert "transport failure" in decision.data["reason"]


# ── The composite carries the deciding member's reason and point ────────

class _Member:
    def __init__(self, score, decision, name, reason, allow, deny):
        self._score, self._decision, self.name = score, decision, name
        self._reason, self._allow, self._deny = reason, allow, deny

    def evaluate(self, context):
        return {"point_estimate": self._score, "decision": self._decision,
                "backend": self.name, "reason": self._reason,
                "threshold_allow": self._allow, "threshold_deny": self._deny,
                "raw_result": {"point_estimate": self._score,
                               "conformal_interval": [self._score, self._score]}}


@pytest.mark.parametrize("combine", ["max", "mean"])
def test_a_composite_decision_carries_the_deciding_members_reason_and_point(combine):
    """In both modes the decision is the strongest member's. The reason and
    the thresholds on the trail are that member's, so the receipt states the
    point the decision was made at, not the top scorer's and not 0.40 / 0.70."""
    quiet = _Member(0.95, "allow", "quiet", "nothing seen", 0.96, 0.99)
    loud = _Member(0.30, "deny", "loud", "matched a deny list", 0.1, 0.2)
    pipeline = _pipeline(CompositeScorer([quiet, loud], combine=combine))
    result = pipeline.intercept(agent_id="a", tool_name="fs.read")
    risk, decision = _risk_and_decision(pipeline, result.action_id)
    assert result.decision == "deny"
    assert decision.data["reason"] == "matched a deny list"
    assert (risk.data["threshold_allow"], risk.data["threshold_deny"]) == (0.1, 0.2)
    receipt = extract_receipt(pipeline.trail, result.action_id)
    assert (receipt.commit.threshold_allow, receipt.commit.threshold_deny) == (0.1, 0.2)
