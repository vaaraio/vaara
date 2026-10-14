"""A decision receipt names the arguments it was decided on without carrying them.

The trail records a salted commitment on each action request. The receipt for
the decision carries the commitment, and the trail's holder opens it for one
decision by handing over that request's arguments and salt.
"""

from __future__ import annotations

import pytest

from vaara.audit import _arguments_commitment as ac
from vaara.audit.decision_receipts import (
    DecisionReceiptSink,
    build_evidence,
    check_disclosed_arguments,
)
from vaara.audit.trail import EventType
from vaara.pipeline import InterceptionPipeline

ARGS = {"path": "/srv/payroll.csv", "amount": 12.5, "to": "acct-7"}


def _decide(tmp_path, params=ARGS):
    pipe = InterceptionPipeline()
    pipe.intercept(agent_id="a", tool_name="file_write", parameters=params)
    records = list(pipe.trail._records)
    sink = DecisionReceiptSink(trail_dir=tmp_path)
    bodies = []
    for record in records:
        if record.event_type == EventType.ACTION_REQUESTED:
            sink(record)
        elif record.event_type in (EventType.DECISION_MADE, EventType.ACTION_BLOCKED):
            bodies.append(sink.mint(record))
    request = [r for r in records if r.event_type == EventType.ACTION_REQUESTED][-1]
    return request, bodies[-1]


def test_the_request_record_carries_a_salt_and_a_commitment(tmp_path):
    request, _ = _decide(tmp_path)
    salt = request.data["arguments_salt"]
    assert len(salt) == 32 and int(salt, 16) >= 0
    assert ac.verify(request.data["arguments_commitment"], request.data["parameters"], salt)


def test_the_receipt_carries_the_commitment_and_not_the_arguments(tmp_path):
    request, body = _decide(tmp_path)
    evidence = body["evidence"]
    assert evidence["argumentsCommitment"] == request.data["arguments_commitment"]
    flat = repr(body)
    assert "payroll" not in flat and "acct-7" not in flat
    assert request.data["arguments_salt"] not in flat


def test_disclosed_arguments_and_salt_open_the_commitment(tmp_path):
    request, body = _decide(tmp_path)
    salt = request.data["arguments_salt"]
    assert check_disclosed_arguments(body["evidence"], dict(ARGS), salt)


@pytest.mark.parametrize("args", [
    {**ARGS, "amount": 12.51},
    {k: v for k, v in ARGS.items() if k != "to"},
    {**ARGS, "extra": True},
])
def test_other_arguments_do_not_open_it(tmp_path, args):
    request, body = _decide(tmp_path)
    assert not check_disclosed_arguments(body["evidence"], args, request.data["arguments_salt"])


def test_another_decisions_salt_does_not_open_it(tmp_path):
    request, body = _decide(tmp_path)
    other, _ = _decide(tmp_path / "other")
    assert other.data["arguments_salt"] != request.data["arguments_salt"]
    assert not check_disclosed_arguments(body["evidence"], ARGS, other.data["arguments_salt"])


def test_same_arguments_commit_differently_per_request(tmp_path):
    first, _ = _decide(tmp_path)
    second, _ = _decide(tmp_path / "second")
    assert first.data["arguments_commitment"] != second.data["arguments_commitment"]


def test_key_order_does_not_change_the_commitment():
    salt = ac.new_salt()
    assert ac.commit({"a": 1, "b": [2, 3]}, salt) == ac.commit({"b": [2, 3], "a": 1}, salt)


def test_a_bare_digest_is_not_the_commitment():
    import hashlib

    import rfc8785

    salt = ac.new_salt()
    bare = "sha256:" + hashlib.sha256(rfc8785.dumps(ARGS)).hexdigest()
    assert ac.commit(ARGS, salt) != bare


@pytest.mark.parametrize("salt", ["", "zz" * 16, "ab" * 15, "AB" * 16])
def test_a_malformed_salt_is_refused(salt):
    with pytest.raises(ValueError):
        ac.commit(ARGS, salt)
    assert not ac.verify("sha256:" + "0" * 64, ARGS, salt)


def test_a_decision_with_no_recorded_request_has_no_commitment(tmp_path):
    pipe = InterceptionPipeline()
    pipe.intercept(agent_id="a", tool_name="file_read", parameters={"path": "x"})
    decision = [r for r in pipe.trail._records
                if r.event_type in (EventType.DECISION_MADE, EventType.ACTION_BLOCKED)][-1]
    evidence = DecisionReceiptSink(trail_dir=tmp_path).mint(decision)["evidence"]
    assert "argumentsCommitment" not in evidence
    assert not check_disclosed_arguments(evidence, {"path": "x"}, ac.new_salt())


def test_a_malformed_commitment_never_reaches_the_receipt():
    pipe = InterceptionPipeline()
    pipe.intercept(agent_id="a", tool_name="file_read", parameters={"path": "x"})
    decision = [r for r in pipe.trail._records
                if r.event_type in (EventType.DECISION_MADE, EventType.ACTION_BLOCKED)][-1]
    assert "argumentsCommitment" not in build_evidence(decision, "md5:abc")


def _sqlite_decision(tmp_path, params=ARGS):
    from vaara.audit.sqlite_backend import SQLiteAuditBackend

    db = tmp_path / "audit.db"
    trail = SQLiteAuditBackend(str(db)).load_trail()
    InterceptionPipeline(trail=trail).intercept(
        agent_id="a", tool_name="file_write", parameters=params)
    receipts = sorted((tmp_path / "receipts").rglob("*.json"))
    receipts = [r for r in receipts if r.parent.name != "receipts"]
    assert receipts, "the sink wrote no decision receipt"
    return db, receipts[-1]


def test_the_commands_disclose_and_verify_one_decisions_arguments(tmp_path, capsys):
    import json

    from vaara.cli import main

    db, receipt = _sqlite_decision(tmp_path)
    out = tmp_path / "disclosure.json"
    assert main(["receipt", "disclose-arguments", str(receipt), "--db", str(db),
                 "--out", str(out)]) == 0
    disclosure = json.loads(out.read_text())
    assert disclosure["arguments"] == ARGS
    assert main(["receipt", "verify-arguments", str(receipt),
                 "--disclosure", str(out)]) == 0
    assert "OK" in capsys.readouterr().out

    disclosure["arguments"] = {**ARGS, "amount": 99}
    out.write_text(json.dumps(disclosure))
    assert main(["receipt", "verify-arguments", str(receipt),
                 "--disclosure", str(out)]) == 1


def test_a_disclosure_for_another_action_is_refused(tmp_path):
    import json

    from vaara.cli import main

    db, receipt = _sqlite_decision(tmp_path)
    out = tmp_path / "d.json"
    assert main(["receipt", "disclose-arguments", str(receipt), "--db", str(db),
                 "--out", str(out)]) == 0
    disclosure = json.loads(out.read_text())
    disclosure["actionId"] = "some-other-action"
    out.write_text(json.dumps(disclosure))
    assert main(["receipt", "verify-arguments", str(receipt),
                 "--disclosure", str(out)]) == 1
