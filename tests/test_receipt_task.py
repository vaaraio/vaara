"""Task binding on execution receipts, and the three-valued check.

MCP Tasks give long-running work a durable taskId and carry it in the
_meta["io.modelcontextprotocol/related-task"] block of every message that
belongs to the task. Transport metadata is not signed, so a receipt that only
names its task there cannot show membership. Here the id rides inside the
signed receiptAsserted block, and the check returns bound, conflict or
unsupported, deciding the unsupported case before any comparison.
"""

from __future__ import annotations

import importlib.util

import pytest

for _mod in ("rfc8785", "cryptography"):
    if importlib.util.find_spec(_mod) is None:
        pytest.skip(
            "attestation extra not installed (pip install 'vaara[attestation]')",
            allow_module_level=True,
        )

from vaara.attestation._attest_types import AttestationError  # noqa: E402
from vaara.attestation._receipt_types import (  # noqa: E402
    ReceiptAsserted,
    receipt_asserted_from_dict,
    receipt_asserted_to_dict,
)
from vaara.attestation.receipt import (  # noqa: E402
    RELATED_TASK_META_KEY,
    OutcomeDerived,
    TaskResult,
    emit_receipt,
    make_back_link,
    parse_receipt,
    related_task_id,
    verify_receipt_signature,
    verify_receipt_task,
)
from vaara.attestation.tool_call_attestation import (  # noqa: E402
    PayloadDerived,
    PlannerDeclared,
    ToolCallBinding,
    emit_attestation,
    make_args_digest,
)

HS_SECRET = b"task-test-secret"


def _attestation():
    payload = PayloadDerived(tool_calls=(ToolCallBinding(
        name="pay",
        server_fingerprint="sha256:" + "2" * 64,
        args=make_args_digest({"amount": "340", "currency": "USD"}),
    ),))
    return emit_attestation(
        planner_declared=PlannerDeclared(intent="settle invoice"),
        payload_derived=payload,
        iss="issuer://test",
        sub="agent:payer",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )


def _emit(att, **overrides):
    kwargs = dict(
        back_link=make_back_link(att),
        outcome_derived=OutcomeDerived(
            status="executed", completed_at="2026-09-19T10:00:00Z"
        ),
        iss="issuer://test",
        sub="agent:payer",
        secret_version="v1",
        alg="HS256",
        signing_material=HS_SECRET,
    )
    kwargs.update(overrides)
    return emit_receipt(**kwargs)


def _asserted(**overrides):
    kwargs = dict(
        iss="i", sub="s", iat="2026-09-19T00:00:00Z", nonce="n",
        secret_version="v1", alg="HS256",
    )
    kwargs.update(overrides)
    return ReceiptAsserted(**kwargs)


# ------------------------------------------------------------- three verdicts

TASK = "786512e2-9e0d-44bd-8f29-789f320fe840"


def test_bound_when_task_matches():
    r = _emit(_attestation(), task_id=TASK)
    res = verify_receipt_task(r, TASK)
    assert res.verdict == "bound"
    assert bool(res) is True
    assert res.receipt_task_id == TASK


def test_conflict_when_task_differs():
    r = _emit(_attestation(), task_id=TASK)
    res = verify_receipt_task(r, "other-task")
    assert res.verdict == "conflict"
    assert bool(res) is False
    assert TASK in res.reason and "other-task" in res.reason


def test_unsupported_when_task_absent():
    r = _emit(_attestation())
    res = verify_receipt_task(r, TASK)
    assert res.verdict == "unsupported"
    assert bool(res) is False
    assert res.receipt_task_id is None


def test_unsupported_and_conflict_are_distinct_values():
    absent = verify_receipt_task(_emit(_attestation()), TASK)
    wrong = verify_receipt_task(_emit(_attestation(), task_id="x"), TASK)
    assert absent.verdict != wrong.verdict
    assert bool(absent) is bool(wrong) is False


def test_empty_expectation_is_the_callers_error_not_a_conflict():
    r = _emit(_attestation(), task_id=TASK)
    with pytest.raises(ValueError):
        verify_receipt_task(r, "")


# ------------------------------------------------------- inside the signature

def test_task_rides_inside_the_signed_envelope():
    r = _emit(_attestation(), task_id=TASK)
    assert verify_receipt_signature(r, verifying_material=HS_SECRET)
    d = r.to_dict()
    assert d["receiptAsserted"]["taskId"] == TASK
    d["receiptAsserted"]["taskId"] = "other-task"
    assert not verify_receipt_signature(parse_receipt(d), verifying_material=HS_SECRET)


def test_replay_under_another_task_is_a_conflict_with_signature_intact():
    r = _emit(_attestation(), task_id=TASK)
    assert verify_receipt_signature(r, verifying_material=HS_SECRET)
    assert verify_receipt_task(r, "another-task").verdict == "conflict"


def test_absent_task_leaves_the_envelope_byte_identical():
    d = _emit(_attestation()).to_dict()
    assert "taskId" not in d["receiptAsserted"]


def test_task_round_trips_through_the_wire():
    r = _emit(_attestation(), task_id=TASK)
    again = parse_receipt(r.to_dict())
    assert again.receipt_asserted.task_id == TASK
    assert verify_receipt_signature(again, verifying_material=HS_SECRET)


@pytest.mark.parametrize("bad", ["", 7, {"taskId": "x"}, ["x"]])
def test_malformed_task_is_refused_on_read(bad):
    d = receipt_asserted_to_dict(_asserted())
    d["taskId"] = bad
    with pytest.raises(AttestationError):
        receipt_asserted_from_dict(d)


def test_empty_task_is_refused_on_emit():
    with pytest.raises(AttestationError):
        _emit(_attestation(), task_id="")


def test_task_and_audience_are_independent():
    r = _emit(_attestation(), task_id=TASK, aud="verifier://a")
    d = r.to_dict()["receiptAsserted"]
    assert d["taskId"] == TASK and d["aud"] == "verifier://a"
    assert verify_receipt_task(r, TASK)


def test_result_is_frozen_and_named():
    res = verify_receipt_task(_emit(_attestation(), task_id=TASK), TASK)
    assert isinstance(res, TaskResult)
    with pytest.raises(Exception):
        res.verdict = "conflict"  # type: ignore[misc]


# ------------------------------------------------ reading the MCP related-task

def test_related_task_id_reads_the_spec_placement():
    params = {"name": "pay", "arguments": {}, "_meta": {RELATED_TASK_META_KEY: {"taskId": TASK}}}
    assert related_task_id(params) == TASK


@pytest.mark.parametrize("params", [
    None,
    "x",
    {},
    {"_meta": None},
    {"_meta": "x"},
    {"_meta": {}},
    {"_meta": {RELATED_TASK_META_KEY: None}},
    {"_meta": {RELATED_TASK_META_KEY: "x"}},
    {"_meta": {RELATED_TASK_META_KEY: {}}},
    {"_meta": {RELATED_TASK_META_KEY: {"taskId": ""}}},
    {"_meta": {RELATED_TASK_META_KEY: {"taskId": 5}}},
])
def test_related_task_id_is_none_for_anything_malformed(params):
    assert related_task_id(params) is None


def test_progress_token_beside_related_task_does_not_interfere():
    params = {"_meta": {"progressToken": "p1", RELATED_TASK_META_KEY: {"taskId": TASK}}}
    assert related_task_id(params) == TASK
