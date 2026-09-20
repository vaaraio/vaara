"""Task binding for execution receipts, and the three-valued check.

MCP Tasks (spec 2025-11-25) give a long-running unit of work a durable
``taskId``, and every request, notification and response that belongs to the
task carries ``_meta["io.modelcontextprotocol/related-task"] = {"taskId": ...}``.
When a supervisor has to show what a sub-agent did under a task, the
execution receipts are the record. A receipt that names the task only in
transport metadata proves nothing about membership: the metadata is not
signed, so a receipt from another task, or from no task, reads the same.

So the task id rides inside ``receiptAsserted`` and therefore inside the
signed preimage, and the check has the same three outcomes as the audience
check, for the same reason: "the receipt names a different task" and "the
receipt names no task" are different findings, and a boolean loses the
second one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from ._receipt_types import ExecutionReceipt

#: The ``_meta`` key MCP Tasks use to associate a message with a task.
RELATED_TASK_META_KEY = "io.modelcontextprotocol/related-task"

TaskVerdict = Literal["bound", "conflict", "unsupported"]


@dataclass(frozen=True)
class TaskResult:
    """Outcome of a task check.

    ``verdict`` is one of:

    * ``"bound"``: the receipt carries ``taskId`` and it equals the expected
      task. Membership is established by the signed bytes.
    * ``"conflict"``: the receipt carries ``taskId`` and it differs. The
      receipt belongs to another task. Refuse and record the refusal.
    * ``"unsupported"``: the receipt carries no ``taskId``. Membership is
      neither established nor contradicted; the record cannot answer.

    Only ``"bound"`` is truthy.
    """

    verdict: TaskVerdict
    receipt_task_id: Optional[str]
    expected_task_id: str
    reason: str

    def __bool__(self) -> bool:
        return self.verdict == "bound"


def verify_receipt_task(
    receipt: ExecutionReceipt, expected_task_id: str
) -> TaskResult:
    """Three-valued task check. Does not verify the signature.

    Run ``verify_receipt_signature`` first. ``expected_task_id`` MUST be
    non-empty: an empty expectation would match nothing and read as a
    conflict, which misreports the caller's mistake as the issuer's.
    """
    if not expected_task_id:
        raise ValueError("expected_task_id must be a non-empty string")
    task_id = receipt.receipt_asserted.task_id
    if task_id is None:
        return TaskResult(
            "unsupported", None, expected_task_id,
            "receipt carries no taskId; membership neither established nor contradicted",
        )
    if task_id == expected_task_id:
        return TaskResult("bound", task_id, expected_task_id, "taskId matches the expected task")
    return TaskResult(
        "conflict", task_id, expected_task_id,
        f"receipt belongs to task {task_id!r}, not {expected_task_id!r}",
    )


def related_task_id(params: Any) -> Optional[str]:
    """The MCP related-task id carried by a request's params, or None.

    Reads ``params._meta["io.modelcontextprotocol/related-task"].taskId``
    exactly as the Tasks spec places it. Anything malformed (a non-object
    ``_meta``, a non-object related-task block, a missing, empty or non-string
    ``taskId``) yields None, so a receipt is emitted unbound and never bound
    to garbage. Unbound is the honest state for a call whose task membership
    could not be read.
    """
    if not isinstance(params, dict):
        return None
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        return None
    related = meta.get(RELATED_TASK_META_KEY)
    if not isinstance(related, dict):
        return None
    task_id = related.get("taskId")
    if isinstance(task_id, str) and task_id:
        return task_id
    return None
