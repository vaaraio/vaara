# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Record before effect: an allowed call whose record did not land does not run.

``InterceptionPipeline.intercept`` does not raise when the audit store refuses
a write. It counts the failure, returns its verdict, and an allowed call runs
with nothing recorded while the chain on disk stays contiguous (ran
2026-10-24, six refused writes, both calls ``allowed=True``). The LLM API
proxy closes that gap for itself in ``_llm_proxy_app._intercept_recorded``,
written after it forwarded 169 prompts in 68 minutes with the trail logging
THE TRAIL IS NOT RECORDING on every one. This module is the same check for
the MCP proxy and the shell proxy, and its failure counter is the one the
Claude Code hook compares on its ``mcp__*`` path.

Before refusing, the gate repairs the store and tries once more, because
every agent that talks through the proxy, including the one that would fix
the trail, stops while it refuses. Repair reads the whole file, so it runs at
most once per interval; between attempts a failing store is refused straight
away. A call the pipeline did not allow is not this gate's business: it is
already stopped.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Repair reads the whole trail file, so it is tried at most this often.
REPAIR_INTERVAL = 30.0

#: Said when the record of the decision did not reach the store.
UNRECORDED = "the decision record did not reach the audit store"


def trail_failures(pipeline: Any) -> int:
    """Writes the pipeline's trail has failed since it was created."""
    count = getattr(getattr(pipeline, "trail", None), "persistence_failures", 0)
    return count if isinstance(count, int) and not isinstance(count, bool) else 0


def trail_path(pipeline: Any) -> Optional[Any]:
    backend = getattr(pipeline.trail, "_chain_backend", lambda: None)()
    if backend is None:
        return None
    return getattr(backend, "db_path", None) or getattr(backend, "_db_path", None)


def refusal(pipeline: Any, why: str, *, who: str) -> str:
    """What an operator reads when ``who`` refuses a call it cannot record."""
    path = trail_path(pipeline)
    where = f" at {path}" if path else ""
    return (
        f"{why}. The audit trail{where} is not recording, and {who} does not "
        "run a call it cannot record. It retries the repair on a later call "
        "by itself. By hand: `vaara trail repair --db "
        f"{path or '<trail>'}`. Do not delete the file, it is the evidence. "
        "To run unrecorded on purpose, restart with --fail-open."
    )


class RecordGate:
    """``pipeline.intercept`` plus the question of whether the record landed."""

    def __init__(self) -> None:
        self._last_repair = 0.0

    def intercept(self, pipeline: Any, **kwargs: Any) -> tuple[Any, Optional[str]]:
        """``(result, None)`` when the records landed, else ``(result, why)``.

        An exception from the pipeline is not caught: a call that raised was
        never allowed.
        """
        before = trail_failures(pipeline)
        result = pipeline.intercept(**kwargs)
        if trail_failures(pipeline) == before:
            return result, None

        why = UNRECORDED
        now = time.time()
        repair = getattr(pipeline.trail, "repair_store", None)
        if repair is None or now - self._last_repair < REPAIR_INTERVAL:
            return result, why
        self._last_repair = now
        try:
            report = repair()
        except Exception as exc:  # repair must never take the proxy down
            logger.error("trail repair raised %s: %s", type(exc).__name__, exc)
            return result, f"{why}, and repair raised {type(exc).__name__}"
        if report is None or not report.ok:
            detail = getattr(report, "error", None) or "no repairable store"
            logger.error("trail repair failed: %s", detail)
            return result, f"{why}, and repair failed ({detail})"
        logger.warning(
            "trail repaired by %s after a failed write; %d record(s) lost%s",
            report.method, len(report.lost_seqs),
            f", damaged file kept at {report.damaged_copy}"
            if report.damaged_copy else "",
        )
        before = trail_failures(pipeline)
        result = pipeline.intercept(**kwargs)
        if trail_failures(pipeline) == before:
            return result, None
        return result, f"{why}, and still after repair by {report.method}"
