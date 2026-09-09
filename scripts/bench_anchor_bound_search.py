# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure the anchor bound search in ``AuditTrail.verify_segment``.

The record lookup is O(1) since the ``_pos_by_record_id`` index. What is left
that grows with the trail is the anchor handling: ``list(self._anchors)`` under
the lock, then a single pass over that list to find the tightest bounding pair.
At the default 32-record anchor cadence a trail carries len/32 anchors, so both
terms grow linearly with the trail.

This script answers the only question worth answering before adding a structure
to maintain: at what trail size does that cost stop being free? Run it before
and after any change and compare the same rows.

Usage:  .venv/bin/python scripts/bench_anchor_bound_search.py [sizes...]
"""

from __future__ import annotations

import sys
import time

from tests.test_timeanchor import _local_tsa_client
from vaara.audit.trail import AuditTrail
from vaara.taxonomy.actions import (
    ActionCategory,
    ActionRequest,
    ActionType,
    BlastRadius,
    Reversibility,
)

_ACTION = ActionType(
    name="bench",
    category=ActionCategory.DATA,
    reversibility=Reversibility.FULLY,
    blast_radius=BlastRadius.LOCAL,
)

ANCHOR_EVERY = 32
REPEATS = 200


def _build(size: int, client) -> tuple[AuditTrail, str]:
    """A trail of ``size`` records with an anchor every 32.

    Returns the trail and the record id of a record near the middle, which is
    the worst realistic case for a scan that keeps the tightest pair: it has
    anchors on both sides and cannot stop early.
    """
    trail = AuditTrail()
    for i in range(size):
        trail.record_action_requested(ActionRequest(
            action_type=_ACTION, tool_name="t", agent_id="a", parameters={"i": i},
        ))
        if (i + 1) % ANCHOR_EVERY == 0:
            trail.anchor_head(client)
    # record_action_requested returns an action id, not a record id, and one
    # action can hold several records. Read the middle record's own id.
    return trail, trail._records[size // 2].record_id


def main(sizes: list[int]) -> None:
    client = _local_tsa_client()
    print(f"{'records':>10} {'anchors':>9} {'verify_segment':>16} {'per anchor':>12}")
    for size in sizes:
        build_start = time.perf_counter()
        trail, rid = _build(size, client)
        built = time.perf_counter() - build_start
        n_anchors = len(trail.anchors)

        # Warm once so the first call does not carry import cost.
        trail.verify_segment(record_id=rid)

        start = time.perf_counter()
        for _ in range(REPEATS):
            result = trail.verify_segment(record_id=rid)
        elapsed = (time.perf_counter() - start) / REPEATS * 1000

        assert result.ok, result.reason
        per_anchor = elapsed / n_anchors * 1000 if n_anchors else 0.0
        print(f"{size:>10,} {n_anchors:>9,} {elapsed:>13.3f} ms {per_anchor:>9.3f} us"
              f"   (built in {built:.1f}s)")


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]] or [5_000, 50_000, 150_000, 500_000]
    main(args)
