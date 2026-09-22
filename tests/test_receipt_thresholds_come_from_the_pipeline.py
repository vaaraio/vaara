# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The thresholds a receipt publishes must be the ones the scorer decided on.

``tests/test_receipts.py`` hand-writes ``threshold_allow`` into the risk
record, so it passes whatever the pipeline does. The pipeline never wrote
them. Every receipt extracted from a live trail therefore fell back to the
legacy pair in ``receipts.py`` and claimed 0.4 / 0.7 for an engine deciding
at 0.55 / 0.85. Found 2026-09-22 on a live box (audit J14). This test goes
through ``InterceptionPipeline`` so the two ends have to meet.
"""
from __future__ import annotations

import pytest

from vaara.audit.receipts import extract_receipt
from vaara.audit.trail import AuditTrail
from vaara.pipeline import InterceptionPipeline
from vaara.scorer.adaptive import AdaptiveScorer
from vaara.taxonomy.actions import create_default_registry


@pytest.mark.parametrize("allow,deny", [(0.55, 0.85), (0.30, 0.55), (0.70, 0.92)])
def test_the_receipt_carries_the_thresholds_the_scorer_used(allow, deny):
    trail = AuditTrail()
    pipe = InterceptionPipeline(
        registry=create_default_registry(),
        scorer=AdaptiveScorer(threshold_allow=allow, threshold_deny=deny),
        trail=trail,
        enforce=True,
    )
    result = pipe.intercept(agent_id="a-1", tool_name="read_file", parameters={"path": "x"})
    receipt = extract_receipt(trail, result.action_id)
    assert receipt is not None
    assert receipt.commit.threshold_allow == pytest.approx(allow)
    assert receipt.commit.threshold_deny == pytest.approx(deny)


def test_the_risk_record_itself_names_both_thresholds():
    trail = AuditTrail()
    pipe = InterceptionPipeline(
        registry=create_default_registry(),
        scorer=AdaptiveScorer(threshold_allow=0.55, threshold_deny=0.85),
        trail=trail,
    )
    result = pipe.intercept(agent_id="a-1", tool_name="read_file", parameters={})
    risk = [r for r in trail._records
            if r.action_id == result.action_id and r.event_type.value == "risk_scored"]
    assert len(risk) == 1
    assert risk[0].data["threshold_allow"] == pytest.approx(0.55)
    assert risk[0].data["threshold_deny"] == pytest.approx(0.85)
