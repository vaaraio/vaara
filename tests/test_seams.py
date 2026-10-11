# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every key a consumer reads off a dict, its producer writes, or the gap has a reason.

#926 fixed ExternalScorer answering `decision` while the pipeline read
`action` with a default, so every remote verdict fell through to the default.
Five other scorers wrote `action`; a check over the union of producers passed.
tests/seams.py compares each producer alone. A new read that some producer
never writes fails here until it is fixed or listed below with the reason the
gap is meant.
"""

from __future__ import annotations

import functools
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import seams  # noqa: E402

_SCORERS = (
    "vaara.scorer.adaptive:AdaptiveScorer.evaluate",
    "vaara.scorer.mc_dropout_gate:MCDropoutGateScorer.evaluate",
    "vaara.scorer.stacked_gate:StackedGateScorer.evaluate",
    "vaara.scorer.trained_gate:TrainedGateScorer.evaluate",
)
_EXTERNAL = "vaara.scorer.composition:ExternalScorer.evaluate"
_INTERCEPT = "vaara.pipeline:InterceptionPipeline.intercept"
_SCORE_ROUTE = "vaara.server.routes:register.<locals>.score"

#: (producer, consumer, key) -> why the producer may leave the key out.
ALLOWED: dict[tuple[str, str, str], str] = {}
for _p in (*_SCORERS, _EXTERNAL):
    ALLOWED[(_p, _INTERCEPT, "policy_id")] = (
        "set by a policy-backed scorer only; the pipeline records it when present")
    ALLOWED[(_p, _INTERCEPT, "modified_parameters")] = (
        "set by a policy that rewrites arguments; absent means run as asked")
for _p in _SCORERS:
    ALLOWED[(_p, _INTERCEPT, "unavailable_input")] = (
        "ExternalScorer's fail-closed deny names what was unreachable; a local "
        "scorer has no remote input to lose")
    ALLOWED[(_p, _INTERCEPT, "violation_type")] = (
        "ExternalScorer's fail-closed deny only")
for _c in (_SCORE_ROUTE, "vaara.server.routes:register"):
    ALLOWED[(_EXTERNAL, _c, "evaluation_ms")] = (
        "the remote answer carries its own timing or none; the route times the call itself")


def _gaps(findings) -> set[tuple[str, str, str]]:
    return {
        (f["producer"], f["consumer"], k)
        for f in findings if f["producer_known"]
        for k in f["missing"]
    }


@functools.cache
def _tree_gaps() -> frozenset[tuple[str, str, str]]:
    return frozenset(_gaps(seams.analyse()))


def test_every_seam_gap_is_listed_with_a_reason():
    found = _tree_gaps()
    new = sorted(found - set(ALLOWED))
    assert not new, "a consumer reads a key its producer never writes:\n" + "\n".join(
        f"  {p} -> {c}: {k!r}" for p, c, k in new)


def test_the_allow_list_holds_no_stale_entry():
    stale = sorted(set(ALLOWED) - _tree_gaps())
    assert not stale, f"no longer a gap, drop from ALLOWED: {stale}"


def test_the_guard_catches_the_926_shape(tmp_path):
    """One producer writes `decision`, its siblings `action`; the consumer reads `action`."""
    pkg = tmp_path / "src" / "demo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "scorers.py").write_text(textwrap.dedent("""
        class Local:
            def evaluate(self, ctx):
                return {"action": "allow", "reason": "ok"}

        class Remote:
            def evaluate(self, ctx):
                return {"decision": "allow", "reason": "remote ok"}
    """))
    (pkg / "pipeline.py").write_text(textwrap.dedent("""
        class Pipeline:
            def intercept(self, ctx):
                result = self.scorer.evaluate(ctx)
                return result.get("action", "deny"), result["reason"]
    """))
    gaps = _gaps(seams.analyse(pkg))
    assert ("demo.scorers:Remote.evaluate", "demo.pipeline:Pipeline.intercept", "action") in gaps
    assert not any(p == "demo.scorers:Local.evaluate" for p, _, _ in gaps)
