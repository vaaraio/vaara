# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Determinism self-test gate for inference replay receipts.

Internal module. Public surface is re-exported from
``vaara.attestation.inference``.

The receipt design reserves ``tier: "replay"`` for outputs that are
byte-reproducible. The 2026-06-15 determinism probe (session 92) established
that qwen3 on stock ollama/ROCm is NOT byte-reproducible at temp=0 greedy: a
rock-stable leading prefix is followed by a single near-tie token that flickers
between two continuations, and the effect survives GPU -> CPU -> single-thread.
See ``research/determinism_findings_20260615.md``.

The honest consequence: never *assume* reproducibility from sampling params.
Instead, measure it. This module turns the empirical test into a reusable gate:
sample the same request K times, and only let a receipt claim ``replay`` if all
K outputs commit to the same digest. Otherwise the honest tier is ``integrity``.

The core (``determinism_verdict`` / ``honest_tier``) is a pure function over an
already-collected list of outputs, so it is unit-testable with no inference
server. Calling the model K times is the caller's (proxy's) job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._inference_emit import make_output_commitment


@dataclass(frozen=True)
class DeterminismVerdict:
    """Result of re-running one request K times and comparing the outputs.

    ``reproducible`` is the gate: True iff all K outputs commit to the same
    digest. ``digest`` carries that shared commitment when reproducible, else
    ``None``. ``stable_prefix_chars`` is a diagnostic -- the length of the
    longest common leading ``content`` prefix across all K samples -- so a near
    miss (long stable prefix, late divergence) is distinguishable from broad
    drift.
    """

    k: int
    reproducible: bool
    distinct: int
    stable_prefix_chars: int
    digest: Optional[str]


def _output_digest(output: dict[str, Any]) -> str:
    """Commit one assembled output with the SHIPPING commitment.

    Using ``make_output_commitment`` (not a parallel hash) guarantees the gate
    measures exactly the bytes a receipt would bind, so a verdict of
    ``reproducible`` means the receipt's own ``outputCommitment`` would match.
    """
    return make_output_commitment(output).projection_digest


def _common_prefix_len(strings: list[str]) -> int:
    """Length of the longest common leading prefix across all strings."""
    if not strings:
        return 0
    shortest = min(strings, key=len)
    for i, ch in enumerate(shortest):
        if any(s[i] != ch for s in strings):
            return i
    return len(shortest)


def determinism_verdict(outputs: list[dict[str, Any]]) -> DeterminismVerdict:
    """Judge whether K re-runs of a request agree, byte-for-byte.

    ``outputs`` is the list of assembled output objects (e.g.
    ``{"content": ..., "thinking": ...}``) from K identical calls. Requires at
    least two samples -- a single sample cannot witness reproducibility, and
    silently passing it would be the exact dishonesty this gate exists to stop.
    """
    if len(outputs) < 2:
        raise ValueError(
            "determinism_verdict needs at least 2 samples to witness "
            f"reproducibility; got {len(outputs)}"
        )
    digests = [_output_digest(o) for o in outputs]
    distinct = len(set(digests))
    reproducible = distinct == 1
    # Diagnostic prefix over the user-visible content only.
    prefix = _common_prefix_len([str(o.get("content", "")) for o in outputs])
    return DeterminismVerdict(
        k=len(outputs),
        reproducible=reproducible,
        distinct=distinct,
        stable_prefix_chars=prefix,
        digest=digests[0] if reproducible else None,
    )


def honest_tier(verdict: DeterminismVerdict) -> str:
    """Map a verdict to the honest receipt tier.

    ``replay`` only when the re-runs actually agreed; otherwise ``integrity``,
    which makes no determinism claim. This is the single chokepoint that keeps a
    ``replay`` label from ever outrunning the evidence for it.
    """
    return "replay" if verdict.reproducible else "integrity"
