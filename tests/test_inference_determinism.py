# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Tests for the determinism self-test gate.

The gate exists to keep a ``tier: "replay"`` label from ever outrunning the
evidence. These tests pin the honest behaviour: agreement -> replay with a
digest; any disagreement -> integrity with no digest; a single sample is
refused outright.
"""

from __future__ import annotations

import pytest

from vaara.attestation.inference import (
    determinism_verdict,
    honest_tier,
    make_output_commitment,
)


def _out(content: str, thinking: str = "") -> dict:
    return {"content": content, "thinking": thinking}


def test_all_identical_is_reproducible_with_digest():
    outs = [_out("the answer is 42. DONE")] * 4
    v = determinism_verdict(outs)
    assert v.reproducible is True
    assert v.k == 4
    assert v.distinct == 1
    # digest matches the shipping commitment a receipt would bind
    assert v.digest == make_output_commitment(outs[0]).projection_digest
    assert honest_tier(v) == "replay"


def test_one_divergent_sample_forces_integrity():
    # Mirrors the live finding: stable prefix, late single-token flicker.
    base = "x" * 420
    outs = [_out(base + "alpha"), _out(base + "alpha"),
            _out(base + "beta"), _out(base + "alpha")]
    v = determinism_verdict(outs)
    assert v.reproducible is False
    assert v.distinct == 2
    assert v.digest is None
    assert v.stable_prefix_chars == 420  # divergence located at char 420
    assert honest_tier(v) == "integrity"


def test_thinking_difference_breaks_reproducibility():
    # Same visible content, different hidden reasoning => different commitment.
    outs = [_out("same", thinking="path A"), _out("same", thinking="path B")]
    v = determinism_verdict(outs)
    assert v.reproducible is False
    assert honest_tier(v) == "integrity"


def test_all_distinct():
    outs = [_out("a"), _out("b"), _out("c")]
    v = determinism_verdict(outs)
    assert v.distinct == 3
    assert v.reproducible is False
    assert v.stable_prefix_chars == 0


def test_single_sample_is_refused():
    # A lone sample cannot witness reproducibility; passing it would be the
    # exact dishonesty the gate prevents.
    with pytest.raises(ValueError):
        determinism_verdict([_out("only one")])


def test_empty_is_refused():
    with pytest.raises(ValueError):
        determinism_verdict([])


def test_common_prefix_handles_length_difference():
    # One output is a strict prefix of the other => prefix == shorter length.
    outs = [_out("hello"), _out("hello world")]
    v = determinism_verdict(outs)
    assert v.reproducible is False
    assert v.stable_prefix_chars == len("hello")
