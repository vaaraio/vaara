"""Tests for synthetic trace generation (cold-start pre-calibration)."""

import tempfile
from pathlib import Path

import pytest

from vaara.pipeline import InterceptionPipeline
from vaara.sandbox.trace_gen import (
    ADVERSARIAL_ARCHETYPE,
    BENIGN_ARCHETYPE,
    CARELESS_ARCHETYPE,
    AgentArchetype,
    SyntheticTrace,
    TraceGenerator,
    TraceStep,
)


class TestTraceGenerator:
    def test_generate_default_count(self):
        gen = TraceGenerator(seed=42)
        traces = gen.generate(n_traces=10)
        assert len(traces) == 10

    def test_generate_zero_traces(self):
        gen = TraceGenerator(seed=42)
        traces = gen.generate(n_traces=0)
        assert len(traces) == 0

    def test_traces_have_steps(self):
        gen = TraceGenerator(seed=42)
        traces = gen.generate(n_traces=5)
        for trace in traces:
            assert len(trace.steps) > 0

    def test_deterministic_with_seed(self):
        gen1 = TraceGenerator(seed=42)
        gen2 = TraceGenerator(seed=42)
        t1 = gen1.generate(n_traces=5)
        t2 = gen2.generate(n_traces=5)
        assert len(t1) == len(t2)
        for a, b in zip(t1, t2):
            assert a.agent_id == b.agent_id
            assert len(a.steps) == len(b.steps)

    def test_different_seeds_different_traces(self):
        gen1 = TraceGenerator(seed=42)
        gen2 = TraceGenerator(seed=99)
        t1 = gen1.generate(n_traces=5)
        t2 = gen2.generate(n_traces=5)
        # At least some traces should differ
        different = any(
            a.agent_id != b.agent_id or len(a.steps) != len(b.steps)
            for a, b in zip(t1, t2)
        )
        assert different

    def test_archetype_distribution(self):
        """Most traces should be benign (60% weight)."""
        gen = TraceGenerator(seed=42)
        traces = gen.generate(n_traces=100)
        benign_count = sum(1 for t in traces if t.archetype == "benign")
        assert benign_count > 40  # Should be ~60 but allow variance

    def test_outcome_range(self):
        """All outcomes should be in [0, 1]."""
        gen = TraceGenerator(seed=42)
        traces = gen.generate(n_traces=20)
        for trace in traces:
            for step in trace.steps:
                assert 0.0 <= step.outcome_severity <= 1.0

    def test_confidence_range(self):
        gen = TraceGenerator(seed=42)
        traces = gen.generate(n_traces=20)
        for trace in traces:
            for step in trace.steps:
                if step.agent_confidence is not None:
                    assert 0.0 <= step.agent_confidence <= 1.0


class TestArchetypeRiskDistributions:
    def test_benign_lower_risk(self):
        """Benign archetype should produce lower mean outcomes."""
        gen = TraceGenerator(
            archetypes=[BENIGN_ARCHETYPE],
            archetype_weights=[1.0],
            seed=42,
        )
        traces = gen.generate(n_traces=50)
        mean = sum(t.mean_outcome for t in traces) / len(traces)
        assert mean < 0.2  # Beta(2, 20) mean = 0.09

    def test_adversarial_higher_risk(self):
        """Adversarial archetype should produce higher mean outcomes."""
        gen = TraceGenerator(
            archetypes=[ADVERSARIAL_ARCHETYPE],
            archetype_weights=[1.0],
            seed=42,
        )
        traces = gen.generate(n_traces=50)
        mean = sum(t.mean_outcome for t in traces) / len(traces)
        assert mean > 0.4  # Beta(5, 3) mean = 0.625

    def test_careless_middle_risk(self):
        gen = TraceGenerator(
            archetypes=[CARELESS_ARCHETYPE],
            archetype_weights=[1.0],
            seed=42,
        )
        traces = gen.generate(n_traces=50)
        mean = sum(t.mean_outcome for t in traces) / len(traces)
        assert 0.15 < mean < 0.5


class TestSequenceInsertion:
    def test_adversarial_has_sequences(self):
        """Adversarial traces (p=0.5) should contain sequence parts."""
        gen = TraceGenerator(
            archetypes=[ADVERSARIAL_ARCHETYPE],
            archetype_weights=[1.0],
            seed=42,
        )
        traces = gen.generate(n_traces=20)
        has_seq = any(
            any(s.is_sequence_part for s in t.steps)
            for t in traces
        )
        assert has_seq

    def test_benign_rarely_has_sequences(self):
        """Benign traces (p=0.02) should rarely contain sequence parts."""
        gen = TraceGenerator(
            archetypes=[BENIGN_ARCHETYPE],
            archetype_weights=[1.0],
            seed=42,
        )
        traces = gen.generate(n_traces=50)
        seq_count = sum(
            1 for t in traces
            if any(s.is_sequence_part for s in t.steps)
        )
        assert seq_count < 10  # Should be ~1 out of 50


class TestPreCalibration:
    def test_pre_calibrate_updates_scorer(self):
        gen = TraceGenerator(seed=42)
        traces = gen.generate(n_traces=10)

        pipeline = InterceptionPipeline()
        assert pipeline.scorer.calibration_size == 0
        assert pipeline.scorer.mwu_update_count == 0

        stats = gen.pre_calibrate(pipeline, traces)

        assert stats["traces_processed"] == 10
        assert stats["steps_processed"] > 0
        assert stats["outcomes_reported"] > 0
        assert stats["calibration_size_after"] > 0
        assert stats["mwu_updates_after"] > 0

    def test_pre_calibrate_achieves_calibration(self):
        """With enough traces, the scorer should become calibrated."""
        gen = TraceGenerator(seed=42)
        traces = gen.generate(n_traces=50)

        pipeline = InterceptionPipeline()
        stats = gen.pre_calibrate(pipeline, traces)

        assert stats["is_calibrated"] is True
        assert stats["calibration_size_after"] >= 30

    def test_pre_calibrate_decisions_distributed(self):
        """Pre-calibration should produce a mix of allow/deny/escalate."""
        gen = TraceGenerator(seed=42)
        traces = gen.generate(n_traces=50)

        pipeline = InterceptionPipeline()
        stats = gen.pre_calibrate(pipeline, traces)

        # Should have at least two different decision types
        decision_types = sum(
            1 for v in stats["decisions"].values() if v > 0
        )
        assert decision_types >= 2

    def test_empty_traces_no_crash(self):
        gen = TraceGenerator(seed=42)
        pipeline = InterceptionPipeline()
        stats = gen.pre_calibrate(pipeline, [])
        assert stats["traces_processed"] == 0


class TestSaveLoad:
    def test_save_jsonl(self):
        gen = TraceGenerator(seed=42)
        traces = gen.generate(n_traces=5)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            count = gen.save_jsonl(traces, path)
            assert count > 0
            assert path.exists()

            lines = path.read_text().strip().split("\n")
            assert len(lines) == count
        finally:
            path.unlink(missing_ok=True)

    def test_load_jsonl_roundtrip(self):
        gen = TraceGenerator(seed=42)
        traces = gen.generate(n_traces=3)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = Path(f.name)

        try:
            count = gen.save_jsonl(traces, path)
            loaded = gen.load_jsonl(path)
            assert len(loaded) == count

            # Check first step
            first_trace_step = traces[0].steps[0]
            assert loaded[0].action_name == first_trace_step.action_name
            assert loaded[0].agent_id == first_trace_step.agent_id
            assert abs(loaded[0].outcome_severity - first_trace_step.outcome_severity) < 0.001
        finally:
            path.unlink(missing_ok=True)


class TestSyntheticTrace:
    def test_mean_outcome(self):
        trace = SyntheticTrace(agent_id="test", archetype="test")
        trace.steps = [
            TraceStep("a", "test", 0.2, 0.5, "test"),
            TraceStep("b", "test", 0.4, 0.5, "test"),
            TraceStep("c", "test", 0.6, 0.5, "test"),
        ]
        assert abs(trace.mean_outcome - 0.4) < 0.001

    def test_max_outcome(self):
        trace = SyntheticTrace(agent_id="test", archetype="test")
        trace.steps = [
            TraceStep("a", "test", 0.2, 0.5, "test"),
            TraceStep("b", "test", 0.9, 0.5, "test"),
            TraceStep("c", "test", 0.1, 0.5, "test"),
        ]
        assert abs(trace.max_outcome - 0.9) < 0.001

    def test_empty_trace_stats(self):
        trace = SyntheticTrace(agent_id="test", archetype="test")
        assert trace.mean_outcome == 0.0
        assert trace.max_outcome == 0.0
