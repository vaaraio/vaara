"""Synthetic trace generation for cold-start training data.

The MWU scorer (§4) starts cold — uniform expert weights, no calibration
data.  During cold start (§8.3), the system correctly routes most actions
through ESCALATE for human review.  But this is expensive: at 10 actions/hour,
calibration takes ~3 hours of human oversight.

This module generates synthetic action traces with KNOWN risk profiles,
enabling pre-calibration of the conformal interval and warm-starting of
MWU weights before any real agent connects.  The key insight:

**Synthetic traces don't need to be realistic — they need to be
representative of the RISK DISTRIBUTION.**

We generate traces for three agent archetypes:
1. **Benign agent** — reads data, makes small transfers, follows protocols.
   Ground-truth outcome ~ Beta(2, 20) → mostly safe, rare minor issues.
2. **Careless agent** — takes risky actions without checking, makes errors.
   Ground-truth outcome ~ Beta(3, 7) → moderate risk, frequent minor issues.
3. **Adversarial agent** — executes known attack patterns, probes boundaries.
   Ground-truth outcome ~ Beta(5, 3) → high risk, frequent severe outcomes.

Each trace is a sequence of (action, outcome) pairs that can be fed
directly to the scorer's learning interface.

The generation process is:
1. Sample an agent archetype
2. Sample a sequence length (Poisson-distributed)
3. For each step, sample an action type weighted by archetype behavior
4. With probability p_sequence, insert a known dangerous sequence pattern
5. Sample ground-truth outcome from the archetype's Beta distribution
6. Record the trace

This produces a calibration dataset where:
- The conformal calibrator sees the full range of risk levels
- MWU experts see outcomes for all signal combinations
- Sequence patterns appear at realistic frequencies

After pre-calibration, the system transitions from rule-based to adaptive
mode immediately when a real agent connects, rather than after 30 outcomes.

Reference: Barber et al. (2023), "Conformal prediction beyond exchangeability"
— synthetic calibration data is valid for conformal guarantees as long as
the calibration scores are marginally exchangeable with test scores.
Since we generate from the same score function, this holds.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from vaara._sanitize import strict_json_dumps
from vaara.pipeline import InterceptionPipeline
from vaara.taxonomy.actions import create_default_registry

# ── Agent archetypes ─────────────────────────────────────────────────────

@dataclass
class AgentArchetype:
    """Behavioral profile for synthetic agent generation."""
    name: str
    # Beta distribution parameters for outcome severity
    alpha: float  # Shape parameter 1
    beta: float   # Shape parameter 2
    # Action type weights — higher = more likely to choose this action
    action_weights: dict[str, float] = field(default_factory=dict)
    # Probability of inserting a dangerous sequence per trace
    sequence_probability: float = 0.0
    # How many actions per trace (Poisson lambda)
    trace_length_lambda: float = 10.0
    # Confidence reporting behavior
    confidence_mean: float = 0.5
    confidence_std: float = 0.2
    description: str = ""


BENIGN_ARCHETYPE = AgentArchetype(
    name="benign",
    alpha=2.0, beta=20.0,  # Mean ~0.09 — mostly safe
    action_weights={
        "data.read": 10.0,
        "data.write": 3.0,
        "comm.api_call": 5.0,
        "tx.approve": 1.0,
        "infra.scale": 2.0,
        "comm.send_email": 2.0,
    },
    sequence_probability=0.02,
    trace_length_lambda=15.0,
    confidence_mean=0.6,
    confidence_std=0.15,
    description="Well-behaved agent — reads data, makes API calls, occasional writes",
)

CARELESS_ARCHETYPE = AgentArchetype(
    name="careless",
    alpha=3.0, beta=7.0,  # Mean ~0.30 — moderate risk
    action_weights={
        "data.read": 5.0,
        "data.write": 5.0,
        "data.delete": 2.0,
        "tx.swap": 3.0,
        "tx.transfer": 2.0,
        "infra.deploy": 2.0,
        "infra.config_change": 3.0,
        "id.create_key": 1.0,
    },
    sequence_probability=0.15,
    trace_length_lambda=20.0,
    confidence_mean=0.7,  # Overconfident
    confidence_std=0.1,
    description="Careless agent — takes risky actions, overconfident, triggers patterns accidentally",
)

ADVERSARIAL_ARCHETYPE = AgentArchetype(
    name="adversarial",
    alpha=5.0, beta=3.0,  # Mean ~0.625 — high risk
    action_weights={
        "data.read": 3.0,
        "data.export": 5.0,
        "data.delete": 3.0,
        "tx.approve": 4.0,
        "tx.swap": 5.0,
        "tx.transfer": 5.0,
        "id.create_key": 3.0,
        "id.grant_permission": 3.0,
        "gov.vote": 2.0,
        "gov.execute_proposal": 2.0,
        "phy.safety_override": 1.0,
    },
    sequence_probability=0.5,
    trace_length_lambda=8.0,
    confidence_mean=0.9,  # Highly overconfident (gaming the system)
    confidence_std=0.05,
    description="Adversarial agent — executes attack patterns, games confidence reporting",
)

DEFAULT_ARCHETYPES = [BENIGN_ARCHETYPE, CARELESS_ARCHETYPE, ADVERSARIAL_ARCHETYPE]


# ── Trace generation ────────────────────────────────────────────────────

@dataclass
class TraceStep:
    """One step in a synthetic trace."""
    action_name: str
    agent_id: str
    outcome_severity: float
    agent_confidence: Optional[float]
    archetype: str
    is_sequence_part: bool = False


@dataclass
class SyntheticTrace:
    """A complete synthetic agent trace with ground-truth outcomes."""
    agent_id: str
    archetype: str
    steps: list[TraceStep] = field(default_factory=list)

    @property
    def mean_outcome(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.outcome_severity for s in self.steps) / len(self.steps)

    @property
    def max_outcome(self) -> float:
        if not self.steps:
            return 0.0
        return max(s.outcome_severity for s in self.steps)


class TraceGenerator:
    """Generate synthetic action traces for cold-start pre-calibration.

    Usage::

        gen = TraceGenerator()
        traces = gen.generate(n_traces=100)

        # Feed traces to a pipeline for pre-calibration
        gen.pre_calibrate(pipeline, traces)

        # Or save for later use
        gen.save_jsonl(traces, Path("traces.jsonl"))
    """

    def __init__(
        self,
        archetypes: Optional[list[AgentArchetype]] = None,
        archetype_weights: Optional[list[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._archetypes = archetypes or DEFAULT_ARCHETYPES
        self._archetype_weights = archetype_weights or [0.6, 0.3, 0.1]
        self._rng = random.Random(seed)
        # Get all registered action names for validation
        registry = create_default_registry()
        self._known_actions = set(registry.all_types.keys())

    def generate(self, n_traces: int = 100) -> list[SyntheticTrace]:
        """Generate n_traces synthetic traces."""
        traces = []
        for i in range(n_traces):
            archetype = self._rng.choices(
                self._archetypes,
                weights=self._archetype_weights,
                k=1,
            )[0]
            trace = self._generate_trace(f"synth-{archetype.name}-{i:04d}", archetype)
            traces.append(trace)
        return traces

    def _generate_trace(self, agent_id: str, archetype: AgentArchetype) -> SyntheticTrace:
        """Generate a single trace from an archetype."""
        trace = SyntheticTrace(agent_id=agent_id, archetype=archetype.name)

        # Sample trace length from Poisson (std = sqrt(lambda) matches Poisson variance)
        import math as _math
        n_steps = max(1, int(self._rng.gauss(archetype.trace_length_lambda, _math.sqrt(archetype.trace_length_lambda))))

        # Decide whether to insert a sequence pattern
        insert_sequence = self._rng.random() < archetype.sequence_probability
        sequence_start = self._rng.randint(0, max(0, n_steps - 3)) if insert_sequence else -1

        # Known sequences to pick from
        sequences = [
            ("data.read", "data.export", "data.delete"),
            ("id.create_key", "id.grant_permission"),
            ("tx.approve", "tx.swap", "tx.transfer"),
        ]
        chosen_sequence = self._rng.choice(sequences) if insert_sequence else ()

        action_names = list(archetype.action_weights.keys())
        action_probs = list(archetype.action_weights.values())

        seq_idx = 0
        for step_i in range(n_steps):
            # Insert sequence actions at the chosen position
            if insert_sequence and sequence_start <= step_i < sequence_start + len(chosen_sequence):
                action_name = chosen_sequence[seq_idx]
                seq_idx += 1
                is_seq = True
                # Sequence actions have elevated outcomes
                outcome = min(1.0, self._beta_sample(archetype.alpha + 2, archetype.beta))
            else:
                action_name = self._rng.choices(action_names, weights=action_probs, k=1)[0]
                is_seq = False
                outcome = self._beta_sample(archetype.alpha, archetype.beta)

            # Sample agent-reported confidence
            confidence = max(0.0, min(1.0,
                self._rng.gauss(archetype.confidence_mean, archetype.confidence_std)
            ))

            trace.steps.append(TraceStep(
                action_name=action_name,
                agent_id=agent_id,
                outcome_severity=outcome,
                agent_confidence=confidence,
                archetype=archetype.name,
                is_sequence_part=is_seq,
            ))

        return trace

    def _beta_sample(self, a: float, b: float) -> float:
        """Sample from Beta(a, b) distribution."""
        return self._rng.betavariate(a, b)

    def pre_calibrate(
        self,
        pipeline: InterceptionPipeline,
        traces: list[SyntheticTrace],
    ) -> dict:
        """Feed synthetic traces through a pipeline for pre-calibration.

        Returns calibration statistics.
        """
        stats = {
            "traces_processed": 0,
            "steps_processed": 0,
            "decisions": {"allow": 0, "deny": 0, "escalate": 0},
            "outcomes_reported": 0,
            "calibration_size_before": pipeline.scorer.calibration_size,
        }

        for trace in traces:
            for step in trace.steps:
                # Intercept the action
                result = pipeline.intercept(
                    agent_id=step.agent_id,
                    tool_name=step.action_name,
                    agent_confidence=step.agent_confidence,
                )
                stats["decisions"][result.decision] += 1
                stats["steps_processed"] += 1

                # Report the outcome (this updates MWU + conformal)
                pipeline.report_outcome(
                    action_id=result.action_id,
                    outcome_severity=step.outcome_severity,
                    description=f"synthetic_{step.archetype}",
                )
                stats["outcomes_reported"] += 1

            stats["traces_processed"] += 1

        stats["calibration_size_after"] = pipeline.scorer.calibration_size
        stats["mwu_updates_after"] = pipeline.scorer.mwu_update_count
        stats["is_calibrated"] = pipeline.scorer.is_calibrated

        return stats

    @staticmethod
    def save_jsonl(traces: list[SyntheticTrace], path: Path) -> int:
        """Save traces to JSONL format for reproducible pre-calibration."""
        # strict_json_dumps matches the rest of the codebase's wire-
        # compliance policy: round() on NaN returns NaN, which json.dumps
        # would emit as the non-RFC `NaN` token; strict dumps scrubs it
        # to null so calibration JSONL consumed by strict parsers stays
        # readable. encoding="utf-8" guarantees cross-platform portability.
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for trace in traces:
                for step in trace.steps:
                    record = {
                        "agent_id": step.agent_id,
                        "action_name": step.action_name,
                        "outcome_severity": round(step.outcome_severity, 6),
                        "agent_confidence": (
                            round(step.agent_confidence, 6)
                            if step.agent_confidence is not None
                            else None
                        ),
                        "archetype": step.archetype,
                        "is_sequence_part": step.is_sequence_part,
                    }
                    f.write(strict_json_dumps(record) + "\n")
                    count += 1
        return count

    @staticmethod
    def load_jsonl(path: Path) -> list[TraceStep]:
        """Load trace steps from JSONL.

        Skips blank lines and logs (then skips) malformed / incomplete
        records rather than killing the whole load — JSONL files are
        often concatenated from multiple sources and a single bad line
        shouldn't discard the rest of a calibration corpus.
        """
        import logging as _logging
        _logger = _logging.getLogger(__name__)

        steps = []
        # encoding="utf-8" mirrors save_jsonl — platform default (cp1252 on
        # Windows, locale-dependent on Linux) can corrupt non-ASCII agent
        # names and archetype descriptions during round-trip, breaking the
        # synthetic-calibration invariant that save(load(x)) == x.
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    _logger.warning(
                        "trace_gen.load_jsonl: skipping malformed JSON at %s:%d (%s)",
                        path, lineno, exc,
                    )
                    continue
                try:
                    steps.append(TraceStep(
                        action_name=data["action_name"],
                        agent_id=data["agent_id"],
                        outcome_severity=data["outcome_severity"],
                        agent_confidence=data.get("agent_confidence"),
                        archetype=data.get("archetype", "unknown"),
                        is_sequence_part=data.get("is_sequence_part", False),
                    ))
                except (KeyError, TypeError) as exc:
                    _logger.warning(
                        "trace_gen.load_jsonl: skipping incomplete record at %s:%d (%s)",
                        path, lineno, exc,
                    )
                    continue
        return steps
