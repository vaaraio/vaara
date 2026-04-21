"""Adaptive risk scorer with conformal prediction and MWU learning.

Complements declarative policy engines (rule-based / YAML / Rego) by
adding calibrated uncertainty to per-action risk:

1. **Conformal prediction** (Angelopoulos & Bates 2022) — distribution-free
   coverage guarantees on risk scores.  The scorer wraps its point estimate
   in a prediction set that contains the true risk with probability >= 1-alpha.
   No distributional assumptions, no retraining needed when the world shifts.

2. **Multiplicative Weight Update (MWU)** — online learning that adapts
   expert weights from action outcomes.  Each "expert" is a risk signal
   (taxonomy base score, sequence pattern, agent history, etc.).  MWU
   tracks which signals actually predict bad outcomes and up-weights them.

3. **Temporal sequence reasoning** — individual actions may be safe but
   sequences can be catastrophic (e.g., read_data → export → delete is a
   data exfiltration pattern even if each action alone is benign).

4. **Cold start → learned transition** — starts with rule-based scoring,
   accumulates calibration data, flips to conformal-wrapped ML scoring
   once coverage guarantee can be maintained.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Context coercion helpers ─────────────────────────────────────────────

def _coerce_unit_float(value: Any, fallback: float) -> float:
    """Coerce an external context value into a finite [0, 1] float.

    Agent SDKs can pass strings, None, NaN, or inf — propagating them into
    MWU/confidence math poisons weights and the conformal residual deque.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(f):
        return fallback
    return min(1.0, max(0.0, f))


def _coerce_optional_unit_float(value: Any) -> Optional[float]:
    """Like _coerce_unit_float but preserves None (signals 'not provided')."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return min(1.0, max(0.0, f))


# ── Decision types ────────────────────────────────────────────────────────

class Decision(str, Enum):
    """Scorer decision — allow, deny, or escalate for human review."""
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"  # Human review required


@dataclass
class RiskAssessment:
    """Full risk assessment for an action request.

    Carries the point estimate, conformal interval, contributing signals,
    and the final decision with thresholds used.
    """
    action_name: str
    agent_id: str
    point_estimate: float          # Combined risk score, 0.0-1.0
    conformal_lower: float         # Lower bound of prediction interval
    conformal_upper: float         # Upper bound — this drives the decision
    decision: Decision
    signals: dict[str, float]      # Named signal contributions
    mwu_weights: dict[str, float]  # Current expert weights (snapshot)
    threshold_allow: float         # Score below this → allow
    threshold_deny: float          # Score above this → deny (between = escalate)
    sequence_risk: float           # Contribution from temporal patterns
    calibration_size: int          # How many calibration points we have
    evaluation_ms: float = 0.0
    explanation: str = ""

    def __post_init__(self) -> None:
        # NaN or +/-Inf in any numeric field means upstream scoring failed
        # (corrupt model bundle, adversarial feature value, div-by-zero in
        # a signal). The safe response is to escalate — but the NaN itself
        # must NOT land in the hash-chained audit trail or the compliance
        # report's risk_score, where it corrupts aggregations and emits
        # non-strict JSON on the wire. Replace with maximum-uncertainty
        # neutral values and force ESCALATE so a human reviews.
        if (not math.isfinite(self.point_estimate)
                or not math.isfinite(self.conformal_lower)
                or not math.isfinite(self.conformal_upper)):
            logger.warning(
                "Non-finite risk values (point=%r lower=%r upper=%r) for "
                "action_name=%s agent_id=%s — forcing ESCALATE and clamping "
                "to [0.0, 1.0] with point=0.5. Upstream scorer likely "
                "produced NaN (corrupt bundle, bad input, or div-by-zero).",
                self.point_estimate, self.conformal_lower, self.conformal_upper,
                self.action_name, self.agent_id,
            )
            self.point_estimate = 0.5
            self.conformal_lower = 0.0
            self.conformal_upper = 1.0
            self.decision = Decision.ESCALATE
            if not self.explanation or "non-finite" not in self.explanation:
                self.explanation = (
                    "escalate: non-finite risk from scorer — review required"
                )

    def to_backend_decision(self) -> dict:
        """Convert to a plain decision dict (allowed/action/reason/backend)."""
        return {
            "allowed": self.decision == Decision.ALLOW,
            "action": self.decision.value,
            "reason": self.explanation,
            "backend": "vaara_adaptive",
            "raw_result": {
                "point_estimate": self.point_estimate,
                "conformal_interval": [self.conformal_lower, self.conformal_upper],
                "signals": self.signals,
                "sequence_risk": self.sequence_risk,
                "calibration_size": self.calibration_size,
            },
            "evaluation_ms": self.evaluation_ms,
            "error": None,
        }


# ── Sequence pattern detection ────────────────────────────────────────────

@dataclass(frozen=True)
class SequencePattern:
    """A dangerous action sequence pattern.

    Matches ordered subsequences within a sliding window.
    """
    name: str
    actions: tuple[str, ...]       # Ordered action names to match
    risk_boost: float              # Additive risk when pattern detected
    window_size: int = 10          # How many recent actions to scan
    description: str = ""


# Built-in dangerous sequences — extensible at runtime
BUILTIN_SEQUENCES = [
    SequencePattern(
        "data_exfiltration",
        ("data.read", "data.export"),
        risk_boost=0.4,
        window_size=5,
        description="Read then export — possible data exfiltration",
    ),
    SequencePattern(
        "data_destruction",
        ("data.read", "data.delete"),
        risk_boost=0.3,
        window_size=5,
        description="Read then delete — possible evidence destruction",
    ),
    SequencePattern(
        "privilege_escalation",
        ("id.create_key", "id.grant_permission"),
        risk_boost=0.5,
        window_size=5,
        description="Create credential then grant permissions — privilege escalation",
    ),
    SequencePattern(
        "financial_drain",
        ("tx.approve", "tx.swap", "tx.transfer"),
        risk_boost=0.6,
        window_size=8,
        description="Approve → swap → transfer — possible fund drainage",
    ),
    SequencePattern(
        "governance_takeover",
        ("id.grant_permission", "gov.vote", "gov.execute_proposal"),
        risk_boost=0.7,
        window_size=10,
        description="Grant access → vote → execute — governance capture",
    ),
    SequencePattern(
        "safety_override_sequence",
        ("infra.config_change", "phy.safety_override"),
        risk_boost=0.8,
        window_size=5,
        description="Config change then safety override — disabling safeguards",
    ),
    SequencePattern(
        "rapid_rebalance",
        ("vault.rebalance", "vault.rebalance"),
        risk_boost=0.3,
        window_size=4,
        description="Repeated rebalance — possible wash trading or manipulation",
    ),
]


# ── MWU Expert System ────────────────────────────────────────────────────

class MWUExperts:
    """Multiplicative Weight Update for online expert aggregation.

    Each expert produces a risk signal in [0, 1].  MWU maintains weights
    that reflect each expert's historical accuracy.  Experts that predicted
    bad outcomes correctly get up-weighted; those that missed get penalized.

    Reference: Arora, Hazan & Kale (2012), "The Multiplicative Weights
    Update Method: a Meta-Algorithm and Applications"
    """

    def __init__(
        self,
        expert_names: list[str],
        eta: float = 0.1,
        min_weight: float = 0.01,
    ) -> None:
        if not expert_names:
            raise ValueError("MWUExperts requires at least one expert name")
        self._eta = eta                # Learning rate
        self._min_weight = min_weight  # Floor to prevent expert death
        self._weights: dict[str, float] = {
            name: 1.0 / len(expert_names) for name in expert_names
        }
        self._update_count = 0

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    @property
    def update_count(self) -> int:
        return self._update_count

    def predict(self, signals: dict[str, float]) -> float:
        """Weighted combination of expert signals."""
        total_weight = sum(self._weights.values())
        if total_weight == 0:
            return 0.5

        score = 0.0
        for name, signal in signals.items():
            if name in self._weights:
                score += (self._weights[name] / total_weight) * signal
        return min(1.0, max(0.0, score))

    def update(self, signals: dict[str, float], outcome: float) -> None:
        """Update weights based on observed outcome.

        Args:
            signals: Expert predictions at decision time.
            outcome: Actual risk realized (0.0 = nothing bad, 1.0 = catastrophic).
                     This is set by the audit trail when action consequences are known.
        """
        for name, signal in signals.items():
            if name not in self._weights:
                continue
            try:
                signal_f = float(signal)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(signal_f):
                continue
            # Loss = |prediction - outcome|.  High loss → weight decreases.
            loss = abs(signal_f - outcome)
            self._weights[name] *= math.exp(-self._eta * loss)
            self._weights[name] = max(self._min_weight, self._weights[name])

        # Re-normalize
        total = sum(self._weights.values())
        if total > 0:
            for name in self._weights:
                self._weights[name] /= total

        self._update_count += 1


# ── Conformal Prediction ─────────────────────────────────────────────────

class ConformalCalibrator:
    """Split conformal prediction for risk score intervals.

    Maintains a calibration set of (predicted_risk, actual_outcome) pairs.
    Produces prediction intervals that contain the true risk with
    probability >= 1 - alpha, with no distributional assumptions.

    Uses FACI-style adaptive alpha when enough data is available:
    the effective alpha tracks coverage errors online, tightening when
    the interval misses and loosening when it's too conservative.

    Reference:
    - Vovk, Gammerman & Shafer (2005), "Algorithmic Learning in a Random World"
    - Gibbs & Candes (2021), "Adaptive Conformal Inference Under Distribution Shift"
    """

    def __init__(
        self,
        alpha: float = 0.10,        # Target miscoverage rate (10%)
        min_calibration: int = 30,   # Min points before conformal kicks in
        max_calibration: int = 2000, # Rolling window size
        gamma: float = 0.005,        # FACI step size for adaptive alpha
    ) -> None:
        self._alpha = alpha
        self._alpha_t = alpha         # Adaptive alpha (FACI)
        self._min_calibration = min_calibration
        self._max_calibration = max_calibration
        self._gamma = gamma
        # Calibration residuals: |predicted - actual|
        self._residuals: deque[float] = deque(maxlen=max_calibration)
        # Lock guards _residuals and _alpha_t. AdaptiveScorer wraps all
        # calls with its own RLock, but ConformalCalibrator can be shared
        # or accessed directly (e.g. in free-threaded Python 3.13+).
        self._lock = threading.Lock()

    @property
    def calibration_size(self) -> int:
        with self._lock:
            return len(self._residuals)

    @property
    def is_calibrated(self) -> bool:
        with self._lock:
            return len(self._residuals) >= self._min_calibration

    @property
    def effective_alpha(self) -> float:
        with self._lock:
            return self._alpha_t

    def add_calibration_point(self, predicted: float, actual: float) -> None:
        """Add a (predicted, actual) pair to the calibration set.

        Called by the audit trail when action outcomes are known.
        Non-finite inputs are dropped — one NaN in the residual deque
        poisons every future quantile (sorted() of NaN is
        implementation-defined and breaks coverage guarantees).
        """
        try:
            predicted_f = float(predicted)
            actual_f = float(actual)
        except (TypeError, ValueError):
            return
        if not (math.isfinite(predicted_f) and math.isfinite(actual_f)):
            return
        residual = abs(predicted_f - actual_f)
        with self._lock:
            self._residuals.append(residual)
            # FACI adaptive alpha update
            if len(self._residuals) >= self._min_calibration:
                # err_t = 1 if actual was outside the interval we would have produced
                quantile = self._get_quantile_locked()
                covered = residual <= quantile
                err_t = 0.0 if covered else 1.0
                # alpha_t+1 = alpha_t + gamma * (alpha - err_t)
                self._alpha_t = self._alpha_t + self._gamma * (self._alpha - err_t)
                self._alpha_t = min(0.5, max(0.001, self._alpha_t))

    def _get_quantile_locked(self) -> float:
        """Conformal quantile — caller must hold self._lock."""
        if not self._residuals:
            return 1.0
        sorted_residuals = sorted(self._residuals)
        n = len(sorted_residuals)
        # Quantile level: ceil((1 - alpha)(n + 1)) / n
        level = math.ceil((1 - self._alpha_t) * (n + 1)) / n
        level = min(1.0, level)
        idx = min(int(level * n), n - 1)
        return sorted_residuals[idx]

    def _get_quantile(self) -> float:
        """Conformal quantile from calibration residuals."""
        with self._lock:
            return self._get_quantile_locked()

    def predict_interval(self, point_estimate: float) -> tuple[float, float]:
        """Produce a conformal prediction interval around the point estimate.

        Returns (lower, upper) where true risk is in [lower, upper]
        with probability >= 1 - alpha.
        """
        with self._lock:
            if len(self._residuals) < self._min_calibration:
                # Not enough data — return conservative interval
                return (max(0.0, point_estimate - 0.3), min(1.0, point_estimate + 0.3))
            q = self._get_quantile_locked()
        lower = max(0.0, point_estimate - q)
        upper = min(1.0, point_estimate + q)
        return (lower, upper)


# ── Agent History Tracker ─────────────────────────────────────────────────

@dataclass
class AgentProfile:
    """Running profile of an agent's behavior."""
    agent_id: str
    total_actions: int = 0
    denied_actions: int = 0
    escalated_actions: int = 0
    bad_outcomes: int = 0        # Actions that resulted in harm
    action_counts: dict[str, int] = field(default_factory=dict)
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=50))
    first_seen: float = 0.0
    last_seen: float = 0.0

    @property
    def denial_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.denied_actions / self.total_actions

    @property
    def bad_outcome_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.bad_outcomes / self.total_actions

    @property
    def age_seconds(self) -> float:
        if self.first_seen == 0:
            return 0.0
        return self.last_seen - self.first_seen

    def record_action(self, action_name: str, timestamp: float) -> None:
        self.total_actions += 1
        self.action_counts[action_name] = self.action_counts.get(action_name, 0) + 1
        self.recent_actions.append((action_name, timestamp))
        if self.first_seen == 0:
            self.first_seen = timestamp
        self.last_seen = timestamp


# ── Main Scorer ───────────────────────────────────────────────────────────

# Default expert names — each produces a [0, 1] risk signal
DEFAULT_EXPERTS = [
    "taxonomy_base",      # From ActionType.base_risk_score
    "agent_history",      # From agent's denial/outcome history
    "sequence_pattern",   # From temporal sequence matching
    "action_frequency",   # Burst detection — too many actions too fast
    "confidence_gap",     # Agent's self-reported confidence vs observed accuracy
]


# Category prefix table for the novel-cluster sequence detector.
# The scorer sees action names as strings so this duplicates the taxonomy's
# category split without importing it (keeps scorer decoupled from registry).
_CATEGORY_PREFIX = {
    "tx.": "financial",
    "vault.": "financial",
    "data.": "data",
    "comm.": "communication",
    "infra.": "infrastructure",
    "id.": "identity",
    "gov.": "governance",
    "phy.": "physical",
}


# Actions that raise the novel-cluster signal when they co-occur across
# different categories within the sliding window. Deliberately limited to
# the most dangerous members of each category — a legitimate workflow can
# touch data.read and infra.scale without tripping it, but one that strings
# data.export + infra.deploy + id.grant_permission together flags as a
# novel multi-domain pattern even if it doesn't match BUILTIN_SEQUENCES.
_HIGH_RISK_ACTIONS = frozenset({
    "tx.sign", "tx.transfer", "tx.swap",
    "data.delete", "data.export",
    "id.grant_permission", "id.create_key", "id.revoke",
    "infra.terminate", "infra.config_change", "infra.deploy",
    "gov.vote", "gov.execute_proposal",
    "phy.safety_override", "phy.actuator",
    "comm.post_public",
})


def _category_of(tool_name: str) -> str:
    """Best-effort category from the tool name prefix."""
    for prefix, category in _CATEGORY_PREFIX.items():
        if tool_name.startswith(prefix):
            return category
    return "unknown"


class AdaptiveScorer:
    """Adaptive risk scorer with conformal guarantees.

    Lifecycle:
        1. **Cold start** — uses taxonomy base scores + rules
        2. **Calibrating** — accumulates outcome data, builds conformal set
        3. **Adaptive** — MWU-weighted experts wrapped in conformal intervals
    """

    def __init__(
        self,
        threshold_allow: float = 0.4,
        threshold_deny: float = 0.7,
        alpha: float = 0.10,
        mwu_eta: float = 0.1,
        sequence_patterns: Optional[list[SequencePattern]] = None,
        burst_window_seconds: float = 60.0,
        burst_threshold: int = 10,
        max_tracked_agents: int = 10_000,
        pre_seed_calibration: bool = True,
    ) -> None:
        """
        Args:
            threshold_allow: Risk scores below this → ALLOW.
            threshold_deny:  Risk scores above this → DENY.
                             Between allow and deny → ESCALATE for human review.
            alpha: Conformal miscoverage rate (0.10 = 90% coverage guarantee).
            mwu_eta: MWU learning rate.
            sequence_patterns: Dangerous action sequences to detect.
            burst_window_seconds: Window for burst detection.
            burst_threshold: Max actions in burst window before flagging.
            max_tracked_agents: LRU cap on unique agent profiles. A
                multi-tenant deploy with attacker-controlled agent_id
                (e.g. "agent-${user_input}") can otherwise exhaust
                memory — ~37KB per profile × unbounded agents. 10K is
                generous for real deployments; tune up for enterprise.
            pre_seed_calibration: If True (default), seed the conformal
                calibrator with 50 synthetic benign prior pairs so the
                interval starts at ~±0.19 instead of the ±0.3 cold-start
                fallback. Real outcomes overwrite the prior within the
                rolling window as traffic arrives. Set False when
                measuring calibration growth from zero.
        """
        if threshold_allow >= threshold_deny:
            raise ValueError(
                f"threshold_allow ({threshold_allow}) must be strictly less than "
                f"threshold_deny ({threshold_deny}); inverted thresholds produce "
                f"no ESCALATE band and nonsensical ALLOW/DENY boundaries"
            )
        self._threshold_allow = threshold_allow
        self._threshold_deny = threshold_deny
        self._burst_window = burst_window_seconds
        self._burst_threshold = burst_threshold
        self._max_tracked_agents = max_tracked_agents

        # MWU expert system
        self._mwu = MWUExperts(DEFAULT_EXPERTS, eta=mwu_eta)

        # Conformal calibrator
        self._conformal = ConformalCalibrator(alpha=alpha)
        if pre_seed_calibration:
            self._seed_conformal_prior()

        # Sequence patterns
        self._sequences = list(sequence_patterns or BUILTIN_SEQUENCES)

        # Agent profiles (agent_id → AgentProfile). OrderedDict provides
        # LRU semantics: move_to_end on access, popitem(last=False) to
        # evict oldest when over _max_tracked_agents.
        self._agents: "OrderedDict[str, AgentProfile]" = OrderedDict()

        # Global action history for cross-agent sequence detection
        self._global_history: deque[tuple[str, str, float]] = deque(maxlen=200)

        # Per-(agent, pattern) last-match state. A pattern that stays in
        # the sliding window would otherwise re-emit the same warning on
        # every evaluate() call, flooding logs and the audit trail with
        # redundant "pattern detected" noise while the agent is just
        # continuing its session. We only log on transitions into
        # "matched" — when the match clears (pattern falls out of the
        # window), the entry is removed so a *new* trip logs again.
        self._seq_match_state: dict[tuple[str, str], bool] = {}

        # Serializes evaluate / dry_run_evaluate / record_outcome so
        # concurrent agent threads don't mutate an AgentProfile's deque
        # mid-iteration (burst_signal and sequence_signal both iterate
        # recent_actions while record_action appends to it). Using an
        # RLock lets nested calls inside the scorer re-enter safely.
        self._lock = threading.RLock()

    def _seed_conformal_prior(self) -> None:
        """Seed the calibrator with 50 synthetic benign (predicted, actual) pairs.

        A just-constructed scorer has zero residuals, so predict_interval()
        returns the ±0.3 conservative fallback until ~30 real outcomes are
        reported. In that window even low-risk actions land above
        threshold_allow and escalate — honest but useless at first-deploy
        time.

        The prior residuals here span 0.10–0.19 (predicted ≈ 0.15–0.24,
        actual = 0.05). The 90th-percentile quantile with 50 samples lands
        at ~0.19, so the starting interval is ±0.19 instead of ±0.3. Real
        traffic then overwrites the prior within max_calibration=2000
        reports without any explicit migration. Operators who want the
        strict cold-start behaviour pass pre_seed_calibration=False.
        """
        for i in range(50):
            predicted = 0.15 + (i % 10) * 0.01
            actual = 0.05
            self._conformal.add_calibration_point(predicted, actual)

    # ── Scorer Backend Protocol ───────────────────────────────────

    @property
    def name(self) -> str:
        """Identifier for this backend."""
        return "vaara_adaptive"

    def evaluate(self, context: dict[str, Any]) -> Any:
        """Evaluate an action context and return a decision dict.

        Args:
            context: Dict with tool_name, agent_id, action_type,
                     base_risk_score, agent_confidence, etc.
                     (produced by ActionRequest.to_policy_context())
        """
        with self._lock:
            return self._evaluate_locked(context)

    def _evaluate_locked(self, context: dict[str, Any]) -> Any:
        start = time.monotonic()

        # Extract fields from context dict
        tool_name = context.get("tool_name", "unknown")
        agent_id = context.get("agent_id", "anonymous")
        base_risk = _coerce_unit_float(context.get("base_risk_score", 0.5), 0.5)
        agent_confidence = _coerce_optional_unit_float(context.get("agent_confidence"))
        reversibility = context.get("reversibility", "partially_reversible")
        blast_radius = context.get("blast_radius", "local")

        # Build risk signals from each expert
        signals = self._compute_signals(
            tool_name=tool_name,
            agent_id=agent_id,
            base_risk=base_risk,
            agent_confidence=agent_confidence,
            reversibility=reversibility,
            blast_radius=blast_radius,
        )

        # MWU-weighted combination
        point_estimate = self._mwu.predict(signals)

        # Conformal interval
        lower, upper = self._conformal.predict_interval(point_estimate)

        # Decision uses the UPPER bound — conservative by design.
        # If the worst-case (within 1-alpha confidence) is safe, allow it.
        # If the best-case is dangerous, deny it.
        decision_score = upper
        if decision_score < self._threshold_allow:
            decision = Decision.ALLOW
        elif decision_score > self._threshold_deny:
            decision = Decision.DENY
        else:
            decision = Decision.ESCALATE

        # Record this action in agent profile
        now = time.time()
        profile = self._get_or_create_agent(agent_id)
        profile.record_action(tool_name, now)
        if decision == Decision.DENY:
            profile.denied_actions += 1
        elif decision == Decision.ESCALATE:
            profile.escalated_actions += 1

        # Record in global history
        self._global_history.append((agent_id, tool_name, now))

        elapsed_ms = (time.monotonic() - start) * 1000

        explanation = (
            f"{decision.value}: risk={point_estimate:.3f} "
            f"[{lower:.3f}, {upper:.3f}] "
            f"(threshold allow<{self._threshold_allow} deny>{self._threshold_deny})"
        )

        assessment = RiskAssessment(
            action_name=tool_name,
            agent_id=agent_id,
            point_estimate=point_estimate,
            conformal_lower=lower,
            conformal_upper=upper,
            decision=decision,
            signals=signals,
            mwu_weights=self._mwu.weights,
            threshold_allow=self._threshold_allow,
            threshold_deny=self._threshold_deny,
            sequence_risk=signals.get("sequence_pattern", 0.0),
            calibration_size=self._conformal.calibration_size,
            evaluation_ms=elapsed_ms,
            explanation=explanation,
        )

        logger.info(
            "Scored %s/%s: %s (%.1fms)",
            agent_id, tool_name, decision.value, elapsed_ms,
        )

        # Return MS-compatible dict
        return assessment.to_backend_decision()

    def dry_run_evaluate(self, context: dict[str, Any]) -> Any:
        """Score without mutating agent profiles, global history, or
        triggering sequence-detector side effects.

        Callers that want to *preview* a decision (pre-screens, compliance
        simulations, what-if queries) should use this instead of evaluate()
        so the learning state and audit-relevant counters stay clean.
        """
        with self._lock:
            return self._dry_run_evaluate_locked(context)

    def _dry_run_evaluate_locked(self, context: dict[str, Any]) -> Any:
        tool_name = context.get("tool_name", "unknown")
        agent_id = context.get("agent_id", "anonymous")
        base_risk = _coerce_unit_float(context.get("base_risk_score", 0.5), 0.5)
        agent_confidence = _coerce_optional_unit_float(context.get("agent_confidence"))
        reversibility = context.get("reversibility", "partially_reversible")
        blast_radius = context.get("blast_radius", "local")

        # _compute_signals is read-only for everything except the
        # sequence detector's warning log — silence that temporarily.
        seq_logger = logging.getLogger(
            "vaara.scorer.adaptive"
        )
        prev_level = seq_logger.level
        seq_logger.setLevel(logging.ERROR)
        try:
            signals = self._compute_signals(
                tool_name=tool_name,
                agent_id=agent_id,
                base_risk=base_risk,
                agent_confidence=agent_confidence,
                reversibility=reversibility,
                blast_radius=blast_radius,
            )
        finally:
            seq_logger.setLevel(prev_level)

        point_estimate = self._mwu.predict(signals)
        lower, upper = self._conformal.predict_interval(point_estimate)
        if upper < self._threshold_allow:
            decision = Decision.ALLOW
        elif upper > self._threshold_deny:
            decision = Decision.DENY
        else:
            decision = Decision.ESCALATE

        return {
            "action": decision.value,
            "raw_result": {
                "point_estimate": point_estimate,
                "conformal_interval": [lower, upper],
            },
        }

    # ── Signal computation ────────────────────────────────────────

    def _compute_signals(
        self,
        tool_name: str,
        agent_id: str,
        base_risk: float,
        agent_confidence: Optional[float],
        reversibility: str,
        blast_radius: str,
    ) -> dict[str, float]:
        """Compute risk signals from each expert."""
        signals: dict[str, float] = {}

        # Expert 1: Taxonomy base risk (from ActionType metadata)
        signals["taxonomy_base"] = base_risk

        # Expert 2: Agent history
        signals["agent_history"] = self._agent_history_signal(agent_id)

        # Expert 3: Temporal sequence patterns
        signals["sequence_pattern"] = self._sequence_signal(agent_id, tool_name)

        # Expert 4: Action frequency / burst detection
        signals["action_frequency"] = self._burst_signal(agent_id)

        # Expert 5: Confidence gap
        signals["confidence_gap"] = self._confidence_gap_signal(
            agent_confidence, base_risk
        )

        return signals

    def _agent_history_signal(self, agent_id: str) -> float:
        """Risk signal from agent's track record."""
        profile = self._agents.get(agent_id)
        if profile is None or profile.total_actions < 5:
            return 0.3  # Unknown agent = moderate risk (not zero, not high)

        # Blend denial rate and bad outcome rate
        # Bad outcomes weigh 3x more than denials (denials might be overly cautious)
        risk = 0.3 * profile.denial_rate + 0.7 * profile.bad_outcome_rate

        # New agents get a slight risk premium that decays with track record
        age_discount = min(1.0, profile.total_actions / 100)
        risk = risk * age_discount + 0.2 * (1 - age_discount)

        return min(1.0, max(0.0, risk))

    def _sequence_signal(self, agent_id: str, current_tool: str) -> float:
        """Detect dangerous action sequences in recent history.

        Runs two layers: (1) exact match against BUILTIN_SEQUENCES, and
        (2) a generic "novel multi-domain cluster" fallback that fires
        when no known pattern matches but the recent window contains
        high-risk actions spanning multiple categories. The fallback
        catches attacks that don't fit a pre-enumerated template
        (e.g., data.export → infra.deploy → gov.vote) without producing
        false positives on single-category legitimate workflows.
        """
        profile = self._agents.get(agent_id)
        if profile is None:
            return 0.0

        recent = list(profile.recent_actions)
        # Append current action for matching
        recent_names = [name for name, _ in recent] + [current_tool]

        max_boost = 0.0
        for pattern in self._sequences:
            matched = self._matches_subsequence(
                recent_names[-pattern.window_size:], pattern.actions
            )
            key = (agent_id, pattern.name)
            was_matched = self._seq_match_state.get(key, False)
            if matched:
                max_boost = max(max_boost, pattern.risk_boost)
                if not was_matched:
                    # Transition: new trip. Log once per entry into the
                    # matched state so the risk signal stays strong
                    # without flooding logs as the window slides.
                    logger.warning(
                        "Sequence pattern '%s' detected for agent %s",
                        pattern.name, agent_id,
                    )
                    self._seq_match_state[key] = True
            elif was_matched:
                # Transition out of matched state — clear so the next
                # trip gets logged again.
                self._seq_match_state.pop(key, None)

        # Fallback: novel high-risk cluster across categories. Only
        # activates when no builtin pattern fired, the current action
        # is itself high-risk, and at least one prior action in the
        # recent window is high-risk in a DIFFERENT category. Two
        # categories is the floor — a lone high-risk action repeating
        # itself is covered by burst detection, not sequence risk.
        if max_boost == 0.0 and current_tool in _HIGH_RISK_ACTIONS:
            window = recent_names[-10:]
            high_risk_categories = {
                _category_of(name) for name in window
                if name in _HIGH_RISK_ACTIONS
            }
            if len(high_risk_categories) >= 2:
                novel_key = (agent_id, "_novel_cluster")
                if not self._seq_match_state.get(novel_key, False):
                    logger.warning(
                        "Novel high-risk multi-domain cluster for agent %s: "
                        "categories=%s",
                        agent_id, sorted(high_risk_categories),
                    )
                    self._seq_match_state[novel_key] = True
                max_boost = 0.25
            else:
                self._seq_match_state.pop((agent_id, "_novel_cluster"), None)

        return min(1.0, max_boost)

    @staticmethod
    def _matches_subsequence(history: list[str], pattern: tuple[str, ...]) -> bool:
        """Check if pattern appears as an ordered subsequence in history."""
        if not pattern:
            return False
        pi = 0
        for action in history:
            if action == pattern[pi]:
                pi += 1
                if pi == len(pattern):
                    return True
        return False

    def _burst_signal(self, agent_id: str) -> float:
        """Detect action bursts — too many actions in a short window."""
        now = time.time()
        cutoff = now - self._burst_window
        profile = self._agents.get(agent_id)
        if profile is None:
            return 0.0

        recent_count = sum(
            1 for _, ts in profile.recent_actions if ts > cutoff
        )

        half = max(1, self._burst_threshold // 2)
        if recent_count <= half:
            return 0.0
        elif recent_count <= self._burst_threshold:
            # Linear ramp from 0 to 0.5
            ratio = (recent_count - half) / half
            return 0.5 * ratio
        else:
            # Over threshold — high risk, capped at 0.9
            over = recent_count - self._burst_threshold
            return min(0.9, 0.5 + 0.1 * over)

    def _confidence_gap_signal(
        self, agent_confidence: Optional[float], base_risk: float
    ) -> float:
        """Risk from gap between agent's self-reported confidence and base risk.

        If an agent claims high confidence on a high-base-risk action,
        something is off — either the agent is miscalibrated or gaming the system.
        Conversely, low confidence on a low-risk action is fine.
        """
        if agent_confidence is None:
            return 0.2  # No confidence reported = slight risk

        # Confidence is [0, 1] where 1 = "I'm sure this is safe"
        # Base risk is [0, 1] where 1 = "this action type is dangerous"
        # A gap where confidence is high but risk is high is suspicious
        gap = max(0.0, agent_confidence - (1.0 - base_risk))
        return min(1.0, gap)

    # ── Agent management ──────────────────────────────────────────

    def _get_or_create_agent(self, agent_id: str) -> AgentProfile:
        if agent_id in self._agents:
            self._agents.move_to_end(agent_id)
            return self._agents[agent_id]
        profile = AgentProfile(agent_id=agent_id)
        self._agents[agent_id] = profile
        # Evict least-recently-used entries when over cap. Also prune
        # matching _seq_match_state keys so the per-(agent, pattern)
        # dict doesn't grow unbounded alongside the evicted agent.
        while len(self._agents) > self._max_tracked_agents:
            evicted_id, _ = self._agents.popitem(last=False)
            stale_keys = [k for k in self._seq_match_state if k[0] == evicted_id]
            for k in stale_keys:
                del self._seq_match_state[k]
        return profile

    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """Get an agent's profile (read-only access for dashboards)."""
        return self._agents.get(agent_id)

    # ── Learning interface (called by audit trail) ────────────────

    def record_outcome(
        self,
        agent_id: str,
        tool_name: str,
        predicted_risk: float,
        actual_outcome: float,
        signals: dict[str, float],
    ) -> None:
        """Record the actual outcome of an action for learning.

        Called by the audit trail when an action's consequences are known.
        Updates both MWU weights and conformal calibration.

        Args:
            agent_id: The agent that took the action.
            tool_name: The tool/action name.
            predicted_risk: What we predicted at decision time.
            actual_outcome: What actually happened (0.0 safe, 1.0 catastrophic).
            signals: The expert signals at decision time (for MWU update).
        """
        # Coerce outcome to a finite unit float up front — NaN/inf would
        # propagate into MWU weights (exp(-eta*NaN) = NaN) and the
        # conformal residual deque, permanently corrupting calibration.
        try:
            actual_outcome = float(actual_outcome)
        except (TypeError, ValueError):
            logger.warning(
                "record_outcome: non-numeric actual_outcome from %s/%s, skipping",
                agent_id, tool_name,
            )
            return
        if not math.isfinite(actual_outcome):
            logger.warning(
                "record_outcome: non-finite actual_outcome from %s/%s, skipping",
                agent_id, tool_name,
            )
            return
        actual_outcome = min(1.0, max(0.0, actual_outcome))

        try:
            predicted_risk = float(predicted_risk)
        except (TypeError, ValueError):
            predicted_risk = 0.5
        if not math.isfinite(predicted_risk):
            predicted_risk = 0.5
        predicted_risk = min(1.0, max(0.0, predicted_risk))

        with self._lock:
            # Update MWU expert weights
            self._mwu.update(signals, actual_outcome)

            # Update conformal calibration set
            self._conformal.add_calibration_point(predicted_risk, actual_outcome)

            # Update agent profile ONLY if the agent is still tracked.
            # A late outcome for an LRU-evicted agent must not resurrect
            # a zero-history profile (which would also evict a currently-
            # active agent from the LRU, silently corrupting its history)
            # — MWU/conformal are global and legitimately updated above,
            # but per-agent counters apply only to live profiles.
            profile = self._agents.get(agent_id)
            if profile is not None:
                self._agents.move_to_end(agent_id)
                if actual_outcome > 0.5:
                    profile.bad_outcomes += 1
            else:
                logger.info(
                    "record_outcome: agent %s no longer tracked "
                    "(evicted or never seen); updating global "
                    "MWU/conformal only, skipping profile update",
                    agent_id,
                )

        logger.info(
            "Outcome recorded: %s/%s predicted=%.3f actual=%.3f "
            "(calibration=%d, mwu_updates=%d)",
            agent_id, tool_name, predicted_risk, actual_outcome,
            self._conformal.calibration_size, self._mwu.update_count,
        )

    # ── Sequence pattern management ───────────────────────────────

    def add_sequence_pattern(self, pattern: SequencePattern) -> None:
        """Register a new dangerous sequence pattern.

        A pattern with the same name replaces the existing one so repeated
        registration (e.g. on hot reload) stays idempotent instead of
        silently accumulating duplicates.
        """
        with self._lock:
            self._sequences = [p for p in self._sequences if p.name != pattern.name]
            self._sequences.append(pattern)

    def remove_sequence_pattern(self, name: str) -> bool:
        """Remove a sequence pattern by name."""
        with self._lock:
            before = len(self._sequences)
            self._sequences = [p for p in self._sequences if p.name != name]
            return len(self._sequences) < before

    # ── Introspection ─────────────────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        """Whether we have enough data for conformal guarantees."""
        return self._conformal.is_calibrated

    @property
    def calibration_size(self) -> int:
        return self._conformal.calibration_size

    @property
    def mwu_weights(self) -> dict[str, float]:
        return self._mwu.weights

    @property
    def mwu_update_count(self) -> int:
        return self._mwu.update_count

    @property
    def thresholds(self) -> tuple[float, float]:
        return (self._threshold_allow, self._threshold_deny)

    def status(self) -> dict[str, Any]:
        """Dashboard-friendly status snapshot."""
        return {
            "calibrated": self.is_calibrated,
            "calibration_size": self.calibration_size,
            "effective_alpha": self._conformal.effective_alpha,
            "mwu_weights": self.mwu_weights,
            "mwu_updates": self.mwu_update_count,
            "thresholds": {"allow": self._threshold_allow, "deny": self._threshold_deny},
            "tracked_agents": len(self._agents),
            "sequence_patterns": len(self._sequences),
        }


# ── Factory ───────────────────────────────────────────────────────────────

def create_default_scorer(**kwargs: Any) -> AdaptiveScorer:
    """Create a scorer with production defaults."""
    return AdaptiveScorer(**kwargs)
