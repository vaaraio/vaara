"""Adaptive risk scorer — ML-based action risk scoring with conformal guarantees."""

from vaara.scorer.adaptive import (
    AdaptiveScorer,
    Decision,
    RiskAssessment,
    create_default_scorer,
)

try:
    from vaara.scorer.action_gate import ActionGate, GateDecision
    from vaara.scorer.trained_gate import TrainedGateScorer, create_trained_gate_scorer

    _TRAINED_GATE_AVAILABLE = True
except Exception:  # joblib/bundle not available in minimal install
    _TRAINED_GATE_AVAILABLE = False

try:
    from vaara.scorer.mc_dropout_gate import (
        MCDropoutGateScorer,
        create_mc_dropout_scorer,
    )

    _MC_DROPOUT_GATE_AVAILABLE = True
except Exception:  # torch/bundle not available in minimal install
    _MC_DROPOUT_GATE_AVAILABLE = False

try:
    from vaara.scorer.stacked_gate import (
        StackedGateScorer,
        create_stacked_scorer,
    )

    _STACKED_GATE_AVAILABLE = True
except Exception:
    _STACKED_GATE_AVAILABLE = False


__all__ = [
    "AdaptiveScorer",
    "Decision",
    "RiskAssessment",
    "create_default_scorer",
]

if _TRAINED_GATE_AVAILABLE:
    __all__.extend(
        ["ActionGate", "GateDecision", "TrainedGateScorer", "create_trained_gate_scorer"]
    )

if _MC_DROPOUT_GATE_AVAILABLE:
    __all__.extend(["MCDropoutGateScorer", "create_mc_dropout_scorer"])

if _STACKED_GATE_AVAILABLE:
    __all__.extend(["StackedGateScorer", "create_stacked_scorer"])
