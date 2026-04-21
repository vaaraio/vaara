"""Trained per-action-step gate backend.

Wraps a frozen action-gate bundle (bootstrap ensemble + conformal q_hat)
and exposes it through the scorer interface used elsewhere in Vaara.

Context shape expected:
    {
      "tool_name": str,
      "agent_id": str,
      "history_pairs": list[(ai_text, env_text)],
      "proposed_action_text": str,
      ...  # other keys are passed through but ignored here
    }

Decisions:
    execute     -> ALLOW
    flag_review -> ESCALATE
    block       -> DENY
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

from vaara.scorer.adaptive import Decision, RiskAssessment

logger = logging.getLogger(__name__)


_DEFAULT_BUNDLE = Path.home() / ".vaara" / "cache" / "perstep_gate_bundle.joblib"


class TrainedGateScorer:
    """Scorer backend that calls a trained per-step ActionGate bundle.

    Data-driven ensemble + conformal interval. Replaces the hand-coded
    base-risk signal when a bundle is available.
    """

    def __init__(self, bundle_path: Optional[str] = None) -> None:
        from vaara.scorer.action_gate import ActionGate

        path = Path(bundle_path) if bundle_path else _DEFAULT_BUNDLE
        if not path.exists():
            raise FileNotFoundError(
                f"Trained gate bundle not found at {path}. "
                "Run freeze_best.py first."
            )
        self._gate = ActionGate.load(str(path))
        self._bundle_path = str(path)
        logger.info(
            "TrainedGateScorer loaded: n_models=%d q_hat=%.3f nlp=%s",
            len(self._gate.models), self._gate.q_hat,
            "yes" if self._gate.nlp_encoder else "no",
        )

    @property
    def name(self) -> str:
        return "vaara_trained_gate"

    @property
    def q_hat(self) -> float:
        return self._gate.q_hat

    @property
    def ensemble_size(self) -> int:
        return len(self._gate.models)

    def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()

        tool_name = context.get("tool_name", "unknown")
        agent_id = context.get("agent_id", "anonymous")
        history = context.get("history_pairs", [])
        proposed = context.get("proposed_action_text", "")

        decision_obj = self._gate.evaluate(history, proposed)

        if decision_obj.verdict == "execute":
            decision = Decision.ALLOW
        elif decision_obj.verdict == "block":
            decision = Decision.DENY
        else:
            decision = Decision.ESCALATE

        risk = decision_obj.error_prob
        q_hat = self._gate.q_hat
        lower = max(0.0, risk - q_hat)
        upper = min(1.0, risk + q_hat)

        elapsed_ms = (time.monotonic() - start) * 1000

        signals = {
            "trained_gate": risk,
            "ensemble_agreement": decision_obj.agreement,
        }

        explanation = (
            f"{decision.value}: err_p={risk:.3f} agree={decision_obj.agreement:.3f} "
            f"[{lower:.3f}, {upper:.3f}]  {decision_obj.top_risk_factor}"
        )

        assessment = RiskAssessment(
            action_name=tool_name,
            agent_id=agent_id,
            point_estimate=risk,
            conformal_lower=lower,
            conformal_upper=upper,
            decision=decision,
            signals=signals,
            mwu_weights={},
            threshold_allow=0.0,
            threshold_deny=1.0,
            sequence_risk=0.0,
            calibration_size=self.ensemble_size,
            evaluation_ms=elapsed_ms,
            explanation=explanation,
        )
        result = assessment.to_backend_decision()
        result["raw_result"]["verdict"] = decision_obj.verdict
        result["raw_result"]["inside_ok_set"] = decision_obj.inside_ok_set
        result["raw_result"]["inside_error_set"] = decision_obj.inside_error_set
        result["raw_result"]["top_risk_factor"] = decision_obj.top_risk_factor
        result["raw_result"]["q_hat"] = q_hat
        result["raw_result"]["bundle"] = self._bundle_path
        result["backend"] = self.name
        return result


def create_trained_gate_scorer(bundle_path: Optional[str] = None) -> TrainedGateScorer:
    return TrainedGateScorer(bundle_path=bundle_path)
