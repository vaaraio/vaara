"""Stacked GBM + MC dropout gate — composes both backends via a trained LR.

The stacked ensemble evaluates both `TrainedGateScorer` (GBM bootstrap) and
`MCDropoutGateScorer` (MC dropout NN) on the same action, then combines their
point estimates through a 2-coefficient logistic regression fit on the
calibration split. On the SWE-agent test set this lifts gated accuracy to
86.79% at 69.3% coverage — the current Pareto-best.

Bundle format (joblib):
    {
      "variant": "stacked",
      "stack_coef": [w_gbm, w_mc],
      "stack_intercept": float,
      "q_hat": float,
      "gbm_bundle_path": str,
      "mc_bundle_path": str,
    }
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Optional

import joblib

from vaara.scorer.adaptive import Decision, RiskAssessment
from vaara.scorer.mc_dropout_gate import MCDropoutGateScorer
from vaara.scorer.trained_gate import TrainedGateScorer

logger = logging.getLogger(__name__)


_DEFAULT_BUNDLE = Path.home() / ".vaara" / "cache" / "stacked_gate_bundle.joblib"


class StackedGateScorer:
    """Scorer backend combining GBM bootstrap + MC dropout via LR stack."""

    def __init__(self, bundle_path: Optional[str] = None) -> None:
        path = Path(bundle_path) if bundle_path else _DEFAULT_BUNDLE
        if not path.exists():
            raise FileNotFoundError(
                f"Stacked bundle not found at {path}. Run freeze_stack.py first."
            )
        bundle = joblib.load(str(path))
        self._w_gbm, self._w_mc = bundle["stack_coef"]
        self._intercept = float(bundle["stack_intercept"])
        self._q_hat = float(bundle["q_hat"])
        self._gbm = TrainedGateScorer(bundle_path=bundle["gbm_bundle_path"])
        self._mc = MCDropoutGateScorer(bundle_path=bundle["mc_bundle_path"])
        self._bundle_path = str(path)
        logger.info(
            "StackedGateScorer loaded: w_gbm=%.3f w_mc=%.3f b=%.3f q_hat=%.3f",
            self._w_gbm, self._w_mc, self._intercept, self._q_hat,
        )

    @property
    def name(self) -> str:
        return "vaara_stacked_gate"

    @property
    def q_hat(self) -> float:
        return self._q_hat

    def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        tool_name = context.get("tool_name", "unknown")
        agent_id = context.get("agent_id", "anonymous")

        gbm_result = self._gbm.evaluate(context)
        mc_result = self._mc.evaluate(context)
        g_p = float(gbm_result["raw_result"]["point_estimate"])
        m_p = float(mc_result["raw_result"]["point_estimate"])

        z = self._w_gbm * g_p + self._w_mc * m_p + self._intercept
        mean_p = 1.0 / (1.0 + math.exp(-z))

        inc_ok = abs(mean_p - 0) <= self._q_hat
        inc_err = abs(mean_p - 1) <= self._q_hat
        if inc_ok and not inc_err:
            verdict = "execute"; decision = Decision.ALLOW
        elif inc_err and not inc_ok:
            verdict = "block"; decision = Decision.DENY
        else:
            verdict = "flag_review"; decision = Decision.ESCALATE

        lower = max(0.0, mean_p - self._q_hat)
        upper = min(1.0, mean_p + self._q_hat)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Disagreement between backends = extra UQ signal
        backend_disagree = abs(g_p - m_p)
        mc_std = float(mc_result["raw_result"].get("mc_std", 0.0))
        agreement = max(0.0, min(1.0, 1.0 - 0.5 * (mc_std + backend_disagree)))

        signals = {
            "stacked_gate": mean_p,
            "gbm_p": g_p,
            "mc_p": m_p,
            "backend_disagree": backend_disagree,
        }
        top_risk = gbm_result["raw_result"].get("top_risk_factor", "")
        explanation = (
            f"{decision.value}: err_p={mean_p:.3f} gbm={g_p:.3f} mc={m_p:.3f} "
            f"disagree={backend_disagree:.3f}  {top_risk}"
        )

        assessment = RiskAssessment(
            action_name=tool_name, agent_id=agent_id,
            point_estimate=mean_p, conformal_lower=lower, conformal_upper=upper,
            decision=decision, signals=signals, mwu_weights={},
            threshold_allow=0.0, threshold_deny=1.0, sequence_risk=0.0,
            calibration_size=0, evaluation_ms=elapsed_ms,
            explanation=explanation,
        )
        result = assessment.to_backend_decision()
        result["raw_result"]["verdict"] = verdict
        result["raw_result"]["inside_ok_set"] = bool(inc_ok)
        result["raw_result"]["inside_error_set"] = bool(inc_err)
        result["raw_result"]["top_risk_factor"] = top_risk
        result["raw_result"]["q_hat"] = self._q_hat
        result["raw_result"]["gbm_p"] = g_p
        result["raw_result"]["mc_p"] = m_p
        result["raw_result"]["backend_disagree"] = backend_disagree
        result["raw_result"]["agreement"] = agreement
        result["raw_result"]["bundle"] = self._bundle_path
        result["backend"] = self.name
        return result


def create_stacked_scorer(bundle_path: Optional[str] = None) -> StackedGateScorer:
    return StackedGateScorer(bundle_path=bundle_path)
