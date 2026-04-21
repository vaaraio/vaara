"""MC dropout NN per-action-step gate.

Alternative to TrainedGateScorer (GBM bootstrap). The MC dropout network
produces real epistemic UQ by sampling dropout masks at inference. On the
SWE-agent benchmark it outperforms GBM on gated accuracy (86.76% vs 85.19%
at similar coverage) and trains in seconds instead of minutes.

Bundle format (torch.save):
    {
      "state_dict": dict,
      "in_dim": int,
      "q_hat": float,
      "variant": "behavioral" | "combined",
      "arch": {"hidden1": int, "hidden2": int, "dropout": float},
      "mc_samples": int,
      "nlp_encoder": dict | None,  # {"vec": TfidfVectorizer, "svd": TruncatedSVD}
    }
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vaara.scorer.action_gate import (
    GateDecision,
    clean_text,
    extract_commands,
    classify_cmd,
)
from vaara.scorer.adaptive import Decision, RiskAssessment

logger = logging.getLogger(__name__)


_DEFAULT_BUNDLE = Path.home() / ".vaara" / "cache" / "mc_dropout_gate_bundle.joblib"


def _build_model(in_dim: int, hidden1: int, hidden2: int, dropout: float):
    import torch
    import torch.nn as nn

    class MCDropoutGate(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln = nn.LayerNorm(d)
            self.fc1 = nn.Linear(d, hidden1)
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.fc3 = nn.Linear(hidden2, 1)
            self.drop = nn.Dropout(dropout)
            self.act = nn.GELU()

        def forward(self, x):
            x = self.ln(x)
            x = self.drop(self.act(self.fc1(x)))
            x = self.drop(self.act(self.fc2(x)))
            return self.fc3(x).squeeze(-1)

    return MCDropoutGate(in_dim)


class MCDropoutGateScorer:
    """Scorer backend that calls a trained MC-dropout NN per-step gate."""

    def __init__(self, bundle_path: Optional[str] = None) -> None:
        import torch

        path = Path(bundle_path) if bundle_path else _DEFAULT_BUNDLE
        if not path.exists():
            raise FileNotFoundError(
                f"MC dropout bundle not found at {path}. "
                "Run mc_dropout_gate.py first."
            )
        # Try weights_only=True (safe, no arbitrary code execution) first.
        # Falls back to weights_only=False only when the bundle contains
        # non-tensor sklearn objects (TfidfVectorizer, TruncatedSVD in
        # nlp_encoder) that torch cannot reconstruct safely. The fallback
        # logs a WARNING — callers should ensure the bundle file is from a
        # trusted source and has not been tampered with.
        try:
            bundle = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception:
            logger.warning(
                "MCDropoutGateScorer: weights_only=True failed for %s "
                "(bundle likely contains sklearn objects); loading with "
                "weights_only=False — ensure bundle is from a trusted source.",
                path,
            )
            bundle = torch.load(str(path), map_location="cpu", weights_only=False)
        self._in_dim = int(bundle["in_dim"])
        self._q_hat = float(bundle["q_hat"])
        self._mc_samples = int(bundle["mc_samples"])
        arch = bundle["arch"]
        self._variant = bundle.get("variant", "combined")
        self._nlp_encoder = bundle.get("nlp_encoder")
        self._model = _build_model(
            self._in_dim, arch["hidden1"], arch["hidden2"], arch["dropout"]
        )
        self._model.load_state_dict(bundle["state_dict"])
        self._model.train()  # keep dropout active for MC sampling
        self._bundle_path = str(path)
        logger.info(
            "MCDropoutGateScorer loaded: in_dim=%d q_hat=%.3f samples=%d variant=%s nlp=%s",
            self._in_dim, self._q_hat, self._mc_samples, self._variant,
            "yes" if self._nlp_encoder else "no",
        )

    @property
    def name(self) -> str:
        return "vaara_mc_dropout_gate"

    @property
    def q_hat(self) -> float:
        return self._q_hat

    @property
    def variant(self) -> str:
        return self._variant

    def _featurize(self, history_pairs, proposed_text):
        """Reproduce the training-time feature vector (30 behavioral + 24 NLP)."""
        from collections import Counter

        import re

        ERROR = re.compile(
            r"traceback|exception|error:|not found|no such file|permission denied|"
            r"command not found|syntax error|undefined variable|cannot|failed to",
            re.IGNORECASE,
        )
        TEST_PASS = re.compile(r"passed|ok\b|success|PASSED", re.IGNORECASE)
        TEST_FAIL = re.compile(r"FAILED|ERRORS|failures=", re.IGNORECASE)

        cmd_types = []
        errors = writes = reads = searches = executes = 0
        test_pass = test_fail = file_refs = 0
        ai_len_sum = env_len_sum = 0.0
        prev_primary = "none"

        for ai_text, env_text in history_pairs:
            cmds = extract_commands(ai_text or "")
            types = [classify_cmd(c) for c in cmds]
            primary = Counter(types).most_common(1)[0][0] if types else "none"
            writes += types.count("write")
            reads += types.count("read")
            searches += types.count("search")
            executes += types.count("execute")
            for cmd in cmds:
                for p in cmd.split()[1:]:
                    if "/" in p or p.endswith((".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".rb")):
                        file_refs += 1
            if ERROR.search(env_text or ""): errors += 1
            if TEST_PASS.search(env_text or ""): test_pass += 1
            if TEST_FAIL.search(env_text or ""): test_fail += 1
            ai_len_sum += len(ai_text or "")
            env_len_sum += len(env_text or "")
            cmd_types.extend(types)
            prev_primary = primary

        K = len(history_pairs)
        denom = max(K, 1)
        hist_unique = len(set(cmd_types)) if cmd_types else 0
        hist_repeat = (sum(1 for i in range(1, len(cmd_types)) if cmd_types[i] == cmd_types[i - 1])
                       / max(len(cmd_types) - 1, 1) if len(cmd_types) > 1 else 0.0)
        hist_err_rate = errors / denom
        denom_c = max(len(cmd_types), 1)
        hist_write_frac = writes / denom_c if cmd_types else 0
        hist_read_frac = reads / denom_c if cmd_types else 0
        hist_search_frac = searches / denom_c if cmd_types else 0
        hist_exec_frac = executes / denom_c if cmd_types else 0
        hist_wandering = ((searches + sum(1 for c in cmd_types if c == "navigate")) / denom_c
                          - hist_write_frac if cmd_types else 0.0)

        cmds = extract_commands(proposed_text)
        types = [classify_cmd(c) for c in cmds]
        primary = Counter(types).most_common(1)[0][0] if types else "none"
        cur_writes = types.count("write")
        cur_reads = types.count("read")
        cur_searches = types.count("search")
        cur_executes = types.count("execute")
        cur_file_refs = 0
        for cmd in cmds:
            for p in cmd.split()[1:]:
                if "/" in p or p.endswith((".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".rb")):
                    cur_file_refs += 1
        cur_ai_len = len(proposed_text)
        is_repeat = 1 if primary == prev_primary and primary != "none" else 0

        feat = np.array([
            K, np.log1p(K), hist_unique, hist_err_rate, hist_write_frac,
            hist_read_frac, hist_search_frac, hist_exec_frac, hist_wandering,
            hist_repeat, test_pass / denom, test_fail / denom,
            np.log1p(file_refs), np.log1p(ai_len_sum / denom),
            np.log1p(env_len_sum / denom),
            len(cmds), cur_writes, cur_reads, cur_searches, cur_executes,
            np.log1p(cur_file_refs), np.log1p(cur_ai_len), is_repeat,
            int(primary == "write"), int(primary == "read"),
            int(primary == "search"), int(primary == "navigate"),
            int(primary == "execute"), int(primary == "submit"),
            int(primary == "other"),
        ], dtype=np.float32)

        if self._nlp_encoder is not None:
            vec = self._nlp_encoder["vec"]
            svd = self._nlp_encoder["svd"]
            nlp_vec = svd.transform(vec.transform([clean_text(proposed_text)]))[0].astype(np.float32)
            feat = np.concatenate([feat, nlp_vec])

        snap = {
            "step_k": K,
            "hist_error_rate": hist_err_rate,
            "cur_executes": cur_executes,
            "cur_primary": primary,
            "is_repeat_primary": bool(is_repeat),
        }
        return feat, snap

    def _mc_sample(self, feat_vec):
        import torch

        X = torch.as_tensor(feat_vec, dtype=torch.float32).unsqueeze(0)
        probs = np.zeros(self._mc_samples, dtype=np.float32)
        with torch.no_grad():
            for s in range(self._mc_samples):
                logits = self._model(X)
                probs[s] = torch.sigmoid(logits).item()
        return probs

    def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
        start = time.monotonic()
        tool_name = context.get("tool_name", "unknown")
        agent_id = context.get("agent_id", "anonymous")
        history = context.get("history_pairs", [])
        proposed = context.get("proposed_action_text", "")

        feat, snap = self._featurize(history, proposed)
        probs = self._mc_sample(feat)
        mean_p = float(probs.mean())
        std_p = float(probs.std())
        import math as _math
        if not _math.isfinite(mean_p) or not _math.isfinite(std_p):
            logger.warning(
                "MC dropout produced non-finite mean_p=%s or std_p=%s "
                "for %s/%s — possible corrupt model weights; defaulting to 0.5",
                mean_p, std_p, agent_id, tool_name,
            )
            mean_p = 0.5 if not _math.isfinite(mean_p) else mean_p
            std_p = 0.5 if not _math.isfinite(std_p) else std_p
        agreement = 1.0 - std_p

        inc_ok = abs(mean_p - 0) <= self._q_hat
        inc_err = abs(mean_p - 1) <= self._q_hat
        if inc_ok and not inc_err:
            verdict = "execute"; decision = Decision.ALLOW
        elif inc_err and not inc_ok:
            verdict = "block"; decision = Decision.DENY
        else:
            verdict = "flag_review"; decision = Decision.ESCALATE

        # Risk attribution (same rules as ActionGate)
        if snap["hist_error_rate"] > 0.3:
            top_risk = f"history has {snap['hist_error_rate']:.0%} error rate"
        elif snap["cur_executes"] > 0:
            top_risk = "action runs code/tests"
        elif snap["is_repeat_primary"]:
            top_risk = "agent is repeating itself"
        else:
            top_risk = "no single dominant risk factor"

        lower = max(0.0, mean_p - self._q_hat)
        upper = min(1.0, mean_p + self._q_hat)
        elapsed_ms = (time.monotonic() - start) * 1000

        signals = {
            "mc_dropout_gate": mean_p,
            "ensemble_agreement": agreement,
        }
        explanation = (
            f"{decision.value}: err_p={mean_p:.3f} agree={agreement:.3f} "
            f"[{lower:.3f}, {upper:.3f}]  {top_risk}"
        )

        assessment = RiskAssessment(
            action_name=tool_name, agent_id=agent_id,
            point_estimate=mean_p, conformal_lower=lower, conformal_upper=upper,
            decision=decision, signals=signals, mwu_weights={},
            threshold_allow=0.0, threshold_deny=1.0, sequence_risk=0.0,
            calibration_size=self._mc_samples, evaluation_ms=elapsed_ms,
            explanation=explanation,
        )
        result = assessment.to_backend_decision()
        result["raw_result"]["verdict"] = verdict
        result["raw_result"]["inside_ok_set"] = bool(inc_ok)
        result["raw_result"]["inside_error_set"] = bool(inc_err)
        result["raw_result"]["top_risk_factor"] = top_risk
        result["raw_result"]["q_hat"] = self._q_hat
        result["raw_result"]["mc_samples"] = self._mc_samples
        result["raw_result"]["mc_std"] = std_p
        result["raw_result"]["variant"] = self._variant
        result["raw_result"]["bundle"] = self._bundle_path
        result["backend"] = self.name
        return result


def create_mc_dropout_scorer(bundle_path: Optional[str] = None) -> MCDropoutGateScorer:
    return MCDropoutGateScorer(bundle_path=bundle_path)
