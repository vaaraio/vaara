"""
Runtime action gate — inference wrapper around the per-step scorer.

Loads a saved ensemble + conformal q_hat and evaluates a proposed action
given history. This is the shape Vaara runtime wrappers ship as a library.

Usage:
    from runtime_gate import ActionGate
    gate = ActionGate.load("cache/perstep_v2.joblib")
    decision = gate.evaluate(history_steps, proposed_step)
    # decision.verdict in {"execute", "flag_review", "block"}
    # decision.error_prob is the ensemble mean probability of error
    # decision.agreement is 1 - std across seeds
    # decision.reason names the dominant risk factor
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np

ERROR_PATTERNS = re.compile(
    r"traceback|exception|error:|not found|no such file|permission denied|"
    r"command not found|syntax error|undefined variable|cannot|failed to",
    re.IGNORECASE,
)
TEST_PASS = re.compile(r"passed|ok\b|success|PASSED", re.IGNORECASE)
TEST_FAIL = re.compile(r"FAILED|ERRORS|failures=", re.IGNORECASE)

CMD_TYPES = ["read", "write", "search", "navigate", "execute", "submit", "other"]


def extract_commands(text: str):
    blocks = re.findall(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
    out = []
    for block in blocks:
        for line in block.strip().split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def classify_cmd(cmd: str) -> str:
    c = cmd.lower().strip()
    if c.startswith(("edit ", "sed ", "patch ", "echo ", "printf ")):
        return "write"
    if c.startswith(("open ", "cat ", "head ", "tail ", "less ")):
        return "read"
    if c.startswith(("find ", "grep ", "rg ", "ag ", "ls ", "tree ", "search ")):
        return "search"
    if c.startswith(("cd ",)):
        return "navigate"
    if c.startswith(("python ", "pytest ", "make ", "npm ", "cargo ", "go ", "ruby ")):
        return "execute"
    if c.startswith(("submit",)):
        return "submit"
    return "other"


def clean_text(t: str) -> str:
    t = re.sub(r"```.*?```", " ", t, flags=re.DOTALL)
    t = re.sub(r"[0-9a-fA-F]{6,}", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


@dataclass
class GateDecision:
    verdict: str                  # execute | flag_review | block
    error_prob: float             # ensemble mean probability of error
    agreement: float              # 1 - ensemble std
    inside_error_set: bool        # conformal set includes ERROR
    inside_ok_set: bool           # conformal set includes OK
    top_risk_factor: str
    feature_snapshot: dict


class ActionGate:
    """Wraps a trained per-step ensemble + conformal q_hat."""

    def __init__(self, models, q_hat: float,
                 nlp_encoder: Optional[dict] = None,
                 feature_names: Optional[list] = None):
        self.models = models
        self.q_hat = float(q_hat)
        self.nlp_encoder = nlp_encoder
        self.feature_names = feature_names or []

    @classmethod
    def load(cls, path: str) -> "ActionGate":
        bundle = joblib.load(path)
        return cls(
            models=bundle["models"],
            q_hat=bundle["q_hat"],
            nlp_encoder=bundle.get("nlp_encoder"),
            feature_names=bundle.get("feature_names", []),
        )

    def save(self, path: str):
        joblib.dump({
            "models": self.models,
            "q_hat": self.q_hat,
            "nlp_encoder": self.nlp_encoder,
            "feature_names": self.feature_names,
        }, path)

    @staticmethod
    def _prefix_stats(history_pairs):
        """Compute running statistics over history (ai_text, env_text) pairs."""
        cmd_types = []
        errors = 0
        writes = reads = searches = executes = 0
        test_pass = test_fail = 0
        file_refs = 0
        ai_len_sum = 0.0
        env_len_sum = 0.0
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
            if ERROR_PATTERNS.search(env_text or ""): errors += 1
            if TEST_PASS.search(env_text or ""): test_pass += 1
            if TEST_FAIL.search(env_text or ""): test_fail += 1
            ai_len_sum += len(ai_text or "")
            env_len_sum += len(env_text or "")
            cmd_types.extend(types)
            prev_primary = primary
        return {
            "cmd_types": cmd_types, "errors": errors,
            "writes": writes, "reads": reads, "searches": searches, "executes": executes,
            "test_pass": test_pass, "test_fail": test_fail,
            "file_refs": file_refs, "ai_len_sum": ai_len_sum, "env_len_sum": env_len_sum,
            "prev_primary": prev_primary,
        }

    def _featurize(self, history_pairs, proposed_ai_text):
        """Return feature vector + cleaned proposed text."""
        h = self._prefix_stats(history_pairs)
        K = len(history_pairs)
        denom = max(K, 1)
        cmd_types = h["cmd_types"]

        hist_unique = len(set(cmd_types)) if cmd_types else 0
        hist_repeat = (sum(1 for i in range(1, len(cmd_types)) if cmd_types[i] == cmd_types[i - 1])
                       / max(len(cmd_types) - 1, 1) if len(cmd_types) > 1 else 0.0)
        hist_err_rate = h["errors"] / denom
        denom_c = max(len(cmd_types), 1)
        hist_write_frac = h["writes"] / denom_c if cmd_types else 0
        hist_read_frac = h["reads"] / denom_c if cmd_types else 0
        hist_search_frac = h["searches"] / denom_c if cmd_types else 0
        hist_exec_frac = h["executes"] / denom_c if cmd_types else 0
        hist_wandering = ((h["searches"] + sum(1 for c in cmd_types if c == "navigate")) / denom_c
                          - hist_write_frac if cmd_types else 0.0)

        # Current proposed action
        cmds = extract_commands(proposed_ai_text)
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
        cur_ai_len = len(proposed_ai_text)
        is_repeat = 1 if primary == h["prev_primary"] and primary != "none" else 0

        feat = np.array([
            K, np.log1p(K), hist_unique, hist_err_rate, hist_write_frac,
            hist_read_frac, hist_search_frac, hist_exec_frac, hist_wandering,
            hist_repeat, h["test_pass"] / denom, h["test_fail"] / denom,
            np.log1p(h["file_refs"]), np.log1p(h["ai_len_sum"] / denom),
            np.log1p(h["env_len_sum"] / denom),
            len(cmds), cur_writes, cur_reads, cur_searches, cur_executes,
            np.log1p(cur_file_refs), np.log1p(cur_ai_len), is_repeat,
            int(primary == "write"), int(primary == "read"),
            int(primary == "search"), int(primary == "navigate"),
            int(primary == "execute"), int(primary == "submit"),
            int(primary == "other"),
        ], dtype=np.float32)

        if self.nlp_encoder is not None:
            vec = self.nlp_encoder["vec"]
            svd = self.nlp_encoder["svd"]
            nlp_vec = svd.transform(vec.transform([clean_text(proposed_ai_text)]))[0].astype(np.float32)
            feat = np.concatenate([feat, nlp_vec])

        snap = {
            "step_k": K,
            "hist_error_rate": hist_err_rate,
            "cur_executes": cur_executes,
            "cur_primary": primary,
            "is_repeat_primary": bool(is_repeat),
        }
        return feat, snap

    def evaluate(self, history_pairs, proposed_ai_text: str) -> GateDecision:
        feat, snap = self._featurize(history_pairs, proposed_ai_text)
        X = feat.reshape(1, -1)
        probs = np.array([m.predict_proba(X)[0, 1] for m in self.models])
        mean_p = float(probs.mean())
        std_p = float(probs.std())
        agreement = 1.0 - std_p

        inc_ok = abs(mean_p - 0) <= self.q_hat
        inc_err = abs(mean_p - 1) <= self.q_hat

        if inc_ok and not inc_err:
            verdict = "execute"
        elif inc_err and not inc_ok:
            verdict = "block"
        else:
            verdict = "flag_review"

        # Risk factor attribution: simplified — pick the feature most over its mean
        if snap["hist_error_rate"] > 0.3:
            top_risk = f"history has {snap['hist_error_rate']:.0%} error rate"
        elif snap["cur_executes"] > 0:
            top_risk = "action runs code/tests"
        elif snap["is_repeat_primary"]:
            top_risk = "agent is repeating itself"
        else:
            top_risk = "no single dominant risk factor"

        return GateDecision(
            verdict=verdict, error_prob=mean_p, agreement=agreement,
            inside_error_set=bool(inc_err), inside_ok_set=bool(inc_ok),
            top_risk_factor=top_risk, feature_snapshot=snap,
        )


def _cli():
    import argparse
    p = argparse.ArgumentParser(description="Score a proposed agent action.")
    p.add_argument("bundle", help="Path to .joblib bundle saved by ActionGate.save()")
    p.add_argument("trajectory_json", help="JSON file with {trajectory: [{role, text}, ...]}")
    p.add_argument("--cutoff", type=int, default=-1,
                   help="Evaluate at step K (uses first K ai-turns as history, K+1 as proposed). -1 = last step.")
    args = p.parse_args()

    gate = ActionGate.load(args.bundle)
    with open(args.trajectory_json) as f:
        traj = json.load(f)["trajectory"]

    # Build (ai, env) pairs
    pairs = []
    i = 0
    while i < len(traj):
        if traj[i].get("role") == "ai":
            env = traj[i + 1] if i + 1 < len(traj) and traj[i + 1].get("role") == "user" else {}
            pairs.append((traj[i].get("text") or "", env.get("text") or ""))
            i += 2
        else:
            i += 1

    if args.cutoff < 0 or args.cutoff >= len(pairs):
        K = len(pairs) - 1
    else:
        K = args.cutoff

    history = pairs[:K]
    proposed = pairs[K][0]

    decision = gate.evaluate(history, proposed)
    print(f"verdict:          {decision.verdict}")
    print(f"error_prob:       {decision.error_prob:.3f}")
    print(f"agreement:        {decision.agreement:.3f}")
    print(f"inside_ok_set:    {decision.inside_ok_set}")
    print(f"inside_error_set: {decision.inside_error_set}")
    print(f"top_risk_factor:  {decision.top_risk_factor}")
    print(f"snapshot:         {decision.feature_snapshot}")


if __name__ == "__main__":
    _cli()
