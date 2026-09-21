# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the gate bundles that the three trained scorer backends load.

``TrainedGateScorer``, ``MCDropoutGateScorer`` and ``StackedGateScorer`` each
open a ``.joblib`` file under ``~/.vaara/cache/``. No such file ships in the
repository and no script in the tree produces one, so before this module the
twelve tests covering those 586 lines skipped on every machine and every CI
leg: the backends were executed nowhere.

The bundles built here are small and synthetic, but they are real artefacts in
the real format. The training rows come out of the shipped featurizer
(``ActionGate._featurize``) run over hand-written trajectories, so the feature
space a bundle is fitted on is the same one ``evaluate()`` produces at call
time. That is the part worth testing: a bundle fitted on some other layout
would load fine and then score nonsense.

These are fixtures, not a training pipeline. They say nothing about how the
production bundle should be fitted or how well it performs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Benign steps: reads and searches. Risky steps: writes and code execution.
# classify_cmd() maps them to different primary-command one-hots, so the two
# groups are separable and a tiny model reaches confident probabilities.
_BENIGN_ACTIONS = [
    "```\ncat README.md\n```",
    "```\nopen src/vaara/cli.py\n```",
    "```\nls src/vaara\n```",
    "```\ngrep -rn parse_args src\n```",
    "```\nfind . -name conftest.py\n```",
    "```\nhead -n 40 pyproject.toml\n```",
]

_RISKY_ACTIONS = [
    "```\nsed -i s/False/True/ src/vaara/policy/engine.py\n```",
    "```\nedit src/vaara/audit/trail.py\n```",
    "```\npython scripts/migrate.py --force\n```",
    "```\npytest tests/ -x\n```",
    "```\nmake install\n```",
    "```\necho token > ~/.vaara/config.json\n```",
]

# Three prefixes: an empty one, a clean one, and one carrying tool errors.
# hist_err_rate and the length features differ across them, so the rows are
# not twelve copies of the same vector with two labels.
_HISTORIES: list[list[tuple[str, str]]] = [
    [],
    [
        ("I'll look at the layout first.", "src/ tests/ pyproject.toml"),
        ("```\nls src/vaara\n```", "cli.py pipeline.py policy/ audit/"),
    ],
    [
        ("```\npython app.py\n```", "Traceback (most recent call last): ImportError"),
        ("```\ncat requirements.txt\n```", "no such file or directory"),
        ("```\npytest\n```", "FAILED tests/test_x.py::test_y - AssertionError"),
    ],
]


def _training_matrix():
    """Featurize the synthetic trajectories through the shipped featurizer."""
    import numpy as np

    from vaara.scorer.action_gate import ActionGate

    # models and q_hat are unused by _featurize; this instance is a featurizer.
    featurizer = ActionGate(models=[], q_hat=0.0)

    rows, labels = [], []
    for history in _HISTORIES:
        for action in _BENIGN_ACTIONS:
            rows.append(featurizer._featurize(history, action)[0])
            labels.append(0)
        for action in _RISKY_ACTIONS:
            rows.append(featurizer._featurize(history, action)[0])
            labels.append(1)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def feature_dim() -> int:
    """Width of the behavioral feature vector, with no NLP encoder attached."""
    return int(_training_matrix()[0].shape[1])


def build_trained_gate_bundle(path: Path, *, q_hat: float = 0.35,
                              n_models: int = 3, seed: int = 0) -> Path:
    """Fit a small bootstrap ensemble and save it in ``ActionGate`` format."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    from vaara.scorer.action_gate import ActionGate

    X, y = _training_matrix()
    rng = np.random.default_rng(seed)

    models = []
    for i in range(n_models):
        # Resample until the draw holds both labels. A single-class fit gives
        # predict_proba a width of one, and ActionGate.evaluate indexes [0, 1].
        for _ in range(100):
            idx = rng.integers(0, len(y), size=len(y))
            if len(set(y[idx].tolist())) == 2:
                break
        else:  # pragma: no cover - 100 draws from a balanced set all one class
            raise RuntimeError("bootstrap resample never held both labels")
        model = LogisticRegression(max_iter=1000, C=1.0 + i, random_state=seed + i)
        model.fit(X[idx], y[idx])
        models.append(model)

    gate = ActionGate(models=models, q_hat=q_hat, nlp_encoder=None,
                      feature_names=[f"f{i}" for i in range(X.shape[1])])
    path.parent.mkdir(parents=True, exist_ok=True)
    gate.save(str(path))
    return path


def build_mc_dropout_bundle(path: Path, *, q_hat: float = 0.35, mc_samples: int = 16,
                            dropout: float = 0.1, epochs: int = 300,
                            seed: int = 0) -> Path:
    """Train the shipped MC-dropout architecture and save it in bundle format."""
    import torch

    from vaara.scorer.mc_dropout_gate import _build_model

    X, y = _training_matrix()
    arch = {"hidden1": 32, "hidden2": 16, "dropout": dropout}

    torch.manual_seed(seed)
    model = _build_model(X.shape[1], arch["hidden1"], arch["hidden2"], arch["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss_fn(model(xt), yt).backward()
        optimizer.step()

    bundle: dict[str, Any] = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "in_dim": int(X.shape[1]),
        "q_hat": float(q_hat),
        "variant": "behavioral",
        "arch": arch,
        "mc_samples": int(mc_samples),
        "nlp_encoder": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, str(path))
    return path


def build_stacked_bundle(path: Path, *, gbm_bundle: Path, mc_bundle: Path,
                         q_hat: float = 0.35, stack_coef: tuple[float, float] = (4.0, 4.0),
                         stack_intercept: float = -4.0) -> Path:
    """Write the LR stack that composes the two bundles above."""
    import joblib

    bundle = {
        "variant": "stacked",
        "stack_coef": [float(stack_coef[0]), float(stack_coef[1])],
        "stack_intercept": float(stack_intercept),
        "q_hat": float(q_hat),
        "gbm_bundle_path": str(gbm_bundle),
        "mc_bundle_path": str(mc_bundle),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, str(path))
    return path


# Contexts the backend tests score. Kept here so all three files agree on what
# a benign step and a risky step look like.
BENIGN_CONTEXT = {
    "tool_name": "read_file",
    "agent_id": "test-agent",
    "history_pairs": _HISTORIES[1],
    "proposed_action_text": "```\ncat README.md\n```",
}

RISKY_CONTEXT = {
    "tool_name": "edit_file",
    "agent_id": "test-agent",
    "history_pairs": _HISTORIES[2],
    "proposed_action_text": "```\nsed -i s/False/True/ src/vaara/policy/engine.py\n```",
}
