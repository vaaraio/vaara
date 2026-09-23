# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Point the whole suite at a throwaway home directory.

``InterceptionPipeline()`` with no trail resolves ``Path.home() / ".vaara" /
"trail" / "audit.db"``, and roughly a dozen other modules bind ``Path.home()``
into a module-level constant for approvals, policy, config, gate bundles and
the proxy trails. There was no conftest, so running ``pytest`` on a machine
that also runs Vaara appended test records to the operator's real audit trail.

For a product whose claim is an evidence chain nobody has edited, a test run
that writes into it is the wrong kind of surprise. It also made the suite
flaky: a live hook writing to the same SQLite file while tests opened it
produced ``disk I/O error`` and 58 failures that had nothing to do with the
code under test.

The redirect happens at import time rather than in a fixture. pytest imports
conftest before it imports any test module, and constants like
``approvals.APPROVALS_DIR`` are bound at *their* import, so a fixture would run
too late to move them.

Set ``VAARA_TEST_USE_REAL_HOME=1`` to opt out, for the rare case of debugging
against a real trail on purpose.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

if not os.environ.get("VAARA_TEST_USE_REAL_HOME"):
    _sandbox = Path(tempfile.mkdtemp(prefix="vaara-test-home-"))
    (_sandbox / ".vaara").mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(_sandbox)
    # Path.home() reads HOME first but falls back to the password database,
    # and pathlib caches nothing, so USERPROFILE matters on Windows runners.
    os.environ["USERPROFILE"] = str(_sandbox)
    # OpenCode's config directory follows XDG_CONFIG_HOME before HOME, and
    # `vaara init` installs a plugin there.
    os.environ["XDG_CONFIG_HOME"] = str(_sandbox / ".config")
    # Anything reading the trail path from the environment follows the same
    # sandbox rather than the operator's file.
    os.environ.setdefault("VAARA_DB", str(_sandbox / ".vaara" / "test-audit.db"))


# --- An operating point that escalates -------------------------------------
#
# A test whose subject is what happens AFTER an escalation needs its sample
# action to land in the escalate band. That used to come free: the scorer's
# constructor defaulted to 0.40 / 0.70, and `tx.transfer` and
# `phy.safety_override` both cleared 0.40 on the conformal upper bound.
#
# The default is now balanced, 0.55 / 0.85, sourced from the mode table. Those
# same actions score under 0.55 and auto-allow, so every approval-path test
# that relied on the default lost its precondition. Twenty-five of them, across
# three files, all failing on the setup line rather than on anything they were
# written to check.
#
# Pinning it here rather than tracking the default on purpose. These tests are
# about the approval machinery, and their meaning should not move when the
# shipped operating point is retuned. What the default IS belongs in
# `tests/test_default_thresholds_are_balanced.py`, which asserts it directly.

ESCALATING_THRESHOLDS = {"threshold_allow": 0.40, "threshold_deny": 0.70}


# --- Gate bundles for the three trained scorer backends ---------------------
#
# Each backend loads a bundle from ``~/.vaara/cache/``. The redirect above
# points that at a fresh temporary home, so the path is empty by construction
# and the twelve tests over those backends skipped on every run. The fixtures
# below build a real bundle in the real format instead. Set the matching
# ``VAARA_*_BUNDLE`` variable to score a production bundle rather than one of
# these; a path that is set but missing is an error, not a silent fallback.


def _bundle_override(env_var: str) -> Path | None:
    value = os.environ.get(env_var)
    if not value:
        return None
    path = Path(value)
    if not path.exists():
        raise RuntimeError(f"{env_var} is set to {path}, which does not exist")
    return path


@pytest.fixture(scope="session")
def trained_gate_bundle(tmp_path_factory):
    pytest.importorskip("sklearn", reason="trained gate needs the ml extra")
    pytest.importorskip("joblib", reason="trained gate needs the ml extra")
    from tests.gate_bundle_factory import build_trained_gate_bundle

    override = _bundle_override("VAARA_TRAINED_GATE_BUNDLE")
    if override is not None:
        return override
    target = tmp_path_factory.mktemp("gate-bundles") / "perstep_gate_bundle.joblib"
    return build_trained_gate_bundle(target)


@pytest.fixture(scope="session")
def mc_dropout_gate_bundle(tmp_path_factory):
    pytest.importorskip("torch", reason="mc dropout gate needs torch")
    from tests.gate_bundle_factory import build_mc_dropout_bundle

    override = _bundle_override("VAARA_MC_DROPOUT_BUNDLE")
    if override is not None:
        return override
    target = tmp_path_factory.mktemp("gate-bundles") / "mc_dropout_gate_bundle.joblib"
    return build_mc_dropout_bundle(target)


@pytest.fixture(scope="session")
def stacked_gate_bundle(tmp_path_factory, trained_gate_bundle, mc_dropout_gate_bundle):
    from tests.gate_bundle_factory import build_stacked_bundle

    override = _bundle_override("VAARA_STACKED_GATE_BUNDLE")
    if override is not None:
        return override
    target = tmp_path_factory.mktemp("gate-bundles") / "stacked_gate_bundle.joblib"
    return build_stacked_bundle(
        target, gbm_bundle=trained_gate_bundle, mc_bundle=mc_dropout_gate_bundle
    )
