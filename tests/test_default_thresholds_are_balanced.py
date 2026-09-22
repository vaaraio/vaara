# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The documented default and the actual default were different numbers.

``vaara.policy.modes`` describes balanced as "Vaara's default operating
point (0.55 / 0.85). Current behaviour when no mode is selected." The
scorer, asked for no mode, ran 0.40 / 0.70, which matches no named mode
at all: eco is 0.40/0.60, balanced 0.55/0.85, performance 0.70/0.92,
strict 0.30/0.55. The constructor pair was an orphan.

The cost, measured on the maintainer's machine 2026-09-22. The Claude
Code plugin config had no ``protection`` key, so ``protection_preset()``
returned None, no policy was applied, and the scorer used 0.40/0.70.
Every MCP call escalated, because an upper conformal bound of 0.427
clears 0.40 and does not clear 0.55. Writing ``protection: balanced``
made the same call return allow. Nothing was misconfigured. The default
was.

The macOS client had it right and was reporting the truth about a
different number: ``readPluginPreset()`` returns "balanced" when the key
is absent, so the UI showed balanced while the engine ran eco-ish
thresholds nobody chose.

One authoritative default, sourced from the mode table.
"""

from __future__ import annotations

import pytest

from vaara.policy.modes import available_modes, get_mode
from vaara.scorer.adaptive import (
    DEFAULT_THRESHOLD_ALLOW,
    DEFAULT_THRESHOLD_DENY,
    AdaptiveScorer,
)

BALANCED = get_mode("balanced")


class TestTheDefaultIsBalanced:
    def test_the_constants_come_from_the_mode_table(self):
        assert DEFAULT_THRESHOLD_ALLOW == BALANCED.escalate
        assert DEFAULT_THRESHOLD_DENY == BALANCED.deny

    def test_a_scorer_built_with_no_arguments_runs_balanced(self):
        scorer = AdaptiveScorer()
        assert scorer._threshold_allow == BALANCED.escalate
        assert scorer._threshold_deny == BALANCED.deny

    def test_the_documented_numbers_have_not_moved(self):
        """modes.py states 0.55 / 0.85 in prose. Pin the prose."""
        assert (BALANCED.escalate, BALANCED.deny) == (0.55, 0.85)

    def test_explicit_thresholds_still_win(self):
        scorer = AdaptiveScorer(threshold_allow=0.1, threshold_deny=0.2)
        assert scorer._threshold_allow == 0.1
        assert scorer._threshold_deny == 0.2


class TestTheOldDefaultWasNobodysMode:
    """0.40 / 0.70 was not a policy choice, it was a pair of literals."""

    def test_no_named_mode_uses_the_old_pair(self):
        for name in available_modes():
            mode = get_mode(name)
            assert (mode.escalate, mode.deny) != (0.40, 0.70), (
                f"mode {name} matches the old constructor defaults; "
                "this test assumed they belonged to no mode"
            )

    def test_selecting_balanced_changes_nothing(self):
        """The claim modes.py makes about itself, as a test.

        "Current Vaara behaviour" has to mean that applying balanced to a
        default scorer is a no-op. It was not.
        """
        from vaara.policy import from_dict
        from vaara.policy.modes import to_policy_dict

        default = AdaptiveScorer()
        explicit = AdaptiveScorer()
        explicit.apply_policy(from_dict(to_policy_dict(get_mode("balanced"))))

        assert (default._threshold_allow, default._threshold_deny) == (
            explicit._threshold_allow, explicit._threshold_deny
        )


class TestTheHookPathAgreesWithTheApp:
    """A missing `protection` key must mean balanced on both sides."""

    def test_the_engine_reads_no_preset_from_an_empty_config(self):
        from vaara.integrations.claude_code_hooks import protection_preset

        assert protection_preset({}) is None

    def test_and_an_absent_preset_still_lands_on_balanced(self):
        """The app shows "balanced" for a missing key. Now that is true."""
        scorer = AdaptiveScorer()
        assert (scorer._threshold_allow, scorer._threshold_deny) == (
            BALANCED.escalate, BALANCED.deny
        )

    @pytest.mark.parametrize("upper", [0.427])
    def test_the_measured_call_no_longer_sits_in_the_escalate_band(self, upper):
        """0.427 was the upper bound that escalated every MCP call.

        It clears 0.40 and does not clear 0.55, which is the whole bug in
        one number.
        """
        assert upper > 0.40
        assert upper < BALANCED.escalate


class TestTheInstallWritesThePreset:
    """A fresh install says its operating point out loud."""

    def test_a_new_config_gets_the_preset(self, tmp_path):
        import json

        from vaara.integrations.init_governance import (
            DEFAULT_PROTECTION_PRESET,
            write_hook_config,
        )

        config = tmp_path / "config.json"
        write_hook_config(config, tmp_path / "audit.db")
        written = json.loads(config.read_text())
        assert written["protection"] == DEFAULT_PROTECTION_PRESET
        assert written["audit_db"] == str(tmp_path / "audit.db")

    def test_the_preset_written_is_a_real_mode(self):
        from vaara.integrations.init_governance import DEFAULT_PROTECTION_PRESET

        assert DEFAULT_PROTECTION_PRESET in available_modes()

    def test_an_operators_choice_is_not_overwritten(self, tmp_path):
        import json

        from vaara.integrations.init_governance import write_hook_config

        config = tmp_path / "config.json"
        config.write_text(json.dumps({"protection": "strict", "mode": "watch"}))
        write_hook_config(config, tmp_path / "audit.db")
        written = json.loads(config.read_text())
        assert written["protection"] == "strict"
        assert written["mode"] == "watch"

    def test_what_the_engine_then_reads_back(self, tmp_path):
        """End to end: install writes it, the hook config reader sees it."""
        import json

        from vaara.integrations.claude_code_hooks import protection_preset
        from vaara.integrations.init_governance import write_hook_config

        config = tmp_path / "config.json"
        write_hook_config(config, tmp_path / "audit.db")
        assert protection_preset(json.loads(config.read_text())) == "balanced"
