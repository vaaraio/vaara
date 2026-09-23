"""/vaara-setup lists the presets in the order their thresholds put them.

The command offered "Paranoid" as the tightest blocking and mapped it to
`eco`, while `strict` holds and denies at lower scores than `eco` on both
thresholds. A user asking for the most protection got less than "Strict".
The order here is read from the mode table, not asserted by hand.
"""
from __future__ import annotations

import re
from pathlib import Path

from vaara.policy.modes import get_mode

SETUP = (Path(__file__).resolve().parents[1]
         / "plugins/claude-code-vaara-governance/commands/vaara-setup.md").read_text()


def _offered_order() -> list[str]:
    line = next(ln for ln in SETUP.splitlines() if "Options mapped to presets" in ln)
    return re.findall(r'\((performance|balanced|eco|strict):', line)


def test_presets_are_offered_loosest_first():
    order = _offered_order()
    assert sorted(order) == ["balanced", "eco", "performance", "strict"]
    thresholds = [(get_mode(n).escalate, get_mode(n).deny) for n in order]
    assert thresholds == sorted(thresholds, reverse=True), list(zip(order, thresholds))


def test_the_option_called_tightest_is_the_tightest_preset():
    tightest = min(("performance", "balanced", "eco", "strict"),
                   key=lambda n: (get_mode(n).deny, get_mode(n).escalate))
    line = next(ln for ln in SETUP.splitlines() if "Options mapped to presets" in ln)
    assert re.search(rf"\({tightest}: the tightest", line)
