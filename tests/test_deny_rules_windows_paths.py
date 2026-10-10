# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Path rules hold for Windows paths.

The patterns are written with ``/``. On Windows a harness hands the hook
``C:\\Users\\me\\.ssh\\id_ed25519``, which no ``\\.ssh/`` pattern matched,
so a read of the user's private key passed the secret-material rule.
"""
from __future__ import annotations

from vaara.deny_rules import load_deny_rules, match_deny_rule, match_deny_rule_any_field

RULES = load_deny_rules()

KEY = "C:\\Users\\me\\.ssh\\id_ed25519"


def test_a_windows_key_path_is_refused_by_tool_name():
    hit = match_deny_rule(RULES, "Read", {"file_path": KEY})
    assert hit is not None and hit[0] == "secret_material_read"


def test_a_windows_key_path_is_refused_through_a_harness_alias():
    hit = match_deny_rule(RULES, "view", {"path": KEY})
    assert hit is not None and hit[0] == "secret_material_read"


def test_a_windows_key_path_is_refused_in_an_mcp_argument():
    hit = match_deny_rule_any_field(RULES, {"path": KEY})
    assert hit is not None and hit[0] == "secret_material_read"


def test_an_ordinary_windows_path_still_passes():
    assert match_deny_rule(RULES, "Read", {"file_path": "C:\\work\\notes.txt"}) is None
