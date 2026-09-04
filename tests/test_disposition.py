# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The disposition vocabulary, and that the three allow-shaped cases separate.

The point of these tests is not that a boolean exists. It is that a relying
party holding only the record can tell a live human approval from a replayed
one WITHOUT reading prose. Every assertion below compares field identity, not
truthiness, which is the rule the transparency_consistency_v0 suite already
encodes.
"""

import pytest

from vaara._disposition import APPROVER, DispositionError, check


class TestVocabulary:
    def test_the_set_is_closed_and_exactly_two(self):
        assert APPROVER == {"human", "policy"}

    @pytest.mark.parametrize("approver", ["human", "policy"])
    def test_both_members_pass_with_matching_flag(self, approver):
        expected = approver == "human"
        assert check(approver, expected) == (approver, expected)

    def test_case_and_whitespace_normalise(self):
        assert check("  HUMAN ", True) == ("human", True)

    @pytest.mark.parametrize(
        "approver", ["counterparty", "auto", "system", "Dr. Smith", ""],
    )
    def test_unknown_approver_is_rejected_not_tolerated(self, approver):
        with pytest.raises(DispositionError, match="closed"):
            check(approver, False)


class TestTheInvariant:
    def test_policy_cannot_claim_a_human_acted(self):
        with pytest.raises(DispositionError, match="must not claim a human"):
            check("policy", True)

    def test_human_approver_may_record_a_false_flag(self):
        # A human reviewed and the disposition was still automatic; allowed,
        # because the flag only ever narrows the claim.
        assert check("human", False) == ("human", False)

    @pytest.mark.parametrize("truthy", [1, "yes", [1], object()])
    def test_truthy_non_bool_cannot_become_a_human_claim(self, truthy):
        # A caller passing 1 rather than True must not silently produce
        # "a human acted". This is the coercion that would defeat the module.
        with pytest.raises(DispositionError, match="must be a bool"):
            check("human", truthy)

    def test_falsy_non_bool_is_also_rejected(self):
        with pytest.raises(DispositionError, match="must be a bool"):
            check("policy", 0)

    def test_non_string_approver_is_rejected(self):
        with pytest.raises(DispositionError, match="must be a string"):
            check(None, False)


class TestNoCoercion:
    """Every failure here would produce a record overstating human involvement,
    so the module raises rather than repairing."""

    def test_check_never_returns_a_repaired_pair(self):
        for approver, flag in [("policy", True), ("nonsense", False)]:
            with pytest.raises(DispositionError):
                check(approver, flag)
