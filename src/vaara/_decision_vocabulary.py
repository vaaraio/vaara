"""The decision vocabulary, and its projection onto the three that are recorded.

AARM Core R4 asks for five decisions: ALLOW, DENY, MODIFY, STEP_UP and DEFER.
The signed decision record carries three, and that set is closed inside
checkers this repository publishes and outside parties run
(``tests/vectors/record_set_v0/_check_independent.py``). A record carrying any
other verdict is graded non-conforming, so renaming one of the three is a
wire-format break rather than a refactor.

MODIFY, STEP_UP and DEFER are therefore policy-layer names that project onto
the coarse three. This module is the single place that projection is defined.
It imports nothing, so the enforcement path (``vaara.pipeline``) and the
evidence path (``vaara.audit.trail``) can both hold the same table without one
depending on the other or on the scorer.

The enum itself lives with the scorer (``vaara.scorer.adaptive.Decision``);
``tests/test_decision_vocabulary.py`` pins the two to each other.
"""

from __future__ import annotations

# The only verdicts a decision record ever carries. Frozen.
COARSE: frozenset[str] = frozenset({"allow", "deny", "escalate"})

# The R4 refinements. Each names WHY a coarse decision was reached; none of
# them is permissive on its own.
REFINEMENTS: frozenset[str] = frozenset({"modify", "step_up", "defer"})

# MODIFY projects to `deny` because the arguments it was asked about do not
# run. The altered arguments come back to the caller and are resubmitted as a
# fresh decision bound to their own digest, so no decision record ever says
# `allow` against arguments other than the ones that executed.
FINE_TO_COARSE: dict[str, str] = {
    "allow": "allow",
    "deny": "deny",
    "escalate": "escalate",
    "modify": "deny",
    "step_up": "escalate",
    "defer": "escalate",
}
