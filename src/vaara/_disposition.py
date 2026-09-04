"""Who disposed of a decision, as a closed vocabulary rather than as prose.

A decision record already says WHAT was decided (``allow``, ``deny``,
``escalate``) and, through ``decision_detail``, which refinement produced it.
It did not say what KIND of party disposed of it, and three cases that a
relying party must treat differently all surfaced as ``allow``:

* policy allowed the action outright;
* a human approved it at escalation;
* a prior human approval was replayed from the trail by
  :meth:`~vaara.audit.trail.AuditTrail.find_prior_approval` inside its window.

The third is a policy disposition wearing a human's earlier decision. The
pipeline has always recorded it honestly, but only in the free-text ``reason``
string ("auto-allowed by prior approval (action_id=..., ...)"), so separating
it from a live human approval meant parsing English. This module makes the
distinction a field.

The vocabulary is CLOSED and is not registry-governed. An unrecognised
``approver`` is not a conforming disposition, which is deliberate: a value an
implementation may invent is a value a relying party cannot branch on.

The invariant that earns the module: ``human_disposed`` is true ONLY when a
human actually acted. A producer MUST NOT claim a human disposed what a policy
did, so ``human_disposed=True`` requires ``approver="human"``.

Imports nothing, so the enforcement path (``vaara.pipeline``) and the evidence
path (``vaara.audit.trail``) can hold the same table without one depending on
the other.
"""

from __future__ import annotations

#: The only approver kinds a conforming disposition carries. Closed.
APPROVER: frozenset[str] = frozenset({"human", "policy"})

#: Recorded when a policy disposed of the action with no human in the loop.
POLICY = "policy"

#: Recorded when a human actually acted on this action, at this decision.
HUMAN = "human"


class DispositionError(ValueError):
    """Raised when a disposition would misreport who acted."""


def check(approver: str, human_disposed: bool) -> tuple[str, bool]:
    """Validate a disposition pair and return it normalised.

    Raises :class:`DispositionError` rather than coercing, because every
    failure mode here is a record that would overstate human involvement.
    A silent coercion would produce exactly the claim this module exists to
    prevent.
    """
    if not isinstance(approver, str):
        raise DispositionError(
            f"approver must be a string, got {type(approver).__name__}"
        )
    if not isinstance(human_disposed, bool):
        # A truthy non-bool (1, "yes", a non-empty object) must not become
        # a claim that a human acted.
        raise DispositionError(
            f"human_disposed must be a bool, got {type(human_disposed).__name__}"
        )
    normalized = approver.strip().lower()
    if normalized not in APPROVER:
        raise DispositionError(
            f"approver {approver!r} is not one of {sorted(APPROVER)}; "
            "the vocabulary is closed and an unknown value is non-conforming"
        )
    if human_disposed and normalized != HUMAN:
        raise DispositionError(
            f"human_disposed=True requires approver='human', got {normalized!r}; "
            "a producer must not claim a human disposed what a policy did"
        )
    return normalized, human_disposed
