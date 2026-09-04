# decision_disposition_v0

Nine cases covering who disposed of a decision, and whether a record can be
made to overstate human involvement.

## What this suite is for

A decision record says WHAT was decided: `allow`, `deny`, `escalate`. Three
different things produce `allow`, and a relying party has to treat them
differently:

1. policy allowed the action outright
2. a human approved it at escalation
3. a prior human approval was replayed by cache lookup inside its window

Case 3 is a policy disposition wearing a human's earlier decision. The human
acted on a different action, at a different time, and consented to a different
instance of the same argument shape.

Before this vocabulary all three surfaced as `allow`, and the only thing
separating them was English in the `reason` field
(`"auto-allowed by prior approval (action_id=..., ...)"`). A record that is
honest only to a reader who parses prose is not machine-checkable, which is the
same defect one level down from the one the `transparency_consistency_v0` suite
exists for.

## The rules, reproduced in `_check_independent.py`

**The closed set.** `approver` is exactly `human` or `policy`. An unknown value
is non-conforming rather than tolerated. A value an implementation may invent
is a value a relying party cannot branch on, so this vocabulary is not
registry-governed and is not expected to grow by registration.

**The honest flag.** `human_disposed` is true ONLY when a human actually acted.

- It must be a real boolean. A truthy `1` or `"yes"` is non-conforming and
  never becomes a human claim.
- `human_disposed: true` requires `approver: "human"`. A producer must not
  claim a human disposed what a policy did.
- The converse is legal. `approver: "human"` with `human_disposed: false`
  records a human who reviewed while the disposition stayed automatic, and the
  flag only ever narrows the claim.

**Silence is valid.** A record carrying neither key is conforming. Records
written before this vocabulary existed carry nothing and must keep hashing
exactly as they did, so absence is never read as "a human acted".

## Files

| File | What it is |
|---|---|
| `cases.json` | The nine inputs |
| `expected.json` | The verdict each one must produce |
| `_check_independent.py` | Standard library only, imports no Vaara code |

## Running it

```
python3 _check_independent.py
```

Exit 0 when every case agrees with `expected.json`. The last line reports the
number of disagreements.

`tests/test_decision_disposition_vectors.py` runs the SHIPPED implementation
over the same cases and requires the same verdicts, so the checker and the
product cannot drift apart without a test going red.

## What the suite asserts about itself

The checker ends by asserting the property the suite exists for: that
`human_approved_at_escalation` and `replayed_prior_approval` both carry
`decision: "allow"` and disagree on `human_disposed`. If a future edit made
those two identical, the vectors would still pass their individual expectations
while testing nothing, so that comparison is checked explicitly.

## Provenance

The requirement was found on 2026-09-04 by reading
`draft-mih-scitt-agent-action-capsule-04` Section 5.5, which states the rule as
a normative MUST for its own format. The property is worth having on its own
terms and the field names here are this project's own; nothing in this suite
claims interoperability with that document.
