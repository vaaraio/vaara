# Northern Lights

A decision that keeps its thread, gets checked at every node it passes, and
arrives at the next one changed by the last.

Routing decisions forward is ordinary and every system does it. Almost none of
them carry a return path: the outcome travelling back along the route the
decision actually took, so the next decision is not made in the same ignorance
as the last one. Without that a decision log is write-only. With it the same
structure becomes a nerve.

```python
from vaara.northern_lights import Brain, Criterion, Decision, Mesh, Node, Verdict
```

## The shape

A node carries its own attributes and its own criteria. The operator sets how
many must match. That count is n, and n criteria across n nodes is the
decision. Nothing here picks those numbers for you.

Three shapes of walk over the same structure:

- **A route.** Stages in order. Each one checks and passes it on or stops it.
- **A broadcast.** Every node hears the same message at the same instant, and
  which of them answered is itself the signal.
- **A discovered route.** No list written in advance. Each node reads which
  direction the subject went and hands the thread to whoever is that way.

A stage is a single node or a panel of them. A panel decides by agreement: two
of three carries it, and the minority is kept in the record.

## Two scales of trust, one rule

```
a criterion has a prior   ->  does this signal count at this node
a node has a weight       ->  does this node's vote count in the panel
```

Both move when an outcome travels back. Both stop counting below a floor. Both
decay toward trust again so one bad week does not become permanent policy. Only
what actually voted is moved, so a criterion that failed a check is never
blamed for a result it had no part in.

In a panel, standing follows being right rather than agreeing. When a majority
carries a call that turns out to be wrong, the objector's weight rises and the
majority's falls, until the lone voice can no longer be outvoted.

## The return path

| channel | where it goes |
|---|---|
| relay | hop by hop along the route the decision took, every node learns |
| direct | straight to the origin from any point on the route, route untouched |
| local | closes at the node itself, the origin is told afterwards |
| lateral | node to node, the origin never hears it |

Four messages travel on those channels, kept separate because they are not the
same message: an outcome, a refusal with its reason, a capability probe, and a
retraction.

A refusal that carries a reason is a record of why not. Most logs say what
happened. This one says who objected and on what ground.

A return path costs three things and this module pays all three. Hop counts
bound the backward wave. A backward message never spawns a forward one, which
is what stops a graph of these oscillating. Decisions carry a sequence because
the two directions cross.

## Conclusions nobody routed

Each node watches its own firing rate against its own baseline, so "more than
usual" means more than usual for that node. A node that objects constantly is
not elevated when it objects.

Watching which nodes are elevated at the same time produces conclusions that no
single node could reach and that nothing routed. One symptom is not a
diagnosis. Several at once, each above its own normal, is.

A conclusion can also carry its negative face: what has to be un-true for it to
hold. Coming in from the cold makes a nose run and raises a cough, and both of
those readings are correct while the conclusion drawn from them is wrong. An
anti-node holds the mundane explanation, and when it is elevated the inference
does not happen. It is a veto and not a vote, so it is never outvoted, and a
blocked conclusion is recorded with what stopped it.

The conclusion is itself a decision, so it routes into the next mesh. That is
how these compose.

## Relationship to the receipt

This is the routing side. A Vaara receipt records that a decision was taken and
lets a third party recompute the verdict from committed bytes. Northern Lights
is about the decision travelling: which nodes saw it, which criteria counted at
each one, who dissented, and what came back afterwards. A decision's route is
carried on the decision itself, which is what makes both credit assignment and
an honest record of a refusal possible.

## Status

Early. The learning rule is a linear step with a floor, chosen because it is
legible in a test rather than because it is correct. Meshes wired into meshes
as a first-class graph, latency tiers, and persistence are not built.
