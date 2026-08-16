# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Northern Lights: a decision that keeps its thread.

    a decision that keeps its thread, gets checked at every node it passes,
    and arrives at the next one changed by the last.

Forward-only routing already exists everywhere. The back edge is what makes
this a nerve instead of a conveyor belt.

Two scales of trust, same mechanic:

    a CRITERION has a prior   -> does this signal count at this node
    a NODE has a weight       -> does this node's vote count in the panel

Both are moved by outcomes travelling backwards, and both stop counting below
a floor. That is the whole learning rule.

Three ways a message goes back, because a nerve is not one channel:

    RELAY   hop by hop along the route the decision took, every node learns
    DIRECT  straight to the brain from wherever the node sits, route untouched
    LOCAL   closes at the node itself, brain told afterwards (reflex arc)

stdlib only, no dependencies.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Protocol


class Verdict(str, Enum):
    PASS = "pass"
    REFUSE = "refuse"


class Backward(str, Enum):
    """What the message says. These are not the same message."""

    OUTCOME = "outcome"           # the call was wrong (or right), priors move
    REFUSAL = "refusal"           # "angle out of range", so the source re-plans
    CAPABILITY = "capability"     # "can you take this?" before committing
    RETRACTION = "retraction"     # already fired, now un-fire it


class Channel(str, Enum):
    """How the message travels. Orthogonal to what it says."""

    RELAY = "relay"      # hop by hop back along the route
    DIRECT = "direct"    # straight to the brain from anywhere on the route
    LOCAL = "local"      # closes at the node, brain told afterwards
    LATERAL = "lateral"  # cell to cell, neighbour to neighbour, brain not involved


Predicate = Callable[[dict, dict], bool]

FLOOR = 0.5      # below this, a criterion or a node stops counting
STEP = 0.25      # how far one outcome moves trust
MAX_HOPS = 32    # a backward message cannot run forever


# ---------------------------------------------------------------------------
# trust
# ---------------------------------------------------------------------------


@dataclass
class Criterion:
    name: str
    test: Predicate
    prior: float = 1.0
    moved_by: list[str] = field(default_factory=list)

    def counts(self, payload: dict, attrs: dict) -> bool:
        """Matched AND still trusted. History enters the decision here."""
        return self.test(payload, attrs) and self.prior >= FLOOR


def _nudge(value: float, delta: float) -> float:
    return max(0.0, min(1.0, value + delta))


# ---------------------------------------------------------------------------
# what a decision carries
# ---------------------------------------------------------------------------


@dataclass
class Vote:
    node: str
    verdict: Verdict
    counted: list[str]
    failed: list[str]
    weighed: bool                 # did this node's vote count at all
    reason: str | None = None


@dataclass
class Hop:
    """One stage of the route. A single node produces one vote, a panel several."""

    stage: str
    verdict: Verdict
    votes: list[Vote]
    reason: str | None = None

    @property
    def dissent(self) -> list[Vote]:
        """The minority. Kept, because a panel that discards it learns nothing."""
        return [v for v in self.votes if v.weighed and v.verdict is not self.verdict]


@dataclass
class Signal:
    """A message on the back channel."""

    kind: Backward
    channel: Channel
    origin: str
    decision_id: str
    seq: int
    reason: str | None = None


@dataclass
class Decision:
    id: str
    payload: dict
    route: list[Hop] = field(default_factory=list)
    seq: int = 0
    retracted: bool = False

    def thread(self) -> list[str]:
        return [hop.stage for hop in self.route]


# ---------------------------------------------------------------------------
# the brain
# ---------------------------------------------------------------------------


@dataclass
class Pattern:
    """A conclusion that no single node can reach.

    Coughing alone is nothing. A running nose alone is nothing. Both of them
    above their own usual, at the same time, is a flu. The nodes never learn
    that word. The brain does, by watching which of them are elevated.
    """

    name: str
    watch: set[str]        # nodes whose elevation counts toward this
    required: int          # how many of them must be elevated at once

    # Anti-nodes. Not more evidence: the negative face of the conclusion,
    # carrying what has to be UN-TRUE for it to hold. A mundane explanation
    # that accounts for the same signals blocks the inference outright, and it
    # beats any number of agreeing observations, because it is a veto and not
    # a vote. Walking in from the cold makes a nose run and raises a cough,
    # and both of those readings are correct while the conclusion is wrong.
    unless: frozenset[str] = frozenset()


@dataclass
class Blocked:
    """A conclusion that was reached and then vetoed. Kept, because knowing
    what was nearly concluded and what stopped it is worth more than silence."""

    pattern: str
    contributors: list[str]
    vetoed_by: list[str]


@dataclass
class Inference:
    """What the brain concluded, and what led it there.

    It is a Decision, so it can be routed into another mesh. That is how the
    babies get connected together.
    """

    pattern: str
    contributors: list[str]
    decision: Decision


class Brain:
    """Where a decision starts, an address any node can reach directly, and the
    only place that sees enough at once to notice a pattern."""

    def __init__(self, name: str = "brain", patterns: Iterable[Pattern] = ()):
        self.name = name
        self.inbox: list[Signal] = []
        self.patterns = list(patterns)
        self.inferences: list[Inference] = []
        self.blocked: list[Blocked] = []
        self._counter = itertools.count(1)

    def receive(self, signal: Signal) -> None:
        self.inbox.append(signal)

    def signals_from(self, node: str) -> list[Signal]:
        return [s for s in self.inbox if s.origin == node]

    def refusals(self) -> list[Signal]:
        return [s for s in self.inbox if s.kind is Backward.REFUSAL]

    def objections_against(self, node: str) -> int:
        """How often this node has been the one saying no."""
        return sum(1 for s in self.refusals() if s.origin == node)

    def infer(self, nodes: Iterable["Node"], tolerance: float = 0.0) -> list[Inference]:
        """Look at who is firing more than usual and see whether that spells something.

        This is not routing. Nothing travelled. The conclusion arises from the
        conjunction of weak signals, none of which is a decision on its own.
        """
        elevated = {n.name for n in nodes if n.elevated(tolerance)}
        found: list[Inference] = []
        for p in self.patterns:
            hits = sorted(p.watch & elevated)
            veto = sorted(p.unless & elevated)
            if veto and len(hits) >= p.required:
                # Enough agreement to conclude it, and a reason it is wrong.
                self.blocked.append(Blocked(p.name, hits, veto))
                continue
            if len(hits) >= p.required:
                d = Decision(
                    id=f"{p.name}-{next(self._counter)}",
                    payload={"inferred": p.name, "from": hits},
                )
                inf = Inference(p.name, hits, d)
                self.inferences.append(inf)
                found.append(inf)
        return found


# ---------------------------------------------------------------------------
# stages: a single node, or a panel of them
# ---------------------------------------------------------------------------


class Stage(Protocol):
    name: str

    def check(self, decision: Decision) -> Hop: ...
    def learn(self, hop: Hop, was_correct: bool, decision_id: str) -> None: ...
    def nodes(self) -> list["Node"]: ...


@dataclass
class Node:
    """Carries its own attributes, its own criteria, and its own standing.

    The operator sets what is measured and how many must match. `weight` is the
    node's standing in a panel, and it is moved by whether the node turned out
    to be right, not by whether it agreed with anyone.
    """

    name: str
    attrs: dict
    criteria: list[Criterion]
    threshold: int
    weight: float = 1.0
    reflex: bool = False          # can this node close a loop locally
    weight_moved_by: list[str] = field(default_factory=list)

    # "more than usual" needs a usual. A node watches its own firing rate
    # against its own baseline, so elevation is relative to this node rather
    # than to some absolute number chosen elsewhere.
    baseline: float = 0.0
    fires: list[bool] = field(default_factory=list)
    sensitivity: int = 0          # raised by a neighbour on the lateral channel

    # Where this node sits. Which nodes answered is itself a signal, and a node
    # can only hand a thread to a neighbour if it knows which way that is.
    at: tuple[float, float] = (0.0, 0.0)

    WINDOW = 12

    def effective_threshold(self) -> int:
        """A sensitised node needs less to fire. That is what priming is."""
        return max(1, self.threshold - self.sensitivity)

    def note(self, fired: bool) -> None:
        self.fires.append(fired)
        del self.fires[:-self.WINDOW]

    def rate(self) -> float:
        return (sum(self.fires) / len(self.fires)) if self.fires else 0.0

    def elevated(self, tolerance: float = 0.0) -> bool:
        """Firing more than usual, by its own reckoning."""
        return bool(self.fires) and self.rate() > self.baseline + tolerance

    def sensitise(self, by: int = 1) -> None:
        self.sensitivity += by

    def settle(self) -> None:
        self.sensitivity = max(0, self.sensitivity - 1)

    # -- forward --

    def vote(self, decision: Decision) -> Vote:
        counted: list[str] = []
        failed: list[str] = []
        for c in self.criteria:
            (counted if c.counts(decision.payload, self.attrs) else failed).append(c.name)
        need = self.effective_threshold()
        ok = len(counted) >= need
        self.note(not ok)          # "firing" here means raising an objection
        reason = None
        if not ok:
            reason = (
                f"{len(counted)} of {need} required criteria held; "
                f"failed: {', '.join(failed) or 'none'}"
            )
        return Vote(
            node=self.name,
            verdict=Verdict.PASS if ok else Verdict.REFUSE,
            counted=counted,
            failed=failed,
            weighed=self.weight >= FLOOR,
            reason=reason,
        )

    def check(self, decision: Decision) -> Hop:
        v = self.vote(decision)
        return Hop(self.name, v.verdict, [v], v.reason)

    def nodes(self) -> list["Node"]:
        return [self]

    # -- backward --

    def learn(self, hop: Hop, was_correct: bool, decision_id: str) -> None:
        """Only criteria that actually voted are moved."""
        for v in hop.votes:
            if v.node != self.name:
                continue
            delta = STEP if was_correct else -STEP
            for c in self.criteria:
                if c.name in v.counted:
                    c.prior = _nudge(c.prior, delta)
                    c.moved_by.append(decision_id)

    def decay(self, toward: float = 1.0, rate: float = 0.1) -> None:
        for c in self.criteria:
            c.prior += (toward - c.prior) * rate
        self.weight += (toward - self.weight) * rate


@dataclass
class Panel:
    """Many nodes look at the same decision. Two of three agree against one.

    A node whose weight has fallen below the floor still votes, and the vote is
    recorded, but it does not count toward the tally. Dissent is always kept.
    """

    name: str
    members: list[Node]
    required: int                 # how many weighed agreements decide it

    def check(self, decision: Decision) -> Hop:
        votes = [n.vote(decision) for n in self.members]
        weighed = [v for v in votes if v.weighed]
        yes = [v for v in weighed if v.verdict is Verdict.PASS]
        verdict = Verdict.PASS if len(yes) >= self.required else Verdict.REFUSE
        reason = None
        if verdict is Verdict.REFUSE:
            objections = [f"{v.node}: {v.reason}" for v in weighed if v.reason]
            reason = (
                f"{len(yes)} of {self.required} required agreements from "
                f"{len(weighed)} weighed of {len(votes)} members"
                + (f"; {' | '.join(objections)}" if objections else "")
            )
        return Hop(self.name, verdict, votes, reason)

    def nodes(self) -> list[Node]:
        return list(self.members)

    def learn(self, hop: Hop, was_correct: bool, decision_id: str) -> None:
        """A node's standing follows whether it was right, not whether it agreed.

        Outcome good  -> whoever matched the outcome gains, the rest lose.
        Outcome bad   -> the majority that carried it loses, the dissenter gains.

        That last case is the point of a panel. A lone objector who turns out to
        be right gets louder, and after enough of it the majority can no longer
        outvote it.
        """
        by_name = {n.name: n for n in self.members}
        for v in hop.votes:
            node = by_name.get(v.node)
            if node is None:
                continue
            agreed_with_hop = v.verdict is hop.verdict
            was_right = agreed_with_hop is was_correct
            node.weight = _nudge(node.weight, STEP if was_right else -STEP)
            node.weight_moved_by.append(decision_id)
            if v.weighed and was_correct is agreed_with_hop:
                node.learn(Hop(self.name, v.verdict, [v]), True, decision_id)
            elif v.weighed:
                node.learn(Hop(self.name, v.verdict, [v]), False, decision_id)


def _bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _apart(x: float, y: float) -> float:
    return abs((x - y + math.pi) % (2 * math.pi) - math.pi)


def _compass(vec: tuple[float, float]) -> str:
    x, y = vec
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return "nowhere"
    return ("right" if x > 0 else "left") if abs(x) >= abs(y) else ("front" if y > 0 else "back")


# ---------------------------------------------------------------------------
# the mesh
# ---------------------------------------------------------------------------


class Mesh:
    """One route of stages, hanging off one brain.

    Meshes compose into graphs of meshes. That is deliberately not this file.
    """

    def __init__(self, name: str, brain: Brain, stages: Iterable[Stage]):
        self.name = name
        self.brain = brain
        self.stages = list(stages)
        self._seq = itertools.count(1)

    def _stage(self, name: str) -> Stage | None:
        return next((s for s in self.stages if s.name == name), None)

    def all_nodes(self) -> list[Node]:
        return [n for s in self.stages for n in s.nodes()]

    # -- forward ------------------------------------------------------------

    def route(self, decision: Decision) -> Verdict:
        decision.seq = next(self._seq)
        decision.route = []
        for stage in self.stages:
            hop = stage.check(decision)
            decision.route.append(hop)
            if hop.verdict is Verdict.REFUSE:
                # A refusal that carries a reason is a receipt of why not, and
                # it goes straight to the brain rather than crawling back.
                self.brain.receive(Signal(
                    kind=Backward.REFUSAL,
                    channel=Channel.LOCAL if self._is_reflex(stage) else Channel.DIRECT,
                    origin=stage.name,
                    decision_id=decision.id,
                    seq=decision.seq,
                    reason=hop.reason,
                ))
                return Verdict.REFUSE
        return Verdict.PASS

    def _is_reflex(self, stage: Stage) -> bool:
        return all(n.reflex for n in stage.nodes())

    def probe(self, decision: Decision) -> dict[str, bool]:
        """Capability probe. Ask before committing, cheaper than dying at stage 4."""
        return {s.name: s.check(decision).verdict is Verdict.PASS for s in self.stages}

    # -- backward -----------------------------------------------------------

    def relay(
        self,
        decision: Decision,
        was_correct: bool,
        kind: Backward = Backward.OUTCOME,
        hops: int = MAX_HOPS,
    ) -> list[str]:
        """Walk the route the decision actually took, in reverse. Everyone learns.

        A backward message never spawns a forward wave, which is what stops a
        graph of meshes oscillating.
        """
        if kind is Backward.RETRACTION:
            decision.retracted = True
        reached: list[str] = []
        for hop in reversed(decision.route):
            if hops <= 0:
                break
            hops -= 1
            stage = self._stage(hop.stage)
            if stage is None:
                continue
            if kind in (Backward.OUTCOME, Backward.RETRACTION):
                stage.learn(hop, was_correct, decision.id)
            reached.append(stage.name)
        self.brain.receive(Signal(kind, Channel.RELAY, reached[-1] if reached else "",
                                  decision.id, decision.seq))
        return reached

    def report(
        self,
        node_name: str,
        decision: Decision,
        kind: Backward = Backward.OUTCOME,
        reason: str | None = None,
    ) -> Signal:
        """One node, anywhere on the route, speaks straight to the brain.

        Nothing between it and the brain sees the message, and the route is not
        touched. This is the half that makes it a nerve rather than a chain.
        """
        signal = Signal(kind, Channel.DIRECT, node_name, decision.id, decision.seq, reason)
        self.brain.receive(signal)
        return signal

    def reflex(self, node_name: str, decision: Decision, reason: str) -> Signal:
        """A loop that closes at the node because the path cannot afford the trip.

        The brain is told afterwards. Not every loop closes at the brain.
        """
        signal = Signal(Backward.REFUSAL, Channel.LOCAL, node_name,
                        decision.id, decision.seq, reason)
        self.brain.receive(signal)
        return signal

    # -- everyone at once ---------------------------------------------------

    def broadcast(self, decision: Decision) -> Hop:
        """One message, every node, same instant.

        Which nodes answer is the signal. Nothing is routed and nothing is
        gated: this asks the whole surface at once and keeps every reply.
        """
        decision.seq = next(self._seq)
        votes = [n.vote(decision) for n in self.all_nodes()]
        felt = [v for v in votes if v.verdict is Verdict.REFUSE]
        hop = Hop(
            "broadcast",
            Verdict.REFUSE if felt else Verdict.PASS,
            votes,
            f"{len(felt)} of {len(votes)} felt it" if felt else None,
        )
        decision.route.append(hop)
        return hop

    def felt_it(self, hop: Hop) -> list[Node]:
        by_name = {n.name: n for n in self.all_nodes()}
        return [by_name[v.node] for v in hop.votes
                if v.verdict is Verdict.REFUSE and v.node in by_name]

    def localise(self, hop: Hop) -> tuple[tuple[float, float], str]:
        """Where did it come from, given who felt it.

        The centroid of the responders against the centroid of the whole
        surface. Wind on the right side of the network is a right-side answer,
        and no single node could have said that.
        """
        answered = self.felt_it(hop)
        if not answered:
            return (0.0, 0.0), "nowhere"
        everyone = self.all_nodes()
        cx = sum(n.at[0] for n in everyone) / len(everyone)
        cy = sum(n.at[1] for n in everyone) / len(everyone)
        ax = sum(n.at[0] for n in answered) / len(answered)
        ay = sum(n.at[1] for n in answered) / len(answered)
        vec = (ax - cx, ay - cy)
        return vec, _compass(vec)

    # -- handing the thread on ----------------------------------------------

    def neighbour_toward(
        self,
        sender: str,
        heading: float,
        cone: float = math.pi / 3,
        visited: Iterable[str] = (),
    ) -> Node | None:
        """Who lies that way. None means the trail ends here."""
        by_name = {n.name: n for n in self.all_nodes()}
        here = by_name.get(sender)
        if here is None:
            return None
        skip = set(visited) | {sender}
        candidates = []
        for n in self.all_nodes():
            if n.name in skip:
                continue
            gap = _apart(_bearing(here.at, n.at), heading)
            if gap < cone:
                span = math.dist(here.at, n.at)
                candidates.append((gap, span, n))
        if not candidates:
            return None
        # Nearest first among those genuinely in that direction. A handoff goes
        # to the next camera along, never over the top of it to a far one.
        candidates.sort(key=lambda c: (round(c[0], 6), c[1]))
        return candidates[0][2]

    def follow(
        self,
        decision: Decision,
        start: str,
        sense: Callable[[Node, Decision], float | None],
        max_hops: int = 8,
    ) -> list[str]:
        """A route discovered as it goes, not written in advance.

        One camera has the subject. When it leaves that field of view the node
        reads the direction it went and hands the thread to whoever is that
        way. The decision keeps its thread across every handoff.
        """
        decision.seq = next(self._seq)
        by_name = {n.name: n for n in self.all_nodes()}
        here = by_name.get(start)
        chain: list[str] = []
        while here is not None and len(chain) < max_hops:
            chain.append(here.name)
            decision.route.append(
                Hop(here.name, Verdict.PASS, [here.vote(decision)])
            )
            heading = sense(here, decision)
            if heading is None:
                break
            nxt = self.neighbour_toward(here.name, heading, visited=chain)
            if nxt is None:
                self.brain.receive(Signal(
                    Backward.REFUSAL, Channel.DIRECT, here.name,
                    decision.id, decision.seq,
                    f"went {heading:.2f} rad and nobody is that way",
                ))
                break
            self.brain.receive(Signal(
                Backward.CAPABILITY, Channel.LATERAL, here.name,
                decision.id, decision.seq, f"handing on to {nxt.name}",
            ))
            here = nxt
        return chain

    def lateral(
        self,
        sender: str,
        receiver: str,
        decision: Decision,
        reason: str | None = None,
    ) -> Signal:
        """Cell to cell. The brain is not involved and the route is not touched.

        The receiving node is sensitised, so it needs less to raise an objection
        than it did a moment ago. A neighbour saying "something is off here" is
        not a verdict, it lowers the bar for the next one.
        """
        by_name = {n.name: n for n in self.all_nodes()}
        if receiver in by_name:
            by_name[receiver].sensitise()
        return Signal(Backward.CAPABILITY, Channel.LATERAL, sender,
                      decision.id, decision.seq, reason)

    def refusal_reason(self, decision: Decision) -> str | None:
        for hop in decision.route:
            if hop.verdict is Verdict.REFUSE:
                return f"{hop.stage}: {hop.reason}"
        return None
