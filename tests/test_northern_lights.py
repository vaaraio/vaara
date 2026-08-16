# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the northern_lights atom.

The first two tests are the theory. If they pass, the claim holds: a decision
arrives at the next node changed by the last, and a panel learns who to listen
to. Everything after them is support.

Run: pytest tests/test_northern_lights.py
"""

import math

from vaara.northern_lights import (
    FLOOR,
    STEP,
    Backward,
    Brain,
    Channel,
    Criterion,
    Decision,
    Mesh,
    Node,
    Panel,
    Pattern,
    Verdict,
)

STRIKES = math.floor((1.0 - FLOOR) / STEP) + 1


def courier(name: str, threshold: int = 2, max_km: float = 5.0, **kw) -> Node:
    """A courier gate: distance, time, effort. The operator picks n."""
    return Node(
        name=name,
        attrs={"max_km": max_km, "max_min": 30},
        criteria=[
            Criterion("distance", lambda p, a: p["km"] <= a["max_km"]),
            Criterion("time", lambda p, a: p["min"] <= a["max_min"]),
            Criterion("effort", lambda p, a: p["effort"] < 0.8),
        ],
        threshold=threshold,
        **kw,
    )


def delivery(did: str = "d1", km: float = 3.0) -> Decision:
    return Decision(id=did, payload={"km": km, "min": 20, "effort": 0.5})


def mesh_of(*stages, brain=None) -> Mesh:
    return Mesh("m", brain or Brain(), stages)


# ==========================================================================
# THE THEORY
# ==========================================================================


def test_the_same_decision_is_judged_differently_after_the_back_edge():
    """History is an input, not just a record.

    Identical payload, twice. The only thing that changed between the runs is
    that an outcome travelled backwards.
    """
    m = mesh_of(courier("pickup"), courier("handoff"))
    first = delivery("d0")
    assert m.route(first) is Verdict.PASS

    for i in range(STRIKES):
        bad = delivery(f"bad{i}")
        # Holds until the last strike, so this is not a one-shot kill switch.
        assert m.route(bad) is Verdict.PASS
        m.relay(bad, was_correct=False)

    again = delivery("d-final")
    assert again.payload == first.payload
    assert m.route(again) is Verdict.REFUSE
    assert "pickup" in m.refusal_reason(again)


def test_a_vindicated_dissenter_eventually_outweighs_the_majority():
    """Two of three agree against one, and the one turns out to be right.

    A panel that only counts heads learns nothing. Standing follows being
    right, so the lone objector gets louder until the majority cannot carry
    a bad call any more.
    """
    lenient_a = courier("a", max_km=99.0)
    lenient_b = courier("b", max_km=99.0)
    strict = courier("c", max_km=1.0, threshold=3)   # the objector
    panel = Panel("panel", [lenient_a, lenient_b, strict], required=2)
    m = mesh_of(panel)

    far = delivery("far", km=50.0)
    assert m.route(far) is Verdict.PASS                    # 2 of 3 carry it
    assert [v.node for v in far.route[0].dissent] == ["c"]  # dissent is kept

    rounds = 0
    while m.route(delivery(f"r{rounds}", km=50.0)) is Verdict.PASS:
        m.relay(delivery(f"r{rounds}", km=50.0), was_correct=False)
        d = delivery(f"x{rounds}", km=50.0)
        m.route(d)
        m.relay(d, was_correct=False)
        rounds += 1
        assert rounds < 20, "the panel never listened to the objector"

    assert strict.weight > lenient_a.weight
    assert lenient_a.weight < FLOOR          # the majority lost its vote


# ==========================================================================
# the nerve route: three ways back
# ==========================================================================


def test_a_node_halfway_along_reaches_the_brain_in_one_message():
    """DIRECT. Nothing between it and the brain sees this, and the route stands."""
    brain = Brain()
    m = mesh_of(courier("n0"), courier("n1"), courier("n2"), brain=brain)
    d = delivery()
    m.route(d)
    before = list(d.thread())

    m.report("n1", d, kind=Backward.OUTCOME, reason="load shifted after handoff")

    assert d.thread() == before
    sig = brain.signals_from("n1")[0]
    assert sig.channel is Channel.DIRECT
    assert brain.signals_from("n0") == []      # n0 never saw it
    assert brain.signals_from("n2") == []


def test_a_relay_walks_the_route_in_reverse_and_everyone_learns():
    m = mesh_of(courier("n0"), courier("n1"), courier("n2"))
    d = delivery()
    m.route(d)
    assert m.relay(d, was_correct=False) == ["n2", "n1", "n0"]
    assert all(c.prior < 1.0 for n in m.all_nodes() for c in n.criteria)


def test_a_reflex_closes_at_the_node_and_tells_the_brain_afterwards():
    """Not every loop closes at the brain. A millisecond path cannot afford it."""
    brain = Brain()
    m = mesh_of(courier("joint", threshold=3, reflex=True), brain=brain)
    d = delivery("d", km=99.0)
    assert m.route(d) is Verdict.REFUSE
    assert brain.inbox[0].channel is Channel.LOCAL
    assert brain.inbox[0].kind is Backward.REFUSAL


def test_a_refusal_reaches_the_brain_without_being_asked():
    brain = Brain()
    m = mesh_of(courier("n0"), courier("gate", threshold=3), brain=brain)
    d = delivery("d", km=99.0)
    m.route(d)
    assert brain.objections_against("gate") == 1
    assert brain.objections_against("n0") == 0
    assert "distance" in brain.refusals()[0].reason


def test_a_backward_message_does_not_spawn_a_forward_wave():
    m = mesh_of(courier("n0"), courier("n1"))
    d = delivery()
    m.route(d)
    before = list(d.thread())
    m.relay(d, was_correct=False)
    assert d.thread() == before


def test_hop_count_bounds_the_backward_wave():
    m = mesh_of(*[courier(f"n{i}") for i in range(4)])
    d = delivery()
    m.route(d)
    assert m.relay(d, was_correct=False, hops=2) == ["n3", "n2"]


# ==========================================================================
# n x n
# ==========================================================================


def test_threshold_is_the_operators_choice_not_the_systems():
    far = delivery("far", km=99.0)
    assert mesh_of(courier("n", threshold=1)).route(far) is Verdict.PASS
    assert mesh_of(courier("n", threshold=3)).route(far) is Verdict.REFUSE


def test_n_criteria_across_n_stages_is_the_decision():
    m = mesh_of(*[courier(f"n{i}") for i in range(3)])
    d = delivery()
    assert m.route(d) is Verdict.PASS
    assert d.thread() == ["n0", "n1", "n2"]
    assert all(len(h.votes[0].counted) == 3 for h in d.route)


def test_the_route_stops_at_the_stage_that_refuses():
    m = mesh_of(courier("n0"), courier("n1", threshold=3), courier("n2"))
    d = delivery("d", km=99.0)
    assert m.route(d) is Verdict.REFUSE
    assert d.thread() == ["n0", "n1"]


def test_a_panel_and_a_single_node_compose_in_one_route():
    panel = Panel("panel", [courier("a"), courier("b"), courier("c")], required=2)
    m = mesh_of(courier("intake"), panel, courier("dispatch"))
    d = delivery()
    assert m.route(d) is Verdict.PASS
    assert d.thread() == ["intake", "panel", "dispatch"]
    assert len(d.route[1].votes) == 3


# ==========================================================================
# the panel
# ==========================================================================


def test_two_of_three_carries_it_and_the_dissent_survives_into_the_record():
    panel = Panel("panel", [courier("a"), courier("b"), courier("c", threshold=3)],
                  required=2)
    m = mesh_of(panel)
    d = delivery("d", km=99.0)
    assert m.route(d) is Verdict.PASS
    hop = d.route[0]
    assert [v.node for v in hop.dissent] == ["c"]
    assert hop.dissent[0].reason is not None


def test_a_node_below_the_floor_still_votes_but_is_not_counted():
    quiet = courier("quiet")
    quiet.weight = FLOOR - 0.01
    panel = Panel("panel", [quiet, courier("b"), courier("c")], required=3)
    m = mesh_of(panel)
    d = delivery()
    assert m.route(d) is Verdict.REFUSE           # only 2 weighed votes exist
    hop = d.route[0]
    assert len(hop.votes) == 3
    assert [v.node for v in hop.votes if not v.weighed] == ["quiet"]


def test_standing_follows_being_right_not_agreeing():
    agreeable = courier("a")
    objector = courier("c", threshold=3)
    panel = Panel("panel", [agreeable, courier("b"), objector], required=2)
    m = mesh_of(panel)
    d = delivery("d", km=99.0)
    m.route(d)                                    # a and b pass, c refuses
    m.relay(d, was_correct=False)                 # the pass was wrong
    assert objector.weight > 1.0 - STEP           # rewarded for holding out
    assert agreeable.weight < 1.0                 # punished for carrying it


def test_a_correct_call_rewards_the_majority_and_costs_the_objector():
    agreeable = courier("a")
    objector = courier("c", threshold=3)
    panel = Panel("panel", [agreeable, courier("b"), objector], required=2)
    m = mesh_of(panel)
    d = delivery("d", km=99.0)
    m.route(d)
    m.relay(d, was_correct=True)
    assert agreeable.weight >= 1.0
    assert objector.weight < 1.0


# ==========================================================================
# credit, decay, retraction, probe
# ==========================================================================


def test_credit_lands_only_on_criteria_that_actually_voted():
    node = courier("gate", threshold=1)
    m = mesh_of(node)
    d = delivery("d", km=99.0)
    m.route(d)                                    # distance failed
    m.relay(d, was_correct=False)
    by_name = {c.name: c for c in node.criteria}
    assert by_name["distance"].prior == 1.0       # never voted, never blamed
    assert by_name["distance"].moved_by == []
    assert by_name["time"].prior < 1.0
    assert by_name["time"].moved_by == ["d"]


def test_retraction_marks_the_decision_and_still_teaches():
    node = courier("gate")
    m = mesh_of(node)
    d = delivery()
    m.route(d)
    m.relay(d, was_correct=False, kind=Backward.RETRACTION)
    assert d.retracted is True
    assert any(c.prior < 1.0 for c in node.criteria)


def test_a_refusal_relay_reports_without_moving_priors():
    node = courier("gate")
    m = mesh_of(node)
    d = delivery()
    m.route(d)
    m.relay(d, was_correct=False, kind=Backward.REFUSAL)
    assert all(c.prior == 1.0 for c in node.criteria)


def test_capability_probe_asks_before_committing():
    m = mesh_of(courier("n0"), courier("n1", threshold=3))
    assert m.probe(delivery("far", km=99.0)) == {"n0": True, "n1": False}


def test_decay_pulls_trust_back_at_both_scales():
    node = courier("gate")
    m = mesh_of(Panel("p", [node, courier("b"), courier("c")], required=2))
    d = delivery()
    m.route(d)
    m.relay(d, was_correct=False)
    hurt_prior = min(c.prior for c in node.criteria)
    hurt_weight = node.weight
    for _ in range(20):
        node.decay()
    assert min(c.prior for c in node.criteria) > hurt_prior
    assert node.weight > hurt_weight


def test_a_criterion_below_the_floor_stops_counting():
    c = Criterion("always", lambda p, a: True, prior=FLOOR - 0.01)
    assert c.test({}, {}) is True
    assert c.counts({}, {}) is False


# ==========================================================================
# the flu: a conclusion nobody routed
# ==========================================================================


def symptom(name: str, trip: float, baseline: float = 0.0) -> Node:
    """A node that objects when its own reading is above `trip`."""
    return Node(
        name=name,
        attrs={"trip": trip},
        criteria=[Criterion("normal", lambda p, a: p.get(a_key(a), 0.0) <= a["trip"])],
        threshold=1,
        baseline=baseline,
    )


def a_key(attrs: dict) -> str:
    return attrs.get("reads", "level")


def reading(did: str, **levels) -> Decision:
    return Decision(id=did, payload=levels)


def test_two_weak_signals_neither_of_which_is_a_decision_produce_a_conclusion():
    """Coughing alone is nothing. A running nose alone is nothing.

    Both above their own usual, at the same time, is a flu. No node knows that
    word, nothing was routed, and the conclusion is a Decision the brain
    originated itself.
    """
    cough = symptom("cough", trip=0.3)
    nose = symptom("nose", trip=0.3)
    brain = Brain(patterns=[Pattern("flu", watch={"cough", "nose"}, required=2)])
    # Symptoms observe in parallel. A serial route would stop at the first one
    # that objects and the second would never see the reading at all.
    m = Mesh("body", brain, [Panel("symptoms", [cough, nose], required=2)])

    # a quiet week: nothing above trip, nobody objects
    for i in range(6):
        m.route(reading(f"ok{i}", level=0.1))
    assert brain.infer(m.all_nodes()) == []

    # now both start firing more than usual
    for i in range(6):
        m.route(reading(f"ill{i}", level=0.9))

    found = brain.infer(m.all_nodes())
    assert [f.pattern for f in found] == ["flu"]
    assert found[0].contributors == ["cough", "nose"]


def test_one_symptom_alone_is_not_enough():
    cough = symptom("cough", trip=0.3)
    nose = symptom("nose", trip=0.9)          # never trips on this data
    brain = Brain(patterns=[Pattern("flu", watch={"cough", "nose"}, required=2)])
    m = Mesh("body", brain, [Panel("symptoms", [cough, nose], required=2)])
    for i in range(6):
        m.route(reading(f"r{i}", level=0.5))  # cough objects, nose does not
    assert cough.elevated() is True
    assert nose.elevated() is False
    assert brain.infer(m.all_nodes()) == []


def test_elevation_is_measured_against_each_nodes_own_usual():
    """"More than usual" needs a usual, and it is per node."""
    noisy = symptom("noisy", trip=0.3, baseline=1.0)   # complaining IS its usual
    quiet = symptom("quiet", trip=0.3, baseline=0.0)
    m = Mesh("body", Brain(), [Panel("both", [noisy, quiet], required=2)])
    for i in range(6):
        m.route(reading(f"r{i}", level=0.5))
    assert quiet.rate() == noisy.rate()       # identical firing
    assert quiet.elevated() is True
    assert noisy.elevated() is False          # for this one, that IS usual


def test_an_inference_is_a_decision_and_can_be_routed_onward():
    """Meshes compose. The conclusion becomes the input to the next mesh."""
    cough = symptom("cough", trip=0.3)
    nose = symptom("nose", trip=0.3)
    brain = Brain(patterns=[Pattern("flu", watch={"cough", "nose"}, required=2)])
    body = Mesh("body", brain, [Panel("symptoms", [cough, nose], required=2)])
    for i in range(6):
        body.route(reading(f"ill{i}", level=0.9))

    inferred = brain.infer(body.all_nodes())[0].decision

    triage = Mesh("triage", Brain(), [
        Node("rest", attrs={}, criteria=[
            Criterion("is-flu", lambda p, a: p.get("inferred") == "flu")],
            threshold=1),
    ])
    assert triage.route(inferred) is Verdict.PASS
    assert inferred.payload["from"] == ["cough", "nose"]


# ==========================================================================
# cell to cell
# ==========================================================================


def test_a_neighbour_can_prime_a_node_without_the_brain_hearing_it():
    brain = Brain()
    a, b = courier("a"), courier("b", threshold=3)
    m = Mesh("m", brain, [a, b])
    d = delivery("d")

    sig = m.lateral("a", "b", d, reason="load felt wrong on my side")

    assert sig.channel is Channel.LATERAL
    assert brain.inbox == []                  # the brain was never told
    assert b.sensitivity == 1
    assert b.effective_threshold() == 2       # it needs less now than it did


def test_sensitisation_changes_the_verdict_on_the_same_payload():
    m = mesh_of(courier("gate", threshold=2))
    gate = m.all_nodes()[0]
    d1 = delivery("d1", km=99.0)              # 2 of 3 criteria hold
    assert m.route(d1) is Verdict.PASS

    gate.threshold = 3                        # a stricter gate refuses it
    assert m.route(delivery("d2", km=99.0)) is Verdict.REFUSE

    gate.sensitise()                          # a neighbour primes it back down
    assert m.route(delivery("d3", km=99.0)) is Verdict.PASS


def test_a_primed_node_settles_again():
    node = courier("n", threshold=3)
    node.sensitise()
    node.sensitise()
    assert node.effective_threshold() == 1
    node.settle()
    assert node.effective_threshold() == 2
    node.settle()
    node.settle()
    assert node.effective_threshold() == 3    # never below its own floor of one


# ==========================================================================
# everyone at once: which nodes answered is the signal
# ==========================================================================


def skin(name: str, x: float, y: float = 0.0) -> Node:
    """Feels a gust when it is near. Objecting means it felt something."""
    return Node(
        name=name, at=(x, y), attrs={"x": x},
        criteria=[Criterion("calm", lambda p, a: abs(a["x"] - p["gust_x"]) > 1.5)],
        threshold=1,
    )


def body_mesh() -> Mesh:
    return Mesh("body", Brain(),
                [skin(f"s{i}", float(i)) for i in range(-3, 4)])


def test_wind_on_the_right_side_is_a_right_side_answer():
    """One message, every node, same instant.

    No single node can say which way the wind came from. The pattern of who
    answered says it, and nothing was routed to work that out.
    """
    m = body_mesh()
    d = Decision(id="gust", payload={"gust_x": 3.0})
    hop = m.broadcast(d)
    assert [n.name for n in m.felt_it(hop)] == ["s2", "s3"]
    vec, side = m.localise(hop)
    assert side == "right"
    assert vec[0] > 0


def test_the_same_gust_from_the_other_side_flips_the_answer():
    m = body_mesh()
    d = Decision(id="gust", payload={"gust_x": -3.0})
    _, side = m.localise(m.broadcast(d))
    assert side == "left"


def test_a_gust_nobody_feels_localises_nowhere():
    m = body_mesh()
    _, side = m.localise(m.broadcast(Decision(id="far", payload={"gust_x": 99.0})))
    assert side == "nowhere"


def test_broadcast_keeps_every_reply_including_the_quiet_ones():
    m = body_mesh()
    hop = m.broadcast(Decision(id="g", payload={"gust_x": 3.0}))
    assert len(hop.votes) == 7            # all N answered, n attributes each
    assert hop.reason == "2 of 7 felt it"


# ==========================================================================
# the CCTV handoff: a route discovered as it goes
# ==========================================================================


def camera(name: str, x: float, y: float = 0.0) -> Node:
    return Node(name=name, at=(x, y), attrs={},
                criteria=[Criterion("in-view", lambda p, a: True)], threshold=1)


def street() -> Mesh:
    return Mesh("street", Brain(), [camera(f"cam{i}", float(i)) for i in range(4)])


def test_the_thread_is_handed_east_from_camera_to_camera():
    """One camera loses him at the edge of its view and says which way he went.

    The route was never written down. It is discovered one handoff at a time,
    and the decision keeps its thread across every one.
    """
    m = street()
    d = Decision(id="suspect", payload={})
    chain = m.follow(d, "cam0", sense=lambda node, dec: 0.0)   # heading east
    assert chain == ["cam0", "cam1", "cam2", "cam3"]
    assert d.thread() == chain


def test_a_handoff_goes_to_the_next_camera_along_not_over_its_head():
    m = street()
    assert m.neighbour_toward("cam0", 0.0).name == "cam1"


def test_the_thread_does_not_walk_back_over_cameras_it_already_used():
    m = street()
    d = Decision(id="s", payload={})
    chain = m.follow(d, "cam1", sense=lambda node, dec: 0.0)
    assert chain == ["cam1", "cam2", "cam3"]
    assert len(set(chain)) == len(chain)


def test_when_nobody_lies_that_way_the_trail_ends_and_the_brain_is_told():
    m = street()
    d = Decision(id="s", payload={})
    chain = m.follow(d, "cam3", sense=lambda node, dec: 0.0)   # nothing further east
    assert chain == ["cam3"]
    assert m.brain.refusals()[0].origin == "cam3"
    assert "nobody is that way" in m.brain.refusals()[0].reason


def test_the_thread_stops_where_the_subject_is_still_in_view():
    m = street()
    d = Decision(id="s", payload={})
    chain = m.follow(d, "cam0", sense=lambda node, dec: None)  # never left the frame
    assert chain == ["cam0"]


# ==========================================================================
# five point detection: many nodes confirm, then the mesh acts
# ==========================================================================


def sensor(name: str, x: float, sees: bool) -> Node:
    return Node(
        name=name, at=(x, 0.0), attrs={"sees": sees},
        criteria=[Criterion("clear", lambda p, a: not (a["sees"] and p["inbound"]))],
        threshold=1,
    )


def test_five_point_confirmation_turns_a_sighting_into_an_action():
    """One camera sees something flying. That is not a decision.

    Five points look at the same track at once. When enough of them agree it is
    inbound, the mesh has a conclusion no sensor owned, and it routes onward as
    a decision of its own.
    """
    points = [sensor(f"p{i}", float(i), sees=i < 4) for i in range(5)]
    brain = Brain(patterns=[Pattern("inbound", watch={f"p{i}" for i in range(5)},
                                    required=3)])
    m = Mesh("air", brain, points)

    track = Decision(id="track", payload={"inbound": True})
    hop = m.broadcast(track)
    confirmed = m.felt_it(hop)
    assert [n.name for n in confirmed] == ["p0", "p1", "p2", "p3"]

    inferred = brain.infer(m.all_nodes())
    assert [f.pattern for f in inferred] == ["inbound"]

    response = Mesh("response", Brain(), [
        Node("engage", attrs={}, criteria=[
            Criterion("is-inbound", lambda p, a: p.get("inferred") == "inbound"),
            Criterion("enough-eyes", lambda p, a: len(p.get("from", [])) >= 3)],
            threshold=2),
    ])
    assert response.route(inferred[0].decision) is Verdict.PASS


def test_one_sensor_alone_does_not_get_to_start_a_war():
    points = [sensor(f"p{i}", float(i), sees=i == 0) for i in range(5)]
    brain = Brain(patterns=[Pattern("inbound", watch={f"p{i}" for i in range(5)},
                                    required=3)])
    m = Mesh("air", brain, points)
    hop = m.broadcast(Decision(id="track", payload={"inbound": True}))
    assert [n.name for n in m.felt_it(hop)] == ["p0"]
    assert brain.infer(m.all_nodes()) == []


# ==========================================================================
# anti-nodes: what has to be un-true
# ==========================================================================


def test_a_mundane_explanation_vetoes_the_conclusion():
    """Walking in from the cold makes a nose run and raises a cough.

    Both readings are correct. The conclusion drawn from them is wrong. An
    anti-node carries what has to be un-true for the conclusion to hold, and
    when it is elevated the inference does not happen.
    """
    cough = symptom("cough", trip=0.3)
    nose = symptom("nose", trip=0.3)
    came_in = symptom("came-in-from-cold", trip=0.3)
    brain = Brain(patterns=[Pattern("flu", watch={"cough", "nose"}, required=2,
                                    unless=frozenset({"came-in-from-cold"}))])
    m = Mesh("body", brain, [Panel("all", [cough, nose, came_in], required=3)])

    for i in range(6):
        m.route(reading(f"r{i}", level=0.9))

    assert cough.elevated() and nose.elevated()      # the greens are real
    assert brain.infer(m.all_nodes()) == []          # and the flu is not
    assert brain.blocked[-1].pattern == "flu"
    assert brain.blocked[-1].contributors == ["cough", "nose"]
    assert brain.blocked[-1].vetoed_by == ["came-in-from-cold"]


def test_without_the_anti_node_the_same_readings_conclude_flu():
    """The control. The only difference is whether the veto is elevated."""
    cough = symptom("cough", trip=0.3)
    nose = symptom("nose", trip=0.3)
    came_in = symptom("came-in-from-cold", trip=0.95)   # not triggered
    brain = Brain(patterns=[Pattern("flu", watch={"cough", "nose"}, required=2,
                                    unless=frozenset({"came-in-from-cold"}))])
    m = Mesh("body", brain, [Panel("all", [cough, nose, came_in], required=3)])
    for i in range(6):
        m.route(reading(f"r{i}", level=0.9))
    assert came_in.elevated() is False
    assert [f.pattern for f in brain.infer(m.all_nodes())] == ["flu"]


def test_one_veto_beats_any_number_of_agreeing_observations():
    """A veto is not a vote and does not get outvoted."""
    watchers = [symptom(f"s{i}", trip=0.3) for i in range(8)]
    came_in = symptom("came-in-from-cold", trip=0.3)
    brain = Brain(patterns=[Pattern("flu", watch={f"s{i}" for i in range(8)},
                                    required=2,
                                    unless=frozenset({"came-in-from-cold"}))])
    m = Mesh("body", brain, [Panel("all", watchers + [came_in], required=9)])
    for i in range(6):
        m.route(reading(f"r{i}", level=0.9))
    assert len([w for w in watchers if w.elevated()]) == 8
    assert brain.infer(m.all_nodes()) == []
    assert brain.blocked[-1].vetoed_by == ["came-in-from-cold"]


def test_a_blocked_conclusion_is_recorded_not_silently_dropped():
    cough = symptom("cough", trip=0.3)
    nose = symptom("nose", trip=0.3)
    came_in = symptom("came-in-from-cold", trip=0.3)
    brain = Brain(patterns=[Pattern("flu", watch={"cough", "nose"}, required=2,
                                    unless=frozenset({"came-in-from-cold"}))])
    m = Mesh("body", brain, [Panel("all", [cough, nose, came_in], required=3)])
    for i in range(6):
        m.route(reading(f"r{i}", level=0.9))
    brain.infer(m.all_nodes())
    assert len(brain.blocked) == 1
    assert brain.inferences == []
