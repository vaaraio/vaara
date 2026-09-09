# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Access records: who opened it, on whose authority, and what came back.

The load-bearing test here is ``test_agent_identity_alone_is_refused``. The
whole reason this record type exists is that a service account name does not
answer "who used this person's data". If that test ever passes with only an
agent identity, the record has stopped answering the question it was built for.
"""
from __future__ import annotations

import pytest

from tests.test_timeanchor import _local_tsa_client
from vaara.audit.access import (
    ACCESS_TOOL,
    CORE_FIELDS,
    accesses_closed_with_no_action,
    accesses_with_no_recorded_outcome,
    digest_returned,
    find_accesses,
    follow_ups,
    record_access,
    record_access_outcome,
)
from vaara.audit.trail import AuditTrail, EventType
from vaara.taxonomy.actions import (
    ActionCategory,
    ActionRequest,
    ActionType,
    BlastRadius,
    Reversibility,
)


def _read(trail, **kw):
    base = dict(
        subject="patient/4711/kayttoloki",
        accessed_by="agent/log-search@istekki",
        on_behalf_of="person/hetu-pseudonym/9f2c",
        basis="hoitosuhteen tarkistus",
    )
    base.update(kw)
    return record_access(trail, **base)


# ── The reason the record type exists ───────────────────────────────────────

def test_agent_identity_alone_is_refused():
    """An access record whose only identity is the agent answers nothing."""
    trail = AuditTrail()
    with pytest.raises(ValueError, match="who used the data"):
        record_access(
            trail,
            subject="patient/4711/kayttoloki",
            accessed_by="agent/log-search@istekki",
            on_behalf_of="",
            basis="hoitosuhteen tarkistus",
        )


def test_both_identities_are_kept_apart():
    trail = AuditTrail()
    _read(trail)

    rec = find_accesses(trail._records)[0]
    params = rec.data["parameters"]

    assert params["accessed_by"] == "agent/log-search@istekki"
    assert params["on_behalf_of"] == "person/hetu-pseudonym/9f2c"
    assert params["accessed_by"] != params["on_behalf_of"]


def test_basis_is_required():
    trail = AuditTrail()
    with pytest.raises(ValueError, match="basis"):
        _read(trail, basis="")


def test_subject_and_agent_are_required():
    trail = AuditTrail()
    with pytest.raises(ValueError, match="what was opened"):
        _read(trail, subject="")
    with pytest.raises(ValueError, match="agent or process"):
        _read(trail, accessed_by="")


# ── It commits to what came back and holds none of it ───────────────────────

def test_returned_set_is_committed_by_digest():
    trail = AuditTrail()
    payload = b'{"rows": ["seen this", "and this"]}'
    _read(trail, returned_sha256=digest_returned(payload), returned_count=2)

    params = find_accesses(trail._records)[0].data["parameters"]
    assert params["returned_sha256"] == digest_returned(payload)
    assert params["returned_count"] == 2


def test_the_payload_never_reaches_the_record():
    """The bytes that were returned must not appear anywhere in the trail."""
    trail = AuditTrail()
    payload = b"DIAGNOSIS-THAT-MUST-NOT-BE-COPIED"
    _read(trail, returned_sha256=digest_returned(payload), returned_count=1)

    blob = repr([r.to_dict() for r in trail._records])
    assert b"DIAGNOSIS-THAT-MUST-NOT-BE-COPIED".decode() not in blob


def test_a_bare_hex_digest_is_refused():
    trail = AuditTrail()
    with pytest.raises(ValueError, match="digest_returned"):
        _read(trail, returned_sha256="a" * 64)


# ── The fields present are enumerable ───────────────────────────────────────

def test_missing_digest_is_stated_rather_than_implied():
    trail = AuditTrail()
    _read(trail)

    present = find_accesses(trail._records)[0].data["parameters"]["fields_present"]
    assert "returned_sha256" not in present
    assert "on_behalf_of" in present
    assert "basis" in present


def test_deployment_fields_are_carried_and_enumerated():
    trail = AuditTrail()
    _read(trail, deployment_fields={"rekisteri": "potilastiedot", "yksikko": "KYS"})

    params = find_accesses(trail._records)[0].data["parameters"]
    assert params["rekisteri"] == "potilastiedot"
    assert "yksikko" in params["fields_present"]
    assert "rekisteri" in params["fields_present"]


def test_deployment_fields_cannot_shadow_a_core_field():
    trail = AuditTrail()
    for reserved in CORE_FIELDS + ("fields_present",):
        with pytest.raises(ValueError, match="shadow"):
            _read(trail, deployment_fields={reserved: "spoofed"})


# ── It inherits the trail's properties rather than reimplementing them ──────

def test_an_access_record_is_chained_like_any_other():
    trail = AuditTrail()
    _read(trail)
    _read(trail, subject="patient/4712/kayttoloki")

    assert trail.verify_chain() is None


def test_removing_an_access_record_breaks_the_chain():
    trail = AuditTrail()
    _read(trail)
    _read(trail, subject="patient/4712/kayttoloki")
    target = find_accesses(trail._records)[0]
    target.data["parameters"]["on_behalf_of"] = "person/somebody-else"

    assert trail.verify_chain() is not None


def test_the_time_is_attested_by_an_anchor_not_asserted():
    """A timestamp the emitter wrote is the emitter's word. Anchors are not."""
    trail = AuditTrail()
    client = _local_tsa_client()
    _read(trail)
    trail.anchor_head(client)
    _read(trail, subject="patient/4712/kayttoloki")
    trail.anchor_head(client)

    accesses = find_accesses(trail._records)
    result = trail.verify_segment(record_id=accesses[0].record_id)

    assert result.ok, result.reason
    assert result.upper_attested_time


# ── Finding them ────────────────────────────────────────────────────────────

def test_who_opened_this_record():
    trail = AuditTrail()
    _read(trail)
    _read(trail, subject="patient/4712/kayttoloki", on_behalf_of="person/other")

    hits = find_accesses(trail._records, subject="patient/4711/kayttoloki")

    assert len(hits) == 1
    assert hits[0].data["parameters"]["on_behalf_of"] == "person/hetu-pseudonym/9f2c"


def test_what_did_this_person_open():
    trail = AuditTrail()
    _read(trail)
    _read(trail, subject="patient/4712/kayttoloki")
    _read(trail, subject="patient/9999/kayttoloki", on_behalf_of="person/other")

    hits = find_accesses(trail._records, on_behalf_of="person/hetu-pseudonym/9f2c")

    assert {h.data["parameters"]["subject"] for h in hits} == {
        "patient/4711/kayttoloki", "patient/4712/kayttoloki",
    }


def test_it_is_its_own_event_type_not_an_ordinary_action():
    trail = AuditTrail()
    _read(trail)

    kinds = {r.event_type for r in trail._records}
    assert EventType.ACCESS_RECORDED in kinds
    rec = find_accesses(trail._records)[0]
    assert rec.tool_name == ACCESS_TOOL


def test_the_narrative_names_both_identities_and_the_basis():
    """"An entity saw it" is the transparency this record type replaces."""
    trail = AuditTrail()
    _read(trail)

    line = find_accesses(trail._records)[0].narrative

    assert "agent/log-search@istekki" in line
    assert "person/hetu-pseudonym/9f2c" in line
    assert "hoitosuhteen tarkistus" in line
    assert "patient/4711/kayttoloki" in line


# ── Did anything follow? ────────────────────────────────────────────────────
#
# The case these came from, Henri 2026-09-09: Maisa.fi says "maintainer has
# seen the message". An anonymous entity, no name, no reason, and no way to
# tell whether anyone acted on it. Naming the reader is half the answer. The
# other half is whether the read produced anything.

_NOTE = ActionType(
    name="write_note",
    category=ActionCategory.DATA,
    reversibility=Reversibility.FULLY,
    blast_radius=BlastRadius.LOCAL,
)


def _acted_on(trail, access_action_id):
    return trail.record_action_requested(ActionRequest(
        action_type=_NOTE, tool_name="write_note", agent_id="nurse/console",
        parameters={"access_ref": access_action_id, "note": "callback booked"},
    ))


def test_an_access_that_produced_something_can_be_followed():
    trail = AuditTrail()
    access_id = _read(trail)
    _acted_on(trail, access_id)

    hits = follow_ups(trail._records, access_id)

    assert hits
    assert all(h.event_type != EventType.ACCESS_RECORDED for h in hits)
    assert accesses_with_no_recorded_outcome(trail._records) == []


def test_an_access_nothing_refers_back_to_is_listed():
    trail = AuditTrail()
    seen_and_dropped = _read(trail)
    acted = _read(trail, subject="patient/4712/kayttoloki")
    _acted_on(trail, acted)

    orphans = accesses_with_no_recorded_outcome(trail._records)

    assert [o.action_id for o in orphans] == [seen_and_dropped]


def test_a_follow_up_for_one_access_does_not_cover_another():
    trail = AuditTrail()
    first = _read(trail)
    second = _read(trail, subject="patient/4712/kayttoloki")
    _acted_on(trail, first)

    assert follow_ups(trail._records, second) == []
    assert [o.action_id for o in accesses_with_no_recorded_outcome(trail._records)] == [second]


def test_the_orphan_list_does_not_claim_nothing_happened():
    """Absence of a follow-on record is not evidence that nothing followed.

    Pinned as a test because the docstring is the only thing stopping a
    reader from treating this list as proof of inaction, and a docstring
    nobody executes is a comment.
    """
    trail = AuditTrail()
    access_id = _read(trail)
    # An action that really happened but was recorded without the back
    # reference. The access still lands in the orphan list.
    trail.record_action_requested(ActionRequest(
        action_type=_NOTE, tool_name="write_note", agent_id="nurse/console",
        parameters={"note": "acted, but nobody linked it"},
    ))

    orphans = accesses_with_no_recorded_outcome(trail._records)

    assert [o.action_id for o in orphans] == [access_id]


# ── Three states, not two ───────────────────────────────────────────────────
#
# Henri, 2026-09-09: there should always be a reason to open a patient file,
# and if there is a reason there is usually an outcome. Either somebody went
# and did something, or somebody was reading for no clinical purpose. Both are
# outcomes and only one of them is lawful.

def test_an_access_can_be_closed_with_an_action():
    trail = AuditTrail()
    access_id = _read(trail)
    record_access_outcome(
        trail, access_id, outcome="uusintapyynto vietiin laakarille",
        action_taken=True, by="nurse/jarvisalo",
    )

    assert follow_ups(trail._records, access_id)
    assert accesses_with_no_recorded_outcome(trail._records) == []
    assert accesses_closed_with_no_action(trail._records) == []


def test_an_access_can_be_closed_with_no_action_and_that_is_a_record():
    trail = AuditTrail()
    access_id = _read(trail)
    record_access_outcome(
        trail, access_id, outcome="selattiin ilman hoidollista tarvetta",
        action_taken=False, by="nurse/unknown",
    )

    idle = accesses_closed_with_no_action(trail._records)

    assert [a.action_id for a in idle] == [access_id]
    # Closed is closed. It leaves the set nobody addressed.
    assert accesses_with_no_recorded_outcome(trail._records) == []


def test_never_closed_is_not_the_same_as_closed_with_no_action():
    """The third value. Nobody said anything is not the same as somebody
    saying nothing came of it, and collapsing them is the failure this
    project refuses everywhere else."""
    trail = AuditTrail()
    silent = _read(trail)
    stated = _read(trail, subject="patient/4712/kayttoloki")
    record_access_outcome(
        trail, stated, outcome="ei toimenpiteita", action_taken=False,
        by="nurse/kirjaaja",
    )

    never_closed = [a.action_id for a in accesses_with_no_recorded_outcome(trail._records)]
    closed_idle = [a.action_id for a in accesses_closed_with_no_action(trail._records)]

    assert never_closed == [silent]
    assert closed_idle == [stated]
    assert never_closed != closed_idle


def test_closing_an_access_needs_a_reason_and_a_name():
    trail = AuditTrail()
    access_id = _read(trail)
    with pytest.raises(ValueError, match="silence this record exists to replace"):
        record_access_outcome(trail, access_id, outcome="", action_taken=True, by="x")
    with pytest.raises(ValueError, match="who is closing"):
        record_access_outcome(trail, access_id, outcome="done", action_taken=True, by="")
    with pytest.raises(ValueError, match="must name the access"):
        record_access_outcome(trail, "", outcome="done", action_taken=True, by="x")


def test_the_outcome_carries_its_own_timestamp_and_name():
    trail = AuditTrail()
    access_id = _read(trail)
    record_access_outcome(
        trail, access_id, outcome="soitettiin potilaalle", action_taken=True,
        by="nurse/jarvisalo", details={"yksikko": "Oulunkyla"},
    )

    closing = follow_ups(trail._records, access_id)[0]
    params = closing.data["parameters"]

    assert params["closed_by"] == "nurse/jarvisalo"
    assert params["action_taken"] is True
    assert params["yksikko"] == "Oulunkyla"
    assert closing.timestamp
