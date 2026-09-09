# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Who opened it: access records that survive the party who wrote them.

The rest of the trail records what an agent DID. Every member of
:class:`~vaara.audit.trail.EventType` before this one is an action:
requested, scored, decided, executed, blocked, escalated, overridden.
Reading was not an event, so nothing in the trail answered the question a
supervisor actually asks first, which is who opened this.

Two things make an access record different from an ordinary access log.

**It names two identities and one of them is a person.** When an agent
performs a read, "who accessed this" has two answers: the agent, and the
principal it acted for. A single actor field collapses them and the
collapse is not recoverable afterwards. Finnish health law is the sharp
case: laki sosiaali- ja terveydenhuollon asiakastietojen kasittelysta
(703/2023) section 11 gives a client the right to be told who used their
data and on what basis, and a service account name does not answer it.
So ``accessed_by`` and ``on_behalf_of`` are both required and neither
defaults.

**It commits to what came back, and holds none of it.** "Agent X opened
patient Y" does not say what X saw, and a dispute is about content. The
record therefore carries a digest over the returned set. It never carries
the set. An access record over health data that copied the data would be
a second unprotected copy of the thing it exists to protect, so
:func:`record_access` takes a digest and has no parameter that could
accept the rows.

Everything else follows from living in the same trail. The record is
hash-chained with its neighbours, so an access cannot be removed without
breaking the chain; it is covered by ``anchor_head``, so the time is
attested by a party outside the operator rather than asserted by the
operator; and ``verify_segment`` proves one access record between two
anchors without walking the trail.

What it does not do, stated so nobody infers it: this proves the system
recorded an access at that moment inside a tamper-evident chain. It does
not prove a human read the screen, and it cannot prove that an access
which was never recorded did not happen. Completeness over a stream of
access records is a separate property and it is not claimed here.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Optional

from vaara.audit.trail import EventType

ACCESS_TOOL = "vaara.access.read"
ACCESS_OUTCOME_TOOL = "vaara.access.outcome"

#: Fields the record always names, so a verifier can enumerate what was
#: captured rather than guessing from what happens to be present. A
#: deployment adds its own through ``deployment_fields``; it cannot remove
#: these.
CORE_FIELDS = (
    "subject",
    "accessed_by",
    "on_behalf_of",
    "basis",
    "returned_sha256",
    "returned_count",
    "authorisation_ref",
    "channel",
)


def digest_returned(payload: bytes) -> str:
    """``sha256:`` digest over the bytes a read returned.

    Compute this at the point the result leaves the store and pass the
    string to :func:`record_access`. Kept as a named helper so the caller
    never has a reason to hand the payload itself to this module.
    """
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def record_access(
    trail,
    *,
    subject: str,
    accessed_by: str,
    on_behalf_of: str,
    basis: str,
    returned_sha256: str = "",
    returned_count: Optional[int] = None,
    authorisation_ref: str = "",
    channel: str = "",
    session_id: str = "",
    deployment_fields: Optional[Mapping[str, Any]] = None,
) -> str:
    """Record one read into ``trail``. Returns the ``action_id``.

    ``subject`` identifies what was opened, in whatever identifier space
    the deployment already uses. It is written to the trail as given, so
    pass a reference rather than content: a record id, a pseudonym, a log
    source name. ``accessed_by`` is the agent, service or process that
    performed the read. ``on_behalf_of`` is the principal it acted for and
    is the field a client's own log request has to be answerable from.
    ``basis`` is why, in the deployment's own vocabulary, and it is the
    "peruste" the Finnish asiakastietolaki asks for beside the identity.

    ``returned_sha256`` commits to what came back; build it with
    :func:`digest_returned`. It is optional because some callers genuinely
    cannot compute it, and when it is absent the record says so through
    ``fields_present`` instead of leaving a reader to assume the read
    returned nothing. ``returned_count`` is the number of items returned,
    which bounds an access even when the digest is missing.

    ``deployment_fields`` carries whatever a sector needs beyond the core
    set. The keys are recorded in ``fields_present`` alongside the core
    ones, so a verifier reading the receipt can state what was captured
    and what was not. An unbounded blob would not survive that question,
    which is why the enumeration exists rather than a free-form note.

    Raises ``ValueError`` when either identity or the basis is missing.
    Those three are the record's reason to exist and a default for any of
    them would produce an access record that answers nothing.
    """
    if not subject:
        raise ValueError("subject must not be empty: an access record names what was opened")
    if not accessed_by:
        raise ValueError("accessed_by must not be empty: name the agent or process that read")
    if not on_behalf_of:
        raise ValueError(
            "on_behalf_of must not be empty: an access record whose only identity is "
            "the agent cannot answer who used the data"
        )
    if not basis:
        raise ValueError("basis must not be empty: record why the read was permitted")
    if returned_sha256 and not returned_sha256.startswith("sha256:"):
        raise ValueError(
            f"returned_sha256 must be a 'sha256:' prefixed digest, got {returned_sha256!r}; "
            "use digest_returned()"
        )

    extra = dict(deployment_fields or {})
    for reserved in CORE_FIELDS + ("fields_present",):
        if reserved in extra:
            raise ValueError(
                f"deployment_fields may not shadow the core field {reserved!r}"
            )

    parameters: dict[str, Any] = {
        "subject": subject,
        "accessed_by": accessed_by,
        "on_behalf_of": on_behalf_of,
        "basis": basis,
        "returned_sha256": returned_sha256,
        "returned_count": returned_count,
        "authorisation_ref": authorisation_ref,
        "channel": channel,
    }
    parameters.update(extra)
    # Present means "carries a value here", so a caller that omitted the
    # returned digest produces a record that states the omission rather
    # than one a reader has to interpret.
    parameters["fields_present"] = sorted(
        k for k, v in parameters.items()
        if v not in ("", None) and k != "fields_present"
    )

    from vaara.pipeline import InterceptionPipeline

    pipeline = InterceptionPipeline(trail=trail, enforce=False)
    result = pipeline.intercept(
        agent_id=accessed_by,
        tool_name=ACCESS_TOOL,
        parameters=parameters,
        session_id=session_id,
        context={"vaara_access": True},
        _event_type_override=EventType.ACCESS_RECORDED,
    )
    return result.action_id


def record_access_outcome(
    trail,
    access_action_id: str,
    *,
    outcome: str,
    action_taken: bool,
    by: str,
    details: Optional[Mapping[str, Any]] = None,
    session_id: str = "",
) -> str:
    """Close an access with what came of it, including nothing.

    Henri's framing, 2026-09-09, and it is the reason this exists: there
    should always be a reason to open a patient file, and if there is a
    reason there is usually an outcome. Either somebody went and did
    something about it, or somebody was reading for no clinical purpose.
    Both are outcomes. Only one of them is lawful, and neither is
    currently sayable.

    ``action_taken`` is the part that carries weight. Passing ``False``
    with an honest ``outcome`` writes a record that says the read led
    nowhere, which is a different and much more useful fact than the read
    simply going unmentioned. A supervisor can find those; a client
    exercising a log request can see one.

    So an access ends in one of three states, and they are three facts
    rather than two:

    * closed with an action, ``action_taken=True``
    * closed with no action, ``action_taken=False``
    * never closed, which is what
      :func:`accesses_with_no_recorded_outcome` returns

    The third is not the second. Nobody has said anything about it, and
    reading that as "nothing happened" is the same collapse this project
    refuses everywhere else: could-not-determine is not the same answer as
    determined-negative.

    Returns the ``action_id`` of the outcome record.
    """
    if not access_action_id:
        raise ValueError("access_action_id must name the access being closed")
    if not outcome:
        raise ValueError(
            "outcome must not be empty: closing an access with a blank reason "
            "is the silence this record exists to replace"
        )
    if not by:
        raise ValueError("by must name who is closing the access")

    parameters: dict[str, Any] = {
        "access_ref": access_action_id,
        "outcome": outcome,
        "action_taken": bool(action_taken),
        "closed_by": by,
    }
    for k, v in dict(details or {}).items():
        if k in parameters:
            raise ValueError(f"details may not shadow {k!r}")
        parameters[k] = v

    from vaara.pipeline import InterceptionPipeline

    pipeline = InterceptionPipeline(trail=trail, enforce=False)
    result = pipeline.intercept(
        agent_id=by,
        tool_name=ACCESS_OUTCOME_TOOL,
        parameters=parameters,
        session_id=session_id,
        context={"vaara_access": True},
    )
    return result.action_id


def accesses_closed_with_no_action(records) -> list:
    """Accesses somebody closed by stating that nothing came of them.

    Distinct from :func:`accesses_with_no_recorded_outcome`, and the
    distinction is the point. Here a named person said the read led
    nowhere. There, nobody said anything at all.

    This is the list worth reading first when the question is whether
    files are being opened without a reason that survives being written
    down.
    """
    closed_idle = {
        (rec.data or {}).get("parameters", {}).get("access_ref")
        for rec in records
        if rec.tool_name == ACCESS_OUTCOME_TOOL
        and (rec.data or {}).get("parameters", {}).get("action_taken") is False
    }
    return [
        rec for rec in records
        if rec.event_type == EventType.ACCESS_RECORDED
        and rec.action_id in closed_idle
    ]


def follow_ups(records, access_action_id: str) -> list:
    """Records that name ``access_action_id`` as what they followed.

    A reader carries a follow-on action's ``access_ref`` in its own
    parameters, which is how "she opened it and then wrote this" becomes
    checkable instead of asserted.
    """
    out = []
    for rec in records:
        if rec.event_type == EventType.ACCESS_RECORDED:
            continue
        params = (rec.data or {}).get("parameters", {}) or {}
        if params.get("access_ref") == access_action_id:
            out.append(rec)
    return out


def accesses_with_no_recorded_outcome(records) -> list:
    """Access records that nothing else in ``records`` refers back to.

    This is the question a person actually asks, and the reason this
    function exists: a portal that says "maintainer has seen the message"
    tells you an entity looked and stops there. It cannot tell you whether
    anything came of it.

    **Read the result precisely.** An access appearing here means NO
    RECORD EXISTS that refers back to it. It does not mean nothing
    happened. A read that produced a decision nobody recorded, and a read
    that produced nothing, land in this list together and this function
    cannot separate them.

    Separating them is what :func:`record_access_outcome` is for: closing
    an access with ``action_taken=False`` moves it out of this list and
    into :func:`accesses_closed_with_no_action`, where somebody has put
    their name to the statement. What stays here is the set nobody
    addressed at all, and that is a third fact rather than a weaker
    version of the second.

    What the list is good for is the direction people care about: it is
    the set of times somebody opened your file and the trail has nothing
    to show for it. Asking about those is a fair question, and today the
    person whose file it was cannot even form it.
    """
    referenced = set()
    for rec in records:
        if rec.event_type == EventType.ACCESS_RECORDED:
            continue
        params = (rec.data or {}).get("parameters", {}) or {}
        ref = params.get("access_ref")
        if ref:
            referenced.add(ref)
    return [
        rec for rec in records
        if rec.event_type == EventType.ACCESS_RECORDED
        and rec.action_id not in referenced
    ]


def find_accesses(records, *, subject: str = "", on_behalf_of: str = "") -> list:
    """Access records in ``records``, optionally narrowed.

    ``subject`` answers "who opened this record", which is the client's
    own question under asiakastietolaki 703/2023 section 11.
    ``on_behalf_of`` answers "what did this person open", which is the
    supervisory direction. Both filters match exactly; no normalisation is
    applied, because the deployment owns the identifier space and guessing
    at case or format here would silently drop rows from a legal answer.
    """
    out = []
    for rec in records:
        if rec.event_type != EventType.ACCESS_RECORDED:
            continue
        params = (rec.data or {}).get("parameters", {}) or {}
        if subject and params.get("subject") != subject:
            continue
        if on_behalf_of and params.get("on_behalf_of") != on_behalf_of:
            continue
        out.append(rec)
    return out
