"""Normalize adjacent MCP records into the SEP-2828 evidence model.

Vaara is the receiving side for agent execution evidence. A SEP-2828
execution record is a signed decision+outcome pair, but the surrounding
MCP ecosystem emits narrower, single-purpose records that each cover one
face of the same event:

  - SEP-2643 structured authorization denials carry the server's
    refusal: the *outcome* of a call that was declined.
  - SEP-2787 tool-call attestations carry the attested request: the
    binding a receipt answers (the exact back-link a conformant receipt
    must pin).
  - SEP-2817 AI invocation audit context carries client-asserted
    *input*: why the agent asked, which model, the user intent, the turn
    id. Its own specification says it MUST NOT be used as authorization
    evidence.

This module reads one such record and maps it onto the SEP-2828 model:
which evidence plane it fills, which SEP-2828 fields it populates, and
what is still missing before the evidence amounts to a complete signed
execution record. It invents nothing and it promotes nothing: an
unsigned client claim stays advisory. It reports honestly what each
source does and does not establish.

The SEP-2643 and SEP-2817 maps are pure standard library and run in the
base install. The SEP-2787 map computes the back-link digest the way the
receipt verifier computes it: sha256 over the JCS canonicalization of the
SEP-2787-modeled fields (parse, then canonicalize), the same value a
conformant receipt pins. Fields outside the modeled schema are not
covered. That step needs the attestation extra and degrades to a reported
gap when the extra is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

SOURCE_SEP2643 = "sep2643"
SOURCE_SEP2787 = "sep2787"
SOURCE_SEP2817 = "sep2817"
SOURCE_UNKNOWN = "unknown"

PLANE_OUTCOME = "outcome"
PLANE_DECISION_ATTESTED = "decision-attested"
PLANE_DECISION_INPUT = "decision-input"

# The reserved request _meta key SEP-2817 defines for invocation audit context.
AI_INVOCATION_KEY = "io.modelcontextprotocol/aiInvocation"


@dataclass(frozen=True)
class NormalizedEvidence:
    """One foreign record mapped onto the SEP-2828 evidence model.

    ``sep2828`` holds only the real SEP-2828 record fields the source
    establishes; ``advisory`` holds context the source carries that is
    not, on its own, a SEP-2828 proof. ``missing`` lists what a complete
    signed record still needs that this source does not supply.
    """

    source_format: str
    source_title: str
    recognized: bool
    evidence_plane: Optional[str] = None
    sep2828: dict[str, Any] = field(default_factory=dict)
    advisory: dict[str, Any] = field(default_factory=dict)
    populated: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceFormat": self.source_format,
            "sourceTitle": self.source_title,
            "recognized": self.recognized,
            "evidencePlane": self.evidence_plane,
            "sep2828": self.sep2828,
            "advisory": self.advisory,
            "populated": list(self.populated),
            "missing": list(self.missing),
            "notes": list(self.notes),
        }


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _denial_authorization(doc: Any) -> Optional[dict[str, Any]]:
    """Locate the SEP-2643 ``authorization`` object inside a denial.

    Accepts the full JSON-RPC error response (authorization under
    ``error.data``), a bare ``data`` object, a bare ``authorization``
    object, or the authorization object itself. Returns it only when it
    carries the REQUIRED ``reason`` string, which distinguishes a denial
    from the other records this module reads.
    """
    if not isinstance(doc, dict):
        return None
    # Nested positions are unambiguously a denial envelope.
    for cand in (
        _as_dict(_as_dict(doc.get("error")).get("data")).get("authorization"),
        _as_dict(doc.get("data")).get("authorization"),
    ):
        if isinstance(cand, dict) and isinstance(cand.get("reason"), str):
            return cand
    # A bare authorization object, or the document itself, must also carry a
    # denial marker, so a stray top-level `reason` on an unrelated record is
    # not read as a denial ahead of the SEP-2817 reader.
    for cand in (doc.get("authorization"), doc):
        if (
            isinstance(cand, dict)
            and isinstance(cand.get("reason"), str)
            and ("authorizationContextId" in cand or "remediationHints" in cand)
        ):
            return cand
    return None


def _ai_invocation(doc: Any) -> Optional[dict[str, Any]]:
    """Locate the SEP-2817 aiInvocation object.

    Accepts a full request (object under ``params._meta[KEY]``), a bare
    ``_meta`` block, a bare object under ``KEY``, or the aiInvocation
    object itself (recognized by one of its four defined fields).
    """
    if not isinstance(doc, dict):
        return None
    candidates = (
        _as_dict(_as_dict(doc.get("params")).get("_meta")).get(AI_INVOCATION_KEY),
        _as_dict(doc.get("_meta")).get(AI_INVOCATION_KEY),
        doc.get(AI_INVOCATION_KEY),
    )
    for cand in candidates:
        if isinstance(cand, dict):
            return cand
    if any(k in doc for k in ("invocationReason", "model", "userIntent", "turnId")):
        return doc
    return None


def _looks_like_attestation(doc: Any) -> bool:
    return isinstance(doc, dict) and all(
        k in doc
        for k in ("plannerDeclared", "issuerAsserted", "payloadDerived", "signature")
    )


def detect_format(doc: Any) -> str:
    """Identify which adjacent-SEP record ``doc`` is, or ``"unknown"``."""
    if _looks_like_attestation(doc):
        return SOURCE_SEP2787
    if _denial_authorization(doc) is not None:
        return SOURCE_SEP2643
    if _ai_invocation(doc) is not None:
        return SOURCE_SEP2817
    return SOURCE_UNKNOWN


def normalize(doc: Any, *, source_format: str = "auto") -> NormalizedEvidence:
    """Map a foreign MCP record onto the SEP-2828 evidence model.

    ``source_format`` is ``"auto"`` (detect) or one of ``"sep2643"``,
    ``"sep2787"``, ``"sep2817"`` to force a reading.
    """
    fmt = detect_format(doc) if source_format == "auto" else source_format
    if fmt == SOURCE_SEP2787:
        return _normalize_attestation(doc)
    if fmt == SOURCE_SEP2643:
        return _normalize_denial(doc)
    if fmt == SOURCE_SEP2817:
        return _normalize_invocation(doc)
    return NormalizedEvidence(
        source_format=SOURCE_UNKNOWN,
        source_title="unrecognized record",
        recognized=False,
        notes=(
            "not a SEP-2643 denial, SEP-2787 attestation, or SEP-2817 "
            "invocation audit context; nothing to normalize",
        ),
    )


def _unrecognized(fmt: str, title: str, why: str) -> NormalizedEvidence:
    return NormalizedEvidence(
        source_format=fmt,
        source_title=title,
        recognized=False,
        notes=(f"does not look like a {title}: {why}",),
    )


def _normalize_denial(doc: Any) -> NormalizedEvidence:
    authz = _denial_authorization(doc)
    if authz is None:
        return _unrecognized(
            SOURCE_SEP2643, "SEP-2643 authorization denial",
            "no authorization object with a 'reason' field",
        )
    advisory: dict[str, Any] = {"reason": authz.get("reason")}
    ctx = authz.get("authorizationContextId")
    if isinstance(ctx, str):
        advisory["authorizationContextId"] = ctx
    hints = authz.get("remediationHints")
    if isinstance(hints, list):
        types = [
            h["type"]
            for h in hints
            if isinstance(h, dict) and isinstance(h.get("type"), str)
        ]
        if types:
            advisory["remediationHintTypes"] = types
    notes = (
        "a denial is the outcome of a refused call: it maps to "
        "outcomeDerived.status = refused",
        "a refused outcome carries no resultCommitment",
        "authorizationContextId is a correlation handle, not authorization "
        "material",
        "the denial carries no completedAt, no signing envelope, and no "
        "back-link; the recording side supplies those",
    )
    return NormalizedEvidence(
        source_format=SOURCE_SEP2643,
        source_title="SEP-2643 authorization denial",
        recognized=True,
        evidence_plane=PLANE_OUTCOME,
        sep2828={"outcomeDerived": {"status": "refused"}},
        advisory=advisory,
        populated=("outcomeDerived.status",),
        missing=(
            "alg", "signature", "backLink", "receiptAsserted",
            "outcomeDerived.completedAt",
        ),
        notes=notes,
    )


def _normalize_invocation(doc: Any) -> NormalizedEvidence:
    inv = _ai_invocation(doc)
    if inv is None:
        return _unrecognized(
            SOURCE_SEP2817, "SEP-2817 AI invocation audit context",
            f"no '{AI_INVOCATION_KEY}' object",
        )
    advisory: dict[str, Any] = {}
    reason = _as_dict(inv.get("invocationReason")).get("text")
    if isinstance(reason, str):
        advisory["invocationReason"] = reason
    model = _as_dict(inv.get("model")).get("name")
    if isinstance(model, str):
        advisory["model"] = model
    ui = _as_dict(inv.get("userIntent"))
    if ui.get("redacted") is True:
        # The source flagged the intent as redacted: surface the flag, never
        # the text, so Vaara does not re-propagate content the source withheld.
        advisory["userIntentRedacted"] = True
    elif isinstance(ui.get("text"), str):
        advisory["userIntent"] = ui["text"]
    turn = inv.get("turnId")
    if isinstance(turn, str):
        advisory["turnId"] = turn
    notes = [
        "SEP-2817 is client-asserted input audit context; per its own "
        "specification it MUST NOT be used as authorization evidence",
        "it maps to the decision-input plane (the agent's stated intent), "
        "the unsigned counterpart of an attested rationale, and populates no "
        "required SEP-2828 field",
    ]
    if "turnId" in advisory:
        notes.append(
            "turnId groups requests from one user turn; it is correlation only"
        )
    return NormalizedEvidence(
        source_format=SOURCE_SEP2817,
        source_title="SEP-2817 AI invocation audit context",
        recognized=True,
        evidence_plane=PLANE_DECISION_INPUT,
        sep2828={},
        advisory=advisory,
        populated=(),
        missing=("alg", "signature", "backLink", "receiptAsserted", "outcomeDerived"),
        notes=tuple(notes),
    )


def _normalize_attestation(doc: Any) -> NormalizedEvidence:
    if not _looks_like_attestation(doc):
        return _unrecognized(
            SOURCE_SEP2787, "SEP-2787 tool-call attestation",
            "missing one of plannerDeclared/issuerAsserted/payloadDerived/signature",
        )
    issuer = _as_dict(doc.get("issuerAsserted"))
    planner = _as_dict(doc.get("plannerDeclared"))
    payload = _as_dict(doc.get("payloadDerived"))

    advisory: dict[str, Any] = {}
    if isinstance(planner.get("intent"), str):
        advisory["intent"] = planner["intent"]
    for f in ("iss", "sub", "alg"):
        if isinstance(issuer.get(f), str):
            advisory[f"attestation_{f}"] = issuer[f]
    calls = payload.get("toolCalls")
    if isinstance(calls, list):
        advisory["toolCalls"] = [
            {"name": c.get("name"), "serverFingerprint": c.get("serverFingerprint")}
            for c in calls
            if isinstance(c, dict)
        ]

    notes = [
        "a SEP-2787 attestation is the attested request a SEP-2828 receipt "
        "answers; it fixes the exact back-link a conformant receipt must pin",
        "plannerDeclared.intent is client-declared and bound by the issuer's "
        "signature, not asserted true by the issuer",
        "the record's own signing (alg, signature, receiptAsserted) is a "
        "separate event by the recording side, not derived from the attestation",
    ]

    back_link, gap = _back_link_from_attestation(doc)
    if back_link is None:
        notes.append(gap or "back-link could not be computed")
        return NormalizedEvidence(
            source_format=SOURCE_SEP2787,
            source_title="SEP-2787 tool-call attestation",
            recognized=True,
            evidence_plane=PLANE_DECISION_ATTESTED,
            sep2828={},
            advisory=advisory,
            populated=(),
            missing=(
                "alg", "signature", "backLink", "receiptAsserted", "outcomeDerived",
            ),
            notes=tuple(notes),
        )
    return NormalizedEvidence(
        source_format=SOURCE_SEP2787,
        source_title="SEP-2787 tool-call attestation",
        recognized=True,
        evidence_plane=PLANE_DECISION_ATTESTED,
        sep2828={"backLink": back_link},
        advisory=advisory,
        populated=("backLink.attestationDigest", "backLink.attestationNonce"),
        missing=("alg", "signature", "receiptAsserted", "outcomeDerived"),
        notes=tuple(notes),
    )


def _back_link_from_attestation(
    doc: Any,
) -> tuple[Optional[dict[str, str]], Optional[str]]:
    """Compute the SEP-2828 back-link a receipt answering this attestation pins.

    Returns ``(back_link, None)`` on success or ``(None, reason)`` when the
    attestation extra is absent or the attestation does not parse. The
    digest is computed the way the receipt verifier computes it: sha256 over
    the JCS canonicalization of the SEP-2787-modeled fields (parse, then
    canonicalize), so it is the exact value a conformant receipt's
    ``backLink.attestationDigest`` must equal. Fields outside the modeled
    schema are not covered.
    """
    try:
        from vaara.attestation.receipt import make_back_link
        from vaara.attestation.sep2787 import AttestationError, parse_attestation
    except ImportError:
        return None, (
            "back-link digest needs the attestation extra "
            "(pip install 'vaara[attestation]')"
        )
    try:
        attestation = parse_attestation(doc)
    except (AttestationError, KeyError, TypeError, ValueError) as exc:
        return None, f"not a parseable SEP-2787 attestation: {exc}"
    try:
        bl = make_back_link(attestation)
    except AttestationError as exc:
        return None, f"back-link digest unavailable: {exc}"
    return (
        {
            "attestationDigest": bl.attestation_digest,
            "attestationNonce": bl.attestation_nonce,
        },
        None,
    )
