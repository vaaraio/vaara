# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The mandate block: a qualified attestation of attributes, carried on a grant.

Internal module. Public surface is in ``vaara.credential``.

Every actor claim in a grant today is self-asserted. ``asserted.iss`` and
``.sub`` are strings the issuer writes about itself. A verifier can check the
signature, recompute the args commitment and confirm the grant is bound to one
attestation instance. What no verifier can check is whether the issuer is the
organisation it says it is.

A Qualified Electronic Attestation of Attributes is an eIDAS trust service
(``EAA/Q``). A supervised, audited provider attests an attribute of a subject,
and an organisation cannot issue one about itself by construction. That is the
whole reason this block is worth carrying: it is the one claim on the record
whose trust comes from a public register rather than from the producer.

Design notes, because both have bitten other people on the SCITT list:

* The attestation is carried **verbatim**, base64 of the bytes the provider
  issued. A normalised copy is not the evidence.
* ``attestationDigest`` is SHA-256 over those **decoded** bytes, not over the
  base64 transport encoding. Two verifiers that read this differently would
  disagree about a valid record, so it is stated here and tested.
* The block is optional and, when absent, adds nothing to the signed preimage.
  A grant without a mandate is byte-identical to one minted before this
  existed.

See ``docs/design/qeaa-attribute-binding-spec.md``. Resolving the provider
against the EU trusted lists is a separate step and is not in this module.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Optional

from vaara.attestation._attest_types import AttestationError

#: The eIDAS service type for a qualified attestation of attributes. Any other
#: trust service is a different instrument, whatever a field says about it.
EAA_Q_SERVICE_TYPE = "http://uri.etsi.org/TrstSvc/Svctype/EAA/Q"

#: How the agent is tied to the attested legal person. ``attestationDigest``
#: means the agent id appears inside the attested attribute set. The spec
#: reserves other values, so an unknown one is a reject rather than a pass.
VALID_BOUND_VIA = frozenset({"attestationDigest"})

MANDATE_KEYS = frozenset({
    "format", "attestationDigest", "attestation",
    "issuer", "subject", "agentBinding",
})
MANDATE_ISSUER_KEYS = frozenset({
    "serviceTypeIdentifier", "territory", "trustListRef", "providerName",
})
MANDATE_SUBJECT_KEYS = frozenset({"legalPersonIdentifier", "attributeSet"})
MANDATE_AGENT_BINDING_KEYS = frozenset({"agentId", "boundVia"})

VALID_FORMATS = frozenset({"eaa-q-jwt"})

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class GrantMandate:
    """A qualified attestation of attributes bound to one grant.

    ``attestation`` is optional: the digest-only path lets the commitment
    travel while the attestation itself is withheld, which is what makes
    selective disclosure a follow-on rather than a redesign.
    """

    attestation_digest: str
    service_type_identifier: str
    territory: str
    trust_list_ref: str
    provider_name: str
    legal_person_identifier: str
    attribute_set: tuple[str, ...]
    agent_id: str
    attestation: Optional[str] = None
    format: str = "eaa-q-jwt"
    bound_via: str = "attestationDigest"

    def __post_init__(self) -> None:
        _validate(self)


def _require_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise AttestationError(f"mandate.{where} must be a non-empty string")
    return value


def _validate(m: GrantMandate) -> None:
    if m.format not in VALID_FORMATS:
        raise AttestationError(
            f"mandate.format {m.format!r} is not a known attestation format"
        )
    if not _DIGEST_RE.match(m.attestation_digest):
        raise AttestationError(
            "mandate.attestationDigest MUST be 'sha256:' followed by 64 hex "
            "characters"
        )
    if m.service_type_identifier != EAA_Q_SERVICE_TYPE:
        raise AttestationError(
            f"mandate.issuer.serviceTypeIdentifier must be {EAA_Q_SERVICE_TYPE} "
            f"(EAA/Q); got {m.service_type_identifier!r}"
        )
    _require_text(m.territory, "issuer.territory")
    _require_text(m.trust_list_ref, "issuer.trustListRef")
    _require_text(m.provider_name, "issuer.providerName")
    _require_text(m.legal_person_identifier, "subject.legalPersonIdentifier")
    if not m.attribute_set or not all(
        isinstance(a, str) and a for a in m.attribute_set
    ):
        raise AttestationError(
            "mandate.subject.attributeSet must be a non-empty list of non-empty "
            "strings; an attestation that attests nothing is not evidence"
        )
    _require_text(m.agent_id, "agentBinding.agentId")
    if m.bound_via not in VALID_BOUND_VIA:
        raise AttestationError(
            f"mandate.agentBinding.boundVia {m.bound_via!r} is reserved; "
            f"known values are {sorted(VALID_BOUND_VIA)}"
        )
    if m.attestation is not None:
        _require_text(m.attestation, "attestation")


def mandate_to_dict(m: GrantMandate) -> dict[str, Any]:
    """Wire form. ``attestation`` is omitted, never nulled, when withheld."""
    d: dict[str, Any] = {
        "agentBinding": {"agentId": m.agent_id, "boundVia": m.bound_via},
        "attestationDigest": m.attestation_digest,
        "format": m.format,
        "issuer": {
            "providerName": m.provider_name,
            "serviceTypeIdentifier": m.service_type_identifier,
            "territory": m.territory,
            "trustListRef": m.trust_list_ref,
        },
        "subject": {
            "attributeSet": list(m.attribute_set),
            "legalPersonIdentifier": m.legal_person_identifier,
        },
    }
    if m.attestation is not None:
        d["attestation"] = m.attestation
    return d


def _closed(d: Any, allowed: frozenset[str], where: str) -> dict[str, Any]:
    if not isinstance(d, dict):
        raise AttestationError(f"{where} must be an object")
    extra = set(d) - allowed
    if extra:
        raise AttestationError(
            f"{where} carries unrecognized field(s) {sorted(extra)!r}; "
            "the signed schema is closed"
        )
    return d


def mandate_from_dict(d: Any) -> GrantMandate:
    """Parse a mandate wire object against the closed schema."""
    d = _closed(d, MANDATE_KEYS, "mandate")
    issuer = _closed(d.get("issuer"), MANDATE_ISSUER_KEYS, "mandate.issuer")
    subject = _closed(d.get("subject"), MANDATE_SUBJECT_KEYS, "mandate.subject")
    agent_binding = _closed(
        d.get("agentBinding"), MANDATE_AGENT_BINDING_KEYS, "mandate.agentBinding"
    )

    attribute_set = subject.get("attributeSet")
    if not isinstance(attribute_set, list):
        raise AttestationError("mandate.subject.attributeSet must be a list")

    attestation = d.get("attestation")
    if attestation is not None and not isinstance(attestation, str):
        raise AttestationError("mandate.attestation must be a string when present")

    return GrantMandate(
        attestation_digest=_require_text(
            d.get("attestationDigest"), "attestationDigest"
        ),
        attestation=attestation,
        format=_require_text(d.get("format"), "format"),
        service_type_identifier=_require_text(
            issuer.get("serviceTypeIdentifier"), "issuer.serviceTypeIdentifier"
        ),
        territory=_require_text(issuer.get("territory"), "issuer.territory"),
        trust_list_ref=_require_text(
            issuer.get("trustListRef"), "issuer.trustListRef"
        ),
        provider_name=_require_text(
            issuer.get("providerName"), "issuer.providerName"
        ),
        legal_person_identifier=_require_text(
            subject.get("legalPersonIdentifier"), "subject.legalPersonIdentifier"
        ),
        attribute_set=tuple(attribute_set),
        agent_id=_require_text(agent_binding.get("agentId"), "agentBinding.agentId"),
        bound_via=_require_text(
            agent_binding.get("boundVia"), "agentBinding.boundVia"
        ),
    )


def verify_mandate_binding(mandate: GrantMandate) -> bool:
    """True iff the carried attestation matches the digest that commits to it.

    Returns True when the attestation is withheld: the digest-only path is a
    supported shape, and there is nothing carried to disagree with. A caller
    that requires the bytes to be present must check for them itself, because
    "not carried" and "carried and wrong" are different conditions and must not
    collapse into one answer.

    Undecodable base64 returns False rather than raising. A malformed
    attestation is a failed binding, not a crash on the verification path.
    """
    if mandate.attestation is None:
        return True
    try:
        raw = base64.b64decode(mandate.attestation, validate=True)
    except (binascii.Error, ValueError):
        return False
    return mandate.attestation_digest == "sha256:" + hashlib.sha256(raw).hexdigest()


def mandate_digest_of(attestation_bytes: bytes) -> str:
    """The digest to put in ``attestationDigest`` for these issued bytes."""
    return "sha256:" + hashlib.sha256(attestation_bytes).hexdigest()


def encode_attestation(attestation_bytes: bytes) -> str:
    """Base64 the provider's bytes for transport, verbatim."""
    return base64.b64encode(attestation_bytes).decode("ascii")
