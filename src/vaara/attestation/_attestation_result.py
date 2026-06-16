"""Express a Vaara attestation verdict as an IETF RATS EAR (Phase 2: neutral verify).

Phases 0 and 1 of the hardware-governance layer bind a SEP-2828 record to a TPM
2.0 quote, to the kernel's IMA log, and to a continuous chain of quotes. The
verdict those verifiers return is Vaara-shaped (``tier``, the ``*_basis`` honesty
fields). This module re-expresses one such verdict as an **EAR** -- the EAT-based
Attestation Result of draft-ietf-rats-ear -- carrying an **AR4SI** trustworthiness
vector (draft-ietf-rats-ar4si). That makes the verifier's output standards-aligned
and root-agnostic: a TPM binding, a TPM chain, and a SEV-SNP enforcement report all
collapse to the same EAR shape a RATS Relying Party already knows how to read.

The mapping is conservative and honest. It asserts only the three AR4SI tier anchors
whose meaning is fixed by the spec -- ``2`` affirming, ``32`` warning, ``96``
contraindicated -- plus ``0``/omission for "no claim made". In v0 the attesting key
and the hardware root are trusted *as supplied* (the TPM EK chain and the AMD KDS
VCEK chain are not validated), so the ``hardware`` and ``instance-identity`` claims
top out at ``warning`` and the overall ``ear_status`` can never read ``affirming``
on the shipped capture path. ``affirming`` is reachable only when a basis reports a
validated root (``ek_chain_verified`` / ``kds_verified``) -- the same un-forgeable-
root capability the reserved ``attested`` tier waits on. The EAR never claims more
than the verdict it was built from.

The decision-semantics limit carries through unchanged: IMA and the launch
measurement attest the *platform*, not that the agent decided X for reason Y. There
is no AR4SI claim for decision semantics, so it is not fabricated as one; it is
recorded as a Vaara verifier-claim and the decision content stays in the signed
SEP-2828 record the EAR references by digest.

The EAR produced here is the unprotected JSON serialization (a plain map, keyless).
The underlying evidence carries its own signatures; this document is the verifier's
appraisal *result*, not a fresh attestation, and says so in its verifier-claims.

Pure standard library. Consumes the ``to_dict()`` of a verdict, so it needs neither
the attestation extra nor any hardware present.
"""

from __future__ import annotations

from typing import Any, Optional

SCHEMA = "vaara.attestation-result/v0"

# The EAR profile this document conforms to (draft-ietf-rats-ear-04), and the
# Vaara profile that signals the verifier-claims extension on the submodule.
EAR_PROFILE = "tag:ietf.org,2026:rats/ear#04"
VAARA_PROFILE = "tag:vaara.io,2026:attestation-result#v0"
VERIFIER_CLAIMS_KEY = "vaara.io/verifier-claims"
VERIFIER_DEVELOPER = "https://vaara.io"

# AR4SI trustworthiness tier anchors (draft-ietf-rats-ar4si section 2.3.2). Only the
# three canonical tier boundaries are asserted; finer per-claim reason codes are not,
# to avoid overstating fidelity. 0 means no claim is being made about that aspect.
NONE = 0
AFFIRMING = 2
WARNING = 32
CONTRAINDICATED = 96

# AR4SI trustworthiness-vector claim keys (JSON names). Claims this version never
# appraises (file-system, runtime-opaque, storage-opaque, sourced-data) are omitted
# from the vector rather than emitted as a hollow ``none``.
CLAIM_INSTANCE_IDENTITY = "instance-identity"
CLAIM_CONFIGURATION = "configuration"
CLAIM_EXECUTABLES = "executables"
CLAIM_HARDWARE = "hardware"

_VALIDATED_TPM_ROOT = "ek_chain_verified"
_VALIDATED_SEV_ROOT = "kds_verified"


def _status_from_vector(vector: "dict[str, int]") -> str:
    """The overall EAR status: no higher trust than the worst claim in the vector."""
    values = list(vector.values())
    if not values:
        return "none"
    worst = max(values)  # higher integer == lower trust across the tier bands
    if worst >= CONTRAINDICATED:
        return "contraindicated"
    if worst >= WARNING:
        return "warning"
    if worst >= AFFIRMING:
        return "affirming"
    return "none"


def _root_tier(*, evidence_ok: bool, root_validated: bool) -> int:
    """Trust in the attesting key / hardware genuineness.

    Contraindicated when the evidence itself does not hold (unparseable or a bad
    signature); affirming only when a basis reports a validated root; otherwise a
    warning -- the key verified but its provenance was taken on trust.
    """
    if not evidence_ok:
        return CONTRAINDICATED
    return AFFIRMING if root_validated else WARNING


def _measured_tier(*, reconciled: bool, pin_basis: str) -> int:
    """Trust in measured state (executables / configuration) from reconcile + pin.

    A pin mismatch is contraindicated. Otherwise affirming only when a reference was
    supplied and matched (``pinned``); a reconciled-but-unpinned measurement is a
    warning (internally consistent, not approved against a reference); a measurement
    that did not reconcile is contraindicated.
    """
    if pin_basis == "pin_mismatch":
        return CONTRAINDICATED
    if not reconciled:
        return CONTRAINDICATED
    return AFFIRMING if pin_basis == "pinned" else WARNING


def _tpm_binding_vector(v: "dict[str, Any]") -> "tuple[dict[str, int], dict[str, Any]]":
    parsed = bool(v.get("parsed"))
    sig_valid = bool(v.get("signature_valid"))
    bound = bool(v.get("bound"))
    evidence_ok = parsed and sig_valid
    root_validated = v.get("ak_chain_basis") == _VALIDATED_TPM_ROOT
    pin_basis = str(v.get("pcr_pin_basis", "unpinned"))

    vector: "dict[str, int]" = {}
    # instance-identity additionally requires the quote to bind THIS record; an
    # otherwise-valid quote that does not carry the record's digest does not speak
    # for it.
    vector[CLAIM_INSTANCE_IDENTITY] = (
        _root_tier(evidence_ok=evidence_ok, root_validated=root_validated)
        if (evidence_ok and bound)
        else CONTRAINDICATED
    )
    vector[CLAIM_HARDWARE] = _root_tier(
        evidence_ok=evidence_ok, root_validated=root_validated
    )
    if v.get("ima_log_entries"):
        vector[CLAIM_EXECUTABLES] = _measured_tier(
            reconciled=bool(v.get("ima_replayed")), pin_basis=pin_basis
        )
    vector[CLAIM_CONFIGURATION] = _measured_tier(
        reconciled=bool(v.get("pcr_digest_recomputed")), pin_basis=pin_basis
    )

    native = {
        "native_tier": v.get("tier"),
        "root_trust_basis": v.get("ak_chain_basis"),
        "pcr_pin_basis": pin_basis,
        "ima_policy_basis": v.get("ima_policy_basis"),
        "freshness_basis": v.get("freshness_basis"),
        "decision_semantics_basis": v.get("decision_logic_basis"),
        "bound_record_digest": v.get("extra_data_expected"),
    }
    return vector, native


def _tpm_chain_vector(v: "dict[str, Any]") -> "tuple[dict[str, int], dict[str, Any]]":
    links_bound = bool(v.get("links_bound"))
    unverified = v.get("tier") == "unverified"
    evidence_ok = links_bound and not unverified
    root_validated = v.get("ak_chain_basis") == _VALIDATED_TPM_ROOT
    pin_basis = str(v.get("pcr_pin_basis", "unpinned"))

    vector: "dict[str, int]" = {}
    vector[CLAIM_INSTANCE_IDENTITY] = _root_tier(
        evidence_ok=evidence_ok, root_validated=root_validated
    )
    vector[CLAIM_HARDWARE] = _root_tier(
        evidence_ok=evidence_ok, root_validated=root_validated
    )
    vector[CLAIM_EXECUTABLES] = _measured_tier(
        reconciled=bool(v.get("ima_append_only")) and evidence_ok, pin_basis=pin_basis
    )
    vector[CLAIM_CONFIGURATION] = _measured_tier(
        reconciled=evidence_ok, pin_basis=pin_basis
    )

    native = {
        "native_tier": v.get("tier"),
        "root_trust_basis": v.get("ak_chain_basis"),
        "pcr_pin_basis": pin_basis,
        "ima_policy_basis": v.get("ima_policy_basis"),
        "freshness_basis": v.get("freshness_basis"),
        "decision_semantics_basis": v.get("decision_logic_basis"),
        "chain_continuous": v.get("tier") == "continuous",
        "n_links": v.get("n_links"),
    }
    return vector, native


def _enforcement_vector(v: "dict[str, Any]") -> "tuple[dict[str, int], dict[str, Any]]":
    parsed = bool(v.get("parsed"))
    sig_valid = bool(v.get("signature_valid"))
    bound = bool(v.get("bound"))
    evidence_ok = parsed and sig_valid
    root_validated = v.get("vcek_chain_basis") == _VALIDATED_SEV_ROOT
    measurement_basis = str(v.get("measurement_basis", "unpinned"))

    vector: "dict[str, int]" = {}
    vector[CLAIM_INSTANCE_IDENTITY] = (
        _root_tier(evidence_ok=evidence_ok, root_validated=root_validated)
        if (evidence_ok and bound)
        else CONTRAINDICATED
    )
    vector[CLAIM_HARDWARE] = _root_tier(
        evidence_ok=evidence_ok, root_validated=root_validated
    )
    if v.get("measurement"):
        # The SEV-SNP launch measurement covers the guest's initial memory: the code
        # image AND its initial configuration. Both claims follow the same pin.
        measured = _measured_tier(reconciled=True, pin_basis=measurement_basis)
        vector[CLAIM_EXECUTABLES] = measured
        vector[CLAIM_CONFIGURATION] = measured

    native = {
        "native_tier": v.get("tier"),
        "root_trust_basis": v.get("vcek_chain_basis"),
        "measurement_basis": measurement_basis,
        # The reported TCB / policy is carried in the report but is not appraised
        # against a reference in v0, so no claim is made about it (it is not warned
        # on -- absence of appraisal, not a found concern).
        "tcb_appraisal": "not_established",
        "enforcement_logic_basis": v.get("enforcement_logic_basis"),
        "bound_record_digest": v.get("report_data_expected"),
    }
    return vector, native


# Verdict schema -> (submodule label, vector builder).
_FAMILIES = {
    "vaara.tpm-binding-attestation/v0": ("tpm", _tpm_binding_vector),
    "vaara.tpm-evidence-chain/v0": ("tpm", _tpm_chain_vector),
    "vaara.enforcement-attestation/v0": ("sev-snp", _enforcement_vector),
}


def build_attestation_result(
    verdict: "dict[str, Any]",
    *,
    issued_at: int,
    verifier_build: str,
    submod_label: Optional[str] = None,
) -> "dict[str, Any]":
    """Build a ``vaara.attestation-result/v0`` EAR document from a Vaara verdict.

    ``verdict`` is the ``to_dict()`` of a TPM-binding, TPM-chain, or SEV-SNP
    enforcement verdict (it is recognised by its ``schema`` field). ``issued_at`` is
    the verifier's appraisal time as integer epoch seconds (the EAR ``iat``; the EAR
    spec forbids floats). ``verifier_build`` identifies the verifier build (e.g.
    ``"vaara 0.71.0"``). The result is a plain dict ready to serialise as JSON.

    Raises ``ValueError`` if ``verdict`` is not a dict or its ``schema`` is not a
    recognised attestation-verdict family.
    """
    if not isinstance(verdict, dict):
        raise ValueError("verdict must be a dict (the to_dict() of a Vaara verdict)")
    schema = verdict.get("schema")
    family = _FAMILIES.get(schema) if isinstance(schema, str) else None
    if family is None:
        raise ValueError(
            f"unrecognised verdict schema {schema!r}; expected one of "
            + ", ".join(sorted(_FAMILIES))
        )
    if not isinstance(issued_at, int) or isinstance(issued_at, bool):
        raise ValueError("issued_at must be an integer (epoch seconds, no floats)")
    if not isinstance(verifier_build, str) or not verifier_build:
        raise ValueError("verifier_build must be a non-empty string")

    default_label, builder = family
    label = submod_label or default_label
    vector, native = builder(verdict)
    status = _status_from_vector(vector)

    verifier_claims = {
        "schema": SCHEMA,
        "source_schema": schema,
        "result_is_unsigned": True,
        "honest_limit": (
            "The platform is attested, not the decision semantics; the decision "
            "content is carried by the signed SEP-2828 record this result references "
            "by digest. affirming requires a validated hardware root and is "
            "unreachable while the root is trusted as supplied."
        ),
    }
    # Drop native fields the family did not populate, keeping the claim block tight.
    verifier_claims.update({k: val for k, val in native.items() if val is not None})

    submod = {
        "eat_profile": VAARA_PROFILE,
        "ear_status": status,
        "ear_trustworthiness_vector": vector,
        VERIFIER_CLAIMS_KEY: verifier_claims,
    }

    return {
        "eat_profile": EAR_PROFILE,
        "iat": issued_at,
        "ear_status": status,
        "ear_verifier_id": {
            "developer": VERIFIER_DEVELOPER,
            "build": verifier_build,
        },
        "submods": {label: submod},
    }
