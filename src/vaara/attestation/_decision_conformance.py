# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Producer-agnostic conformance check for SEP-2828 decision records.

The sibling of ``_receipt_conformance``. A decision record is the
*before* half of an execution record: the governing server's verdict
(``allow`` / ``block`` / ``escalate``) and the basis for it. The basis can
be carried three ways, strongest to weakest for reading the "why": a
native ``rationale`` (the rule that fired, a human reason, and the
declared intent the call was judged against, all in this record); an
optional content-addressed ``evidenceRef``; and a ``backLink`` digest to
the prior attestation the outcome record answers. This checks that a
candidate is a well-formed decision record, with no signing key and no
matching attestation, so a neutral party can judge a record someone else
produced.

Like the receipt check it covers the wire schema only: required fields,
types, supported ``alg``, valid verdict, ``sha256:<hex>`` digest format,
and that ``issuerAsserted.alg`` matches the envelope ``alg``. The
signature and the back-link to a held attestation are out of scope here;
they need external material and live in the decision verifier.

Pure standard library; importable without the ``attestation`` extra, so
it runs in the base install beside ``verify-records``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from vaara.attestation._receipt_conformance import (
    ADVISORY,
    REQUIRED,
    _DIGEST_RE,
    _HEX_RE,
    _SIG_HEX_LEN,
    VALID_ALGS,
    ConformanceCheck,
    ConformanceReport,
)

VALID_VERDICTS: frozenset[str] = frozenset({"allow", "block", "escalate"})

DECISION_SCHEMA_NAME = "sep2828-decision-record"

# Risk-basis fields are decimal strings on the wire (floats are banned at
# the JCS boundary). When present they SHOULD be strings.
_RISK_FIELDS = ("riskScore", "thresholdAllow", "thresholdBlock")


def check_decision_conformance(doc: Any) -> ConformanceReport:
    """Check a parsed JSON value against the SEP-2828 decision-record schema.

    Returns a report listing every applicable check; never raises on a
    malformed record. ``conforms`` is true iff every applicable
    ``required`` check passed. The report's ``status`` slot carries the
    decision verdict (the salient enum), mirroring the receipt report.
    """
    checks: list[ConformanceCheck] = []

    def add(check_id: str, ok: bool, severity: str, detail: str) -> None:
        checks.append(ConformanceCheck(check_id, ok, severity, detail))

    def finish(alg: Any, verdict: Any) -> ConformanceReport:
        conforms = all(c.ok for c in checks if c.severity == REQUIRED)
        return ConformanceReport(
            conforms=conforms,
            checks=tuple(checks),
            alg=alg if isinstance(alg, str) else None,
            status=verdict if isinstance(verdict, str) else None,
        )

    if not isinstance(doc, dict):
        add("top_level_object", False, REQUIRED, "record MUST be a JSON object")
        return finish(None, None)
    add("top_level_object", True, REQUIRED, "record is a JSON object")

    version = doc.get("version")
    add("version", version == 1, REQUIRED, f"version MUST be 1; got {version!r}")

    alg = doc.get("alg")
    add("alg_supported", alg in VALID_ALGS, REQUIRED,
        f"alg MUST be one of {sorted(VALID_ALGS)}; got {alg!r}")

    sig = doc.get("signature")
    sig_hex = isinstance(sig, str) and bool(_HEX_RE.match(sig)) and len(sig) % 2 == 0
    add("signature_hex", sig_hex, REQUIRED,
        "signature MUST be an even-length lowercase hex string")
    if sig_hex and isinstance(sig, str) and isinstance(alg, str) and alg in _SIG_HEX_LEN:
        want = _SIG_HEX_LEN[alg]
        add("signature_length", len(sig) == want, ADVISORY,
            f"{alg} signature SHOULD be {want} hex chars; got {len(sig)}")

    _check_back_link(doc.get("backLink"), add)

    ia = doc.get("issuerAsserted")
    if not isinstance(ia, dict):
        add("issuer_asserted_present", False, REQUIRED,
            "issuerAsserted MUST be an object")
    else:
        ia_fields = ("alg", "iat", "iss", "nonce", "secretVersion", "sub")
        missing = [f for f in ia_fields if f not in ia]
        add("issuer_asserted_present", not missing, REQUIRED,
            f"issuerAsserted MUST carry {list(ia_fields)}; missing {missing}")
        add("issuer_asserted_alg_matches", ia.get("alg") == alg, REQUIRED,
            "issuerAsserted.alg MUST equal the top-level alg")

    verdict = _check_decision_derived(doc.get("decisionDerived"), add)
    return finish(alg, verdict)


def _check_back_link(bl: Any, add: Any) -> None:
    if not isinstance(bl, dict):
        add("back_link_present", False, REQUIRED, "backLink MUST be an object")
        return
    add("back_link_present",
        "attestationDigest" in bl and "attestationNonce" in bl, REQUIRED,
        "backLink MUST carry attestationDigest and attestationNonce")
    ad = bl.get("attestationDigest")
    add("back_link_digest_format",
        isinstance(ad, str) and bool(_DIGEST_RE.match(ad)), REQUIRED,
        "backLink.attestationDigest MUST be 'sha256:<64 lowercase hex>'")
    add("back_link_nonce_type", isinstance(bl.get("attestationNonce"), str),
        REQUIRED, "backLink.attestationNonce MUST be a string")


def _check_decision_derived(dd: Any, add: Any) -> Optional[str]:
    if not isinstance(dd, dict):
        add("decision_present", False, REQUIRED, "decisionDerived MUST be an object")
        return None
    verdict = dd.get("decision")
    add("decision_present", "decision" in dd and "decidedAt" in dd, REQUIRED,
        "decisionDerived MUST carry decision and decidedAt")
    add("decision_valid", verdict in VALID_VERDICTS, REQUIRED,
        f"decision MUST be one of {sorted(VALID_VERDICTS)}; got {verdict!r}")
    add("decided_at_type", isinstance(dd.get("decidedAt"), str), REQUIRED,
        "decisionDerived.decidedAt MUST be a string")
    for field in _RISK_FIELDS:
        if field in dd:
            add(f"{field}_is_string", isinstance(dd[field], str), ADVISORY,
                f"decisionDerived.{field} SHOULD be a decimal string, not a number")
    if "evidenceRef" in dd:
        _check_evidence_ref(dd["evidenceRef"], add)
    if "rationale" in dd:
        _check_rationale(dd["rationale"], add)
    if "binding" in dd:
        _check_binding(dd["binding"], dd.get("rationale"), add)
    if "decisionProof" in dd:
        _check_decision_proof(dd["decisionProof"], dd.get("binding"), add)
    return verdict if isinstance(verdict, str) else None


def _check_decision_proof(p: Any, binding: Any, add: Any) -> None:
    """Check the decisionProof envelope shape (keyless).

    ``decisionProof`` is the wire format for a succinct proof that the
    verdict is the correct output of the committed policy on the committed
    intent and inputs, without revealing them. This checks the envelope
    only: the proof system name, that ``publicInputs`` carries a
    well-formed ``bindingDigest``, that the proof is hex, and that the
    verifier parameters are pinned by digest so the proof names an exact
    circuit. Verifying the proof itself needs the proving system and lives
    behind the attestation extra, the same split signatures and anchors
    use. Where a ``binding`` is present, the proof's public bindingDigest
    MUST equal it, so the record self-attests the proof is about this exact
    commitment with no proving system in the loop.
    """
    if not isinstance(p, dict):
        add("decision_proof_object", False, REQUIRED,
            "decisionDerived.decisionProof MUST be an object")
        return
    add("decision_proof_object", True, REQUIRED,
        "decisionDerived.decisionProof is an object")
    add("decision_proof_system",
        isinstance(p.get("proofSystem"), str) and bool(p.get("proofSystem")), REQUIRED,
        "decisionProof.proofSystem MUST be a non-empty string")
    pi = p.get("publicInputs")
    add("decision_proof_public_inputs_object", isinstance(pi, dict), REQUIRED,
        "decisionProof.publicInputs MUST be an object")
    bd = pi.get("bindingDigest") if isinstance(pi, dict) else None
    add("decision_proof_binding_digest_format",
        isinstance(bd, str) and bool(_DIGEST_RE.match(bd)), REQUIRED,
        "decisionProof.publicInputs.bindingDigest MUST be 'sha256:<64 lowercase hex>'")
    pf = p.get("proof")
    add("decision_proof_bytes",
        isinstance(pf, str) and bool(_HEX_RE.match(pf)) and len(pf) % 2 == 0, REQUIRED,
        "decisionProof.proof MUST be an even-length lowercase hex string")
    vpd = p.get("verifierParamsDigest")
    add("decision_proof_params_digest",
        isinstance(vpd, str) and bool(_DIGEST_RE.match(vpd)), REQUIRED,
        "decisionProof.verifierParamsDigest MUST be 'sha256:<64 lowercase hex>'")
    if isinstance(binding, dict) and isinstance(binding.get("bindingDigest"), str) \
            and isinstance(bd, str):
        add("decision_proof_binds_commitment", bd == binding["bindingDigest"], REQUIRED,
            "decisionProof.publicInputs.bindingDigest MUST equal "
            "decisionDerived.binding.bindingDigest")


def _check_rationale(r: Any, add: Any) -> None:
    """Check the optional native decision rationale.

    The legible, in-record answer to "why this verdict": the rule that
    fired, a human reason, and the declared intent the call was judged
    against. Unlike ``evidenceRef`` (a content-addressed pointer) and the
    ``backLink`` (a digest of a separate attestation), this carries the
    intent and reason in the 2828 record itself, so the differential
    between one call and another reads without dereferencing anything.
    Fires only when present; every field is required once it does.
    """
    if not isinstance(r, dict):
        add("rationale_object", False, REQUIRED,
            "decisionDerived.rationale MUST be an object")
        return
    add("rationale_object", True, REQUIRED, "decisionDerived.rationale is an object")
    add("rationale_rule",
        isinstance(r.get("rule"), str) and bool(r.get("rule")), REQUIRED,
        "rationale.rule MUST be a non-empty string (the policy rule that fired)")
    add("rationale_reason",
        isinstance(r.get("reason"), str) and bool(r.get("reason")), REQUIRED,
        "rationale.reason MUST be a non-empty string (a human-legible reason)")
    add("rationale_declared_intent",
        isinstance(r.get("declaredIntent"), str) and bool(r.get("declaredIntent")),
        REQUIRED,
        "rationale.declaredIntent MUST be a non-empty string (the intent judged)")
    if "intentSatisfied" in r:
        add("rationale_intent_satisfied_type",
            isinstance(r["intentSatisfied"], bool), ADVISORY,
            "rationale.intentSatisfied SHOULD be a boolean when present")


def _check_binding(b: Any, rationale: Any, add: Any) -> None:
    """Check the optional decision binding: commitments that pin the verdict.

    ``binding`` carries content-addressed commitments to the exact policy,
    intent, and inputs a verdict was computed over, plus a single
    ``bindingDigest`` a succinct proof of decision correctness (a later
    layer) opens. Digest-only, so the sensitive policy or intent need not
    ship. Where the declared intent is also in the record, its digest is
    recomputed from the bytes here, keyless, the same self-proving pattern
    the receipt uses for its projection digest. Fires only when present.
    """
    if not isinstance(b, dict):
        add("binding_object", False, REQUIRED,
            "decisionDerived.binding MUST be an object")
        return
    add("binding_object", True, REQUIRED, "decisionDerived.binding is an object")
    for key, check_id in (
        ("policyDigest", "binding_policy_digest"),
        ("intentDigest", "binding_intent_digest"),
        ("inputsDigest", "binding_inputs_digest"),
        ("bindingDigest", "binding_commitment_digest"),
    ):
        val = b.get(key)
        add(check_id, isinstance(val, str) and bool(_DIGEST_RE.match(val)), REQUIRED,
            f"binding.{key} MUST be 'sha256:<64 lowercase hex>'")
    declared = rationale.get("declaredIntent") if isinstance(rationale, dict) else None
    if isinstance(declared, str) and isinstance(b.get("intentDigest"), str):
        want = "sha256:" + hashlib.sha256(declared.encode("utf-8")).hexdigest()
        add("binding_intent_self_consistent", b["intentDigest"] == want, REQUIRED,
            "binding.intentDigest MUST equal sha256 over rationale.declaredIntent bytes")


def _check_evidence_ref(er: Any, add: Any) -> None:
    """Check the optional content-addressed evidence reference on the basis.

    Fires only when ``evidenceRef`` is present; the field itself is optional.
    The digest is the binding (``sha256:<hex>``), and ``canonicalization``
    plus ``schema`` are what let an independent implementation recompute the
    address and interpret the referenced object, so all three are required
    once the reference exists. ``ref`` is an optional non-authoritative
    locator.
    """
    if not isinstance(er, dict):
        add("evidence_ref_object", False, REQUIRED,
            "decisionDerived.evidenceRef MUST be an object")
        return
    add("evidence_ref_object", True, REQUIRED,
        "decisionDerived.evidenceRef is an object")
    dg = er.get("digest")
    add("evidence_ref_digest_format",
        isinstance(dg, str) and bool(_DIGEST_RE.match(dg)), REQUIRED,
        "evidenceRef.digest MUST be 'sha256:<64 lowercase hex>'")
    add("evidence_ref_canonicalization",
        isinstance(er.get("canonicalization"), str) and bool(er.get("canonicalization")),
        REQUIRED, "evidenceRef.canonicalization MUST be a non-empty string")
    add("evidence_ref_schema",
        isinstance(er.get("schema"), str) and bool(er.get("schema")),
        REQUIRED, "evidenceRef.schema MUST be a non-empty string")
    if "ref" in er:
        add("evidence_ref_ref_type", isinstance(er["ref"], str) and bool(er["ref"]),
            ADVISORY, "evidenceRef.ref SHOULD be a non-empty string when present")
