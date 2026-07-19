# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Producer-agnostic conformance check for SEP-2828 execution records.

Given any JSON that claims to be a SEP-2828 execution receipt, this
answers a question that needs no signing key and no matching
attestation: is it a well-formed record, and is it internally
consistent? Every check is reported, pass or fail, so the output is a
conformance *report* rather than a first-error abort. This is the check
a neutral party runs on a record someone else produced, before anyone
reaches for a key.

The split is deliberate. The cryptographic checks live in the receipt
verifier and need external material the producer does not ship inside
the record: the signature needs the verifying key, the back-link needs
the attestation it answers, the result commitment needs the runtime
result object. Conformance needs none of those. It covers the wire
schema and the one binding a record proves about *itself*: that
``resultCommitment.projectionDigest`` is the SHA-256 of the projection
bytes beside it, recomputable from the record alone with nothing but a
hash function.

Pure standard library (``hashlib`` + ``re``); importable without the
``attestation`` extra, so conformance runs in the base install.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

# In lockstep with _sep2787_types.VALID_ALGS / _receipt_types.VALID_STATUSES.
# Inlined rather than imported so conformance carries no parser dependency.
VALID_ALGS: frozenset[str] = frozenset({"HS256", "ES256", "RS256"})
VALID_STATUSES: frozenset[str] = frozenset({"executed", "refused", "errored"})

SCHEMA_NAME = "sep2828-execution-record"
SCHEMA_VERSION = 1
REQUIRED = "required"
ADVISORY = "advisory"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_RE = re.compile(r"^[0-9a-f]+$")
# Fixed-width hex signature lengths. RS256 is modulus-width (variable).
_SIG_HEX_LEN: dict[str, int] = {"HS256": 64, "ES256": 128}
# Allowlisted post-quantum hybrid suites, in lockstep with _receipt_pq.
_HYBRID_SUITES: frozenset[str] = frozenset(
    {"ES256+ML-DSA-65", "RS256+ML-DSA-65"}
)


@dataclass(frozen=True)
class ConformanceCheck:
    """One conformance assertion. ``required`` gates conformance; ``advisory`` does not."""

    id: str
    ok: bool
    severity: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "ok": self.ok, "severity": self.severity, "detail": self.detail}


@dataclass(frozen=True)
class ConformanceReport:
    """Outcome of checking a candidate record against the SEP-2828 schema."""

    conforms: bool
    checks: tuple[ConformanceCheck, ...]
    alg: Optional[str]
    status: Optional[str]

    @property
    def required_failed(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.checks if c.severity == REQUIRED and not c.ok)

    @property
    def advisories(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.checks if c.severity == ADVISORY and not c.ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_NAME,
            "schemaVersion": SCHEMA_VERSION,
            "conforms": self.conforms,
            "alg": self.alg,
            "status": self.status,
            "requiredFailed": list(self.required_failed),
            "advisories": list(self.advisories),
            "checks": [c.to_dict() for c in self.checks],
        }


def _sha256_digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def check_record_conformance(doc: Any) -> ConformanceReport:
    """Check a parsed JSON value against the SEP-2828 record schema.

    Returns a report listing every applicable check; never raises on a
    malformed record. ``conforms`` is true iff every applicable
    ``required`` check passed (advisory failures are reported, not gating).
    """
    checks: list[ConformanceCheck] = []

    def add(check_id: str, ok: bool, severity: str, detail: str) -> None:
        checks.append(ConformanceCheck(check_id, ok, severity, detail))

    def finish(alg: Any, status: Any) -> ConformanceReport:
        conforms = all(c.ok for c in checks if c.severity == REQUIRED)
        return ConformanceReport(
            conforms=conforms,
            checks=tuple(checks),
            alg=alg if isinstance(alg, str) else None,
            status=status if isinstance(status, str) else None,
        )

    if not isinstance(doc, dict):
        add("top_level_object", False, REQUIRED, "record MUST be a JSON object")
        return finish(None, None)
    add("top_level_object", True, REQUIRED, "record is a JSON object")

    version = doc.get("version")
    add("version", version == SCHEMA_VERSION, REQUIRED,
        f"version MUST be {SCHEMA_VERSION}; got {version!r}")

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

    bl = doc.get("backLink")
    if not isinstance(bl, dict):
        add("back_link_present", False, REQUIRED, "backLink MUST be an object")
    else:
        add("back_link_present",
            "attestationDigest" in bl and "attestationNonce" in bl, REQUIRED,
            "backLink MUST carry attestationDigest and attestationNonce")
        ad = bl.get("attestationDigest")
        add("back_link_digest_format",
            isinstance(ad, str) and bool(_DIGEST_RE.match(ad)), REQUIRED,
            "backLink.attestationDigest MUST be 'sha256:<64 lowercase hex>'")
        add("back_link_nonce_type", isinstance(bl.get("attestationNonce"), str),
            REQUIRED, "backLink.attestationNonce MUST be a string")

    ra = doc.get("receiptAsserted")
    if not isinstance(ra, dict):
        add("receipt_asserted_present", False, REQUIRED,
            "receiptAsserted MUST be an object")
    else:
        ra_fields = ("alg", "iat", "iss", "nonce", "secretVersion", "sub")
        missing = [f for f in ra_fields if f not in ra]
        add("receipt_asserted_present", not missing, REQUIRED,
            f"receiptAsserted MUST carry {list(ra_fields)}; missing {missing}")
        add("receipt_asserted_alg_matches", ra.get("alg") == alg, REQUIRED,
            "receiptAsserted.alg MUST equal the top-level alg")

    od = doc.get("outcomeDerived")
    status: Any = None
    if not isinstance(od, dict):
        add("outcome_present", False, REQUIRED, "outcomeDerived MUST be an object")
    else:
        status = od.get("status")
        add("outcome_present", "status" in od and "completedAt" in od, REQUIRED,
            "outcomeDerived MUST carry status and completedAt")
        add("status_valid", status in VALID_STATUSES, REQUIRED,
            f"status MUST be one of {sorted(VALID_STATUSES)}; got {status!r}")
        add("completed_at_type", isinstance(od.get("completedAt"), str), REQUIRED,
            "outcomeDerived.completedAt MUST be a string")
        if od.get("resultCommitment") is not None:
            _check_result_commitment(od["resultCommitment"], status, add)
        dd = od.get("decisionDigest")
        if dd is not None:
            add("decision_digest_format",
                isinstance(dd, str) and bool(_DIGEST_RE.match(dd)), REQUIRED,
                "outcomeDerived.decisionDigest MUST be 'sha256:<64 lowercase hex>'")

    _check_pq_fields(doc, ra if isinstance(ra, dict) else {}, add)
    return finish(alg, status)


def _check_pq_fields(
    doc: dict[str, Any], ra: dict[str, Any],
    add: Callable[[str, bool, str, str], None],
) -> None:
    """Schema and self-evident-downgrade checks for the optional PQ fields.

    Advisory only: a record without these fields is a perfectly conforming
    classical record, and the quantum-resistance *tier* (``pq_verdict``) is a
    verification judgment, not a schema gate. What is schema-visible here is
    that a record committing to a hybrid ``sigSuite`` should carry the
    ``pqSignature`` it promised; absence is a downgrade the record shows about
    itself, reported but not gating.
    """
    suite = ra.get("sigSuite")
    if suite is not None:
        add("sig_suite_type", isinstance(suite, str), ADVISORY,
            "receiptAsserted.sigSuite SHOULD be a string when present")
    pq = doc.get("pqSignature")
    if pq is not None:
        shape_ok = isinstance(pq, dict) and all(
            isinstance(pq.get(k), str) and pq.get(k) for k in ("alg", "keyid", "sig")
        )
        add("pq_signature_shape", shape_ok, ADVISORY,
            "pqSignature SHOULD be an object with non-empty string alg, keyid, sig")
        sig = pq.get("sig") if isinstance(pq, dict) else None
        if isinstance(sig, str):
            add("pq_signature_hex",
                bool(_HEX_RE.match(sig)) and len(sig) % 2 == 0, ADVISORY,
                "pqSignature.sig SHOULD be an even-length lowercase hex string")
    if isinstance(suite, str) and suite in _HYBRID_SUITES:
        add("committed_suite_has_pq_signature",
            doc.get("pqSignature") is not None, ADVISORY,
            "receiptAsserted.sigSuite commits to a hybrid suite but no "
            "pqSignature is present (a schema-visible downgrade)")


def _check_result_commitment(
    rc: Any, status: Any, add: Callable[[str, bool, str, str], None]
) -> None:
    """Schema + self-consistency for an optional result commitment.

    The projection variant carries the one check a record proves about
    itself with no external input: the digest equals SHA-256 over the
    projection bytes. The ref variant points outward, so only its digest
    *format* is checkable here.
    """
    if not isinstance(rc, dict):
        add("result_commitment_shape", False, REQUIRED, "resultCommitment MUST be an object")
        return
    if "projection" in rc:
        proj, pdg = rc.get("projection"), rc.get("projectionDigest")
        shape_ok = isinstance(proj, str) and isinstance(pdg, str)
        add("result_commitment_shape", shape_ok, REQUIRED,
            "projection commitment MUST carry string projection and projectionDigest")
        if isinstance(proj, str) and isinstance(pdg, str):
            add("result_commitment_digest_format", bool(_DIGEST_RE.match(pdg)), REQUIRED,
                "resultCommitment.projectionDigest MUST be 'sha256:<64 lowercase hex>'")
            add("result_commitment_self_consistent",
                _sha256_digest(proj.encode("utf-8")) == pdg, REQUIRED,
                "projectionDigest MUST equal sha256 over the projection bytes")
    elif "ref" in rc:
        dg = rc.get("digest")
        add("result_commitment_shape",
            isinstance(rc.get("ref"), str) and isinstance(dg, str), REQUIRED,
            "ref commitment MUST carry string ref and digest")
        add("result_commitment_digest_format",
            isinstance(dg, str) and bool(_DIGEST_RE.match(dg)), REQUIRED,
            "resultCommitment.digest MUST be 'sha256:<64 lowercase hex>'")
    else:
        add("result_commitment_shape", False, REQUIRED,
            "resultCommitment MUST carry either 'projection' or 'ref'")
    if status == "refused":
        add("refused_has_no_result", False, ADVISORY,
            "a refused call SHOULD NOT carry a resultCommitment")
