# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Henri Sirkkavaara
"""Same-box model-diversity cross-check: a second model corroborates the first.

Tier 1's second half. The weld (``_inference_chain_verify``) proves
*platform continuity* for a model turn, the model ran on a hardware root the
operator cannot forge. It deliberately makes no claim about whether the output is
*right*: byte-replay is provably not deliverable on this stack (see
``_inference_determinism`` and ``research/determinism_findings_20260615.md``), so a
receipt's honest tier is ``integrity``, never a determinism guarantee.

This module adds the orthogonal corroboration the determinism gate cannot: a
*second local model of different identity* re-derives a judgment over the same
prompt and the subject's output, and that judgment is signed and bound to the exact
subject receipt. Because the subject receipt is itself bound to the TPM/fTPM root
through the session manifest, a ``vaara.inference-crosscheck/v0`` record ties an
independent equivalence opinion to the same hardware root.

The honesty model is the whole point. This is corroboration, not proof:

- The check is *semantic equivalence*, an opinion from a fallible model, never a
  byte-replay or a correctness proof. The schema names ``method:
  "semantic-equivalence"`` in the signed preimage so the claim cannot be read as
  more than it is.
- It is only meaningful when the verifier model differs from the subject model. A
  model judging its own output is not diversity; ``diverse`` records that fact and
  the composite verdict refuses to call a non-diverse check corroboration.
- The cross-check binds to the subject *receipt digest*, and refuses to be built
  over a response whose output commitment does not match the receipt. You cannot
  cross-check a substituted output and have it count for the signed, hardware-bound
  receipt.

The core (``build_crosscheck`` / ``parse_crosscheck`` / ``verify_crosscheck``) is a
pure function over an injected :class:`Judge`, so the whole wire path is unit
testable with a :class:`StubJudge` and no inference server, exactly as the TPM weld
tests run on ``MockTPMQuoter`` with no TPM. :class:`OllamaJudge` is the live judge;
it only runs when a server is present.

Schema ``vaara.inference-crosscheck/v0``.
"""

from __future__ import annotations

import argparse
import hmac
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from vaara.attestation._inference_emit import (
    inference_receipt_digest,
    make_output_commitment,
    verify_inference_back_link,
)
from vaara.attestation._inference_types import (
    InferenceAttestation,
    InferenceReceipt,
    ModelDerived,
    model_derived_from_dict,
    model_derived_to_dict,
)
from vaara.attestation._receipt_types import (
    ReceiptAsserted,
    receipt_asserted_from_dict,
    receipt_asserted_to_dict,
)
from vaara.attestation._attest_canonical import canonical_json, new_nonce, now_iso8601
from vaara.attestation._attest_signing import (
    sign_es256,
    sign_hs256,
    sign_rs256,
    verify_es256,
    verify_hs256,
    verify_rs256,
)
from vaara.attestation._attest_types import (
    VALID_ALGS,
    Algorithm,
    ArgsCommitment,
    ArgsProjection,
    AttestationError,
    args_from_dict,
    args_to_dict,
)

CROSSCHECK_SCHEMA = "vaara.inference-crosscheck/v0"

# The judge's verdict vocabulary. "uncertain" is the honest fail-safe: an
# unparseable or hedged judge reply maps here, never silently to "equivalent".
CROSSCHECK_AGREEMENTS: frozenset[str] = frozenset(
    {"equivalent", "divergent", "uncertain"}
)
# The only method this schema version expresses. Named in the signed preimage so
# the claim can never be mistaken for a determinism or correctness proof.
CROSSCHECK_METHOD = "semantic-equivalence"

_RECORD_KEYS = frozenset(
    {"schema", "version", "alg", "subject", "verifier", "crosscheck",
     "receiptAsserted", "signature"}
)
_SUBJECT_KEYS = frozenset({"receiptDigest", "model"})
_VERIFIER_KEYS = frozenset({"model", "method", "judgmentCommitment"})
_CROSSCHECK_KEYS = frozenset({"agreement", "diverse", "checkedAt"})


# --- signing dispatch (mirrors _inference_emit, kept local so the module is
# self-contained over the shared SEP-2787 primitives) -----------------------


def _sign(payload: bytes, *, alg: Algorithm, signing_material: Any) -> str:
    if alg == "HS256":
        if not isinstance(signing_material, (bytes, bytearray)):
            raise AttestationError("HS256 requires bytes shared_secret")
        return sign_hs256(payload, shared_secret=bytes(signing_material))
    if alg == "ES256":
        return sign_es256(payload, private_key=signing_material)
    if alg == "RS256":
        return sign_rs256(payload, private_key=signing_material)
    raise AttestationError(f"unsupported alg: {alg!r}")


def _verify(
    payload: bytes, *, alg: Algorithm, signature_hex: str, verifying_material: Any
) -> bool:
    if alg == "HS256":
        if not isinstance(verifying_material, (bytes, bytearray)):
            return False
        return verify_hs256(
            payload, signature_hex=signature_hex, shared_secret=bytes(verifying_material)
        )
    if alg == "ES256":
        return verify_es256(
            payload, signature_hex=signature_hex, public_key=verifying_material
        )
    if alg == "RS256":
        return verify_rs256(
            payload, signature_hex=signature_hex, public_key=verifying_material
        )
    return False


# --- the judge abstraction --------------------------------------------------


@dataclass(frozen=True)
class JudgeOutcome:
    """What the verifier model returned about the subject's output.

    ``agreement`` is one of :data:`CROSSCHECK_AGREEMENTS`. ``raw_judgment`` is the
    judge model's reply text, committed (not stored) so the judgment is auditable
    as an opinion without leaking it into the record. ``model`` is the verifier
    model's resolved identity, which must differ from the subject's for the check
    to count as diverse.
    """

    agreement: str
    raw_judgment: str
    model: ModelDerived


class Judge(Protocol):
    """A second model that judges whether a response answers a request.

    Implementations call a *different* local model. Tests inject a stub;
    production injects :class:`OllamaJudge`.
    """

    def __call__(self, *, messages: Any, candidate_response: Any) -> JudgeOutcome: ...


# --- the cross-check record -------------------------------------------------


@dataclass(frozen=True)
class CrossCheckRecord:
    """Signed ``vaara.inference-crosscheck/v0`` envelope.

    The signature covers the JCS-canonical encoding of every block except the
    signature itself.
    """

    version: int
    alg: Algorithm
    subject_receipt_digest: str
    subject_model: ModelDerived
    verifier_model: ModelDerived
    judgment_commitment: ArgsCommitment
    agreement: str
    diverse: bool
    checked_at: str
    receipt_asserted: ReceiptAsserted
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CROSSCHECK_SCHEMA,
            "version": self.version,
            "alg": self.alg,
            "subject": {
                "receiptDigest": self.subject_receipt_digest,
                "model": model_derived_to_dict(self.subject_model),
            },
            "verifier": {
                "model": model_derived_to_dict(self.verifier_model),
                "method": CROSSCHECK_METHOD,
                "judgmentCommitment": args_to_dict(self.judgment_commitment),
            },
            "crosscheck": {
                "agreement": self.agreement,
                "diverse": self.diverse,
                "checkedAt": self.checked_at,
            },
            "receiptAsserted": receipt_asserted_to_dict(self.receipt_asserted),
            "signature": self.signature,
        }


def _signing_payload(record: CrossCheckRecord) -> bytes:
    """JCS-canonical encoding of the record blocks, signature excluded."""
    full = record.to_dict()
    full.pop("signature", None)
    return canonical_json(full)


def response_matches_receipt(receipt: InferenceReceipt, response: Any) -> bool:
    """True iff ``response`` recomputes the receipt's output commitment.

    Recomputes ``make_output_commitment(response)`` with the SHIPPING commitment
    and constant-time compares the projection digest to the one the receipt
    bound. A refusal receipt (no output commitment) can never match, so a
    cross-check cannot be attached to an outcome that produced no output.
    """
    oc = receipt.outcome_derived.output_commitment
    if oc is None or not isinstance(oc, ArgsProjection):
        return False
    recomputed = make_output_commitment(response)
    return hmac.compare_digest(oc.projection_digest, recomputed.projection_digest)


def build_crosscheck(
    *,
    attestation: InferenceAttestation,
    receipt: InferenceReceipt,
    messages: Any,
    response: Any,
    judge: Judge,
    secret_version: str,
    alg: Algorithm,
    signing_material: Any,
    iss: str = "vaara-infer-crosscheck",
    sub: Optional[str] = None,
    nonce: Optional[str] = None,
    iat: Optional[str] = None,
    checked_at: Optional[str] = None,
    version: int = 1,
) -> CrossCheckRecord:
    """Run the second model over a subject inference and sign the verdict.

    Refuses (raises :class:`AttestationError`) before ever calling the judge if
    the receipt does not back-link the attestation, or if ``response`` does not
    match the receipt's output commitment. Those two gates are what make the
    signed verdict cover the *real* hardware-bound receipt rather than a
    substitute. ``messages`` and ``response`` are the runtime prompt and the
    assembled output object the proxy committed; they are passed to the judge and
    used for the binding check, never stored in the record.
    """
    if alg not in VALID_ALGS:
        raise AttestationError(f"unsupported alg: {alg!r}")
    if not verify_inference_back_link(receipt, attestation=attestation):
        raise AttestationError(
            "receipt does not back-link the supplied attestation; "
            "subject model identity cannot be trusted"
        )
    if not response_matches_receipt(receipt, response):
        raise AttestationError(
            "supplied response does not match the receipt's outputCommitment; "
            "refusing to cross-check a substituted output"
        )

    subject_model = attestation.model_derived
    outcome = judge(messages=messages, candidate_response=response)
    if outcome.agreement not in CROSSCHECK_AGREEMENTS:
        raise AttestationError(
            f"judge returned an unknown agreement {outcome.agreement!r}; "
            f"expected one of {sorted(CROSSCHECK_AGREEMENTS)}"
        )

    verifier_model = outcome.model
    # Diversity keys on the weights pin (gguf metadata), not the manifest digest:
    # the live OllamaJudge resolves a model's manifest differently from the proxy,
    # so a same-model judge would show a different manifest and falsely read as
    # diverse. The gguf-metadata hash is computed identically on both sides, so
    # same weights -> same hash -> not diverse, which is the honest signal.
    diverse = subject_model.gguf_metadata_hash != verifier_model.gguf_metadata_hash
    judgment_commitment = make_output_commitment(outcome.raw_judgment)

    receipt_asserted = ReceiptAsserted(
        iss=iss,
        sub=sub or verifier_model.model_ref,
        iat=iat or now_iso8601(),
        nonce=nonce or new_nonce(),
        secret_version=secret_version,
        alg=alg,
    )
    unsigned = CrossCheckRecord(
        version=version,
        alg=alg,
        subject_receipt_digest=inference_receipt_digest(receipt),
        subject_model=subject_model,
        verifier_model=verifier_model,
        judgment_commitment=judgment_commitment,
        agreement=outcome.agreement,
        diverse=diverse,
        checked_at=checked_at or now_iso8601(),
        receipt_asserted=receipt_asserted,
        signature="",
    )
    signature = _sign(
        _signing_payload(unsigned), alg=alg, signing_material=signing_material
    )
    return CrossCheckRecord(
        version=unsigned.version,
        alg=unsigned.alg,
        subject_receipt_digest=unsigned.subject_receipt_digest,
        subject_model=unsigned.subject_model,
        verifier_model=unsigned.verifier_model,
        judgment_commitment=unsigned.judgment_commitment,
        agreement=unsigned.agreement,
        diverse=unsigned.diverse,
        checked_at=unsigned.checked_at,
        receipt_asserted=unsigned.receipt_asserted,
        signature=signature,
    )


def parse_crosscheck(doc: Any) -> CrossCheckRecord:
    """Validate a record against the closed schema and self-consistency.

    Beyond the closed key sets and digest/enum shapes, this enforces that the
    stored ``diverse`` flag agrees with the two model identities: a record that
    claims ``diverse: true`` while the subject and verifier manifest digests are
    equal is rejected here rather than allowed to verify. Returns the
    reconstructed record; raises :class:`AttestationError` otherwise.
    """
    if not isinstance(doc, dict):
        raise AttestationError("crosscheck record must be a JSON object")
    extra = set(doc) - _RECORD_KEYS
    if extra:
        raise AttestationError(
            f"crosscheck record carries unrecognized field(s) {sorted(extra)!r}; "
            "the schema is closed"
        )
    if doc.get("schema") != CROSSCHECK_SCHEMA:
        raise AttestationError(
            f"unexpected schema {doc.get('schema')!r}; expected {CROSSCHECK_SCHEMA!r}"
        )
    for required in _RECORD_KEYS:
        if required not in doc:
            raise AttestationError(
                f"crosscheck record missing required field {required!r}"
            )
    if doc["alg"] not in VALID_ALGS:
        raise AttestationError(f"unsupported alg {doc['alg']!r}")

    subject = doc["subject"]
    verifier = doc["verifier"]
    crosscheck = doc["crosscheck"]
    for block, keys, name in (
        (subject, _SUBJECT_KEYS, "subject"),
        (verifier, _VERIFIER_KEYS, "verifier"),
        (crosscheck, _CROSSCHECK_KEYS, "crosscheck"),
    ):
        if not isinstance(block, dict):
            raise AttestationError(f"crosscheck {name} block must be an object")
        block_extra = set(block) - keys
        if block_extra:
            raise AttestationError(
                f"crosscheck {name} block carries unrecognized field(s) "
                f"{sorted(block_extra)!r}"
            )
        for required in keys:
            if required not in block:
                raise AttestationError(
                    f"crosscheck {name} block missing required field {required!r}"
                )

    if not str(subject["receiptDigest"]).startswith("sha256:"):
        raise AttestationError("subject.receiptDigest MUST be a 'sha256:' digest")
    if verifier["method"] != CROSSCHECK_METHOD:
        raise AttestationError(
            f"unexpected verifier.method {verifier['method']!r}; "
            f"expected {CROSSCHECK_METHOD!r}"
        )
    if crosscheck["agreement"] not in CROSSCHECK_AGREEMENTS:
        raise AttestationError(f"invalid agreement {crosscheck['agreement']!r}")
    if not isinstance(crosscheck["diverse"], bool):
        raise AttestationError("crosscheck.diverse MUST be a boolean")
    if not isinstance(crosscheck["checkedAt"], str) or not crosscheck["checkedAt"]:
        raise AttestationError("crosscheck.checkedAt MUST be a non-empty string")

    subject_model = model_derived_from_dict(subject["model"])
    verifier_model = model_derived_from_dict(verifier["model"])
    expected_diverse = (
        subject_model.gguf_metadata_hash != verifier_model.gguf_metadata_hash
    )
    if crosscheck["diverse"] != expected_diverse:
        raise AttestationError(
            "crosscheck.diverse does not agree with the subject/verifier weights "
            "(gguf metadata) digests (the flag was altered after the record was built)"
        )

    return CrossCheckRecord(
        version=doc["version"],
        alg=doc["alg"],
        subject_receipt_digest=subject["receiptDigest"],
        subject_model=subject_model,
        verifier_model=verifier_model,
        judgment_commitment=args_from_dict(verifier["judgmentCommitment"]),
        agreement=crosscheck["agreement"],
        diverse=crosscheck["diverse"],
        checked_at=crosscheck["checkedAt"],
        receipt_asserted=receipt_asserted_from_dict(doc["receiptAsserted"]),
        signature=doc["signature"],
    )


def verify_crosscheck_signature(
    record: CrossCheckRecord, *, verifying_material: Any
) -> bool:
    """Verify the record signature only (a cross-check is durable, no TTL)."""
    return _verify(
        _signing_payload(record),
        alg=record.alg,
        signature_hex=record.signature,
        verifying_material=verifying_material,
    )


def verify_crosscheck(
    record: CrossCheckRecord,
    *,
    subject_receipt: Optional[InferenceReceipt] = None,
    verifying_material: Any = None,
) -> dict[str, Any]:
    """Composite verdict over a parsed cross-check record.

    ``subject_receipt`` (when supplied) is the receipt the record claims to
    cover; its digest is recomputed and constant-time compared, so a record
    cannot be re-pointed at a different receipt. ``verifying_material`` keys the
    signature check; ``None`` runs structural checks only. ``corroborated`` is
    the strict AND of: a valid signature, a diverse verifier, an ``equivalent``
    agreement, and (when a receipt is given) a matching receipt digest. It is
    corroboration, never proof: see the module docstring.
    """
    checks: dict[str, Any] = {
        "schema": True,
        "agreement": record.agreement,
        "diverse": record.diverse,
        "method": CROSSCHECK_METHOD,
    }
    if verifying_material is not None:
        checks["signatureValid"] = verify_crosscheck_signature(
            record, verifying_material=verifying_material
        )
    if subject_receipt is not None:
        expected = inference_receipt_digest(subject_receipt)
        checks["receiptBinds"] = hmac.compare_digest(
            record.subject_receipt_digest, expected
        )

    sig_ok = checks["signatureValid"] if "signatureValid" in checks else True
    receipt_ok = checks["receiptBinds"] if "receiptBinds" in checks else True
    corroborated = bool(
        sig_ok and receipt_ok and record.diverse and record.agreement == "equivalent"
    )
    checks["corroborated"] = corroborated
    checks["reason"] = _reason(
        checks,
        record,
        has_key=verifying_material is not None,
        has_receipt=subject_receipt is not None,
    )
    return checks


def _reason(
    checks: dict[str, Any],
    record: CrossCheckRecord,
    *,
    has_key: bool,
    has_receipt: bool,
) -> str:
    if has_key and not checks.get("signatureValid"):
        return "the cross-check signature did not verify: the verdict is not authentic"
    if has_receipt and not checks.get("receiptBinds"):
        return (
            "the record's subject receipt digest does not match the supplied "
            "receipt: this cross-check covers a different inference"
        )
    if not record.diverse:
        return (
            "the verifier model has the same identity as the subject model: a "
            "model judging its own output is not a diversity check"
        )
    if record.agreement != "equivalent":
        return (
            f"the independent verifier judged the output {record.agreement!r}, "
            "not equivalent"
        )
    return (
        "an independent local model of different identity judged the subject "
        "output equivalent to a correct answer; this is corroboration, not proof, "
        "and makes no determinism claim. The subject receipt is hardware-bound "
        "through its session manifest; this binds the second opinion to that root"
    )


def parse_agreement(raw: str) -> str:
    """Extract the agreement token from a judge reply; fail safe to 'uncertain'.

    Reads the token that *begins* the ``VERDICT:`` value (the format the judge is
    instructed to use), case-insensitively. A reply with no ``VERDICT:`` marker,
    or one whose value does not start with a known token, returns ``"uncertain"``.
    Matching on the leading token (not a free-text scan) means a hedged reply like
    "not equivalent" is never misread as ``"equivalent"``.
    """
    low = raw.lower()
    marker = "verdict:"
    idx = low.find(marker)
    if idx == -1:
        return "uncertain"
    head = low[idx + len(marker):].strip()
    for token in ("equivalent", "divergent", "uncertain"):
        if head.startswith(token):
            return token
    return "uncertain"


# --- the live judge ---------------------------------------------------------


class OllamaJudge:
    """A live :class:`Judge` backed by a second local model on an ollama server.

    Only used outside tests. Calls ``/api/chat`` with a strict-verdict system
    prompt and resolves the verifier model identity from ``/api/show`` the same
    way the proxy does, so the recorded verifier identity is a real weights pin,
    not a name. Uses stdlib ``urllib`` so the module has no new dependency.
    """

    _SYSTEM = (
        "You are an independent verifier. A different AI model produced a response "
        "to the user's request below. Judge only whether the response is a correct "
        "and adequate answer to the request. Reply with exactly one line beginning "
        "'VERDICT: equivalent' (it correctly and adequately answers), "
        "'VERDICT: divergent' (it is wrong or inadequate), or 'VERDICT: uncertain' "
        "(you cannot tell). You may add a short reason after the verdict."
    )

    def __init__(
        self,
        *,
        model: str,
        upstream: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        num_gpu: "Optional[int]" = None,
    ) -> None:
        self._model = model
        self._upstream = upstream.rstrip("/")
        self._timeout = timeout
        # ``num_gpu=0`` runs the judge CPU-only so a light verifier never evicts a
        # large subject model from a small GPU. The verdict is short, so CPU is
        # quick; identity resolution and the diverse gate are unaffected.
        self._num_gpu = num_gpu

    def __call__(self, *, messages: Any, candidate_response: Any) -> JudgeOutcome:
        raw = self._chat(self._render(messages, candidate_response))
        return JudgeOutcome(
            agreement=parse_agreement(raw),
            raw_judgment=raw,
            model=self._resolve_model(),
        )

    @staticmethod
    def _render(messages: Any, candidate_response: Any) -> str:
        if isinstance(messages, list):
            rendered = "\n".join(
                f"{m.get('role', '?')}: {m.get('content', '')}"
                for m in messages
                if isinstance(m, dict)
            )
        else:
            rendered = str(messages)
        body = candidate_response
        if isinstance(candidate_response, dict):
            body = candidate_response.get("content", json.dumps(candidate_response))
        return f"REQUEST:\n{rendered}\n\nCANDIDATE RESPONSE:\n{body}\n\nGive your VERDICT."

    def _chat(self, prompt: str) -> str:
        import urllib.request

        options: "dict[str, Any]" = {"temperature": 0}
        if self._num_gpu is not None:
            options["num_gpu"] = self._num_gpu
        payload = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": options,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{self._upstream}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data.get("message", {}).get("content", ""))

    def _resolve_model(self) -> ModelDerived:
        import urllib.request

        from vaara.integrations._infer_proxy_model import (
            fallback_model_derived,
            stable_hash,
        )

        try:
            req = urllib.request.Request(
                f"{self._upstream}/api/show",
                data=json.dumps({"model": self._model}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                show = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return fallback_model_derived(self._model)
        model_info = show.get("model_info") or {}
        details = show.get("details") or {}
        gguf_hash = stable_hash(model_info) if model_info else stable_hash(details)
        manifest = stable_hash({"model": self._model, "ggufMetadataHash": gguf_hash})
        return ModelDerived(
            model_ref=self._model,
            manifest_digest=manifest,
            gguf_metadata_hash=gguf_hash,
            quantization=details.get("quantization_level"),
            param_count=details.get("parameter_size"),
        )


# --- CLI: verify a cross-check record ---------------------------------------


def _load_verifying_material(pubkey: Optional[str], secret: Optional[str]) -> Any:
    if pubkey and secret:
        raise ValueError("pass only one of --pubkey / --secret")
    if secret:
        return Path(secret).expanduser().read_bytes()
    if pubkey:
        from cryptography.hazmat.primitives import serialization

        return serialization.load_pem_public_key(
            Path(pubkey).expanduser().read_bytes()
        )
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify-crosscheck",
        description=(
            "Verify a signed vaara.inference-crosscheck/v0 record: an independent "
            "second model's equivalence opinion bound to a subject inference receipt."
        ),
    )
    parser.add_argument("record", help="Path to a cross-check record JSON.")
    parser.add_argument(
        "--receipt",
        default=None,
        help="Subject InferenceReceipt JSON; enables the receipt-digest binding check.",
    )
    parser.add_argument("--pubkey", default=None, help="PEM public key (ES256/RS256).")
    parser.add_argument("--secret", default=None, help="Raw shared-secret file (HS256).")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    record_path = Path(args.record).expanduser()
    if not record_path.is_file():
        print(f"verify-crosscheck: not a file: {record_path}", file=sys.stderr)
        return 2
    try:
        material = _load_verifying_material(args.pubkey, args.secret)
    except (ValueError, OSError) as exc:
        print(f"verify-crosscheck: {exc}", file=sys.stderr)
        return 2

    try:
        record = parse_crosscheck(json.loads(record_path.read_text(encoding="utf-8")))
    except (AttestationError, ValueError) as exc:
        print(f"verify-crosscheck: {exc}", file=sys.stderr)
        return 1

    subject_receipt = None
    if args.receipt:
        from vaara.attestation.inference import parse_inference_receipt

        subject_receipt = parse_inference_receipt(
            json.loads(Path(args.receipt).expanduser().read_text(encoding="utf-8"))
        )

    verdict = verify_crosscheck(
        record, subject_receipt=subject_receipt, verifying_material=material
    )
    if args.json:
        print(json.dumps(verdict, indent=2))
    else:
        label = "CORROBORATED" if verdict["corroborated"] else "NOT CORROBORATED"
        print(
            f"{record_path.name}: {label}  (agreement={verdict['agreement']}, "
            f"diverse={verdict['diverse']})"
        )
        for key in ("signatureValid", "receiptBinds"):
            if key in verdict:
                print(f"    [{'pass' if verdict[key] else 'FAIL':4s}] {key}")
        print(f"    {verdict['reason']}")
    return 0 if verdict["corroborated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
