#!/usr/bin/env python3
"""Independent checker for the v0 normalize vectors.

A second implementation of the SEP-2643 / SEP-2787 / SEP-2817 -> SEP-2828
normalization, written from the specs alone and importing no Vaara code.
For each committed input it reproduces the normalized mapping (which
evidence plane, which fields populated, what is still missing) and
compares against ``expected.json``.

The SEP-2643 and SEP-2817 maps are pure standard library. The SEP-2787
map reconstructs the SEP-2787-modeled envelope (dropping unmodeled fields
and injecting the ArgsRef canonicalization default, exactly as the wire
schema defines) and digests that under RFC 8785, which is the value the
receipt verifier pins; that case is skipped, not failed, when ``rfc8785``
is not installed.

Run: ``python tests/vectors/normalize_v0/_check_independent.py``.
Exit 0 means every runnable case matched its expected mapping.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
AI_KEY = "io.modelcontextprotocol/aiInvocation"
# Declarative profiles are data: this checker reads the same JSON specs the
# product ships and reproduces the field-mapping with its own code below,
# importing nothing from Vaara.
PROFILE_DIR = HERE.parents[2] / "src" / "vaara" / "attestation" / "profiles"

try:
    import rfc8785

    HAVE_JCS = True
except ImportError:  # pragma: no cover - exercised only in a base install
    HAVE_JCS = False


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _denial(doc: Any) -> Optional[dict[str, Any]]:
    if not isinstance(doc, dict):
        return None
    for cand in (
        _as_dict(_as_dict(doc.get("error")).get("data")).get("authorization"),
        _as_dict(doc.get("data")).get("authorization"),
    ):
        if isinstance(cand, dict) and isinstance(cand.get("reason"), str):
            return cand
    for cand in (doc.get("authorization"), doc):
        if (
            isinstance(cand, dict)
            and isinstance(cand.get("reason"), str)
            and ("authorizationContextId" in cand or "remediationHints" in cand)
        ):
            return cand
    return None


def _invocation(doc: Any) -> Optional[dict[str, Any]]:
    if not isinstance(doc, dict):
        return None
    for cand in (
        _as_dict(_as_dict(doc.get("params")).get("_meta")).get(AI_KEY),
        _as_dict(doc.get("_meta")).get(AI_KEY),
        doc.get(AI_KEY),
    ):
        if isinstance(cand, dict):
            return cand
    if any(k in doc for k in ("invocationReason", "model", "userIntent", "turnId")):
        return doc
    return None


def _is_attestation(doc: Any) -> bool:
    keys = ("plannerDeclared", "issuerAsserted", "payloadDerived", "signature")
    return isinstance(doc, dict) and all(k in doc for k in keys)


def _result(source, title, recognized, plane, sep2828, advisory, populated, missing,
            notes):
    return {
        "sourceFormat": source,
        "sourceTitle": title,
        "recognized": recognized,
        "evidencePlane": plane,
        "sep2828": sep2828,
        "advisory": advisory,
        "populated": populated,
        "missing": missing,
        "notes": notes,
    }


def normalize(doc: Any) -> dict[str, Any]:
    if _is_attestation(doc):
        return _attestation(doc)
    if _denial(doc) is not None:
        return _denial_map(doc)
    if _invocation(doc) is not None:
        return _invocation_map(doc)
    declarative = _declarative_map(doc)
    if declarative is not None:
        return declarative
    return _result(
        "unknown", "unrecognized record", False, None, {}, {}, [], [],
        ["not a SEP-2643 denial, SEP-2787 attestation, or SEP-2817 "
         "invocation audit context; nothing to normalize"],
    )


def _resolve(doc: Any, path: str) -> Any:
    """Independent reimplementation of the contract's dotted/[index] path."""
    cur = doc
    for raw in path.split("."):
        key, _, rest = raw.partition("[")
        if key:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        for tok in (t for t in rest.split("[") if t):
            idx = int(tok.rstrip("]"))
            if not isinstance(cur, list) or idx >= len(cur):
                return None
            cur = cur[idx]
    return cur


def _rule_ok(doc: Any, rule: dict[str, Any]) -> bool:
    value = _resolve(doc, rule["path"])
    if "equals" in rule:
        return value == rule["equals"]
    if "startsWith" in rule:
        return isinstance(value, str) and value.startswith(rule["startsWith"])
    if "in" in rule:
        return value in rule["in"]
    if "exists" in rule:
        return (value is not None) == bool(rule["exists"])
    return False


def _spec_lift(doc: Any, mapping: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, source in mapping.items():
        if isinstance(source, dict) and "const" in source:
            out[key] = source["const"]
        elif isinstance(source, str):
            value = _resolve(doc, source)
            if value is not None:
                out[key] = value
    return out


def _declarative_map(doc: Any) -> dict[str, Any] | None:
    """Reproduce any matching declarative profile from its shipped JSON spec."""
    if not isinstance(doc, dict) or not PROFILE_DIR.is_dir():
        return None
    specs = [json.loads(p.read_text()) for p in sorted(PROFILE_DIR.glob("*.json"))]
    specs.sort(key=lambda s: (s.get("priority", 100), s["sourceFormat"]))
    for spec in specs:
        detect = spec["detect"]
        if not all(_rule_ok(doc, r) for r in detect.get("all", [])):
            continue
        any_rules = detect.get("any", [])
        if any_rules and not any(_rule_ok(doc, r) for r in any_rules):
            continue
        sep2828: dict[str, Any] = {}
        populated: list[str] = []
        for dotted, value in _spec_lift(doc, spec.get("sep2828", {})).items():
            cur = sep2828
            parts = dotted.split(".")
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = value
            populated.append(dotted)
        return _result(
            spec["sourceFormat"], spec["sourceTitle"], True,
            spec.get("evidencePlane"), sep2828,
            _spec_lift(doc, spec.get("advisory", {})),
            sorted(populated), list(spec.get("missing", [])),
            list(spec.get("notes", [])),
        )
    return None


def _denial_map(doc: Any) -> dict[str, Any]:
    authz = _denial(doc)
    assert authz is not None
    advisory: dict[str, Any] = {"reason": authz.get("reason")}
    if isinstance(authz.get("authorizationContextId"), str):
        advisory["authorizationContextId"] = authz["authorizationContextId"]
    hints = authz.get("remediationHints")
    if isinstance(hints, list):
        types = [h["type"] for h in hints
                 if isinstance(h, dict) and isinstance(h.get("type"), str)]
        if types:
            advisory["remediationHintTypes"] = types
    return _result(
        "sep2643", "SEP-2643 authorization denial", True, "outcome",
        {"outcomeDerived": {"status": "refused"}}, advisory,
        ["outcomeDerived.status"],
        ["alg", "signature", "backLink", "receiptAsserted",
         "outcomeDerived.completedAt"],
        ["a denial is the outcome of a refused call: it maps to "
         "outcomeDerived.status = refused",
         "a refused outcome carries no resultCommitment",
         "authorizationContextId is a correlation handle, not authorization "
         "material",
         "the denial carries no completedAt, no signing envelope, and no "
         "back-link; the recording side supplies those"],
    )


def _invocation_map(doc: Any) -> dict[str, Any]:
    inv = _invocation(doc)
    assert inv is not None
    advisory: dict[str, Any] = {}
    if isinstance(_as_dict(inv.get("invocationReason")).get("text"), str):
        advisory["invocationReason"] = inv["invocationReason"]["text"]
    if isinstance(_as_dict(inv.get("model")).get("name"), str):
        advisory["model"] = inv["model"]["name"]
    ui = _as_dict(inv.get("userIntent"))
    if ui.get("redacted") is True:
        advisory["userIntentRedacted"] = True
    elif isinstance(ui.get("text"), str):
        advisory["userIntent"] = ui["text"]
    if isinstance(inv.get("turnId"), str):
        advisory["turnId"] = inv["turnId"]
    notes = [
        "SEP-2817 is client-asserted input audit context; per its own "
        "specification it MUST NOT be used as authorization evidence",
        "it maps to the decision-input plane (the agent's stated intent), "
        "the unsigned counterpart of an attested rationale, and populates no "
        "required SEP-2828 field",
    ]
    if "turnId" in advisory:
        notes.append(
            "turnId groups requests from one user turn; it is correlation only")
    return _result(
        "sep2817", "SEP-2817 AI invocation audit context", True,
        "decision-input", {}, advisory, [],
        ["alg", "signature", "backLink", "receiptAsserted", "outcomeDerived"],
        notes,
    )


def _modeled_args(a: dict[str, Any]) -> dict[str, Any]:
    if "ref" in a:
        return {
            "ref": a.get("ref"),
            "digest": a.get("digest"),
            "canonicalization": a.get("canonicalization", "jcs"),
        }
    return {"projection": a.get("projection"), "projectionDigest": a.get("projectionDigest")}


def _modeled_attestation(doc: Any) -> dict[str, Any]:
    """Reconstruct the SEP-2787-modeled envelope the receipt verifier digests.

    Keep only the modeled fields, inject the ArgsRef canonicalization default,
    drop everything else. Digesting this (not the raw doc) is what makes the
    cross-check faithful to the production back-link computation.
    """
    issuer = _as_dict(doc.get("issuerAsserted"))
    planner = _as_dict(doc.get("plannerDeclared"))
    payload = _as_dict(doc.get("payloadDerived"))
    declared: dict[str, Any] = {"intent": planner.get("intent")}
    if planner.get("requestedCapability") is not None:
        declared["requestedCapability"] = planner["requestedCapability"]
    return {
        "version": doc.get("version"),
        "alg": doc.get("alg"),
        "signature": doc.get("signature"),
        "plannerDeclared": declared,
        "issuerAsserted": {
            k: issuer.get(k)
            for k in ("alg", "expSeconds", "iat", "iss", "nonce", "secretVersion", "sub")
        },
        "payloadDerived": {
            "toolCalls": [
                {
                    "name": c.get("name"),
                    "serverFingerprint": c.get("serverFingerprint"),
                    "args": _modeled_args(_as_dict(c.get("args"))),
                }
                for c in payload.get("toolCalls", [])
                if isinstance(c, dict)
            ]
        },
    }


def _attestation(doc: Any) -> dict[str, Any]:
    issuer = _as_dict(doc.get("issuerAsserted"))
    planner = _as_dict(doc.get("plannerDeclared"))
    payload = _as_dict(doc.get("payloadDerived"))
    advisory: dict[str, Any] = {}
    if isinstance(planner.get("intent"), str):
        advisory["intent"] = planner["intent"]
    for f in ("iss", "sub", "alg"):
        if isinstance(issuer.get(f), str):
            advisory[f"attestation_{f}"] = issuer[f]
    if isinstance(payload.get("toolCalls"), list):
        advisory["toolCalls"] = [
            {"name": c.get("name"), "serverFingerprint": c.get("serverFingerprint")}
            for c in payload["toolCalls"] if isinstance(c, dict)
        ]
    digest = "sha256:" + hashlib.sha256(
        rfc8785.dumps(_modeled_attestation(doc))
    ).hexdigest()
    return _result(
        "sep2787", "SEP-2787 tool-call attestation", True, "decision-attested",
        {"backLink": {"attestationDigest": digest,
                      "attestationNonce": issuer.get("nonce")}},
        advisory,
        ["backLink.attestationDigest", "backLink.attestationNonce"],
        ["alg", "signature", "receiptAsserted", "outcomeDerived"],
        ["a SEP-2787 attestation is the attested request a SEP-2828 receipt "
         "answers; it fixes the exact back-link a conformant receipt must pin",
         "plannerDeclared.intent is client-declared and bound by the issuer's "
         "signature, not asserted true by the issuer",
         "the record's own signing (alg, signature, receiptAsserted) is a "
         "separate event by the recording side, not derived from the attestation"],
    )


def main() -> int:
    expected = json.loads((HERE / "expected.json").read_text())
    failures = skipped = 0
    for name in sorted(expected):
        want = expected[name]
        doc = json.loads((HERE / "inputs" / f"{name}.json").read_text())
        if want["sourceFormat"] == "sep2787" and not HAVE_JCS:
            print(f"[SKIP] {name}: rfc8785 not installed")
            skipped += 1
            continue
        got = normalize(doc)
        ok = got == want
        failures += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {name}")
        if not ok:
            print(f"   got: {json.dumps(got, sort_keys=True)}")
    total = len(expected) - skipped
    print(f"\n{total - failures}/{total} cases matched ({skipped} skipped).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
