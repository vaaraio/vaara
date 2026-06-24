#!/usr/bin/env python3
"""Independent checker for the agent_decision_v0 conformance vector.

Imports only the standard library plus ``cryptography`` (Ed25519) and
``rfc8785`` (JCS). It imports **no Vaara code**: producer and auditor share no
implementation, so a passing check means the vector stands on the bytes alone.

It reproduces the verdict offered on in-toto/attestation#554 for the proposed
``agent-decision/v0.1`` predicate:

  paeSha256          recompute the DSSE pre-authentication encoding from the
                     envelope payload and confirm its sha256 matches expected.
  signatureVerifies  verify the Ed25519 signature over that PAE with the
                     published public key.
  normalized         reproduce the SEP-2828 mapping (which evidence plane, which
                     fields populated, what is still missing) from the shipped
                     declarative profile spec, with this checker's own code, and
                     confirm it matches expected.

The mapping is reproduced from ``src/vaara/attestation/profiles/agent_decision.json``
(the same data the product ships), not from any Vaara function, so the profile
is no more self-confirming than a hand-written one.

Run: ``python tests/vectors/agent_decision_v0/_check_independent.py``.
Exit 0 means every verdict matched ``expected.json``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import rfc8785
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

HERE = Path(__file__).resolve().parent
PROFILE = (
    HERE.parents[2]
    / "src" / "vaara" / "attestation" / "profiles" / "agent_decision.json"
)


def _pae(payload_type: str, body: bytes) -> bytes:
    t = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (len(t), t, len(body), body)


# --- declarative-profile reproduction (own code, reads the shipped spec) ------


def _resolve(doc: Any, path: str) -> Any:
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


def _lift(doc: Any, mapping: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, source in mapping.items():
        if isinstance(source, dict) and "const" in source:
            out[key] = source["const"]
        elif isinstance(source, str):
            value = _resolve(doc, source)
            if value is not None:
                out[key] = value
    return out


def _normalize_from_spec(doc: Any, spec: dict[str, Any]) -> dict[str, Any]:
    detect = spec["detect"]
    if not all(_rule_ok(doc, r) for r in detect.get("all", [])):
        raise SystemExit("statement does not match the agent-decision profile")
    any_rules = detect.get("any", [])
    if any_rules and not any(_rule_ok(doc, r) for r in any_rules):
        raise SystemExit("statement does not match the agent-decision profile")
    sep2828: dict[str, Any] = {}
    populated: list[str] = []
    for dotted, value in _lift(doc, spec.get("sep2828", {})).items():
        cur = sep2828
        parts = dotted.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
        populated.append(dotted)
    return {
        "sourceFormat": spec["sourceFormat"],
        "sourceTitle": spec["sourceTitle"],
        "recognized": True,
        "evidencePlane": spec.get("evidencePlane"),
        "sep2828": sep2828,
        "advisory": _lift(doc, spec.get("advisory", {})),
        "populated": sorted(populated),
        "missing": list(spec.get("missing", [])),
        "notes": list(spec.get("notes", [])),
    }


def main() -> int:
    expected = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    envelope = json.loads((HERE / "envelope.json").read_text(encoding="utf-8"))
    spec = json.loads(PROFILE.read_text(encoding="utf-8"))
    pub = load_pem_public_key((HERE / "keys" / "ed25519_public.pem").read_bytes())

    payload = base64.standard_b64decode(envelope["payload"])
    statement = json.loads(payload)
    pae = _pae(envelope["payloadType"], payload)

    pae_sha256 = "sha256:" + hashlib.sha256(pae).hexdigest()

    sig_ok = False
    if isinstance(pub, Ed25519PublicKey):
        sig = base64.standard_b64decode(envelope["signatures"][0]["sig"])
        try:
            pub.verify(sig, pae)
            sig_ok = True
        except InvalidSignature:
            sig_ok = False

    # The payload bytes must be the JCS canonical form, or an independent reader
    # cannot recompute the same PAE: re-canonicalize and require byte-equality.
    payload_is_canonical = payload == rfc8785.dumps(statement)

    normalized = _normalize_from_spec(statement, spec)

    got = {
        "predicateType": statement.get("predicateType"),
        "paeSha256": pae_sha256,
        "signatureVerifies": sig_ok,
        "decision": statement.get("predicate", {}).get("decision"),
        "normalized": normalized,
    }

    checks = {
        "payload_is_jcs_canonical": payload_is_canonical,
        "predicateType_matches": got["predicateType"] == expected["predicateType"],
        "paeSha256_matches": got["paeSha256"] == expected["paeSha256"],
        "signature_verifies": got["signatureVerifies"] is True
        and expected["signatureVerifies"] is True,
        "decision_matches": got["decision"] == expected["decision"],
        "normalized_matches": got["normalized"] == expected["normalized"],
    }

    for name, ok in checks.items():
        print(f"[{'OK' if ok else 'FAIL'}] {name}")
    passed = all(checks.values())
    print(f"\n{'all verdicts matched expected' if passed else 'MISMATCH vs expected'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
