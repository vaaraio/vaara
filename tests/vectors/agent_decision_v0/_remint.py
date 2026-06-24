#!/usr/bin/env python3
"""Independent re-mint of the agent_decision_v0 conformance vector.

The *second producer* asked for on ards-project/ard-spec#7. Where
``_check_independent.py`` verifies the committed vector with no Vaara import,
this re-derives it from scratch and proves the committed bytes reproduce
exactly. It shares no code with ``_generate.py`` (which builds ``normalized`` by
calling ``vaara.attestation.receipt.normalize``); here the mapping is recomputed
from the shipped declarative profile with this file's own spec interpreter.
Imports: the standard library plus ``cryptography`` (Ed25519) and ``rfc8785``
(JCS) — no Vaara, no generator, no issuer in the loop.

From the source input alone it re-derives statement.json (JCS-sorted form),
envelope.json (JCS payload, DSSE PAE, deterministic Ed25519 signature),
expected.json (paeSha256, decision, SEP-2828 mapping), and the public key, then
compares each byte-for-byte with the committed file. A tamper pass mutates the
decision and confirms the content address moves (fails closed).

Run: ``python tests/vectors/agent_decision_v0/_remint.py``. Exit 0 means every
committed byte reproduced from scratch and the tamper check failed closed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import rfc8785
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "vectors" / "normalize_v0" / "inputs" / "agent_decision.json"
PROFILE = HERE.parents[2] / "src" / "vaara" / "attestation" / "profiles" / "agent_decision.json"

# Published test inputs (values, not behaviour). Ed25519 is deterministic, so the
# seed alone pins both the public key and the signature.
SEED_HEX = "a1" * 32
PAYLOAD_TYPE = "application/vnd.in-toto+json"
KEYID = "vaara-agent-decision-conformance-ed25519-k1"


# --- own spec interpreter (regex tokeniser; not the checker's split/partition) -

_TOKENS = re.compile(r"[^.\[\]]+|\[\d+\]")


def _dig(node: Any, path: str) -> Any:
    """Resolve a dotted/indexed path like ``predicate.tool_calls[1].name``."""
    for token in _TOKENS.findall(path):
        if token.startswith("["):
            index = int(token[1:-1])
            if not isinstance(node, list) or index >= len(node):
                return None
            node = node[index]
        else:
            if not isinstance(node, dict) or token not in node:
                return None
            node = node[token]
    return node


def _rule_holds(doc: Any, rule: dict[str, Any]) -> bool:
    found = _dig(doc, rule["path"])
    if "equals" in rule:
        return found == rule["equals"]
    if "startsWith" in rule:
        return isinstance(found, str) and found.startswith(rule["startsWith"])
    if "in" in rule:
        return found in rule["in"]
    if "exists" in rule:
        return (found is not None) == bool(rule["exists"])
    return False


def _map_block(doc: Any, block: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, source in block.items():
        if isinstance(source, dict) and "const" in source:
            result[name] = source["const"]
            continue
        value = _dig(doc, source) if isinstance(source, str) else None
        if value is not None:
            result[name] = value
    return result


def _normalize(doc: Any, spec: dict[str, Any]) -> dict[str, Any]:
    detect = spec.get("detect", {})
    if not all(_rule_holds(doc, r) for r in detect.get("all", [])):
        raise SystemExit("source does not match the agent-decision profile")
    any_rules = detect.get("any", [])
    if any_rules and not any(_rule_holds(doc, r) for r in any_rules):
        raise SystemExit("source does not match the agent-decision profile")

    sep2828: dict[str, Any] = {}
    populated: list[str] = []
    for dotted, value in _map_block(doc, spec.get("sep2828", {})).items():
        cursor = sep2828
        keys = dotted.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
        populated.append(dotted)

    return {
        "sourceFormat": spec["sourceFormat"],
        "sourceTitle": spec["sourceTitle"],
        "recognized": True,
        "evidencePlane": spec.get("evidencePlane"),
        "sep2828": sep2828,
        "advisory": _map_block(doc, spec.get("advisory", {})),
        "populated": sorted(populated),
        "missing": list(spec.get("missing", [])),
        "notes": list(spec.get("notes", [])),
    }


# --- re-mint -----------------------------------------------------------------


def _pae(payload_type: str, body: bytes) -> bytes:
    head = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (len(head), head, len(body), body)


def _canonical_text(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def _remint(statement: dict[str, Any], spec: dict[str, Any]) -> dict[str, bytes]:
    priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(SEED_HEX))
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    payload = rfc8785.dumps(statement)
    pae = _pae(PAYLOAD_TYPE, payload)
    sig = priv.sign(pae)

    envelope = {
        "payloadType": PAYLOAD_TYPE,
        "payload": base64.standard_b64encode(payload).decode("ascii"),
        "signatures": [{"keyid": KEYID, "sig": base64.standard_b64encode(sig).decode("ascii")}],
    }
    expected = {
        "predicateType": statement["predicateType"],
        "paeSha256": "sha256:" + hashlib.sha256(pae).hexdigest(),
        "signatureVerifies": True,
        "decision": statement["predicate"]["decision"],
        "normalized": _normalize(statement, spec),
    }
    return {
        "statement.json": _canonical_text(statement).encode("utf-8"),
        "envelope.json": _canonical_text(envelope).encode("utf-8"),
        "expected.json": _canonical_text(expected).encode("utf-8"),
        "keys/ed25519_public.pem": pub_pem,
    }


def main() -> int:
    statement = json.loads(SOURCE.read_text(encoding="utf-8"))
    spec = json.loads(PROFILE.read_text(encoding="utf-8"))

    minted = _remint(statement, spec)
    ok = True
    for rel, produced in minted.items():
        committed = (HERE / rel).read_bytes()
        match = produced == committed
        ok = ok and match
        print(f"[{'OK' if match else 'FAIL'}] reproduced {rel} ({len(produced)} bytes)")

    forged = json.loads(SOURCE.read_text(encoding="utf-8"))
    forged["predicate"]["decision"] = "allow"
    forged_sha = "sha256:" + hashlib.sha256(_pae(PAYLOAD_TYPE, rfc8785.dumps(forged))).hexdigest()
    committed_sha = json.loads((HERE / "expected.json").read_text())["paeSha256"]
    tamper_closed = forged_sha != committed_sha
    ok = ok and tamper_closed
    print(f"[{'OK' if tamper_closed else 'FAIL'}] tamper fails closed (forged paeSha256 differs)")

    print(f"\n{'re-minted byte-for-byte from source, no Vaara in the loop' if ok else 'RE-MINT MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
