#!/usr/bin/env python3
"""Independent re-mint of the acp_checkout_v0 conformance vector.

The *second producer* asked for on ards-project/ard-spec#7. Where
``_check_independent.py`` verifies the committed vector with no Vaara import,
this re-derives the verdict from scratch and proves the committed bytes
reproduce exactly. It shares no code with ``_generate.py`` (which builds
``normalized`` by calling ``vaara.attestation.receipt.normalize``); here the
mapping is recomputed from the shipped declarative profile with this file's own
spec interpreter. Imports: the standard library plus ``rfc8785`` (JCS) — no
Vaara, no generator, no issuer in the loop.

ACP sessions are not signed, so the recomputable anchor is a content commitment.
From the statement alone it re-derives expected.json (jcsSha256 over the JCS
canonical bytes, the terminal status, and the SEP-2828 mapping) and compares it
byte-for-byte with the committed file. A tamper pass mutates the status and
confirms the content address moves (fails closed).

Run: ``python tests/vectors/acp_checkout_v0/_remint.py``. Exit 0 means the
committed verdict reproduced from scratch and the tamper check failed closed.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import rfc8785

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "statement.json"
PROFILE = HERE.parents[2] / "src" / "vaara" / "attestation" / "profiles" / "acp_checkout.json"


# --- own spec interpreter (regex tokeniser; not the checker's split/partition) -

_TOKENS = re.compile(r"[^.\[\]]+|\[\d+\]")


def _dig(node: Any, path: str) -> Any:
    """Resolve a dotted/indexed path like ``line_items[0].totals[0].amount``."""
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
        raise SystemExit("source does not match the acp-checkout profile")
    any_rules = detect.get("any", [])
    if any_rules and not any(_rule_holds(doc, r) for r in any_rules):
        raise SystemExit("source does not match the acp-checkout profile")

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


def _jcs_sha256(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(rfc8785.dumps(obj)).hexdigest()


def _canonical_text(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def main() -> int:
    statement = json.loads(SOURCE.read_text(encoding="utf-8"))
    spec = json.loads(PROFILE.read_text(encoding="utf-8"))

    expected = {
        "jcsSha256": _jcs_sha256(statement),
        "normalized": _normalize(statement, spec),
        "status": statement.get("status"),
    }
    produced = _canonical_text(expected).encode("utf-8")
    committed = (HERE / "expected.json").read_bytes()
    match = produced == committed
    print(f"[{'OK' if match else 'FAIL'}] reproduced expected.json ({len(produced)} bytes)")

    forged = json.loads(SOURCE.read_text(encoding="utf-8"))
    forged["status"] = "canceled"
    tamper_closed = _jcs_sha256(forged) != expected["jcsSha256"]
    ok = match and tamper_closed
    print(f"[{'OK' if tamper_closed else 'FAIL'}] tamper fails closed (forged jcsSha256 differs)")

    print(f"\n{'re-minted byte-for-byte from statement, no Vaara in the loop' if ok else 'RE-MINT MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
