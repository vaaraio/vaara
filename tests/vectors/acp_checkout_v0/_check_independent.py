#!/usr/bin/env python3
"""Independent checker for the acp_checkout_v0 conformance vector.

Imports only the standard library plus ``rfc8785`` (JCS). It imports **no Vaara
code**: producer and auditor share no implementation, so a passing check means
the vector stands on the bytes alone.

ACP checkout sessions are not signed (the protocol authenticates the API call
at the transport, not the session record), so there is no signature to verify.
The checker reproduces two things:

  jcsSha256   the sha256 of the JCS (RFC 8785) canonical statement bytes, a
              content commitment any reader recomputes from the document.
  normalized  the SEP-2828 mapping (which evidence plane, which advisory fields
              lifted, what a complete signed receipt is still missing) reproduced
              from the shipped declarative profile spec, with this checker's own
              code.

The mapping is reproduced from ``src/vaara/attestation/profiles/acp_checkout.json``
(the same data the product ships), not from any Vaara function, so the profile
is no more self-confirming than a hand-written one.

Run: ``python tests/vectors/acp_checkout_v0/_check_independent.py``.
Exit 0 means every verdict matched ``expected.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import rfc8785

HERE = Path(__file__).resolve().parent
PROFILE = (
    HERE.parents[2]
    / "src" / "vaara" / "attestation" / "profiles" / "acp_checkout.json"
)


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
        raise SystemExit("statement does not match the acp-checkout profile")
    any_rules = detect.get("any", [])
    if any_rules and not any(_rule_ok(doc, r) for r in any_rules):
        raise SystemExit("statement does not match the acp-checkout profile")
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
    statement = json.loads((HERE / "statement.json").read_text(encoding="utf-8"))
    spec = json.loads(PROFILE.read_text(encoding="utf-8"))

    jcs = rfc8785.dumps(statement)
    jcs_sha256 = "sha256:" + hashlib.sha256(jcs).hexdigest()
    normalized = _normalize_from_spec(statement, spec)

    checks = {
        "jcsSha256_matches": jcs_sha256 == expected["jcsSha256"],
        "status_matches": statement.get("status") == expected["status"],
        "normalized_matches": normalized == expected["normalized"],
    }

    for name, ok in checks.items():
        print(f"[{'OK' if ok else 'FAIL'}] {name}")
    passed = all(checks.values())
    print(f"\n{'all verdicts matched expected' if passed else 'MISMATCH vs expected'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
