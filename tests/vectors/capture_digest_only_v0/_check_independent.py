#!/usr/bin/env python3
"""Independent checker for the vaara.capture/v0 vectors.

Imports only the standard library plus ``rfc8785``. It does NOT import Vaara.
From the policy, the test key and the input arguments alone it reproduces, for
every case, the arguments a store holds under digest-only capture:

  masked       each declared field is replaced by
               {"captured": "digest-only", "digest": "hmac-sha256:<hex>",
               "reference": ...} with
               digest = HMAC-SHA256(key, "vaara.capture.v1\\n" || path || "\\n" || JCS(value)),
               every other field is left as it was, and no declared value
               appears anywhere in the output.
  disclosure   an operator who holds the key shows that a value is the one
               that was passed by recomputing its digest; any other value, or
               the same value at another path, does not match.

Run: tests/vectors/capture_digest_only_v0/_check_independent.py
Exit 0 means every verdict matched the committed cases.
"""

from __future__ import annotations

import fnmatch
import hashlib
import hmac
import json
import sys
from pathlib import Path

import rfc8785

HERE = Path(__file__).resolve().parent
DOMAIN = b"vaara.capture.v1\n"


def digest(key: bytes, path: str, value) -> str:
    message = DOMAIN + path.encode() + b"\n" + rfc8785.dumps(value)
    return "hmac-sha256:" + hmac.new(key, message, hashlib.sha256).hexdigest()


def mask_value(key: bytes, path: str, value, reference: str, hidden: list) -> dict:
    hidden.append(value)
    out = {"captured": "digest-only", "digest": digest(key, path, value)}
    if reference:
        out["reference"] = reference
    return out


def walk(key: bytes, node, steps: list[str], at: list[str], reference: str, hidden: list):
    if not steps:
        return mask_value(key, ".".join(at), node, reference, hidden)
    head, rest = steps[0], steps[1:]
    if isinstance(node, dict):
        names = list(node) if head == "*" else ([head] if head in node else [])
        out = dict(node)
        for name in names:
            out[name] = walk(key, node[name], rest, at + [name], reference, hidden)
        return out
    if isinstance(node, list) and head == "*":
        return [walk(key, item, rest, at + [str(i)], reference, hidden)
                for i, item in enumerate(node)]
    return node


def expected_masked(doc: dict, tool: str, parameters, hidden: list):
    key = bytes.fromhex(doc["key_hex"])
    out = parameters
    for rule in doc["policy"]["rules"]:
        if not fnmatch.fnmatchcase(tool, rule.get("tool", "*")):
            continue
        for field in rule["fields"]:
            out = walk(key, out, field.split("."), [], rule.get("reference", ""), hidden)
    return out


def leaves(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from leaves(v)
    elif isinstance(node, list):
        for v in node:
            yield from leaves(v)
    else:
        yield node


def main() -> int:
    doc = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    failures = []
    for case in doc["cases"]:
        hidden: list = []
        got = expected_masked(doc, case["tool"], case["parameters"], hidden)
        if got != case["expected"]:
            failures.append(f"{case['name']}: masked arguments differ from the committed case")
        committed = json.dumps(case["expected"], ensure_ascii=False)
        for value in hidden:
            for leaf in leaves(value):
                if isinstance(leaf, str) and len(leaf) >= 4 and leaf in committed:
                    failures.append(f"{case['name']}: a declared value {leaf!r} is in the case in clear")
    key = bytes.fromhex(doc["key_hex"])
    for d in doc["disclosures"]:
        matches = hmac.compare_digest(digest(key, d["path"], d["value"]), d["digest"])
        if matches != d["matches"]:
            failures.append(f"disclosure at {d['path']}: expected matches={d['matches']}")
    for line in failures:
        print("FAIL", line)
    if failures:
        return 1
    print(f"ok: {len(doc['cases'])} cases, {len(doc['disclosures'])} disclosures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
