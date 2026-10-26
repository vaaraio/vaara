#!/usr/bin/env python3
"""Regenerate the capture_digest_only_v0 conformance vectors.

This script imports Vaara to MINT ``cases.json``; the sibling
``_check_independent.py`` imports no Vaara and only RECOMPUTES it. The key is a
fixed test key, so the output is byte-stable across runs.

Run: tests/vectors/capture_digest_only_v0/_generate.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from vaara import _capture

HERE = Path(__file__).resolve().parent

# A fixed TEST key. A real deployment's key is random and never published.
KEY_HEX = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"

POLICY = {
    "version": 1,
    "rules": [
        {"tool": "mcp__ehr__*", "fields": ["patient.id", "note"], "reference": "ehr"},
        {"tool": "mcp__lab__*", "fields": ["rows.*.ssn"]},
        {"tool": "shell.exec", "fields": ["*"], "reference": "shell"},
    ],
}

INPUTS = [
    ("nested_field_and_reference", "mcp__ehr__lookup", {
        "patient": {"id": "010170-123A", "ward": "B2"},
        "note": "Mäkelä: chest pain since monday",
        "limit": 5,
    }),
    ("every_item_of_a_list", "mcp__lab__results", {
        "rows": [{"ssn": "010170-123A", "n": 1}, {"ssn": "010170-123A", "n": 2}],
        "unit": "mmol/l",
    }),
    ("a_lone_wildcard_keeps_the_keys", "shell.exec", {
        "command": "grep 010170-123A /srv/records.txt",
        "timeout": 12.5,
        "flags": {"dry": True, "extra": None},
    }),
    ("a_tool_no_rule_names", "mcp__other__lookup", {
        "patient": {"id": "010170-123A"},
        "note": "left as the host sent it",
    }),
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        policy_file = Path(tmp) / "capture.json"
        policy_file.write_text(json.dumps(POLICY), encoding="utf-8")
        key_file = Path(tmp) / "keys" / "capture-hmac.key"
        key_file.parent.mkdir()
        key_file.write_text(KEY_HEX, encoding="utf-8")
        os.environ["VAARA_CAPTURE_POLICY"] = str(policy_file)
        cases = [
            {"name": name, "tool": tool, "parameters": params,
             "expected": _capture.mask(tool, params)}
            for name, tool, params in INPUTS
        ]
    doc = {
        "format": "vaara.capture/v0",
        "key_hex": KEY_HEX,
        "policy": POLICY,
        "cases": cases,
        "disclosures": [
            {"path": "patient.id", "value": "010170-123A",
             "digest": cases[0]["expected"]["patient"]["id"]["digest"], "matches": True},
            {"path": "patient.id", "value": "010170-123B",
             "digest": cases[0]["expected"]["patient"]["id"]["digest"], "matches": False},
            {"path": "note", "value": "010170-123A",
             "digest": cases[0]["expected"]["patient"]["id"]["digest"], "matches": False},
        ],
    }
    (HERE / "cases.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(cases)} cases to {HERE / 'cases.json'}")


if __name__ == "__main__":
    main()
