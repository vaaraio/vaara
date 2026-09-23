"""The TypeScript client's types say what the HTTP API actually sends.

`appendAuditEvent` and `getActionChain` were typed with `record_id` and
`record_hash`, while the server and `docs/openapi.yaml` both use `event_id`,
`event_hash` and `chain_position`. A caller reading `.record_id` got
`undefined` and no compiler said anything. This compares every interface in
`clients/ts/src/types.ts` that shares a name with an OpenAPI schema: each
field the schema requires must be in the interface, and each field the
interface declares must exist in the schema.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
TYPES = (ROOT / "clients" / "ts" / "src" / "types.ts").read_text()
SCHEMAS = yaml.safe_load((ROOT / "docs" / "openapi.yaml").read_text())[
    "components"]["schemas"]


def _interface(name: str) -> dict[str, bool]:
    """Top-level fields of a TS interface, mapped to whether they are optional."""
    m = re.search(r"export interface %s \{(.*?)\n\}" % name, TYPES, re.S)
    assert m, name
    fields: dict[str, bool] = {}
    depth = 0
    for line in m.group(1).split("\n"):
        if depth == 0:
            f = re.match(r"\s+([A-Za-z_]+)(\??):", line)
            if f:
                fields[f.group(1)] = bool(f.group(2))
        depth += line.count("{") - line.count("}")
    return fields


def _shared() -> list[str]:
    names = re.findall(r"export interface ([A-Za-z]+) \{", TYPES)
    return [n for n in names if n in SCHEMAS]


def test_the_comparison_covers_the_audit_types():
    assert {"AuditEventResponse", "ScoreResponse", "VerifyResponse"} <= set(_shared())


@pytest.mark.parametrize("name", _shared())
def test_interface_matches_schema(name):
    schema = SCHEMAS[name]
    props = set(schema.get("properties", {}))
    required = set(schema.get("required", []))
    fields = _interface(name)
    assert required <= set(fields), f"{name} lacks {sorted(required - set(fields))}"
    assert set(fields) <= props, f"{name} declares {sorted(set(fields) - props)}"


def test_chain_event_matches_the_chain_schema():
    items = SCHEMAS["AuditChain"]["properties"]["events"]["items"]
    fields = _interface("AuditChainEvent")
    assert set(items["required"]) <= set(fields)
    assert set(fields) <= set(items["properties"])
