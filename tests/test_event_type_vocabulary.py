# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One event-type vocabulary across the trail, the server, the spec, the
TypeScript client and the schema document.

The seam audit of 2026-10-11 found five lists: the trail's EventType had 16
values, the server's request Literal 12, docs/openapi.yaml 9, the TypeScript
EventType 9, and docs/audit_event_schema.md 16. Worse, two of the server's
extra values were markers only the trail itself should write: a client could
POST `key_lifecycle`, which counts as SOC 2 CC6.2 and CC8.1 evidence and is
surfaced by `vaara verify` as a custodian rotation. Opening the Literal to all
16 would have let a client POST `repair_gap`, and a later repair_gap naming
missing seqs turns a broken chain into one with declared gaps.

The rule held here: the trail's own markers (SYSTEM_EVENT_TYPES) are never
accepted from outside; every other type is, and every list names exactly it.
"""

from __future__ import annotations

import re
import typing
from pathlib import Path

import pytest

from vaara.audit.trail import CLIENT_EVENT_TYPES, SYSTEM_EVENT_TYPES, EventType

ROOT = Path(__file__).resolve().parents[1]
CLIENT = {e.value for e in CLIENT_EVENT_TYPES}
ALL = {e.value for e in EventType}


def test_the_split_covers_every_type_once():
    assert {e.value for e in SYSTEM_EVENT_TYPES} == {
        "anchor_gap", "repair_gap", "repair_recovered", "trail_forked", "key_lifecycle",
        "witness_configured", "record_erased", "cage_lifecycle", "trail_not_recording",
        "trail_rotated",
    }
    assert CLIENT | {e.value for e in SYSTEM_EVENT_TYPES} == ALL
    assert not CLIENT & {e.value for e in SYSTEM_EVENT_TYPES}


def test_server_request_literal_is_the_client_set():
    pytest.importorskip("fastapi")
    from vaara.server import schemas as S

    field = S.AuditEventRequest.model_fields["event_type"]
    assert set(typing.get_args(field.annotation)) == CLIENT


def test_openapi_enum_is_the_client_set():
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load((ROOT / "docs" / "openapi.yaml").read_text(encoding="utf-8"))
    enum = spec["components"]["schemas"]["AuditEventRequest"]["properties"]["event_type"]["enum"]
    assert set(enum) == CLIENT
    assert len(enum) == len(set(enum))


def test_typescript_event_type_is_the_client_set():
    types = (ROOT / "clients" / "ts" / "src" / "types.ts").read_text(encoding="utf-8")
    m = re.search(r"export type EventType =(.*?);", types, re.S)
    assert m
    assert set(re.findall(r'"([a-z_]+)"', m.group(1))) == CLIENT


def test_schema_document_names_every_type():
    doc = (ROOT / "docs" / "audit_event_schema.md").read_text(encoding="utf-8")
    section = doc.split("## Event types", 1)[1].split("\n## ", 1)[0]
    assert set(re.findall(r"^\| `([a-z_]+)` \|", section, re.M)) == ALL


@pytest.mark.parametrize("event_type", sorted(e.value for e in SYSTEM_EVENT_TYPES))
def test_server_refuses_a_system_marker(event_type):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from vaara.server import create_app

    client = TestClient(create_app())
    r = client.post("/v1/audit/events", json={
        "event_type": event_type, "action_id": "a1",
        "payload": {"lost_seqs": [3]},
    })
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("event_type", sorted(CLIENT))
def test_server_takes_every_client_type(event_type):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from vaara.server import create_app

    client = TestClient(create_app())
    r = client.post("/v1/audit/events", json={"event_type": event_type, "action_id": "a1"})
    assert r.status_code == 201, r.text
