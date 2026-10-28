# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The SOC 2 CC8.1 row says a policy reload is not on the trail.

A reload swaps the live policy and writes no record, so the row must not
let a reader take Vaara's trail as evidence of policy change control.
The second half pins the behaviour: if a reload ever writes a record,
this test fails and the row has to change with it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from vaara.audit.trail import AuditTrail
from vaara.policy.controller import PolicyController
from vaara.policy.loader import from_dict
from vaara.policy.registry import PolicyRegistry

ROOT = Path(__file__).resolve().parents[1]


def _cc81_row() -> str:
    text = (ROOT / "docs" / "COMPLIANCE.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| **CC8.1**"))
    return re.sub(r"\s+", " ", row)


def _policy(escalate: float, deny: float) -> dict:
    return {
        "version": "0.1",
        "domains": ["eu_ai_act"],
        "action_classes": {},
        "thresholds": {"default": {"escalate": escalate, "deny": deny}},
        "sequences": {},
        "escalation": {"routes": [{"operator_group": "on_call", "if": []}]},
    }


def test_the_cc81_row_names_the_reload_limit():
    row = _cc81_row()
    assert "writes no record" in row
    for path in ("`PolicyController.reload`", "`POST /v1/policy/reload`", "`PolicyRegistry`"):
        assert path in row, path
    assert "deployer's own change control" in row


def test_a_reload_writes_no_trail_record():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from vaara.server import create_app

    audit = AuditTrail()
    registry = PolicyRegistry()
    registry.register("", PolicyController(from_dict(_policy(0.55, 0.85))))
    registry.register("tenant-a", PolicyController(from_dict(_policy(0.55, 0.85))))
    client = TestClient(create_app(audit=audit, policy_registry=registry))

    before = audit.size
    for tenant in ("", "tenant-a"):
        r = client.post(
            "/v1/policy/reload",
            json={"body": _policy(0.4, 0.9), "tenant_id": tenant},
        )
        assert r.status_code == 200, r.text
    registry.reload("tenant-a", _policy(0.3, 0.95))
    assert audit.size == before
