# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The published TypeScript client must agree with the server it talks to.

``clients/ts`` ships to npm as ``@vaara/client`` and its own test suite mocks
``fetch``, so it only ever proved the client agrees with itself. This drives
the real FastAPI app with the exact field names and shapes ``types.ts``
declares, which is where the disagreements were:

* ``OutcomeRequest`` declared ``description`` where the server takes
  ``notes``, and the server model is ``extra="forbid"``, so the documented
  call was a 422;
* ``reportOutcome`` was typed as returning ``{ok: true}`` from an endpoint
  that answers 204 with no body;
* ``ScoreRequest`` had no ``tenant_id``, so a TypeScript caller could not
  attribute a call to a tenant at all;
* ``ScoreResponse`` and ``RiskBlock`` omitted fields the server always sends
  and invented two it never sends.

The parsing here is deliberately crude: types.ts is the published artefact,
and a test that imported a generated model instead would stop checking the
thing that ships.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - server extra not installed
    pytest.skip(
        "server extra not installed (pip install 'vaara[server]')",
        allow_module_level=True,
    )

from vaara.server import schemas as S

TYPES_TS = Path(__file__).resolve().parents[1] / "clients" / "ts" / "src" / "types.ts"
CLIENT_TS = Path(__file__).resolve().parents[1] / "clients" / "ts" / "src" / "client.ts"


def _interface_fields(name: str) -> set[str]:
    """Field names declared on a TypeScript interface in types.ts."""
    source = TYPES_TS.read_text()
    match = re.search(
        rf"export interface {name} \{{(.*?)\n\}}", source, re.DOTALL,
    )
    assert match, f"types.ts has no interface {name}"
    return set(re.findall(r"^\s{2}(\w+)\??:", match.group(1), re.MULTILINE))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("VAARA_DB", str(tmp_path / "audit.db"))
    from vaara.server.app import create_app

    return TestClient(create_app())


# ── request models: what the client sends must be accepted ────────────────

def test_outcome_request_field_names_match_the_server():
    assert _interface_fields("OutcomeRequest") <= set(
        S.OutcomeRequest.model_fields
    ), (
        "types.ts declares a field OutcomeRequest does not accept; the model "
        "is extra=forbid, so sending it is a 422"
    )


def test_score_request_field_names_match_the_server():
    assert _interface_fields("ScoreRequest") <= set(S.ScoreRequest.model_fields)


def test_audit_event_request_field_names_match_the_server():
    assert _interface_fields("AuditEventRequest") <= set(
        S.AuditEventRequest.model_fields
    )


def test_score_request_can_carry_a_tenant():
    """Multi-tenant attribution is unreachable from TypeScript without it."""
    assert "tenant_id" in _interface_fields("ScoreRequest")


def test_the_documented_outcome_call_is_accepted(client):
    scored = client.post("/v1/score", json={
        "tool_name": "tx.transfer", "agent_id": "agent-007",
        "base_risk_score": 0.6,
    })
    assert scored.status_code == 200, scored.text
    action_id = scored.json()["action_id"]

    fields = _interface_fields("OutcomeRequest")
    payload = {"action_id": action_id, "outcome_severity": 0.1}
    for optional in fields - {"action_id", "outcome_severity"}:
        payload[optional] = "it went fine"
    response = client.post("/v1/score/outcome", json=payload)
    assert response.status_code != 422, response.text


def test_a_scored_request_with_a_tenant_is_accepted(client):
    response = client.post("/v1/score", json={
        "tool_name": "tx.transfer", "agent_id": "a", "tenant_id": "acme",
    })
    assert response.status_code == 200, response.text


# ── response models: what the client promises must arrive ─────────────────

def test_score_response_declares_every_field_the_server_sends(client):
    response = client.post("/v1/score", json={
        "tool_name": "tx.transfer", "agent_id": "agent-007",
    })
    assert response.status_code == 200, response.text
    declared = _interface_fields("ScoreResponse")
    missing = set(response.json()) - declared
    assert not missing, f"types.ts omits fields the server always sends: {missing}"


def test_score_response_declares_nothing_the_server_never_sends(client):
    response = client.post("/v1/score", json={
        "tool_name": "tx.transfer", "agent_id": "agent-007",
    })
    invented = _interface_fields("ScoreResponse") - set(response.json())
    assert not invented, (
        f"types.ts promises fields the server does not return: {invented}"
    )


def test_risk_block_declares_every_field_the_server_sends(client):
    response = client.post("/v1/score", json={
        "tool_name": "tx.transfer", "agent_id": "agent-007",
    })
    missing = set(response.json()["risk"]) - _interface_fields("RiskBlock")
    assert not missing, f"RiskBlock omits: {missing}"


def test_report_outcome_is_not_typed_as_returning_a_body():
    """/v1/score/outcome answers 204; the client returned undefined as {ok:true}."""
    source = CLIENT_TS.read_text()
    signature = re.search(r"async reportOutcome\([^)]*\):\s*([^{]+)\{", source)
    assert signature, "client.ts has no reportOutcome"
    assert "ok: true" not in signature.group(1), (
        "reportOutcome claims a body from a 204 endpoint, so callers reading "
        ".ok get a TypeError on undefined"
    )


def test_outcome_endpoint_really_answers_204(client):
    scored = client.post("/v1/score", json={
        "tool_name": "tx.transfer", "agent_id": "agent-007",
    })
    action_id = scored.json()["action_id"]
    response = client.post("/v1/score/outcome", json={
        "action_id": action_id, "outcome_severity": 0.1,
    })
    assert response.status_code == 204
    assert response.content == b""


# ── route coverage ────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "ScoreRequest", "OutcomeRequest", "AuditEventRequest", "ScoreResponse",
])
def test_openapi_yaml_matches_the_server_models(name):
    """docs/openapi.yaml is the spec integrators read before writing code."""
    yaml = pytest.importorskip("yaml")

    spec = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docs" / "openapi.yaml").read_text()
    )
    documented = set(
        (spec["components"]["schemas"][name].get("properties") or {})
    )
    actual = set(getattr(S, name).model_fields)
    assert documented == actual, (
        f"openapi.yaml {name}: only-in-doc={sorted(documented - actual)}, "
        f"only-in-code={sorted(actual - documented)}"
    )


def test_every_path_the_client_calls_exists_on_the_server(client):
    paths = set(re.findall(r'"(/v1/[^"`]*)"', CLIENT_TS.read_text()))
    paths |= {
        p.replace("${encodeURIComponent(actionId)}", "{action_id}")
        for p in re.findall(r"`(/v1/[^`]*)`", CLIENT_TS.read_text())
    }
    served = set(client.app.openapi()["paths"])
    assert paths <= served, f"client calls routes the server does not serve: {paths - served}"
