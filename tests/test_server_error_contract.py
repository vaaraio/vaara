# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A malformed request is answered the way docs/openapi.yaml says.

The spec documents ``400`` with the ``Error`` body for a bad request on
``/v1/score``, ``/v1/score/outcome``, ``/v1/audit/events`` and
``/v1/policy/reload``, and the TypeScript client maps ``error.code`` to
``VaaraError.code`` (``invalid_request`` among them). Without a handler for
FastAPI's request validation every malformed body answered ``422
{"detail": [...]}``, so the published contract was false on four routes and
the TypeScript client saw a generic ``http_error``. The generated OpenAPI
said nothing about error bodies either, since no route declared one.
"""

from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient

    from vaara.server import create_app
except ImportError:
    pytest.skip(
        "server extra not installed (pip install 'vaara[server]')",
        allow_module_level=True,
    )

from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _spec():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((ROOT / "docs" / "openapi.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def client():
    return TestClient(create_app())


MALFORMED = [
    ("/v1/score", {"tool_name": "x"}),                         # agent_id missing
    ("/v1/score", {"tool_name": "x", "agent_id": "a", "agent_confidence": 1.5}),
    ("/v1/score/outcome", {"action_id": "a", "outcome_severity": "high"}),
    ("/v1/audit/events", {"event_type": "action_requested"}),  # action_id missing
    ("/v1/policy/reload", {"path": 7}),
]


@pytest.mark.parametrize("path,body", MALFORMED)
def test_a_malformed_body_answers_400_invalid_request(client, path, body):
    r = client.post(path, json=body)
    assert r.status_code == 400, r.text
    error = r.json()["error"]
    assert error["code"] == "invalid_request"
    assert isinstance(error["message"], str) and error["message"]
    assert isinstance(error["details"]["errors"], list) and error["details"]["errors"]
    assert "detail" not in r.json()


def test_a_body_that_is_not_json_answers_400_invalid_request(client):
    r = client.post("/v1/score", content=b"{not json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 400, r.text
    assert r.json()["error"]["code"] == "invalid_request"


def test_the_validation_message_names_the_field(client):
    r = client.post("/v1/score", json={"tool_name": "x"})
    error = r.json()["error"]
    assert "agent_id" in error["message"]
    assert any("agent_id" in str(e.get("loc", [])) for e in error["details"]["errors"])


# ── The generated OpenAPI declares the error bodies the spec documents ───

def _generated():
    return create_app().openapi()


def _error_schema_name(generated, path, method, status):
    response = generated["paths"][path][method]["responses"][status]
    ref = response["content"]["application/json"]["schema"]["$ref"]
    return ref.rsplit("/", 1)[-1]


@pytest.mark.parametrize("path", [
    "/v1/score", "/v1/score/outcome", "/v1/audit/events", "/v1/policy/reload",
])
def test_the_generated_openapi_declares_400_with_the_error_body(path):
    generated = _generated()
    assert _error_schema_name(generated, path, "post", "400") == "ErrorResponse"
    schema = generated["components"]["schemas"]["ErrorResponse"]
    assert schema["required"] == ["error"]
    body = generated["components"]["schemas"]["ErrorBody"]
    assert {"code", "message"} <= set(body["required"])


def test_the_generated_openapi_documents_every_error_status_the_spec_does():
    """Every non-2xx status docs/openapi.yaml documents for a route the
    server serves is declared on that route with the error body, so the
    generated schema and the published one agree about failures."""
    generated = _generated()
    for path, methods in _spec()["paths"].items():
        if path not in generated["paths"]:
            continue
        for method, operation in methods.items():
            for status, response in operation.get("responses", {}).items():
                if status.startswith("2"):
                    continue
                ref = response.get("content", {}).get("application/json", {}).get(
                    "schema", {}).get("$ref", "")
                if not ref.endswith("/Error"):
                    continue
                assert _error_schema_name(generated, path, method, status) == "ErrorResponse", (
                    f"{method.upper()} {path} {status}"
                )


def test_the_generated_openapi_no_longer_promises_a_422_detail_body():
    """FastAPI adds a 422 HTTPValidationError to every route with a body. The
    server answers 400 Error instead, so the generated schema must not keep
    promising the body it never sends. /v1/policy/reload keeps its own 422:
    policy_invalid, with the Error body."""
    generated = _generated()
    for path, methods in generated["paths"].items():
        for method, operation in methods.items():
            response = operation["responses"].get("422")
            if response is None:
                continue
            ref = response["content"]["application/json"]["schema"]["$ref"]
            assert not ref.endswith("HTTPValidationError"), f"{method.upper()} {path}"
