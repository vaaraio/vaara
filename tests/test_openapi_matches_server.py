# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""docs/openapi.yaml describes the server that ships.

The spec calls itself authoritative, so what the server serves and what the
spec documents are compared mechanically: every route the server registers
is a path in the spec with the same methods, and every request or response
model that shares a name with a schema declares the same fields. The seam
audit of 2026-10-11 found `POST /v1/anchor` live and x402-gated for a year
with no entry in the spec, and `tenant_id` carried by the policy reload
request and response with no mention in the spec or the TypeScript types,
so the TypeScript client could not make the tenant-scoped reload the server
serves. The TypeScript README also pinned its coverage to v0.15.0 on a
2.10.0 package.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
pytest.importorskip("fastapi")

from vaara.server import create_app  # noqa: E402
from vaara.server import schemas as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPEC = yaml.safe_load((ROOT / "docs" / "openapi.yaml").read_text(encoding="utf-8"))
SCHEMAS = SPEC["components"]["schemas"]


def _served() -> dict[str, set[str]]:
    routes: dict[str, set[str]] = {}
    for route in create_app().routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", None)
        if not path.startswith("/v1/") or not methods:
            continue
        routes.setdefault(path, set()).update(m.lower() for m in methods if m != "HEAD")
    return routes


def test_every_served_route_is_in_the_spec():
    served = _served()
    documented = {path: set(ops) for path, ops in SPEC["paths"].items()}
    missing = {path: ops for path, ops in served.items() if path not in documented}
    assert not missing, f"served but not in docs/openapi.yaml: {missing}"
    for path, ops in served.items():
        assert ops <= documented[path], f"{path}: served {ops}, documented {documented[path]}"


def test_every_documented_route_is_served():
    served = _served()
    undocumented = [p for p in SPEC["paths"] if p not in served]
    assert not undocumented, f"documented but not served: {undocumented}"


def _models() -> dict[str, type]:
    from pydantic import BaseModel
    return {
        name: obj for name, obj in vars(S).items()
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel
    }


def _schema_for(name: str) -> dict | None:
    # The server's ErrorResponse is the spec's Error.
    return SCHEMAS.get({"ErrorResponse": "Error"}.get(name, name))


@pytest.mark.parametrize("name", sorted(n for n in _models() if _schema_for(n)))
def test_a_model_and_its_schema_declare_the_same_fields(name):
    model = _models()[name]
    schema = _schema_for(name)
    fields = set(model.model_fields)
    props = set(schema.get("properties", {}))
    assert fields == props, (
        f"{name}: server has {sorted(fields - props)} the spec lacks; "
        f"spec has {sorted(props - fields)} the server lacks"
    )
    always_sent = {
        field for field, info in model.model_fields.items() if info.is_required()
    }
    required = set(schema.get("required", []))
    assert always_sent <= required, (
        f"{name}: the server requires {sorted(always_sent - required)}; the spec does not"
    )


def test_the_anchor_route_and_its_bodies_are_documented():
    assert "/v1/anchor" in SPEC["paths"]
    operation = SPEC["paths"]["/v1/anchor"]["post"]
    assert operation["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/AnchorRequest")
    assert set(operation["responses"]) >= {"200", "400", "402", "502", "503"}
    assert operation["responses"]["402"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/X402Challenge")


def test_the_tenant_scoped_reload_is_documented_and_typed():
    assert "tenant_id" in SCHEMAS["PolicyReloadRequest"]["properties"]
    assert "tenant_id" in SCHEMAS["PolicyReloadResponse"]["properties"]
    types = (ROOT / "clients" / "ts" / "src" / "types.ts").read_text(encoding="utf-8")
    request = re.search(r"export interface PolicyReloadRequest \{(.*?)\n\}", types, re.S)
    response = re.search(r"export interface PolicyReloadResponse \{(.*?)\n\}", types, re.S)
    assert request and "tenant_id?: string" in request.group(1)
    assert response and "tenant_id: string" in response.group(1)


def test_the_ts_readme_pins_no_stale_version():
    readme = (ROOT / "clients" / "ts" / "README.md").read_text(encoding="utf-8")
    pinned = re.findall(r"\bv\d+\.\d+\.(?:\d+|x)\b", readme)
    assert not pinned, (
        f"clients/ts/README.md names a version ({pinned}); the package version is the "
        f"statement, and prose pinned to one release goes stale at the next"
    )
