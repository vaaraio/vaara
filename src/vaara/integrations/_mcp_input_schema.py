# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check tools/call arguments against the tool's advertised inputSchema.

An MCP server publishes an ``inputSchema`` (JSON Schema) for every tool in
its ``tools/list`` response. The proxy keeps the schemas it saw and checks
each ``tools/call`` against them before the call reaches the risk scorer,
so arguments outside the declared shape are refused before execution and
the refusal names the parameter that failed.

The checker covers the keywords that declare a parameter's shape: ``type``,
``required``, ``properties``, ``additionalProperties``, ``enum``, ``const``,
``minimum``, ``maximum``, ``exclusiveMinimum``, ``exclusiveMaximum``,
``minLength``, ``maxLength``, ``pattern``, ``items``, ``minItems`` and
``maxItems``. Composition and references (``anyOf``, ``oneOf``, ``allOf``,
``not``, ``$ref``, ``if``/``then``) are not evaluated: a value under one of
those is accepted as far as that keyword goes. ``unchecked_keywords``
reports which of them a schema uses, so the record can say the check was
partial rather than imply it was complete.
"""

from __future__ import annotations

import re
from typing import Any

_MAX_DEPTH = 32
_MAX_VIOLATIONS = 20

_UNCHECKED = frozenset({
    "anyOf", "oneOf", "allOf", "not", "$ref", "if", "then", "else",
    "patternProperties", "dependentRequired", "dependentSchemas",
    "prefixItems", "contains", "uniqueItems", "multipleOf", "format",
    "propertyNames", "minProperties", "maxProperties",
})


def _type_ok(value: Any, expected: str) -> bool:
    # bool is an int subclass in Python; JSON Schema keeps them apart.
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer()
        )
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    # An unknown type name declares nothing checkable.
    return True


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check(value: Any, schema: Any, path: str, out: list[str], depth: int) -> None:
    if len(out) >= _MAX_VIOLATIONS or depth > _MAX_DEPTH:
        return
    if schema is False:
        out.append(f"{path}: no value is allowed here")
        return
    if not isinstance(schema, dict):
        return

    expected = schema.get("type")
    if isinstance(expected, str):
        types = [expected]
    elif isinstance(expected, list):
        types = [t for t in expected if isinstance(t, str)]
    else:
        types = []
    if types and not any(_type_ok(value, t) for t in types):
        out.append(f"{path}: expected {' or '.join(types)}, got {_json_type(value)}")
        return

    if "const" in schema and value != schema["const"]:
        out.append(f"{path}: must equal {schema['const']!r}")
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        shown = ", ".join(repr(e) for e in enum[:8])
        out.append(f"{path}: must be one of {shown}")

    if _is_number(value):
        lo, hi = schema.get("minimum"), schema.get("maximum")
        xlo, xhi = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
        if _is_number(lo) and value < lo:
            out.append(f"{path}: {value} is below the minimum {lo}")
        if _is_number(hi) and value > hi:
            out.append(f"{path}: {value} is above the maximum {hi}")
        if _is_number(xlo) and value <= xlo:
            out.append(f"{path}: {value} must be greater than {xlo}")
        if _is_number(xhi) and value >= xhi:
            out.append(f"{path}: {value} must be less than {xhi}")

    if isinstance(value, str):
        lo, hi = schema.get("minLength"), schema.get("maxLength")
        if isinstance(lo, int) and len(value) < lo:
            out.append(f"{path}: shorter than {lo} characters")
        if isinstance(hi, int) and len(value) > hi:
            out.append(f"{path}: longer than {hi} characters")
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                if re.search(pattern, value) is None:
                    out.append(f"{path}: does not match the pattern {pattern!r}")
            except re.error:
                # A pattern Python cannot compile declares nothing checkable.
                pass

    if isinstance(value, list):
        lo, hi = schema.get("minItems"), schema.get("maxItems")
        if isinstance(lo, int) and len(value) < lo:
            out.append(f"{path}: fewer than {lo} items")
        if isinstance(hi, int) and len(value) > hi:
            out.append(f"{path}: more than {hi} items")
        items = schema.get("items")
        if isinstance(items, (dict, bool)):
            for i, item in enumerate(value):
                _check(item, items, f"{path}[{i}]", out, depth + 1)

    if isinstance(value, dict):
        props = schema.get("properties")
        props = props if isinstance(props, dict) else {}
        required = schema.get("required")
        if isinstance(required, list):
            for name in required:
                if isinstance(name, str) and name not in value:
                    out.append(f"{_join(path, name)}: required and missing")
        extra = schema.get("additionalProperties", True)
        for name, sub in value.items():
            if name in props:
                _check(sub, props[name], _join(path, name), out, depth + 1)
            elif extra is False:
                out.append(f"{_join(path, name)}: not a declared parameter")
            elif isinstance(extra, dict):
                _check(sub, extra, _join(path, name), out, depth + 1)


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def check_arguments(arguments: dict, schema: Any) -> list[str]:
    """Violations of ``schema`` by ``arguments``, empty when they conform.

    Each entry names the parameter path and what is wrong with it. At most
    twenty are returned.
    """
    out: list[str] = []
    _check(arguments, schema, "", out, 0)
    return [v if not v.startswith(": ") else "arguments" + v for v in out]


def unchecked_keywords(schema: Any) -> list[str]:
    """Keywords in ``schema`` this checker does not evaluate, sorted."""
    found: set[str] = set()

    def walk(node: Any, depth: int) -> None:
        if depth > _MAX_DEPTH or not isinstance(node, dict):
            return
        found.update(k for k in node if k in _UNCHECKED)
        for key in ("properties",):
            sub = node.get(key)
            if isinstance(sub, dict):
                for child in sub.values():
                    walk(child, depth + 1)
        for key in ("items", "additionalProperties"):
            walk(node.get(key), depth + 1)

    walk(schema, 0)
    return sorted(found)
