"""Internal JSON-safe coercion for audit-trail hashing.

Real agent tools routinely pass datetimes, bytes, dataclasses, and
sets as tool arguments. The audit-trail hash requires strict JSON,
so sanitize at the pipeline/integration boundary rather than mutating
the hash input (which would break chain verification).

Recursion is bounded to guard against circular refs.
"""

from __future__ import annotations

import json as _json
import math
from typing import Any


def json_safe(value: Any, _depth: int = 0) -> Any:
    if _depth >= 10:
        return f"<truncated depth>{type(value).__name__}</truncated>"
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # Strict JSON (RFC 8259) rejects NaN/+Inf/-Inf. Scrub at the sanitize
        # choke point so audit hash input, JSONL exports, and MCP wire all
        # see the same canonical form. Mirrors mcp_server._scrub_nonfinite.
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return f"<bytes len={len(value)}>"
    if isinstance(value, (list, tuple)):
        return [json_safe(v, _depth + 1) for v in value]
    if isinstance(value, (set, frozenset)):
        # Sort after recursive sanitisation to guarantee canonical
        # ordering. Raw set iteration is PYTHONHASHSEED-dependent for
        # str/bytes/complex-hash element types — two processes writing
        # the same {"tags": {"risky","high"}} context would produce
        # different JSON, different record hashes, and break cross-
        # process reproducibility of the regulator evidence trail.
        # Sort by json-dumped form for a stable total ordering across
        # mixed-type elements.
        import json as _json
        return sorted(
            (json_safe(v, _depth + 1) for v in value),
            key=lambda x: _json.dumps(x, sort_keys=True, default=str),
        )
    if isinstance(value, dict):
        return {str(k): json_safe(v, _depth + 1) for k, v in value.items()}
    # Fallback for arbitrary objects. `repr(value)` is the obvious choice
    # but default `__repr__` bakes the memory address into the string
    # (`<pkg.Foo object at 0x7f...>`) — two live instances of the same
    # class produce different audit hashes, and the hash is not
    # reproducible from original inputs (regulator cannot recompute).
    # Prefer a canonical type-name form; if the object has a custom
    # __repr__ (dataclass, Enum, pydantic), that's already deterministic
    # and more informative, so use it. Heuristic: default object __repr__
    # is identified by the "<...at 0x" signature.
    try:
        r = repr(value)
    except Exception:
        return f"<unreprable {type(value).__name__}>"
    if r.startswith("<") and " at 0x" in r and r.endswith(">"):
        return f"<{type(value).__module__}.{type(value).__qualname__}>"
    return r


def _scrub_nonfinite(obj: Any) -> Any:
    # Recursive NaN/+Inf/-Inf scrub. `json_safe` already does this at the
    # pipeline/integration ingress boundary, but export paths that accept
    # pre-constructed `AuditRecord` objects (direct callers, DB reloads
    # of legacy records, trace-gen JSONL replays) bypass that guard — so
    # strict_json_dumps below runs it again at the wire boundary. Cheap
    # idempotent pass; matches the sqlite_backend helper added in Loop 36.
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _scrub_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_nonfinite(v) for v in obj]
    return obj


def strict_json_dumps(obj: Any, **kwargs: Any) -> str:
    """Strict-JSON (RFC 8259) serialiser used on every regulator-facing
    wire boundary. Scrubs non-finite floats to None, then dumps with
    allow_nan=False so any residual non-finite raises instead of emitting
    `NaN` / `Infinity` tokens that Go encoding/json, Rust serde_json,
    strict Node JSON.parse, and most JSON-RPC validators reject."""
    kwargs.setdefault("default", str)
    kwargs["allow_nan"] = False
    return _json.dumps(_scrub_nonfinite(obj), **kwargs)
