# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The engine's decision state, and the digest a decision record carries for it.

A receipt that says "the policy decided" without saying which policy cannot be
checked against anything. The pipeline describes what it decides by (the Vaara
version, enforce or shadow, and the scorer's configured state) as one JSON
object, ``vaara.engine-policy/v0``, and every decision record carries
``policy_digest``, the SHA-256 of its RFC 8785 form. The first decision a
pipeline records, and the first after the state changes, also carries the
state itself as ``policy_state``, so the trail explains its own digests.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from typing import Any

SCHEMA = "vaara.engine-policy/v0"


def jsonable(value: Any) -> Any:
    """``value`` as plain JSON: dataclasses and mappings as objects, enums by
    value, sequences as lists, sets sorted, anything else by its type name."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, enum.Enum):
        return jsonable(value.value)
    if isinstance(value, (set, frozenset)):
        return sorted((jsonable(v) for v in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return type(value).__name__


def engine_state(*, version: str, enforce: bool, scorer: Any, authority: Any = None,
                 require_delegatee_binding: bool = False,
                 unknown_parent: str = "deny",
                 grantless_child: str = "deny") -> dict[str, Any]:
    """The ``vaara.engine-policy/v0`` object for a pipeline."""
    describe = getattr(scorer, "policy_state", None)
    scorer_state = describe() if callable(describe) else {"scorer": type(scorer).__name__}
    state: dict[str, Any] = {
        "schema": SCHEMA,
        "vaara": version,
        "enforce": bool(enforce),
        "scorer": scorer_state,
    }
    if authority is not None:
        state["authority"] = jsonable(authority)
    # Present only when on, so every digest taken without it stays the same.
    if require_delegatee_binding:
        state["require_delegatee_binding"] = True
    # Same rule: only the non-default answer for an unknown parent grant.
    if unknown_parent != "deny":
        state["unknown_parent"] = unknown_parent
    if grantless_child != "deny":
        state["grantless_child"] = grantless_child
    return state


def digest(state: dict[str, Any]) -> str:
    """``sha256:`` and the SHA-256 of the state's RFC 8785 form."""
    from vaara.attestation._decision_binding import policy_digest

    return policy_digest(state)
