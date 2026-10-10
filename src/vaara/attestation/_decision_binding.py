# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Binding and rationale digests for the SEP-2828 decision record.

Object digests are over RFC 8785 JCS bytes; the intent digest is over the
UTF-8 bytes of the declared intent, and the binding digest over the joined
digests and verdict, as each function below states.

The digests here are the commitments the zero-knowledge decisionProof opens
against: the policy, the declared intent, the evaluation inputs, and the
resulting binding.
"""

from __future__ import annotations

import hashlib
from typing import Any


def _canonical_bytes(obj: dict[str, Any]) -> bytes:
    """RFC 8785 JCS bytes.

    ``rfc8785`` is a base dependency. A digest written over anything else
    would not recompute in another implementation, so a missing library is
    an error here rather than a quiet fallback to sorted JSON.
    """
    try:
        import rfc8785
    except ImportError as exc:  # pragma: no cover - a broken install
        raise ImportError(
            "decision binding digests need rfc8785 (RFC 8785 JCS); "
            "reinstall vaara with its dependencies"
        ) from exc
    return rfc8785.dumps(obj)


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def policy_digest(canonical_policy: dict[str, Any]) -> str:
    return _sha(_canonical_bytes(canonical_policy))


def intent_digest(declared_intent: str) -> str:
    return _sha(declared_intent.encode("utf-8"))


def inputs_digest(canonical_inputs: dict[str, Any]) -> str:
    return _sha(_canonical_bytes(canonical_inputs))


def binding_digest(policy_d: str, intent_d: str, inputs_d: str, verdict: str) -> str:
    """The value the proof's public input opens against. Domain-separated join of
    the three input digests and the verdict, so no field can be shifted into
    another without changing the result."""
    preimage = "\x1f".join([policy_d, intent_d, inputs_d, verdict]).encode("utf-8")
    return _sha(preimage)


def build_binding(
    canonical_policy: dict[str, Any],
    declared_intent: str,
    canonical_inputs: dict[str, Any],
    verdict: str,
) -> dict[str, str]:
    p = policy_digest(canonical_policy)
    i = intent_digest(declared_intent)
    x = inputs_digest(canonical_inputs)
    return {
        "policyDigest": p,
        "intentDigest": i,
        "inputsDigest": x,
        "bindingDigest": binding_digest(p, i, x, verdict),
    }


def build_rationale(
    rule: str,
    reason: str,
    declared_intent: str,
    intent_satisfied: bool | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rule": rule,
        "reason": reason,
        "declaredIntent": declared_intent,
    }
    if intent_satisfied is not None:
        out["intentSatisfied"] = intent_satisfied
    return out
