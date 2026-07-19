# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vaara policy modes — preset threshold bundles.

Each mode is a preset operating point for the risk thresholds Vaara uses to
decide allow / escalate / deny. Modes do not change Vaara's primitives; they
bundle a default-threshold choice that maps to a concrete cost / risk / latency
profile, in the spirit of a CPU/GPU power profile.

Four built-in modes:

- eco: tight deny threshold catches borderline risk early, cutting agent loops
  short. Pairs with regex-first gating (sub-millisecond) to short-circuit
  before any model forward pass. Lowest watts per call.
- balanced: Vaara's default operating point (0.55 / 0.85). Current behaviour
  when no mode is selected.
- performance: loose thresholds let more through. For high-throughput internal
  pipelines where the deployer keeps tight action-class overrides on the few
  classes that matter (financial, infrastructure).
- strict: lowest deny threshold, escalate-on-doubt. For incident response,
  regulator audit prep, or production lockdown windows.

A mode emits a minimal valid Policy document — ``thresholds.default`` tuned
per mode, ``action_classes`` / ``sequences`` / ``escalation`` left for the
deployer to add. The emitted document round-trips through
``vaara.policy.from_dict`` / ``from_json`` / ``from_yaml`` like any other
policy artifact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

from vaara.policy.schema import SCHEMA_VERSION


@dataclass(frozen=True)
class Mode:
    """A preset operating point for Vaara risk thresholds."""
    name: str
    escalate: float
    deny: float
    description: str
    watt_profile: str


_MODES: tuple[Mode, ...] = (
    Mode(
        name="eco",
        escalate=0.40,
        deny=0.60,
        description=(
            "Tight deny threshold; cuts agent loops short on borderline risk. "
            "Pair with regex-first gating to short-circuit before any model pass."
        ),
        watt_profile=(
            "Lowest. Fewer downstream LLM calls per blocked tool invocation; "
            "more requests resolved at the regex gate."
        ),
    ),
    Mode(
        name="balanced",
        escalate=0.55,
        deny=0.85,
        description="Default operating point. Current Vaara behaviour.",
        watt_profile="Baseline. No optimization either direction.",
    ),
    Mode(
        name="performance",
        escalate=0.70,
        deny=0.92,
        description=(
            "Loose thresholds; lets more through. For high-throughput "
            "pipelines where the deployer keeps tight action-class overrides "
            "on the few classes that matter."
        ),
        watt_profile=(
            "Higher per call. Fewer early denials means more downstream tool "
            "calls and more LLM round trips."
        ),
    ),
    Mode(
        name="strict",
        escalate=0.30,
        deny=0.55,
        description=(
            "Escalate-on-doubt. For incident response, audit prep, or "
            "production lockdown windows."
        ),
        watt_profile=(
            "Mixed. More escalations (cheap, human-bound) but rarely runs "
            "the full scorer ensemble before a verdict."
        ),
    ),
)

_BY_NAME: Mapping[str, Mode] = {m.name: m for m in _MODES}


def available_modes() -> tuple[str, ...]:
    """Return the names of the built-in modes in canonical order."""
    return tuple(m.name for m in _MODES)


def get_mode(name: str) -> Mode:
    """Return the Mode by name. Raises KeyError on unknown name."""
    try:
        return _BY_NAME[name]
    except KeyError as e:
        valid = ", ".join(repr(n) for n in available_modes())
        raise KeyError(
            f"unknown mode {name!r}, expected one of [{valid}]"
        ) from e


def to_policy_dict(mode: Mode) -> dict:
    """Return a minimal Vaara policy dict for ``mode``.

    Round-trips through ``vaara.policy.from_dict``. Action classes, sequences,
    and escalation routes are left empty for the deployer to fill in.
    """
    return {
        "version": SCHEMA_VERSION,
        "domains": ["eu_ai_act"],
        "action_classes": {},
        "thresholds": {
            "default": {"escalate": mode.escalate, "deny": mode.deny},
        },
    }


def emit_json(name: str, *, indent: int = 2) -> str:
    """Emit a mode's policy as a JSON string with a trailing newline."""
    return json.dumps(to_policy_dict(get_mode(name)), indent=indent) + "\n"


def emit_yaml(name: str) -> str:
    """Emit a mode's policy as YAML. Requires ``vaara[yaml]``."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "emit_yaml requires the [yaml] extra. "
            "Install with: pip install 'vaara[yaml]'"
        ) from e
    dumped: str = yaml.safe_dump(
        to_policy_dict(get_mode(name)),
        sort_keys=False,
    )
    return dumped
