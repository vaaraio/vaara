# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vaara policy dataclasses.

All types are frozen so a loaded Policy is a value object — passing it
around the pipeline is safe without defensive copying. Mapping fields
on Policy are wrapped in MappingProxyType in __post_init__ so callers
cannot mutate them in place either.
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import dataclass

from vaara.taxonomy.actions import (
    ActionCategory,
    BlastRadius,
    RegulatoryDomain,
    Reversibility,
    UrgencyClass,
)

SCHEMA_VERSION = "0.1"


class PolicyError(ValueError):
    """Raised when a policy document fails to parse or validate."""


@dataclass(frozen=True)
class Thresholds:
    """Risk-score thresholds. Below escalate = allow. Above deny = deny."""
    escalate: float
    deny: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.escalate <= 1.0:
            raise PolicyError(f"thresholds.escalate must be in [0,1], got {self.escalate}")
        if not 0.0 <= self.deny <= 1.0:
            raise PolicyError(f"thresholds.deny must be in [0,1], got {self.deny}")
        if self.escalate >= self.deny:
            raise PolicyError(
                f"thresholds.escalate ({self.escalate}) must be < deny ({self.deny})"
            )


@dataclass(frozen=True)
class ActionClassDef:
    """Declarative action-class definition — what a tool name classifies as."""
    name: str
    category: ActionCategory
    reversibility: Reversibility
    blast_radius: BlastRadius
    urgency: UrgencyClass
    regulatory: tuple[str, ...] = ()


@dataclass(frozen=True)
class SequencePattern:
    """A dangerous action sequence. If matched in window, risk gets boosted."""
    name: str
    pattern: tuple[str, ...]
    risk_boost: float
    window_seconds: int
    regulatory: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_boost <= 1.0:
            raise PolicyError(f"sequence {self.name!r} risk_boost must be in [0,1]")
        if self.window_seconds <= 0:
            raise PolicyError(f"sequence {self.name!r} window_seconds must be > 0")
        if not self.pattern:
            raise PolicyError(f"sequence {self.name!r} pattern must be non-empty")


@dataclass(frozen=True)
class EscalationRoute:
    """Escalation route. Empty `if_articles` = default fallback."""
    operator_group: str
    if_articles: tuple[str, ...] = ()


@dataclass(frozen=True)
class Policy:
    """Loaded, validated Vaara policy."""
    version: str
    domains: tuple[RegulatoryDomain, ...]
    action_classes: Mapping[str, ActionClassDef]
    thresholds_default: Thresholds
    thresholds_overrides: Mapping[str, Mapping[str, float]]
    sequences: tuple[SequencePattern, ...]
    escalation_routes: tuple[EscalationRoute, ...]

    def __post_init__(self) -> None:
        # Wrap mapping fields so callers cannot mutate Policy in place.
        # Nested override dicts are wrapped too so
        # `policy.thresholds_overrides["x"]["escalate"] = 0.9` raises.
        object.__setattr__(
            self, "action_classes",
            types.MappingProxyType(dict(self.action_classes)),
        )
        nested_frozen = {
            k: types.MappingProxyType(dict(v))
            for k, v in self.thresholds_overrides.items()
        }
        object.__setattr__(
            self, "thresholds_overrides",
            types.MappingProxyType(nested_frozen),
        )

    def threshold_for(self, action_class_name: str) -> Thresholds:
        """Resolve thresholds for a given action class, with override fallback.

        Override is partial — supplying just `deny` keeps the default
        `escalate`. This lets a deployer tighten one knob without
        re-stating the other.
        """
        override = self.thresholds_overrides.get(action_class_name, {})
        return Thresholds(
            escalate=override.get("escalate", self.thresholds_default.escalate),
            deny=override.get("deny", self.thresholds_default.deny),
        )

    def escalation_route_for(self, articles: set[str]) -> str:
        """Return operator group whose `if_articles` overlap with the input.

        Falls back to the default route (empty `if_articles`) if no match.
        Returns "on_call" if no default is defined either.
        """
        for route in self.escalation_routes:
            if route.if_articles and any(a in articles for a in route.if_articles):
                return route.operator_group
        for route in self.escalation_routes:
            if not route.if_articles:
                return route.operator_group
        return "on_call"
