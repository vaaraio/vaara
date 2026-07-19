# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Server state container — scorer + audit trail + policy registry singletons."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from vaara.audit.trail import AuditTrail
from vaara.policy.controller import PolicyController
from vaara.policy.registry import PolicyRegistry
from vaara.scorer.adaptive import AdaptiveScorer


@dataclass
class _ActionInfo:
    agent_id: str
    tool_name: str
    predicted_risk: float
    signals: dict[str, float] = field(default_factory=dict)
    tenant_id: str = ""


class ServerState:
    """Holds scorer + audit-trail singletons. Lifetime = process lifetime."""

    def __init__(
        self,
        scorer: Optional[AdaptiveScorer] = None,
        audit: Optional[AuditTrail] = None,
        policy_controller: Optional[PolicyController] = None,
        policy_registry: Optional[PolicyRegistry] = None,
    ) -> None:
        if policy_controller is not None and policy_registry is not None:
            raise ValueError(
                "Pass either policy_controller (single-tenant legacy) or "
                "policy_registry (multi-tenant v0.40), not both. Mixing "
                "the two silently splits threshold sources between the "
                "default slot and per-tenant overrides.",
            )
        self.scorer = scorer or AdaptiveScorer()
        self.audit = audit or AuditTrail()
        # v0.40: a single PolicyRegistry holds all tenant policies. The
        # single-tenant entry point (`policy_controller=...`) lands in the
        # empty-string "" slot for back-compat with v0.39 callers.
        if policy_registry is None and policy_controller is not None:
            policy_registry = PolicyRegistry()
            policy_registry.register("", policy_controller)
        self.policy_registry = policy_registry
        self.policy_controller = policy_controller
        if policy_controller is not None:
            policy_controller.add_listener(self.scorer.apply_policy)
        elif policy_registry is not None:
            default = policy_registry.get("")
            if default is not None:
                default.add_listener(self.scorer.apply_policy)
                self.policy_controller = default
        # v0.40 per-tenant threshold dispatch: the scorer asks the
        # registry for the calling tenant's policy on every evaluate.
        # An exact-match miss falls back to the scorer-bound defaults
        # (which the default-slot listener keeps fresh on reload). We
        # use get_exact rather than get so the scorer's own default
        # path stays the single fallback channel.
        if policy_registry is not None:
            def _lookup(tid: str):
                ctrl = policy_registry.get_exact(tid)
                return ctrl.policy if ctrl is not None else None
            self.scorer.set_policy_lookup(_lookup)
        self._lock = threading.Lock()
        # action_id to info captured at score time so outcome reports can
        # feed the MWU update without the client having to resend context.
        self._actions: dict[str, _ActionInfo] = {}

    def remember_action(
        self,
        action_id: str,
        agent_id: str,
        tool_name: str,
        predicted_risk: float,
        signals: dict[str, float],
        tenant_id: str = "",
    ) -> None:
        with self._lock:
            self._actions[action_id] = _ActionInfo(
                agent_id=agent_id,
                tool_name=tool_name,
                predicted_risk=predicted_risk,
                signals=signals,
                tenant_id=tenant_id,
            )

    def lookup_action(self, action_id: str) -> Optional[_ActionInfo]:
        with self._lock:
            return self._actions.get(action_id)
