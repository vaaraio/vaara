"""Server state container — scorer + audit trail singletons."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

from vaara.audit.trail import AuditTrail
from vaara.scorer.adaptive import AdaptiveScorer


@dataclass
class _ActionInfo:
    agent_id: str
    tool_name: str
    predicted_risk: float
    signals: dict[str, float] = field(default_factory=dict)


class ServerState:
    """Holds scorer + audit-trail singletons. Lifetime = process lifetime."""

    def __init__(
        self,
        scorer: Optional[AdaptiveScorer] = None,
        audit: Optional[AuditTrail] = None,
    ) -> None:
        self.scorer = scorer or AdaptiveScorer()
        self.audit = audit or AuditTrail()
        self._lock = threading.Lock()
        # action_id → info captured at score time so outcome reports can
        # feed the MWU update without the client having to resend context.
        self._actions: dict[str, _ActionInfo] = {}

    def remember_action(
        self,
        action_id: str,
        agent_id: str,
        tool_name: str,
        predicted_risk: float,
        signals: dict[str, float],
    ) -> None:
        with self._lock:
            self._actions[action_id] = _ActionInfo(
                agent_id=agent_id,
                tool_name=tool_name,
                predicted_risk=predicted_risk,
                signals=signals,
            )

    def lookup_action(self, action_id: str) -> Optional[_ActionInfo]:
        with self._lock:
            return self._actions.get(action_id)
