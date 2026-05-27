"""Hot policy reload.

The controller owns the live ``Policy`` and the listeners that need to
re-bind to its fields when it changes. ``reload(source)`` parses and
validates the new document before any listener runs; if validation
fails, the old policy stays in place. If validation succeeds, the swap
plus notification runs under a single write lock so listeners observe
the same generation of the policy.

Listeners (e.g. ``AdaptiveScorer.apply_policy``) are responsible for
rebinding their own internal fields atomically under their own lock.
In-flight ``evaluate`` calls that already read the old thresholds keep
running against the old thresholds; the next call sees the new ones.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

from vaara.policy.loader import from_dict, from_json, from_yaml
from vaara.policy.schema import Policy, PolicyError


PolicyListener = Callable[[Policy], None]


@dataclass(frozen=True)
class ReloadResult:
    """Outcome of a single ``PolicyController.reload`` call."""
    version: int
    thresholds_default_escalate: float
    thresholds_default_deny: float
    sequence_count: int
    action_class_count: int
    escalation_route_count: int


class PolicyController:
    """Holds the live ``Policy`` and atomically applies replacements."""

    def __init__(self, policy: Policy) -> None:
        self._policy = policy
        self._version = 1
        self._listeners: list[PolicyListener] = []
        self._lock = threading.RLock()

    @property
    def policy(self) -> Policy:
        with self._lock:
            return self._policy

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def add_listener(self, listener: PolicyListener) -> None:
        """Register a callable invoked on every successful reload.

        The new policy is also applied to the listener immediately so a
        component registered after construction picks up the current
        state without a separate manual call.
        """
        with self._lock:
            self._listeners.append(listener)
            listener(self._policy)

    def reload(
        self, source: Union[str, Path, dict, Policy], *, format: Optional[str] = None
    ) -> ReloadResult:
        """Parse, validate, and apply a new policy.

        ``format`` may be ``"json"`` or ``"yaml"`` to force the parser.
        When omitted, ``.yaml``/``.yml`` paths use the YAML loader, dicts
        bypass parsing, and everything else goes through JSON.

        Already-validated ``Policy`` instances may be passed directly; the
        registry path (``vaara.policy.registry.PolicyRegistry``) uses this
        to swap a per-tenant policy that was parsed in bulk.

        Raises ``PolicyError`` if the source is malformed; in that case
        the previously loaded policy remains live.
        """
        new_policy = source if isinstance(source, Policy) else _load(source, format)
        with self._lock:
            self._policy = new_policy
            self._version += 1
            for fn in self._listeners:
                fn(new_policy)
            return ReloadResult(
                version=self._version,
                thresholds_default_escalate=new_policy.thresholds_default.escalate,
                thresholds_default_deny=new_policy.thresholds_default.deny,
                sequence_count=len(new_policy.sequences),
                action_class_count=len(new_policy.action_classes),
                escalation_route_count=len(new_policy.escalation_routes),
            )


def _load(source: Union[str, Path, dict], fmt: Optional[str]) -> Policy:
    if isinstance(source, dict):
        return from_dict(source)
    if fmt == "yaml":
        return from_yaml(source)
    if fmt == "json":
        return from_json(source)
    if isinstance(source, Path):
        if source.suffix in (".yaml", ".yml"):
            return from_yaml(source)
        return from_json(source)
    # str: treat as path if it looks like one; otherwise JSON content.
    if isinstance(source, str):
        if source.lstrip().startswith("{"):
            return from_json(source)
        path = Path(source)
        if path.suffix in (".yaml", ".yml"):
            return from_yaml(path)
        return from_json(path)
    raise PolicyError(f"unsupported policy source type: {type(source).__name__}")
