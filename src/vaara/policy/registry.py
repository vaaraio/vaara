"""Per-tenant policy registry.

A ``PolicyRegistry`` owns one ``PolicyController`` per tenant_id, with the
empty string ("") reserved for the default / fallback policy used when a
request carries no tenant scope or no tenant-specific policy is loaded.

Filename convention for ``load_directory``:

* ``default.yaml`` / ``default.json`` → tenant_id ""
* ``TENANT.yaml`` / ``TENANT.json``   → tenant_id "TENANT"

This is the v0.40 multi-tenant policy plane. Single-tenant deployments
keep using ``vaara serve --policy PATH``, which lands in the "" slot.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Union

from vaara.policy.controller import PolicyController, ReloadResult
from vaara.policy.loader import from_json, from_yaml
from vaara.policy.schema import Policy, PolicyError


_DEFAULT_TENANT = ""
_POLICY_SUFFIXES = (".yaml", ".yml", ".json")


def _filename_to_tenant(stem: str) -> str:
    return "" if stem.lower() == "default" else stem


def _load_path(path: Path) -> Policy:
    if path.suffix in (".yaml", ".yml"):
        return from_yaml(path)
    return from_json(path)


class PolicyRegistry:
    """Holds one PolicyController per tenant. Thread-safe."""

    def __init__(self) -> None:
        self._controllers: dict[str, PolicyController] = {}
        self._lock = threading.RLock()

    def __contains__(self, tenant_id: str) -> bool:
        with self._lock:
            return tenant_id in self._controllers

    def tenants(self) -> list[str]:
        with self._lock:
            return sorted(self._controllers.keys())

    def get(self, tenant_id: str) -> Optional[PolicyController]:
        """Return the controller for ``tenant_id``, falling back to the
        default ("") slot. Returns None if neither is registered.
        """
        with self._lock:
            if tenant_id in self._controllers:
                return self._controllers[tenant_id]
            return self._controllers.get(_DEFAULT_TENANT)

    def get_exact(self, tenant_id: str) -> Optional[PolicyController]:
        """Return only an exact-match controller, no default fallback."""
        with self._lock:
            return self._controllers.get(tenant_id)

    def register(self, tenant_id: str, controller: PolicyController) -> None:
        with self._lock:
            self._controllers[tenant_id] = controller

    def reload(
        self,
        tenant_id: str,
        source: Union[str, Path, dict],
        *,
        format: Optional[str] = None,
    ) -> ReloadResult:
        """Reload one tenant's policy. Creates the slot if missing."""
        with self._lock:
            controller = self._controllers.get(tenant_id)
            if controller is None:
                policy = _materialise(source, format)
                controller = PolicyController(policy)
                self._controllers[tenant_id] = controller
                return ReloadResult(
                    version=controller.version,
                    thresholds_default_escalate=policy.thresholds_default.escalate,
                    thresholds_default_deny=policy.thresholds_default.deny,
                    sequence_count=len(policy.sequences),
                    action_class_count=len(policy.action_classes),
                    escalation_route_count=len(policy.escalation_routes),
                )
            return controller.reload(source, format=format)

    def load_directory(self, directory: Union[str, Path]) -> list[str]:
        """Load every ``*.yaml``/``*.yml``/``*.json`` file in ``directory`` as
        one tenant's policy. Returns the list of tenant_ids loaded.

        Raises ``PolicyError`` if any file fails to parse — partial loads
        are not allowed, the registry stays untouched.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise PolicyError(f"policy directory does not exist: {directory}")

        candidates: list[tuple[str, Path]] = []
        for entry in sorted(directory.iterdir()):
            if entry.suffix not in _POLICY_SUFFIXES or not entry.is_file():
                continue
            candidates.append((_filename_to_tenant(entry.stem), entry))

        if not candidates:
            raise PolicyError(f"policy directory holds no policy files: {directory}")

        parsed: list[tuple[str, Policy]] = [
            (tenant_id, _load_path(path)) for tenant_id, path in candidates
        ]
        with self._lock:
            for tenant_id, policy in parsed:
                existing = self._controllers.get(tenant_id)
                if existing is None:
                    self._controllers[tenant_id] = PolicyController(policy)
                else:
                    existing.reload(policy)
        return [tenant_id for tenant_id, _ in parsed]


def _materialise(source: Union[str, Path, dict], fmt: Optional[str]) -> Policy:
    """Mirror PolicyController._load for the new-tenant fast path."""
    from vaara.policy.loader import from_dict
    if isinstance(source, dict):
        return from_dict(source)
    if fmt == "yaml":
        return from_yaml(source)
    if fmt == "json":
        return from_json(source)
    if isinstance(source, Path):
        return _load_path(source)
    if isinstance(source, str):
        if source.lstrip().startswith("{"):
            return from_json(source)
        return _load_path(Path(source))
    raise PolicyError(f"unsupported policy source type: {type(source).__name__}")


__all__ = ["PolicyRegistry"]
