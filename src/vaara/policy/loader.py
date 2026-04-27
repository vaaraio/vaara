"""Policy loaders. JSON is core (stdlib). YAML requires the `vaara[yaml]` extra."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from vaara.taxonomy.actions import (
    ActionCategory,
    BlastRadius,
    RegulatoryDomain,
    Reversibility,
    UrgencyClass,
)
from vaara.policy.schema import (
    SCHEMA_VERSION,
    ActionClassDef,
    EscalationRoute,
    Policy,
    PolicyError,
    SequencePattern,
    Thresholds,
)


def from_dict(data: dict) -> Policy:
    """Validate and convert a policy dict into a Policy instance.

    Raises PolicyError with a field path on any validation failure.
    """
    if not isinstance(data, dict):
        raise PolicyError(f"policy must be a mapping, got {type(data).__name__}")

    version = data.get("version")
    if version != SCHEMA_VERSION:
        raise PolicyError(
            f"policy version {version!r} not supported (expected {SCHEMA_VERSION!r})"
        )

    domains = tuple(
        _coerce_enum(d, RegulatoryDomain, f"domains[{i}]")
        for i, d in enumerate(data.get("domains") or [])
    )

    action_classes: dict[str, ActionClassDef] = {}
    for name, raw in (data.get("action_classes") or {}).items():
        if not isinstance(raw, dict):
            raise PolicyError(f"action_classes.{name}: must be a mapping")
        action_classes[name] = ActionClassDef(
            name=name,
            category=_coerce_enum(
                raw.get("category"), ActionCategory, f"action_classes.{name}.category"
            ),
            reversibility=_coerce_enum(
                raw.get("reversibility"), Reversibility, f"action_classes.{name}.reversibility"
            ),
            blast_radius=_coerce_enum(
                raw.get("blast_radius"), BlastRadius, f"action_classes.{name}.blast_radius"
            ),
            urgency=_coerce_enum(
                raw.get("urgency"), UrgencyClass, f"action_classes.{name}.urgency"
            ),
            regulatory=tuple(raw.get("regulatory") or ()),
        )

    thr_block = data.get("thresholds") or {}
    default_block = thr_block.get("default") or {"escalate": 0.55, "deny": 0.85}
    thresholds_default = Thresholds(
        escalate=float(default_block.get("escalate", 0.55)),
        deny=float(default_block.get("deny", 0.85)),
    )
    thresholds_overrides = {
        k: {kk: float(vv) for kk, vv in (v or {}).items()}
        for k, v in thr_block.items()
        if k != "default"
    }

    sequences = tuple(
        SequencePattern(
            name=name,
            pattern=tuple(raw.get("pattern") or ()),
            risk_boost=float(raw.get("risk_boost", 0.0)),
            window_seconds=int(raw.get("window_seconds", 60)),
            regulatory=tuple(raw.get("regulatory") or ()),
        )
        for name, raw in (data.get("sequences") or {}).items()
    )

    escalation_routes = tuple(
        EscalationRoute(
            operator_group=raw.get("operator_group") or raw.get("default") or "on_call",
            if_articles=tuple(raw.get("if") or ()),
        )
        for raw in (data.get("escalation", {}).get("routes") or [])
    )

    return Policy(
        version=version,
        domains=domains,
        action_classes=action_classes,
        thresholds_default=thresholds_default,
        thresholds_overrides=thresholds_overrides,
        sequences=sequences,
        escalation_routes=escalation_routes,
    )


def _coerce_enum(value, enum_cls, path: str):
    """Convert a string value to an enum member, or raise PolicyError."""
    if value is None:
        raise PolicyError(f"{path}: required field missing")
    try:
        return enum_cls(value)
    except ValueError:
        valid = ", ".join(repr(m.value) for m in enum_cls)
        raise PolicyError(f"{path}: {value!r} is not one of [{valid}]") from None


def from_json(source: Union[str, Path, dict]) -> Policy:
    """Load a policy from a JSON string, JSON file path, or already-parsed dict."""
    if isinstance(source, dict):
        return from_dict(source)
    if isinstance(source, Path):
        try:
            text = source.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise PolicyError(f"policy file not found: {source}") from e
    elif isinstance(source, str) and not source.lstrip().startswith("{"):
        # Treat as path. If it isn't a path either, surface as PolicyError
        # so callers don't have to guard against FileNotFoundError separately.
        try:
            text = Path(source).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise PolicyError(
                f"input is neither a JSON object nor a readable file path: {source!r}"
            ) from None
    else:
        text = source
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PolicyError(f"invalid JSON: {e}") from None
    return from_dict(data)


def from_yaml(source: Union[str, Path]) -> Policy:
    """Load a policy from a YAML file path or YAML string. Requires `vaara[yaml]`."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "from_yaml requires the [yaml] extra. "
            "Install with: pip install 'vaara[yaml]'"
        ) from e

    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    elif isinstance(source, str) and "\n" not in source and Path(source).is_file():
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = source

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise PolicyError(f"invalid YAML: {e}") from None
    return from_dict(data)
