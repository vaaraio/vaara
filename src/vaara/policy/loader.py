# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Policy loaders. JSON is core (stdlib). YAML requires the `vaara[yaml]` extra."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import TypeVar, Union

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

EnumT = TypeVar("EnumT", bound=Enum)

_ALLOWED_THRESHOLD_KEYS = frozenset({"escalate", "deny"})


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
        for i, d in enumerate(_require_sequence(data.get("domains"), "domains"))
    )

    action_classes: dict[str, ActionClassDef] = {}
    for name, raw in _require_mapping(data.get("action_classes"), "action_classes").items():
        raw_dict = _require_mapping(raw, f"action_classes.{name}")
        action_classes[name] = ActionClassDef(
            name=name,
            category=_coerce_enum(
                raw_dict.get("category"), ActionCategory, f"action_classes.{name}.category"
            ),
            reversibility=_coerce_enum(
                raw_dict.get("reversibility"), Reversibility, f"action_classes.{name}.reversibility"
            ),
            blast_radius=_coerce_enum(
                raw_dict.get("blast_radius"), BlastRadius, f"action_classes.{name}.blast_radius"
            ),
            urgency=_coerce_enum(
                raw_dict.get("urgency"), UrgencyClass, f"action_classes.{name}.urgency"
            ),
            regulatory=tuple(_require_sequence(
                raw_dict.get("regulatory"), f"action_classes.{name}.regulatory"
            )),
        )

    thr_block = _require_mapping(data.get("thresholds"), "thresholds")
    default_block = _require_mapping(thr_block.get("default"), "thresholds.default")
    _reject_unknown_keys(default_block, _ALLOWED_THRESHOLD_KEYS, "thresholds.default")
    if not default_block:
        default_block = {"escalate": 0.55, "deny": 0.85}
    thresholds_default = Thresholds(
        escalate=float(default_block.get("escalate", 0.55)),
        deny=float(default_block.get("deny", 0.85)),
    )
    thresholds_overrides: dict[str, dict[str, float]] = {}
    for key, value in thr_block.items():
        if key == "default":
            continue
        override = _require_mapping(value, f"thresholds.{key}")
        _reject_unknown_keys(override, _ALLOWED_THRESHOLD_KEYS, f"thresholds.{key}")
        thresholds_overrides[key] = {kk: float(vv) for kk, vv in override.items()}
        # Validate the merged-with-default thresholds upfront so a bad override
        # doesn't sit dormant until threshold_for() is queried.
        try:
            Thresholds(
                escalate=thresholds_overrides[key].get(
                    "escalate", thresholds_default.escalate
                ),
                deny=thresholds_overrides[key].get(
                    "deny", thresholds_default.deny
                ),
            )
        except PolicyError as e:
            raise PolicyError(f"thresholds.{key}: {e}") from None

    sequences_block = _require_mapping(data.get("sequences"), "sequences")
    sequences_list: list[SequencePattern] = []
    for name, raw in sequences_block.items():
        raw_dict = _require_mapping(raw, f"sequences.{name}")
        pattern_seq = _require_sequence(
            raw_dict.get("pattern"), f"sequences.{name}.pattern"
        )
        sequences_list.append(SequencePattern(
            name=name,
            pattern=tuple(pattern_seq),
            risk_boost=float(raw_dict.get("risk_boost", 0.0)),
            window_seconds=int(raw_dict.get("window_seconds", 60)),
            regulatory=tuple(_require_sequence(
                raw_dict.get("regulatory"), f"sequences.{name}.regulatory"
            )),
        ))
    sequences = tuple(sequences_list)

    escalation_block = _require_mapping(data.get("escalation"), "escalation")
    routes_seq = _require_sequence(
        escalation_block.get("routes"), "escalation.routes"
    )
    escalation_routes_list: list[EscalationRoute] = []
    for i, raw in enumerate(routes_seq):
        raw_dict = _require_mapping(raw, f"escalation.routes[{i}]")
        escalation_routes_list.append(EscalationRoute(
            operator_group=raw_dict.get("operator_group") or raw_dict.get("default") or "on_call",
            if_articles=tuple(_require_sequence(
                raw_dict.get("if"), f"escalation.routes[{i}].if"
            )),
        ))
    escalation_routes = tuple(escalation_routes_list)

    return Policy(
        version=version,
        domains=domains,
        action_classes=action_classes,
        thresholds_default=thresholds_default,
        thresholds_overrides=thresholds_overrides,
        sequences=sequences,
        escalation_routes=escalation_routes,
    )


def _coerce_enum(value: object, enum_cls: type[EnumT], path: str) -> EnumT:
    """Convert a string value to an enum member, or raise PolicyError."""
    if value is None:
        raise PolicyError(f"{path}: required field missing")
    try:
        return enum_cls(value)
    except ValueError:
        valid = ", ".join(repr(m.value) for m in enum_cls)
        raise PolicyError(f"{path}: {value!r} is not one of [{valid}]") from None


def _require_mapping(value: object, path: str) -> dict:
    """Return value as a dict, or raise PolicyError. None becomes empty dict."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PolicyError(
            f"{path}: must be a mapping, got {type(value).__name__}"
        )
    return value


def _reject_unknown_keys(
    mapping: dict, allowed: frozenset[str], path: str
) -> None:
    """Raise PolicyError on the first unknown key in mapping.

    Catches typos like `deni` for `deny` that would otherwise float-coerce
    cleanly and then silently fall back to the default at query time.
    """
    extras = sorted(set(mapping) - allowed)
    if extras:
        raise PolicyError(
            f"{path}: unknown key(s) {extras!r}, allowed: {sorted(allowed)!r}"
        )


def _require_sequence(value: object, path: str) -> list:
    """Return value as a list, or raise PolicyError.

    Strings are rejected explicitly: tuple("abc") would silently produce
    ('a','b','c') and a malformed `pattern: "abc"` should not become three
    one-character pattern entries.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise PolicyError(
            f"{path}: must be a list, got {type(value).__name__}"
        )
    return list(value)


def _read_policy_text(path: Path, *, fallback_msg: str | None = None) -> str:
    """Read a policy file as utf-8 text, normalising every OS error to PolicyError.

    FileNotFoundError, IsADirectoryError, PermissionError, UnicodeDecodeError,
    and the catch-all OSError all surface as PolicyError so callers don't have
    to guard against OS-specific exception types around a public loader API.
    fallback_msg overrides the default "policy file not found" framing for
    the str-treated-as-path branch where a missing file means the input was
    neither inline content nor a readable path.
    """
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        if fallback_msg is not None:
            raise PolicyError(fallback_msg) from None
        raise PolicyError(f"policy file not found: {path}") from e
    except IsADirectoryError as e:
        raise PolicyError(f"policy path is a directory, not a file: {path}") from e
    except PermissionError as e:
        raise PolicyError(f"policy file not readable (permissions): {path}") from e
    except UnicodeDecodeError as e:
        raise PolicyError(f"policy file is not valid utf-8: {path}: {e}") from e
    except OSError as e:
        raise PolicyError(f"policy file unreadable: {path}: {e}") from e


def from_json(source: Union[str, Path, dict]) -> Policy:
    """Load a policy from a JSON string, JSON file path, or already-parsed dict."""
    if isinstance(source, dict):
        return from_dict(source)
    if isinstance(source, Path):
        text = _read_policy_text(source)
    elif isinstance(source, str) and not source.lstrip().startswith("{"):
        # Treat as path. If unreadable for any reason (missing, dir, perms,
        # bad utf-8), surface as PolicyError so callers don't have to guard
        # against OS-specific error types around a public API.
        text = _read_policy_text(
            Path(source),
            fallback_msg=(
                f"input is neither a JSON object nor a readable file path: {source!r}"
            ),
        )
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
        import yaml  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "from_yaml requires the [yaml] extra. "
            "Install with: pip install 'vaara[yaml]'"
        ) from e

    if isinstance(source, Path):
        text = _read_policy_text(source)
    elif isinstance(source, str) and "\n" not in source:
        # An attacker-controlled string longer than the OS path limit makes
        # Path(...).is_file() raise OSError(ENAMETOOLONG) directly, bypassing
        # this loader's PolicyError contract. Treat any stat() failure as
        # "not a path" and fall through to YAML parsing.
        try:
            looks_like_path = Path(source).is_file()
        except OSError:
            looks_like_path = False
        text = _read_policy_text(Path(source)) if looks_like_path else source
    else:
        text = source

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise PolicyError(f"invalid YAML: {e}") from None
    return from_dict(data)
