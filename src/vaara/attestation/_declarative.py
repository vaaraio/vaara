"""Data-only source profiles: bind a foreign format without writing code.

The built-in profiles in ``_normalize`` are Python because two of them must
*compute* something (the SEP-2787 map derives a back-link digest, a crypto
op, not a field copy). Most foreign formats need no computation: they need to
be recognized, to have a few fields lifted into advisory context, and to
report honestly what a complete signed execution record still lacks. That is
field-mapping, and field-mapping is data.

This module compiles a declarative spec (a plain ``dict``, authored as JSON)
into the same frozen ``SourceProfile`` the registry already holds, so a new
format is a file an implementer drops in, not a dispatch branch they patch.
The hot path (``detect_format`` / ``normalize``) never knows whether a profile
came from Python or JSON. A spec asserts only what the source establishes and
cannot fabricate a signature or back-link, because those are computed, not
copied. See ``docs/source-profile-contract.md`` for the full contract.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from vaara.attestation._normalize import (
    NormalizedEvidence,
    SourceProfile,
    register,
)

logger = logging.getLogger(__name__)

# Where the shipped declarative specs live, next to this module.
BUILTIN_PROFILE_DIR = Path(__file__).resolve().parent / "profiles"


class ProfileSpecError(ValueError):
    """A declarative profile spec is malformed."""


def resolve_path(doc: Any, path: str) -> Any:
    """Resolve a dotted path with optional ``[index]`` segments against ``doc``.

    ``"predicate.builder.id"`` walks dict keys; ``"subject[0].digest"`` walks a
    list index then more keys. Any missing key, out-of-range index, or type
    mismatch yields ``None`` rather than raising, so a profile that maps a field
    a document happens not to carry simply omits it instead of failing ingest.
    """
    cur = doc
    for raw in path.split("."):
        key, indices = _split_indices(raw)
        if key:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        for idx in indices:
            if not isinstance(cur, (list, tuple)) or idx >= len(cur) or idx < 0:
                return None
            cur = cur[idx]
    return cur


def _split_indices(segment: str) -> tuple[str, list[int]]:
    """Split ``"subject[0][1]"`` into ``("subject", [0, 1])``."""
    if "[" not in segment:
        return segment, []
    key, _, rest = segment.partition("[")
    indices: list[int] = []
    for part in rest.split("["):
        token = part.rstrip("]")
        if not token.isdigit():
            raise ProfileSpecError(f"bad list index in path segment {segment!r}")
        indices.append(int(token))
    return key, indices


_KNOWN_OPS = ("equals", "startsWith", "in", "exists")


def _validate_rule(rule: Any) -> None:
    """Check a detect rule's shape at compile time so a malformed spec fails
    loudly when it loads, not mid-ingest against a live document."""
    if not isinstance(rule, dict) or "path" not in rule:
        raise ProfileSpecError(f"detect rule needs a 'path': {rule!r}")
    if not any(op in rule for op in _KNOWN_OPS):
        raise ProfileSpecError(f"detect rule has no known operator: {rule!r}")
    if "in" in rule and not isinstance(rule["in"], list):
        raise ProfileSpecError(f"'in' takes a list: {rule!r}")


def _eval_rule(doc: Any, rule: dict[str, Any]) -> bool:
    """Evaluate one detect rule: a ``path`` plus one operator (``equals``,
    ``startsWith``, ``in``, or ``exists``)."""
    value = resolve_path(doc, rule["path"])
    if "equals" in rule:
        return value == rule["equals"]
    if "startsWith" in rule:
        return isinstance(value, str) and value.startswith(rule["startsWith"])
    if "in" in rule:
        if not isinstance(rule["in"], list):
            raise ProfileSpecError(f"'in' takes a list: {rule!r}")
        return value in rule["in"]
    if "exists" in rule:
        return (value is not None) == bool(rule["exists"])
    raise ProfileSpecError(f"detect rule has no known operator: {rule!r}")


def _make_detector(detect: dict[str, Any]):
    """Build a detector from ``{all: [...], any: [...]}``: every ``all`` rule
    must match and at least one ``any`` rule must match. At least one group is
    required (enforced here)."""
    all_rules = detect.get("all", [])
    any_rules = detect.get("any", [])
    if not isinstance(all_rules, list) or not isinstance(any_rules, list):
        raise ProfileSpecError("detect 'all'/'any' must be lists")
    if not all_rules and not any_rules:
        raise ProfileSpecError("detect needs a non-empty 'all' or 'any' group")
    for rule in (*all_rules, *any_rules):
        _validate_rule(rule)

    def detector(doc: Any) -> bool:
        if not isinstance(doc, dict):
            return False
        if not all(_eval_rule(doc, r) for r in all_rules):
            return False
        if any_rules and not any(_eval_rule(doc, r) for r in any_rules):
            return False
        return True

    return detector


def _lift(doc: Any, mapping: dict[str, Any]) -> dict[str, Any]:
    """Resolve a ``{outKey: source}`` mapping where source is a path string or
    a ``{"const": value}`` literal; drop path keys that resolve to None."""
    out: dict[str, Any] = {}
    for out_key, source in mapping.items():
        if isinstance(source, dict) and "const" in source:
            out[out_key] = source["const"]
        elif isinstance(source, str):
            value = resolve_path(doc, source)
            if value is not None:
                out[out_key] = value
        else:
            raise ProfileSpecError(
                f"mapping for {out_key!r} must be a path or {{const: ...}}"
            )
    return out


def _set_nested(target: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``target['a']['b'] = value`` for a dotted ``"a.b"`` key."""
    cur = target
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def _make_normalizer(spec: dict[str, Any], detector):
    source_format = spec["sourceFormat"]
    source_title = spec["sourceTitle"]
    evidence_plane = spec.get("evidencePlane")
    advisory_map = spec.get("advisory", {}) or {}
    sep2828_map = spec.get("sep2828", {}) or {}
    missing = tuple(spec.get("missing", ()))
    notes = tuple(spec.get("notes", ()))
    if not isinstance(advisory_map, dict) or not isinstance(sep2828_map, dict):
        raise ProfileSpecError("'advisory' and 'sep2828' must be objects")

    def normalizer(doc: Any) -> NormalizedEvidence:
        # A forced source_format can reach a normalizer whose document does not
        # match; re-check so a forced read stays honest.
        if not detector(doc):
            return NormalizedEvidence(
                source_format=source_format, source_title=source_title,
                recognized=False,
                notes=(f"document does not match the {source_format!r} profile",),
            )
        sep2828: dict[str, Any] = {}
        populated: list[str] = []
        for dotted, value in _lift(doc, sep2828_map).items():
            _set_nested(sep2828, dotted, value)
            populated.append(dotted)
        return NormalizedEvidence(
            source_format=source_format, source_title=source_title,
            recognized=True, evidence_plane=evidence_plane,
            sep2828=sep2828, advisory=_lift(doc, advisory_map),
            populated=tuple(sorted(populated)), missing=missing, notes=notes,
        )

    return normalizer


def compile_profile(spec: dict[str, Any]) -> SourceProfile:
    """Compile a declarative spec dict into a ``SourceProfile``."""
    for required in ("sourceFormat", "sourceTitle", "detect"):
        if required not in spec:
            raise ProfileSpecError(f"profile spec missing {required!r}")
    if not isinstance(spec["detect"], dict):
        raise ProfileSpecError("'detect' must be an object with 'all'/'any'")
    priority = spec.get("priority", 100)
    if not isinstance(priority, int):
        raise ProfileSpecError("'priority' must be an integer")
    detector = _make_detector(spec["detect"])
    return SourceProfile(
        source_format=spec["sourceFormat"], source_title=spec["sourceTitle"],
        detector=detector, normalizer=_make_normalizer(spec, detector),
        priority=priority,
    )


def load_profile_file(path: Path) -> SourceProfile:
    """Compile and register the declarative profile at ``path``."""
    profile = compile_profile(json.loads(Path(path).read_text()))
    register(profile)
    return profile


def load_builtin_declarative_profiles() -> list[str]:
    """Compile and register every shipped spec in ``profiles/``; return the
    source-format ids. A spec that fails to compile is logged and skipped so
    one bad file cannot break ingest for the rest."""
    if not BUILTIN_PROFILE_DIR.is_dir():
        return []
    registered: list[str] = []
    for path in sorted(BUILTIN_PROFILE_DIR.glob("*.json")):
        try:
            registered.append(load_profile_file(path).source_format)
        except (ProfileSpecError, ValueError) as exc:
            logger.warning("skipping declarative profile %s: %s", path.name, exc)
    return registered
