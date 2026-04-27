"""Vaara policy schema — JSON-native, YAML via the optional `vaara[yaml]` extra.

The policy expresses, in declarative form, the deployer's choices for:
- Action-class taxonomy (category, reversibility, blast radius, urgency, regulatory tags)
- Risk thresholds (default plus per-action-class overrides)
- Sequence patterns that boost risk when matched in a sliding history window
- Escalation routes mapping regulatory-article sets to operator groups

Internal model stays zero-dependency. JSON loading uses stdlib. YAML loading is
gated on `vaara[yaml]` so the core library stays free of third-party imports.

The companion JSON Schema document at `docs/policy_schema.json` is the citable
spec for compliance-evidence purposes. Hand-rolled validation in `loader.py`
mirrors the schema, with clean error paths for human readers.
"""

from vaara.policy.schema import (
    SCHEMA_VERSION,
    ActionClassDef,
    EscalationRoute,
    Policy,
    PolicyError,
    SequencePattern,
    Thresholds,
)
from vaara.policy.loader import from_dict, from_json, from_yaml

__all__ = [
    "SCHEMA_VERSION",
    "ActionClassDef",
    "EscalationRoute",
    "Policy",
    "PolicyError",
    "SequencePattern",
    "Thresholds",
    "from_dict",
    "from_json",
    "from_yaml",
]
