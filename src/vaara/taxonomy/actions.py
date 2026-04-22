"""Action taxonomy for AI agent execution layer.

Classifies every action an AI agent can take by:
- Category (financial, data, communication, infrastructure, identity, governance, physical)
- Reversibility (fully, partially, irreversible)
- Blast radius (self, local, shared, global)

Each action type carries regulatory-domain tags (EU AI Act, GDPR, DORA,
NIS2, MiFID II, HIPAA, SOC 2, product liability) so audit records can be
mapped article-by-article at runtime.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ActionCategory(str, Enum):
    """Top-level action category."""
    FINANCIAL = "financial"
    DATA = "data"
    COMMUNICATION = "communication"
    INFRASTRUCTURE = "infrastructure"
    IDENTITY = "identity"
    GOVERNANCE = "governance"
    PHYSICAL = "physical"
    UNKNOWN = "unknown"


class Reversibility(str, Enum):
    """How reversible is the action's effect?"""
    FULLY = "fully_reversible"
    PARTIALLY = "partially_reversible"
    IRREVERSIBLE = "irreversible"


class BlastRadius(str, Enum):
    """How many entities are affected?"""
    SELF = "self"           # Only the agent itself
    LOCAL = "local"         # Agent + immediate counterparty
    SHARED = "shared"       # Multiple users / shared state
    GLOBAL = "global"       # Public / broadcast / on-chain


class UrgencyClass(str, Enum):
    """Can the action wait for human review?"""
    DEFERRABLE = "deferrable"       # Can wait hours/days
    TIMELY = "timely"               # Should happen within minutes
    IMMEDIATE = "immediate"         # Time-critical, blocking on review costs money
    IRREVOCABLE = "irrevocable"     # Once started, cannot be stopped


# ── Regulatory relevance flags ──────────────────────────────────────────────

class RegulatoryDomain(str, Enum):
    """Which regulatory frameworks potentially apply."""
    EU_AI_ACT = "eu_ai_act"
    GDPR = "gdpr"
    MIFID2 = "mifid2"
    DORA = "dora"
    NIS2 = "nis2"
    PRODUCT_LIABILITY = "product_liability"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    NONE = "none"


# ── Action classification ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ActionType:
    """Immutable classification of an action type.

    Each registered action type carries its risk metadata so the scorer
    and compliance engine can use it without re-deriving.
    """
    name: str
    category: ActionCategory
    reversibility: Reversibility
    blast_radius: BlastRadius
    urgency: UrgencyClass = UrgencyClass.DEFERRABLE
    regulatory_domains: frozenset[RegulatoryDomain] = field(default_factory=frozenset)
    description: str = ""

    @property
    def base_risk_score(self) -> float:
        """Heuristic base risk from static metadata, 0.0 to 1.0.

        This is the floor — the adaptive scorer adds learned adjustments.
        """
        rev_score = {
            Reversibility.FULLY: 0.1,
            Reversibility.PARTIALLY: 0.4,
            Reversibility.IRREVERSIBLE: 0.8,
        }[self.reversibility]

        blast_score = {
            BlastRadius.SELF: 0.0,
            BlastRadius.LOCAL: 0.1,
            BlastRadius.SHARED: 0.3,
            BlastRadius.GLOBAL: 0.5,
        }[self.blast_radius]

        urgency_score = {
            UrgencyClass.DEFERRABLE: 0.0,
            UrgencyClass.TIMELY: 0.1,
            UrgencyClass.IMMEDIATE: 0.2,
            UrgencyClass.IRREVOCABLE: 0.3,
        }[self.urgency]

        return min(1.0, (rev_score + blast_score + urgency_score) / 1.6)


# ── Action request envelope ────────────────────────────────────────────────

@dataclass
class ActionRequest:
    """Envelope for an action an agent wants to execute.

    This is the unit that passes through the interception pipeline:
    taxonomy → scorer → policy engine → audit logger → execute or block.
    """
    agent_id: str
    tool_name: str
    action_type: ActionType
    parameters: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    confidence: Optional[float] = None  # Agent's self-reported confidence
    session_id: str = ""
    parent_action_id: Optional[str] = None  # For action chains
    sequence_position: int = 0  # Position in current action sequence
    timestamp_utc: str = ""

    def to_policy_context(self) -> dict:
        """Convert to a plain dict of policy-evaluation fields.

        Extra keys from ``context`` are merged first so the core fields
        (agent_id, tool_name, base_risk_score, etc.) cannot be overridden
        by caller-supplied context — otherwise a caller could spoof its
        own identity or risk metadata to the scorer.
        """
        return {
            **self.context,
            "agent_id": self.agent_id,
            "tool_name": self.tool_name,
            "action_type": self.action_type.category.value,
            "reversibility": self.action_type.reversibility.value,
            "blast_radius": self.action_type.blast_radius.value,
            "urgency": self.action_type.urgency.value,
            "base_risk_score": self.action_type.base_risk_score,
            "agent_confidence": self.confidence,
            "session_id": self.session_id,
            "parent_action_id": self.parent_action_id,
            "sequence_position": self.sequence_position,
            "parameters": self.parameters,
        }


# ── Built-in action type registry ──────────────────────────────────────────

class ActionRegistry:
    """Registry of known action types with their risk metadata.

    Extensible — users register domain-specific action types at startup.
    """

    def __init__(self) -> None:
        self._types: dict[str, ActionType] = {}
        self._tool_mappings: dict[str, str] = {}  # tool_name → action_type name
        # Plugins (custom domain packs) can call register / map_tool
        # from different threads during application startup, and classify()
        # runs on every intercepted action. A lock keeps the two dicts
        # internally consistent without serializing hot-path classify()
        # beyond dict lookups. Under free-threaded 3.13t dict mutation
        # during iteration is no longer implicitly safe, so the lock
        # guards both mutation and the prefix iteration in classify().
        self._lock = threading.RLock()

    def register(self, action_type: ActionType) -> None:
        """Register an action type. Silently replaces an existing type
        with the same name but logs a warning so plugins with clashing
        taxonomies are visible at startup instead of failing mysteriously
        later."""
        with self._lock:
            if action_type.name in self._types and self._types[action_type.name] is not action_type:
                logger.warning(
                    "ActionRegistry: replacing action type %r "
                    "(previous %r was registered; new one wins)",
                    action_type.name, self._types[action_type.name],
                )
            self._types[action_type.name] = action_type

    def map_tool(self, tool_name: str, action_type_name: str) -> None:
        """Map a framework tool name to a registered action type."""
        with self._lock:
            if action_type_name not in self._types:
                raise KeyError(f"Action type '{action_type_name}' not registered")
            self._tool_mappings[tool_name] = action_type_name

    def classify(self, tool_name: str, parameters: Optional[dict] = None) -> ActionType:
        """Classify a tool call into an action type.

        Resolution order:
          1. Exact mapping (map_tool / auto-mapped builtin).
          2. Longest prefix match from registered mappings.
          3. Keyword heuristic over the built-in taxonomy — catches
             framework tools with names like "send_slack_alert" or
             "delete_customer_row" that the user never explicitly
             mapped. Better than UNKNOWN for the common case.
          4. UNKNOWN_ACTION as a last resort.
        """
        with self._lock:
            if tool_name in self._tool_mappings:
                return self._types[self._tool_mappings[tool_name]]
            # Try prefix matching for namespaced tools (e.g., "fs.write" → "fs")
            for prefix in sorted(self._tool_mappings, key=len, reverse=True):
                if tool_name.startswith(prefix):
                    return self._types[self._tool_mappings[prefix]]
            # Keyword heuristic — only resolves to built-in action types
            # that are actually registered in this registry. A custom
            # registry without the default builtins falls through to
            # UNKNOWN rather than returning a type it doesn't know.
            guessed = _heuristic_classify(tool_name)
            if guessed and guessed in self._types:
                return self._types[guessed]
            return UNKNOWN_ACTION

    def get(self, name: str) -> Optional[ActionType]:
        with self._lock:
            return self._types.get(name)

    @property
    def all_types(self) -> dict[str, ActionType]:
        with self._lock:
            return dict(self._types)


# ── Heuristic keyword classification ───────────────────────────────────────

# Keyword → built-in action-type name. Scanned (in order) against a
# lowercased, non-alphanumeric-stripped tool name when explicit mapping
# and prefix matching both miss. This is the "LangChain @tool with a
# custom name" path: a tool called "send_slack_alert" shouldn't fall to
# UNKNOWN just because the user didn't call map_tool(). The table
# deliberately covers the clear cases only — ambiguous names still fall
# to UNKNOWN (better than miscategorising a finance tool as comms).
# Order matters: more specific keywords come first so "delete_email"
# matches data.delete before comm.send_email's "email".
_HEURISTIC_KEYWORDS: tuple[tuple[str, str], ...] = (
    # Financial
    ("transfer", "tx.transfer"),
    ("withdraw", "tx.transfer"),
    ("swap", "tx.swap"),
    ("approve_token", "tx.approve"),
    ("sign_tx", "tx.sign"),
    ("sign_transaction", "tx.sign"),
    ("rebalance", "vault.rebalance"),
    # Data
    ("delete", "data.delete"),
    ("drop_table", "data.delete"),
    ("export", "data.export"),
    ("download", "data.export"),
    ("upload", "data.export"),
    ("write_file", "data.write"),
    ("write", "data.write"),
    ("save", "data.write"),
    ("read_file", "data.read"),
    ("read", "data.read"),
    ("fetch", "data.read"),
    ("query", "data.read"),
    ("search", "data.read"),
    # Communication
    ("email", "comm.send_email"),
    ("slack", "comm.send_email"),
    ("post", "comm.post_public"),
    ("tweet", "comm.post_public"),
    ("publish", "comm.post_public"),
    ("api_call", "comm.api_call"),
    ("http", "comm.api_call"),
    ("request", "comm.api_call"),
    # Infrastructure
    ("deploy", "infra.deploy"),
    ("terminate", "infra.terminate"),
    ("kill", "infra.terminate"),
    ("scale", "infra.scale"),
    ("config", "infra.config_change"),
    ("settings", "infra.config_change"),
    # Identity
    ("grant_permission", "id.grant_permission"),
    ("grant_role", "id.grant_permission"),
    ("grant", "id.grant_permission"),
    ("create_key", "id.create_key"),
    ("api_key", "id.create_key"),
    ("revoke", "id.revoke"),
    # Governance
    ("execute_proposal", "gov.execute_proposal"),
    ("vote", "gov.vote"),
    # Physical
    ("safety_override", "phy.safety_override"),
    ("actuator", "phy.actuator"),
    # Privilege escalation — all map to id.grant_permission (highest-risk id action)
    ("assume_role", "id.grant_permission"),
    ("impersonate", "id.grant_permission"),
    ("escalate", "id.grant_permission"),
    ("sudo", "id.grant_permission"),
    ("chmod", "id.grant_permission"),
    ("chown", "id.grant_permission"),
    ("authorized_key", "id.grant_permission"),
    ("bypass_mfa", "id.revoke"),
    ("disable_mfa", "id.revoke"),
    ("disable_2fa", "id.revoke"),
    ("modify_rbac", "id.grant_permission"),
    ("setuid", "id.grant_permission"),
    ("privileged", "id.grant_permission"),
    ("change_password", "id.grant_permission"),
    ("add_user_to_group", "id.grant_permission"),
    # Destructive variants (shell/sql/infra verbs missing from the base data.delete set)
    ("execute_sql", "data.delete"),
    ("run_shell", "data.delete"),
    ("shell", "data.delete"),
    ("rm_rf", "data.delete"),
    ("wipe", "data.delete"),
    ("destroy", "data.delete"),
    ("format_disk", "data.delete"),
    ("format", "data.delete"),
    ("purge", "data.delete"),
    ("truncate", "data.delete"),
    ("drop_database", "data.delete"),
    ("drop_db", "data.delete"),
    ("dd_", "data.delete"),
    ("reset_hard", "data.delete"),
    ("force_push", "data.delete"),
    ("selfdestruct", "data.delete"),
    ("terraform_destroy", "data.delete"),
    ("revoke_ssl", "data.delete"),
    ("clear_logs", "data.delete"),
    # Cred exfil — tool-name hints that a secret is being touched
    ("credentials", "data.read"),
    ("secret", "data.read"),
    ("env_dump", "data.read"),
    ("kms_decrypt", "data.read"),
    ("get_secret", "data.read"),
    ("read_env", "data.read"),
    # SSRF-ish tool names (narrow — parameter-level detection is a follow-up PR)
    ("port_scan", "comm.api_call"),
    ("webhook_relay", "comm.api_call"),
    ("webhook_register", "comm.api_call"),
    # Data exfil variants
    ("s3_copy", "data.export"),
    ("cloud_sync", "data.export"),
    ("bigquery_export", "data.export"),
    ("mongo_dump", "data.export"),
    ("elastic_query", "data.read"),
    ("screenshot", "data.export"),
    ("clipboard_get", "data.read"),
)


def _heuristic_classify(tool_name: str) -> Optional[str]:
    """Return the best-guess action_type name for an unmapped tool, or None.

    Normalises by lowercasing and keeping only [a-z0-9_]. A tool called
    "SendSlackMessage" normalises to "sendslackmessage" and matches the
    "slack" keyword. The first keyword in _HEURISTIC_KEYWORDS that appears
    as a substring wins — order matters for overlapping keywords.
    """
    if not tool_name:
        return None
    normalized = "".join(
        c if c.isalnum() or c == "_" else "_" for c in tool_name.lower()
    )
    for keyword, action_type_name in _HEURISTIC_KEYWORDS:
        if keyword in normalized:
            return action_type_name
    return None


# ── Sentinel for unclassified actions ───────────────────────────────────────

UNKNOWN_ACTION = ActionType(
    name="unknown",
    category=ActionCategory.UNKNOWN,
    reversibility=Reversibility.PARTIALLY,
    blast_radius=BlastRadius.LOCAL,
    urgency=UrgencyClass.DEFERRABLE,
    description="Unclassified action — treated as medium risk by default",
)


# ── Default action types ───────────────────────────────────────────────────

BUILTIN_ACTIONS = [
    # Financial
    ActionType("tx.sign", ActionCategory.FINANCIAL, Reversibility.IRREVERSIBLE,
               BlastRadius.SHARED, UrgencyClass.IRREVOCABLE,
               frozenset({RegulatoryDomain.MIFID2, RegulatoryDomain.DORA}),
               "Sign and broadcast a blockchain transaction"),
    ActionType("tx.transfer", ActionCategory.FINANCIAL, Reversibility.IRREVERSIBLE,
               BlastRadius.SHARED, UrgencyClass.IRREVOCABLE,
               frozenset({RegulatoryDomain.MIFID2, RegulatoryDomain.DORA}),
               "Transfer funds between accounts"),
    ActionType("tx.approve", ActionCategory.FINANCIAL, Reversibility.PARTIALLY,
               BlastRadius.SHARED, UrgencyClass.TIMELY,
               frozenset({RegulatoryDomain.MIFID2}),
               "Approve token spending allowance"),
    ActionType("tx.swap", ActionCategory.FINANCIAL, Reversibility.IRREVERSIBLE,
               BlastRadius.SHARED, UrgencyClass.IMMEDIATE,
               frozenset({RegulatoryDomain.MIFID2, RegulatoryDomain.DORA}),
               "Execute a token swap on DEX"),
    ActionType("vault.rebalance", ActionCategory.FINANCIAL, Reversibility.PARTIALLY,
               BlastRadius.SHARED, UrgencyClass.TIMELY,
               frozenset({RegulatoryDomain.MIFID2, RegulatoryDomain.EU_AI_ACT}),
               "Rebalance vault allocation weights"),

    # Data
    ActionType("data.read", ActionCategory.DATA, Reversibility.FULLY,
               BlastRadius.SELF, UrgencyClass.DEFERRABLE,
               frozenset({RegulatoryDomain.GDPR}),
               "Read data from storage"),
    ActionType("data.write", ActionCategory.DATA, Reversibility.PARTIALLY,
               BlastRadius.LOCAL, UrgencyClass.DEFERRABLE,
               frozenset({RegulatoryDomain.GDPR}),
               "Write data to storage"),
    ActionType("data.delete", ActionCategory.DATA, Reversibility.IRREVERSIBLE,
               BlastRadius.SHARED, UrgencyClass.DEFERRABLE,
               frozenset({RegulatoryDomain.GDPR}),
               "Delete data permanently"),
    ActionType("data.export", ActionCategory.DATA, Reversibility.IRREVERSIBLE,
               BlastRadius.GLOBAL, UrgencyClass.DEFERRABLE,
               frozenset({RegulatoryDomain.GDPR, RegulatoryDomain.HIPAA}),
               "Export data to external system"),

    # Communication
    ActionType("comm.send_email", ActionCategory.COMMUNICATION, Reversibility.IRREVERSIBLE,
               BlastRadius.LOCAL, UrgencyClass.TIMELY,
               frozenset(), "Send an email"),
    ActionType("comm.post_public", ActionCategory.COMMUNICATION, Reversibility.IRREVERSIBLE,
               BlastRadius.GLOBAL, UrgencyClass.DEFERRABLE,
               frozenset(), "Post to public channel (social, forum)"),
    ActionType("comm.api_call", ActionCategory.COMMUNICATION, Reversibility.PARTIALLY,
               BlastRadius.LOCAL, UrgencyClass.TIMELY,
               frozenset(), "Call external API"),

    # Infrastructure
    ActionType("infra.deploy", ActionCategory.INFRASTRUCTURE, Reversibility.PARTIALLY,
               BlastRadius.SHARED, UrgencyClass.TIMELY,
               frozenset({RegulatoryDomain.NIS2}),
               "Deploy code or configuration"),
    ActionType("infra.config_change", ActionCategory.INFRASTRUCTURE, Reversibility.PARTIALLY,
               BlastRadius.SHARED, UrgencyClass.DEFERRABLE,
               frozenset({RegulatoryDomain.NIS2}),
               "Modify system configuration"),
    ActionType("infra.scale", ActionCategory.INFRASTRUCTURE, Reversibility.FULLY,
               BlastRadius.SHARED, UrgencyClass.TIMELY,
               frozenset(), "Scale resources up or down"),
    ActionType("infra.terminate", ActionCategory.INFRASTRUCTURE, Reversibility.IRREVERSIBLE,
               BlastRadius.SHARED, UrgencyClass.IMMEDIATE,
               frozenset({RegulatoryDomain.NIS2}),
               "Terminate process or resource"),

    # Identity
    ActionType("id.grant_permission", ActionCategory.IDENTITY, Reversibility.PARTIALLY,
               BlastRadius.SHARED, UrgencyClass.DEFERRABLE,
               frozenset({RegulatoryDomain.SOC2}),
               "Grant permissions or roles"),
    ActionType("id.create_key", ActionCategory.IDENTITY, Reversibility.PARTIALLY,
               BlastRadius.LOCAL, UrgencyClass.DEFERRABLE,
               frozenset({RegulatoryDomain.SOC2}),
               "Create API key or credential"),
    ActionType("id.revoke", ActionCategory.IDENTITY, Reversibility.PARTIALLY,
               BlastRadius.SHARED, UrgencyClass.IMMEDIATE,
               frozenset({RegulatoryDomain.SOC2}),
               "Revoke access or credential"),

    # Governance
    ActionType("gov.vote", ActionCategory.GOVERNANCE, Reversibility.IRREVERSIBLE,
               BlastRadius.GLOBAL, UrgencyClass.TIMELY,
               frozenset({RegulatoryDomain.EU_AI_ACT}),
               "Cast governance vote"),
    ActionType("gov.execute_proposal", ActionCategory.GOVERNANCE, Reversibility.IRREVERSIBLE,
               BlastRadius.GLOBAL, UrgencyClass.IRREVOCABLE,
               frozenset({RegulatoryDomain.EU_AI_ACT}),
               "Execute approved governance proposal"),

    # Physical / IoT
    ActionType("phy.actuator", ActionCategory.PHYSICAL, Reversibility.IRREVERSIBLE,
               BlastRadius.LOCAL, UrgencyClass.IMMEDIATE,
               frozenset({RegulatoryDomain.PRODUCT_LIABILITY}),
               "Control physical actuator"),
    ActionType("phy.safety_override", ActionCategory.PHYSICAL, Reversibility.IRREVERSIBLE,
               BlastRadius.SHARED, UrgencyClass.IRREVOCABLE,
               frozenset({RegulatoryDomain.PRODUCT_LIABILITY, RegulatoryDomain.EU_AI_ACT}),
               "Override safety system"),
]


def create_default_registry() -> ActionRegistry:
    """Create registry with all built-in action types.

    Each builtin action type is also auto-mapped as a tool name,
    so classify("data.read") returns the data.read ActionType without
    needing an explicit map_tool() call.
    """
    registry = ActionRegistry()
    for action_type in BUILTIN_ACTIONS:
        registry.register(action_type)
        registry.map_tool(action_type.name, action_type.name)
    return registry
