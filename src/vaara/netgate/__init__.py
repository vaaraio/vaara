# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara net-gate``: agents' network access, governed where it leaves.

The tool-call gate decides what an agent asks to do. An agent that opens a
socket itself never asks. The net-gate is the other half: the agent's only way
out, where every connection is attributed to one agent by its own credential,
checked against that agent's mandate and the operator's policy, recorded on
the trail, and ended the moment the agent is revoked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from vaara.taxonomy.actions import (
    ActionCategory,
    ActionType,
    BlastRadius,
    RegulatoryDomain,
    Reversibility,
    UrgencyClass,
)

from ._gateway import NetGate
from ._mandate import AllowRule, MandateStore
from ._wg import client_config, split_allowed_ips

NET_CONNECT = ActionType(
    name="net.connect",
    category=ActionCategory.COMMUNICATION,
    reversibility=Reversibility.PARTIALLY,
    blast_radius=BlastRadius.SHARED,
    urgency=UrgencyClass.DEFERRABLE,
    regulatory_domains=frozenset({RegulatoryDomain.EU_AI_ACT, RegulatoryDomain.GDPR}),
    description="An agent opens an outbound network connection through the gateway",
)

NET_CLOSE = ActionType(
    name="net.close",
    category=ActionCategory.COMMUNICATION,
    reversibility=Reversibility.FULLY,
    blast_radius=BlastRadius.SELF,
    urgency=UrgencyClass.DEFERRABLE,
    regulatory_domains=frozenset(),
    description="A gateway connection ended: byte counts, duration and why",
)

NET_ACTIONS = [NET_CONNECT, NET_CLOSE]


def build_pipeline(db: Optional[Path] = None, policy: Optional[Any] = None):
    """A pipeline with the net actions registered and the trail at ``db``."""
    from vaara.audit.sqlite_backend import SQLiteAuditBackend
    from vaara.pipeline import InterceptionPipeline
    from vaara.taxonomy.actions import create_default_registry

    registry = create_default_registry()
    for at in NET_ACTIONS:
        registry.register(at)
        registry.map_tool(at.name, at.name)
    trail = None
    if db:
        db.parent.mkdir(parents=True, exist_ok=True)
        trail = SQLiteAuditBackend(str(db)).load_trail()
    pipeline = InterceptionPipeline(registry=registry, trail=trail)
    if policy is not None:
        pipeline.scorer.apply_policy(policy)
    return pipeline


__all__ = ["NetGate", "MandateStore", "AllowRule", "build_pipeline",
           "client_config", "split_allowed_ips", "NET_ACTIONS"]
