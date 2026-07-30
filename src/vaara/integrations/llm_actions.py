# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LLM proxy action types for governing prompts sent to LLM providers."""

from __future__ import annotations

from vaara.taxonomy.actions import ActionCategory, ActionType, BlastRadius, \
    RegulatoryDomain, Reversibility, UrgencyClass

LLM_PROMPT = ActionType(
    name="llm.prompt",
    category=ActionCategory.COMMUNICATION,
    reversibility=Reversibility.FULLY,
    blast_radius=BlastRadius.LOCAL,
    urgency=UrgencyClass.DEFERRABLE,
    regulatory_domains=frozenset({RegulatoryDomain.GDPR}),
    description="Send a prompt to an LLM provider",
)

LLM_PROMPT_EXFIL = ActionType(
    name="llm.prompt.exfil",
    category=ActionCategory.COMMUNICATION,
    reversibility=Reversibility.IRREVERSIBLE,
    blast_radius=BlastRadius.GLOBAL,
    urgency=UrgencyClass.IMMEDIATE,
    regulatory_domains=frozenset({RegulatoryDomain.GDPR, RegulatoryDomain.EU_AI_ACT}),
    description="Prompt containing potentially sensitive data sent to LLM provider",
)

LLM_ACTIONS = [LLM_PROMPT, LLM_PROMPT_EXFIL]
