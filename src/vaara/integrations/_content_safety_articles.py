"""Canonical mapping from cloud-guardrail findings to EU AI Act articles.

This module is the value, not the SDK glue around it. Each provider
returns category-typed findings using its own vocabulary (Bedrock
``topicPolicy``, Azure ``Hate``, GCP ``responsible_ai``). The adapters
in this package translate those into a single Vaara vocabulary and the
EU AI Act articles the deployer needs evidence against, so the same
``ContentSafetyFinding`` flows downstream regardless of which provider
emitted it.

The mapping is a published artefact. A deployer can read it, dispute
it, and override it without modifying adapter code.

Article references come from Regulation (EU) 2024/1689 (the AI Act).
The CSAM row references the Digital Omnibus political agreement of
May 2026, which adds a CSAM-specific prohibition effective 2 December
2026 alongside the Art. 5 prohibited-practices regime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CategoryMapping:
    """One row of the provider-category to article mapping table."""

    provider: str
    provider_category: str
    vaara_category: str
    ai_act_articles: tuple[str, ...]
    owasp_llm: tuple[str, ...]
    notes: str = ""


def _row(p: str, pc: str, vc: str, arts: tuple[str, ...],
         owasp: tuple[str, ...], notes: str = "") -> CategoryMapping:
    return CategoryMapping(p, pc, vc, arts, owasp, notes)


_BEDROCK = "aws-bedrock-guardrails"
_AZURE = "azure-content-safety"
_GCP = "gcp-model-armor"


BEDROCK_MAPPINGS: tuple[CategoryMapping, ...] = (
    _row(_BEDROCK, "topicPolicy", "prohibited_topic", ("Art. 5",), ("LLM08",),
         "Denied-topic block. Topic identifies what the deployer marked as prohibited content."),
    _row(_BEDROCK, "contentPolicy.HATE", "hate", ("Art. 5",), ("LLM05",),
         "Confidence HIGH treated as Art. 5 evidence; MEDIUM recorded but not asserted."),
    _row(_BEDROCK, "contentPolicy.INSULTS", "hate", ("Art. 5",), ("LLM05",)),
    _row(_BEDROCK, "contentPolicy.SEXUAL", "sexual", ("Art. 5",), ("LLM05",)),
    _row(_BEDROCK, "contentPolicy.VIOLENCE", "violence", ("Art. 5",), ("LLM05",)),
    _row(_BEDROCK, "contentPolicy.MISCONDUCT", "misconduct", ("Art. 5",), ("LLM05",)),
    _row(_BEDROCK, "contentPolicy.PROMPT_ATTACK", "adversarial", ("Art. 15",), ("LLM01",),
         "Prompt-attack detection. Art. 15 robustness signal."),
    _row(_BEDROCK, "wordPolicy", "word_block", ("Art. 5",), ("LLM05",),
         "Custom denied words. Article mapping is heuristic; deployer-defined list."),
    _row(_BEDROCK, "sensitiveInformationPolicy", "pii", ("Art. 10",), ("LLM02",),
         "PII redaction or block. Type field carries the PII class."),
    _row(_BEDROCK, "contextualGroundingPolicy", "grounding", ("Art. 13", "Art. 15"), ("LLM09",),
         "Grounding score below threshold — hallucination signal."),
)


AZURE_MAPPINGS: tuple[CategoryMapping, ...] = (
    _row(_AZURE, "Hate", "hate", ("Art. 5",), ("LLM05",),
         "Severity ladder 0/2/4/6. Vaara treats >=4 as Art. 5 evidence; 2 recorded but not asserted."),
    _row(_AZURE, "SelfHarm", "self_harm", ("Art. 5",), ("LLM05",)),
    _row(_AZURE, "Sexual", "sexual", ("Art. 5",), ("LLM05",)),
    _row(_AZURE, "Violence", "violence", ("Art. 5",), ("LLM05",)),
    _row(_AZURE, "PromptShield.UserPrompt", "adversarial", ("Art. 15",), ("LLM01",),
         "Direct jailbreak attempt detected by Prompt Shields on user prompt."),
    _row(_AZURE, "PromptShield.Documents", "adversarial", ("Art. 15",), ("LLM01",),
         "Indirect prompt injection detected by Prompt Shields on attached documents."),
    _row(_AZURE, "ProtectedMaterial.Text", "protected_material", ("Art. 53",), ("LLM02",),
         "Protected (copyrighted) text in model output. Art. 53(1)(c) copyright policy evidence."),
    _row(_AZURE, "ProtectedMaterial.Code", "protected_material", ("Art. 53",), ("LLM02",)),
    _row(_AZURE, "Groundedness", "grounding", ("Art. 13", "Art. 15"), ("LLM09",),
         "Groundedness Detection API — ungrounded responses against supplied sources."),
)


GCP_MAPPINGS: tuple[CategoryMapping, ...] = (
    _row(_GCP, "responsible_ai.hate_speech", "hate", ("Art. 5",), ("LLM05",),
         "Confidence MEDIUM_AND_ABOVE or HIGH treated as Art. 5 evidence."),
    _row(_GCP, "responsible_ai.harassment", "hate", ("Art. 5",), ("LLM05",)),
    _row(_GCP, "responsible_ai.sexually_explicit", "sexual", ("Art. 5",), ("LLM05",)),
    _row(_GCP, "responsible_ai.dangerous", "violence", ("Art. 5",), ("LLM05",)),
    _row(_GCP, "pi_and_jailbreak", "adversarial", ("Art. 15",), ("LLM01",),
         "Prompt-injection and jailbreak detection."),
    _row(_GCP, "malicious_uris", "malicious_uri", ("Art. 15",), ("LLM05",),
         "Malicious URL detected. Robustness/cybersecurity signal."),
    _row(_GCP, "sdp", "pii", ("Art. 10",), ("LLM02",),
         "Sensitive Data Protection integration. infoType field carries the PII class."),
    _row(_GCP, "csam", "csam", ("Art. 5", "Digital Omnibus CSAM (effective 2 Dec 2026)"), (),
         "CSAM. Always hard-block per Art. 5 plus the CSAM-specific obligation from the Digital Omnibus."),
)


_INDEX: dict[tuple[str, str], CategoryMapping] = {
    (m.provider, m.provider_category): m
    for m in BEDROCK_MAPPINGS + AZURE_MAPPINGS + GCP_MAPPINGS
}


def lookup(provider: str, provider_category: str) -> Optional[CategoryMapping]:
    """Return the canonical mapping, or None for unmapped categories.

    Adapter should still record the raw provider response and use
    ``vaara_category="unmapped"`` so evidence surfaces without
    article-level annotation.
    """
    return _INDEX.get((provider, provider_category))


def all_mappings_for(provider: str) -> tuple[CategoryMapping, ...]:
    """Every published mapping for a provider. Used by doc generation and tests."""
    return tuple(m for m in _INDEX.values() if m.provider == provider)
