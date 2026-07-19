# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pydantic models matching docs/openapi.yaml v1 contract.

These models are the wire format. The internal `AdaptiveScorer.evaluate(ctx)`
takes and returns dicts; this module is the bridge between the spec and the
internal types.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


_Reversibility = Literal["reversible", "partially_reversible", "irreversible"]
_BlastRadius = Literal["local", "account", "organization", "global"]
_Decision = Literal["allow", "escalate", "deny"]
_EventType = Literal[
    "action_requested",
    "risk_scored",
    "decision_made",
    "action_executed",
    "action_blocked",
    "escalation_sent",
    "escalation_resolved",
    "outcome_recorded",
    "policy_override",
]


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(max_length=512)
    agent_id: str = Field(max_length=256)
    action_type: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    agent_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    base_risk_score: Optional[float] = Field(default=None, ge=0, le=1)
    reversibility: Optional[_Reversibility] = None
    blast_radius: Optional[_BlastRadius] = None
    session_id: Optional[str] = Field(default=None, max_length=256)
    parent_action_id: Optional[str] = Field(default=None, max_length=128)
    tenant_id: str = Field(default="", max_length=256)
    context: dict[str, Any] = Field(default_factory=dict)


class RiskBlock(BaseModel):
    point: float = Field(ge=0, le=1)
    lower: float = Field(ge=0, le=1)
    upper: float = Field(ge=0, le=1)
    alpha: float = Field(ge=0, le=1)
    bucket: Optional[str] = None


class Thresholds(BaseModel):
    allow: float
    deny: float


class ScoreResponse(BaseModel):
    action_id: str
    decision: _Decision
    risk: RiskBlock
    signals: dict[str, float] = Field(default_factory=dict)
    mwu_weights: dict[str, float] = Field(default_factory=dict)
    thresholds: Thresholds
    sequence_risk: float = 0.0
    calibration_size: int = 0
    evaluation_ms: float = 0.0
    explanation: str = ""


class OutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    outcome_severity: float = Field(ge=0, le=1)
    notes: Optional[str] = None


class AuditEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: _EventType
    action_id: str
    agent_id: Optional[str] = None
    tool_name: Optional[str] = None
    tenant_id: str = Field(default="", max_length=256)
    payload: dict[str, Any] = Field(default_factory=dict)


class AuditEventResponse(BaseModel):
    event_id: str
    chain_position: int
    event_hash: str
    previous_hash: str
    timestamp: str


class AuditChainEvent(BaseModel):
    event_id: str
    event_type: str
    chain_position: int
    event_hash: str
    previous_hash: str
    timestamp: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AuditChain(BaseModel):
    action_id: str
    events: list[AuditChainEvent]


class VerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_event_id: Optional[str] = None
    to_event_id: Optional[str] = None


class FirstBreak(BaseModel):
    event_id: str
    chain_position: int
    expected_previous_hash: str
    actual_previous_hash: str


class VerifyResponse(BaseModel):
    valid: bool
    events_checked: int
    first_break: Optional[FirstBreak] = None


class ScorerInfo(BaseModel):
    type: str
    calibration_size: int
    threshold_allow: float
    threshold_deny: float
    alpha: float


class Capabilities(BaseModel):
    score: bool
    audit: bool
    outcome_feedback: bool


class ServerInfo(BaseModel):
    name: str
    version: str
    vaara_version: str
    capabilities: Capabilities
    scorer: Optional[ScorerInfo] = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


class DetectInjectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(max_length=100_000)
    threshold: Optional[float] = Field(default=None, ge=0, le=1)


class DetectInjectionResponse(BaseModel):
    detected: bool
    score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    bundle_version: str
    backend: str


class DetectPIIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(max_length=100_000)


class DetectPIIFinding(BaseModel):
    category: str
    value: str
    offset: int
    length: int


class DetectPIIResponse(BaseModel):
    detected: bool
    categories: list[str]
    findings: list[DetectPIIFinding]


class PolicyReloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = Field(default=None, max_length=4096)
    body: Optional[dict[str, Any]] = None
    format: Optional[Literal["json", "yaml"]] = None
    tenant_id: str = Field(default="", max_length=256)


class PolicyReloadResponse(BaseModel):
    version: int
    thresholds_default: dict[str, float]
    sequence_count: int
    action_class_count: int
    escalation_route_count: int
    tenant_id: str = ""
