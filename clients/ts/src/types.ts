/**
 * Wire types for the Vaara HTTP API (v1).
 *
 * Authoritative source: docs/openapi.yaml in the Vaara repository.
 * These types track the v1 contract; if a field is added there, it
 * lands here at the next minor release.
 */

export type Decision = "allow" | "escalate" | "deny";

export type Reversibility =
  | "reversible"
  | "partially_reversible"
  | "irreversible";

export type BlastRadius = "local" | "account" | "organization" | "global";

export type EventType =
  | "action_requested"
  | "risk_scored"
  | "decision_made"
  | "action_executed"
  | "action_blocked"
  | "escalation_sent"
  | "escalation_resolved"
  | "outcome_recorded"
  | "policy_override";

export interface ScoreRequest {
  tool_name: string;
  agent_id: string;
  action_type?: string;
  parameters?: Record<string, unknown>;
  agent_confidence?: number;
  base_risk_score?: number;
  reversibility?: Reversibility;
  blast_radius?: BlastRadius;
  session_id?: string;
  parent_action_id?: string;
  context?: Record<string, unknown>;
}

export interface RiskBlock {
  point: number;
  lower: number;
  upper: number;
}

export interface ScoreResponse {
  action_id: string;
  decision: Decision;
  risk: RiskBlock;
  signals?: Record<string, number>;
  backend?: string;
  composition?: {
    members: string[];
    mode: string;
  };
}

export interface OutcomeRequest {
  action_id: string;
  outcome_severity: number;
  description?: string;
}

export interface AuditEventRequest {
  event_type: EventType;
  action_id: string;
  agent_id?: string;
  tool_name?: string;
  data?: Record<string, unknown>;
  regulatory_articles?: string[];
}

export interface AuditEventResponse {
  record_id: string;
  record_hash: string;
  previous_hash: string;
  timestamp: string;
}

export interface VerifyResponse {
  valid: boolean;
  events_checked: number;
  first_break?: {
    event_id: string;
    chain_position: number;
    expected_previous_hash: string;
    actual_previous_hash: string;
  } | null;
}

export interface PolicyReloadRequest {
  path?: string;
  body?: Record<string, unknown>;
  format?: "json" | "yaml";
}

export interface PolicyReloadResponse {
  version: number;
  thresholds_default: { escalate: number; deny: number };
  sequence_count: number;
  action_class_count: number;
  escalation_route_count: number;
}

export interface DetectInjectionRequest {
  text: string;
  threshold?: number;
}

export interface DetectInjectionResponse {
  detected: boolean;
  score: number;
  threshold: number;
  bundle_version: string;
  backend: "vaara_adversarial" | "heuristic";
}

export interface DetectPIIRequest {
  text: string;
}

export type PIICategory =
  | "email"
  | "phone"
  | "ssn"
  | "ipv4"
  | "credit_card"
  | "iban";

export interface DetectPIIFinding {
  category: PIICategory;
  value: string;
  offset: number;
  length: number;
}

export interface DetectPIIResponse {
  detected: boolean;
  categories: PIICategory[];
  findings: DetectPIIFinding[];
}

export interface ServerInfo {
  name: string;
  version: string;
  vaara_version: string;
  capabilities: {
    score?: boolean;
    audit?: boolean;
    outcome_feedback?: boolean;
  };
  scorer?: {
    type?: string;
    calibration_size?: number;
    threshold_allow?: number;
    threshold_deny?: number;
    alpha?: number;
  };
}
