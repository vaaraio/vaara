/**
 * @vaara/client — TypeScript HTTP client for Vaara.
 *
 * Vaara is a runtime AI agent governance kernel: conformal risk
 * scoring, hash-chained audit trail, EU AI Act article-evidence
 * model. Python implementation lives at
 * https://github.com/vaaraio/vaara; this package is the JavaScript /
 * TypeScript client for its v1 HTTP API.
 */

export { VaaraClient } from "./client.js";
export type { VaaraClientOptions } from "./client.js";
export {
  VaaraError,
  VaaraTransportError,
  type VaaraErrorBody,
} from "./errors.js";
export type {
  AuditEventRequest,
  AuditEventResponse,
  BlastRadius,
  Decision,
  DetectInjectionRequest,
  DetectInjectionResponse,
  DetectPIIFinding,
  DetectPIIRequest,
  DetectPIIResponse,
  EventType,
  OutcomeRequest,
  PIICategory,
  PolicyReloadRequest,
  PolicyReloadResponse,
  Reversibility,
  RiskBlock,
  ScoreRequest,
  ScoreResponse,
  ServerInfo,
  VerifyResponse,
} from "./types.js";
