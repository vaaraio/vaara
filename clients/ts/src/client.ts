/**
 * VaaraClient — typed wrapper around the Vaara HTTP API v1.
 *
 * The client is a thin pass-through to the wire contract; it does no
 * caching, no retries, no batching. Server-side error responses become
 * `VaaraError`; network failures become `VaaraTransportError`.
 *
 * Example:
 *
 * ```ts
 * import { VaaraClient } from "@vaara/client";
 *
 * const vaara = new VaaraClient({ baseUrl: "http://localhost:8000" });
 *
 * const result = await vaara.score({
 *   tool_name: "tx.transfer",
 *   agent_id: "agent-007",
 *   base_risk_score: 0.6,
 * });
 *
 * if (result.decision === "deny") throw new Error("blocked");
 * ```
 */

import { VaaraError, VaaraTransportError, type VaaraErrorBody } from "./errors.js";
import type {
  AuditEventRequest,
  AuditEventResponse,
  DetectInjectionRequest,
  DetectInjectionResponse,
  DetectPIIRequest,
  DetectPIIResponse,
  OutcomeRequest,
  PolicyReloadRequest,
  PolicyReloadResponse,
  ScoreRequest,
  ScoreResponse,
  ServerInfo,
  VerifyResponse,
} from "./types.js";

export interface VaaraClientOptions {
  /** Base URL of the Vaara HTTP server, e.g. `http://localhost:8000`. */
  baseUrl: string;
  /** Per-request timeout in milliseconds. Default 10_000. */
  timeoutMs?: number;
  /** Extra headers sent on every request. */
  headers?: Record<string, string>;
  /** Optional custom fetch implementation (defaults to global `fetch`). */
  fetch?: typeof fetch;
}

export class VaaraClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly headers: Record<string, string>;
  private readonly fetchImpl: typeof fetch;

  constructor(options: VaaraClientOptions) {
    if (!options.baseUrl) {
      throw new Error("VaaraClient requires a baseUrl");
    }
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.timeoutMs = options.timeoutMs ?? 10_000;
    this.headers = {
      "content-type": "application/json",
      accept: "application/json",
      ...(options.headers ?? {}),
    };
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  // ── Score + outcome ──────────────────────────────────────────────

  async score(req: ScoreRequest): Promise<ScoreResponse> {
    return this.post<ScoreResponse>("/v1/score", req);
  }

  async reportOutcome(req: OutcomeRequest): Promise<{ ok: true }> {
    return this.post<{ ok: true }>("/v1/score/outcome", req);
  }

  // ── Audit ────────────────────────────────────────────────────────

  async appendAuditEvent(req: AuditEventRequest): Promise<AuditEventResponse> {
    return this.post<AuditEventResponse>("/v1/audit/events", req);
  }

  async getActionChain(actionId: string): Promise<{
    action_id: string;
    events: Array<{
      record_id: string;
      event_type: string;
      record_hash: string;
      previous_hash: string;
      timestamp: string;
      payload: Record<string, unknown>;
    }>;
  }> {
    return this.get(
      `/v1/audit/actions/${encodeURIComponent(actionId)}/chain`,
    );
  }

  async verifyAuditChain(): Promise<VerifyResponse> {
    return this.post<VerifyResponse>("/v1/audit/verify", {});
  }

  // ── Policy ───────────────────────────────────────────────────────

  async reloadPolicy(req: PolicyReloadRequest): Promise<PolicyReloadResponse> {
    return this.post<PolicyReloadResponse>("/v1/policy/reload", req);
  }

  // ── Detect ───────────────────────────────────────────────────────

  async detectInjection(
    req: DetectInjectionRequest,
  ): Promise<DetectInjectionResponse> {
    return this.post<DetectInjectionResponse>("/v1/detect/injection", req);
  }

  async detectPII(req: DetectPIIRequest): Promise<DetectPIIResponse> {
    return this.post<DetectPIIResponse>("/v1/detect/pii", req);
  }

  // ── Server identity ──────────────────────────────────────────────

  async serverInfo(): Promise<ServerInfo> {
    return this.get<ServerInfo>("/v1/server");
  }

  async health(): Promise<{ status: "ok" }> {
    return this.get<{ status: "ok" }>("/v1/health");
  }

  // ── Internal HTTP helpers ────────────────────────────────────────

  private async post<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>("POST", path, body);
  }

  private async get<T>(path: string): Promise<T> {
    return this.request<T>("GET", path);
  }

  private async request<T>(
    method: "GET" | "POST",
    path: string,
    body?: unknown,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await this.fetchImpl(url, {
        method,
        headers: this.headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
      const text = await response.text();
      let parsed: unknown = undefined;
      if (text.length > 0) {
        try {
          parsed = JSON.parse(text);
        } catch {
          throw new VaaraTransportError(
            `non-JSON response (status=${response.status})`,
          );
        }
      }
      if (!response.ok) {
        const err = (parsed as { error?: VaaraErrorBody } | undefined)?.error;
        if (err && typeof err.code === "string" && typeof err.message === "string") {
          throw new VaaraError(response.status, err);
        }
        throw new VaaraError(response.status, {
          code: "http_error",
          message: text || response.statusText,
        });
      }
      return parsed as T;
    } catch (err) {
      if (err instanceof VaaraError || err instanceof VaaraTransportError) {
        throw err;
      }
      throw new VaaraTransportError(
        err instanceof Error ? err.message : String(err),
        err,
      );
    } finally {
      clearTimeout(timer);
    }
  }
}
