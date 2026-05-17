/**
 * Vaara client errors.
 *
 * `VaaraError` wraps a server-side error response (4xx/5xx) so callers
 * can pattern-match on the code (`policy_invalid`, `policy_not_configured`,
 * `invalid_request`, …) without parsing the body themselves.
 *
 * `VaaraTransportError` is raised on network failures and on responses
 * that did not return JSON. Treat as fail-closed: do not assume the
 * server saw the request.
 */

export interface VaaraErrorBody {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export class VaaraError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: Record<string, unknown>;

  constructor(status: number, body: VaaraErrorBody) {
    super(`Vaara API ${status} ${body.code}: ${body.message}`);
    this.name = "VaaraError";
    this.status = status;
    this.code = body.code;
    this.details = body.details ?? {};
  }
}

export class VaaraTransportError extends Error {
  override readonly cause?: unknown;

  constructor(message: string, cause?: unknown) {
    super(`Vaara transport error: ${message}`);
    this.name = "VaaraTransportError";
    this.cause = cause;
  }
}
