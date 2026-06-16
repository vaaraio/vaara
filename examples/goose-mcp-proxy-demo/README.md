# Vaara MCP proxy as a Goose extension

This example shows how to add Vaara's runtime governance layer in front of any stdio MCP server you use as a [Goose](https://github.com/block/goose) extension, without changing the upstream server or your Goose setup.

Target reader: anyone running Goose Desktop or Goose CLI with one or more MCP servers configured. You want every `tools/call`, `resources/read`, and `prompts/get` your Goose agent makes to land in a hash-chained, tamper-evident audit trail that runs locally and can be exported to a regulator-facing report. Goose's built-in security features (prompt-injection detection, tool permissions, allowlist) keep working untouched. Vaara adds a second layer at the protocol boundary.

## Architecture

```
Goose (Desktop or CLI)
    │
    │  JSON-RPC over stdio
    ▼
Vaara MCP proxy ────────────► Vaara interception pipeline
    │                              │
    │  subprocess stdio             ▼
    ▼                          AuditTrail (hash-chained, SQLite)
Upstream MCP server
(github-mcp-server, filesystem, sqlite, your choice)
    │
    ▼
Underlying API or filesystem
```

Every MCP method that lands on the proxy gets intercepted, scored where applicable, and recorded. Allowed calls forward to the upstream transparently. Blocked calls return an MCP `isError: true` response with the block reason. Initialization handshake forwards unchanged.

## Prerequisites

- Goose Desktop or Goose CLI installed
- Python 3.10+
- One stdio-capable MCP server you want Goose to use (this demo uses [`github/github-mcp-server`](https://github.com/github/github-mcp-server), but any stdio MCP server works the same way)
- For the GitHub variant: a [GitHub Personal Access Token](https://github.com/settings/personal-access-tokens/new) with the scopes you intend to grant the agent

## Three-step setup

### 1. Install Vaara

```bash
pip install vaara
```

### 2. Register Vaara as a Goose Command-line Extension

Run `goose configure` and select **Add Extension then Command-line Extension**. Use these values:

- **Name:** `github-via-vaara` (or whatever name you prefer)
- **Command:** `python -m vaara.integrations.mcp_proxy --upstream /path/to/github-mcp-server --upstream-arg stdio --db /path/to/github_audit.db`
- **Timeout:** `300`
- **Environment variable:** `GITHUB_PERSONAL_ACCESS_TOKEN` set to your token

The interactive prompt walks through each field. After the last prompt Goose writes the extension into `~/.config/goose/config.yaml`.

If you prefer to edit the config directly, append the snippet from [`goose_config.example.yaml`](goose_config.example.yaml) under the top-level `extensions:` key. Both routes produce the same result.

For the **Goose Desktop UI**, the same values go into Settings then Extensions then Add custom extension. Type is `Standard IO`. The command field takes the full Vaara invocation as a single string.

### 3. Restart Goose and use it normally

Open Goose and ask for GitHub work the same way you would normally: read this repo, open a PR, comment on issue #42, rerun the failed workflow. Every tool call gets intercepted, scored, and recorded.

## What the audit trail captures

After a session, query the audit database:

```bash
sqlite3 /path/to/github_audit.db ".dump audit_records" | head -40
```

Each tool call produces a hash-chained sequence of records. A typical session against a PR-creation tool produces:

- An `action_requested` record naming the upstream tool, agent_id, parameters, and base risk score.
- A `risk_scored` record with the conformal interval and the contributing expert signals.
- A `decision_made` record with `ALLOW` / `DENY` / `ESCALATE` and the reason.
- An `action_executed` or `action_blocked` record depending on the decision.
- An `outcome_recorded` record after the upstream returns, carrying the severity signal back to the scorer.

Each record links to the previous via SHA-256. The chain is tamper-evident. `vaara audit verify` checks it.

## How this composes with Goose's existing security features

Goose already ships with a few related controls. They sit at different layers and compose cleanly:

- **`SECURITY_PROMPT_ENABLED` and the prompt-injection classifier.** These run at the LLM input boundary, before the model sees user content. Vaara runs at the MCP tool-call boundary, after the model has decided what to do.
- **Tool permissions and goose-mode (auto, approve, chat, smart_approve).** Control whether a tool runs at all. Vaara records what ran and provides a downstream evidence chain for the ones that did.
- **`.gooseignore`.** Filesystem scoping for the built-in Developer extension. Vaara is upstream-agnostic and records at the protocol layer regardless of which extension produced the call.
- **`GOOSE_TELEMETRY_ENABLED` and the OpenTelemetry exporter.** Operational metrics. Vaara is the per-action governance and evidence layer, distinct from aggregate telemetry.

Running both stacks together gives you input-side detection, tool-permission gating, and an output-side hash-chained record.

## Customising policy

The default policy ships fail-closed for unknown tool names. To tune for the GitHub upstream:

- Start with a read-only policy that allows `get_*`, `list_*`, `search_*` tools and routes mutating tools (`create_*`, `update_*`, `delete_*`, `merge_*`, `push_*`) to ESCALATE.
- Tighten further on the destructive end: any `delete_*` or force-push-equivalent operation goes DENY unless explicitly approved.
- Pass `--allow-tool NAME` / `--deny-tool NAME` repeatedly to the proxy command for inline filtering. Filtered tools are dropped from the `tools/list` response Goose sees, and any matching `tools/call` is rejected at the proxy without contacting the upstream.

See the main Vaara README "Policy" section for the full policy file shape.

## Generating evidence for downstream review

After the audit DB has captured live activity:

```bash
vaara compliance report --db /path/to/github_audit.db --format json
vaara compliance report --db /path/to/github_audit.db --format pdf
vaara trail export --db /path/to/github_audit.db
```

The PDF output is the format a Notified Body or internal compliance reviewer reads. The `trail export` produces a Sigstore-signed envelope suitable for regulator handoff. For organisations subject to EU AI Act high-risk-use obligations on agentic systems, Article 12 (logging) and Article 14 (human oversight) apply at the tool-call layer, not at the LLM layer.

## Troubleshooting

- **Goose says the extension failed to activate.** Check `~/.config/goose/logs/` for the proxy's stderr. The upstream MCP server's stderr is forwarded through the proxy, so missing API tokens and binary-not-found errors surface there.
- **`github-mcp-server: command not found`.** Build it with `go install github.com/github/github-mcp-server/cmd/github-mcp-server@latest` and point `--upstream` at the resulting binary in `$(go env GOBIN)` or `$(go env GOPATH)/bin`. Or use the OCI image and swap the upstream command for `docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server`.
- **Every tool call is blocked.** Default fail-closed policy. Define a policy file scoped to the upstream tool catalog, or relax for read-only tools as a starting point.
- **Timeout warnings on long-running tools.** Raise the `timeout:` value in the Goose extension config. Vaara does not impose its own timeout on the upstream.

## The same pattern in front of any MCP server

The proxy is MCP-protocol-level, not GitHub-specific. The three-step setup above works identically when the upstream is something else. Useful upstreams for Goose users:

- **`@modelcontextprotocol/server-filesystem`.** Filesystem read and write tools, governed at the path level. Command becomes `npx -y @modelcontextprotocol/server-filesystem /path/to/scope`.
- **`@modelcontextprotocol/server-sqlite`.** Database access, governed at the query level.
- **Microsoft Graph MCP, Salesforce MCP, ServiceNow MCP, Databricks MCP.** Enterprise SaaS surfaces where per-call audit has direct compliance value.
- **Sibling demos in the Vaara repo.** [`examples/github-mcp-proxy-demo/`](../github-mcp-proxy-demo/) and [`examples/sap-mcp-proxy-demo/`](../sap-mcp-proxy-demo/) follow the same recipe with different upstreams.

## License notes

Vaara is AGPL-3.0-or-later. Goose is Apache-2.0 (Block, Inc.). The upstream MCP server you wrap retains its own license. The Vaara proxy adds no licensing constraint on top of the upstream.
