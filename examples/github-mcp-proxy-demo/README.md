# Vaara MCP proxy in front of GitHub's official MCP server

This example shows how to add Vaara's runtime governance layer in front of [`github/github-mcp-server`](https://github.com/github/github-mcp-server) (GitHub's official MCP server, MIT-licensed) without changing the upstream server or your GitHub setup.

Target reader: any developer running Claude Code, Cursor, VS Code Copilot, Claude Desktop, or another MCP client against GitHub. You already have a GitHub Personal Access Token, Docker installed, and a working MCP config pointing at the GitHub MCP server. You want every repo read, file edit, PR creation, issue update, and workflow trigger your agent makes to land in a hash-chained audit trail aligned to EU AI Act Articles 12 and 14.

## Architecture

```
Claude Code (MCP client)
    │
    │  JSON-RPC over stdio
    ▼
Vaara MCP proxy ────────────► Vaara interception pipeline
    │                              │
    │  subprocess stdio             ▼
    ▼                          AuditTrail (hash-chained, SQLite)
GitHub MCP server
(ghcr.io/github/github-mcp-server, run via Docker)
    │
    ▼
GitHub API (repos, issues, PRs, Actions, code scanning)
```

Every `tools/call` request from the MCP client routes through Vaara before reaching the upstream GitHub MCP server. Allowed calls forward transparently. Blocked calls return an MCP `isError: true` response with the block reason. Other MCP methods (initialize, tools/list, resources, notifications) forward unchanged.

## Prerequisites

- Python 3.10+
- The `github-mcp-server` binary on disk. Either build from source (`go install github.com/github/github-mcp-server/cmd/github-mcp-server@latest` produces a stdio-capable binary, no external runtime required) or use the OCI image `ghcr.io/github/github-mcp-server` if you already run Docker.
- A [GitHub Personal Access Token](https://github.com/settings/personal-access-tokens/new) with the scopes you want the agent to be able to use (read-only is a sensible starting point, widen later as your trust grows)
- Claude Code or any other MCP-capable client

## Three-step setup

### 1. Install Vaara

```bash
pip install vaara
```

### 2. Replace your MCP server entry with the Vaara proxy

Before. Claude Code config pointing directly at the GitHub MCP server binary:

```json
{
  "mcpServers": {
    "github": {
      "command": "/path/to/github-mcp-server",
      "args": ["stdio"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

After. Claude Code config pointing at the Vaara proxy, which spawns the same GitHub MCP server binary as a subprocess:

```json
{
  "mcpServers": {
    "github-via-vaara": {
      "command": "python",
      "args": [
        "-m", "vaara.integrations.mcp_proxy",
        "--upstream", "/path/to/github-mcp-server",
        "--upstream-arg", "stdio",
        "--db", "/path/to/github_audit.db"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

The proxy inherits the environment, so the GitHub PAT flows through to the upstream MCP server unchanged. The upstream sees the same env it would see in the direct setup.

If you run the upstream via Docker instead of a local binary, swap `--upstream /path/to/github-mcp-server --upstream-arg stdio` for `--upstream docker --upstream-arg run --upstream-arg -i --upstream-arg --rm --upstream-arg -e --upstream-arg GITHUB_PERSONAL_ACCESS_TOKEN --upstream-arg ghcr.io/github/github-mcp-server`. The proxy shape is the same. The only difference is whether Docker wraps the binary or not.

A full example config lives at [`claude_code_config.example.json`](claude_code_config.example.json) in this directory.

### 3. Restart your MCP client and use it normally

Open Claude Code (or whichever client you use) and ask for GitHub work the same way you would normally: "show me the open issues on owner/repo", "open a PR from this branch", "rerun the failed workflow on main", "comment on PR #42". Every tool call gets intercepted, scored, and recorded.

## What the audit trail captures

After a session, query the audit database:

```bash
sqlite3 /path/to/github_audit.db ".dump audit_records" | head -40
```

Each tool call produces a hash-chained sequence of records. A typical session against a PR-creation tool produces:

- An `action_requested` record naming the upstream tool (e.g., `create_pull_request`), agent_id, parameters (owner, repo, base, head, title), and base risk score.
- A `risk_scored` record with the conformal interval and the contributing expert signals.
- A `decision_made` record with `ALLOW` / `DENY` / `ESCALATE` and the reason.
- An `action_executed` or `action_blocked` record depending on the decision.
- An `outcome_recorded` record after the upstream returns, carrying the severity signal back to the scorer.

Escalations to a human reviewer (Article 14) produce additional `escalation_sent` and `escalation_resolved` records on top of this sequence. The chain integrity (each record links to the previous via SHA-256) makes the trail tamper-evident.

## Why this matters for GitHub specifically

GitHub MCP tool calls cluster around a few categories where runtime evidence has real load:

- **Code modification.** `create_or_update_file`, `push_files`, `merge_pull_request`. Auditable record of which agent made which change at which time, separable from the human commit graph.
- **Privilege escalation surfaces.** `update_repository`, branch protection edits, Actions secret reads, workflow dispatches. Useful to gate on policy and route to human review.
- **Notification and identity exposure.** Comments, reviews, mentions. The agent posts in your name, and the trail records that fact.
- **Supply-chain-adjacent operations.** Dependabot alert reads, security advisory access, release publication. Sensitive enough that a hash-chained log is the minimum bar.

For organisations subject to EU AI Act high-risk-use obligations on agentic systems, Article 12 (logging) and Article 14 (human oversight) apply at the tool-call layer, not at the LLM layer. The proxy is where that evidence gets generated.

## Customising policy

The default policy ships fail-closed for unknown tool names. To tune for the GitHub MCP catalog:

- Start with a read-only policy that allows `get_*`, `list_*`, `search_*` tools and routes mutating tools (`create_*`, `update_*`, `delete_*`, `merge_*`, `push_*`) to ESCALATE.
- Tighten further on the destructive end: any `delete_*` or force-push-equivalent operation goes DENY unless explicitly approved.
- Override the default agent_id by passing `--agent-id <your-id>` to the proxy.
- Per-call overrides via the `_vaara_agent_id` argument key on a `tools/call` request (the proxy strips it before forwarding).

See the main Vaara README "Policy" section for the full policy file shape.

## Generating evidence for AI Act conformity

After the audit DB has captured live activity:

```bash
vaara compliance report --db /path/to/github_audit.db --format json
vaara compliance report --db /path/to/github_audit.db --format pdf
vaara trail export --db /path/to/github_audit.db
```

The PDF output is the format a Notified Body or internal compliance auditor reads. The `trail export` produces a Sigstore-signed envelope suitable for regulator handoff.

## Troubleshooting

- **The proxy hangs on startup.** The upstream is probably failing its initialize handshake. Check the proxy's stderr (the upstream's stderr is forwarded through). Common cause: missing or invalid `GITHUB_PERSONAL_ACCESS_TOKEN`.
- **`github-mcp-server: command not found`.** Build it with `go install github.com/github/github-mcp-server/cmd/github-mcp-server@latest` and point `--upstream` at the resulting binary in `$(go env GOBIN)` or `$(go env GOPATH)/bin`.
- **Every tool call is blocked.** Default fail-closed policy. Define a policy file scoped to the GitHub MCP tool catalog, or relax for read-only tools as a starting point.
- **Rate limits surface as upstream errors.** GitHub rate limits hit the upstream, not the proxy. The proxy records the failure in the audit chain and returns the upstream's error to the client.

## The same pattern in front of any MCP server

The proxy is MCP-protocol-level, not GitHub-specific. The three-step setup above works identically when the upstream is something else. Sibling demos and the broader landscape:

- [`examples/sap-mcp-proxy-demo/`](../sap-mcp-proxy-demo/). Vaara in front of community SAP MCP servers (SAP ADT, SAP BTP, SAP Mobile Development Kit).
- **Microsoft Graph MCP.** Email, calendar, OneDrive, Teams. Tool calls into the M365 surface that frequently touch GDPR territory.
- **Salesforce / ServiceNow MCP.** CRM and ITSM. HR and operational-safety obligations.
- **AWS / GCP / Azure cloud MCP servers.** Infrastructure operations. Production change auditability.
- **Databricks MCP.** Data platform. Tool calls that read or transform regulated data.

Contributions adding example configs for any community MCP server are welcome. The expected shape is a sibling directory under `examples/` following the same three-step recipe as this one.

## License notes

Vaara is Apache-2.0. The GitHub MCP server is MIT-licensed by GitHub. The Vaara proxy adds no licensing constraint on top of the upstream.
