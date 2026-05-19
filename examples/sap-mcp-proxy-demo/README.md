# Vaara MCP proxy in front of a SAP MCP server

This example shows how to add Vaara's runtime governance layer in front of an existing SAP MCP server (SAP ADT, SAP Graph API, SAP Cloud ALM, any community-built server) without changing the upstream server or your SAP system.

Target reader: an SAP architect or developer already running Claude Code (or any MCP client) against a SAP MCP server. You already have SAP access, an MCP server choice, and a working Claude Code config. You want to add audit, policy gating, and hash-chained evidence aligned to EU AI Act Articles 12 and 14 to every tool call your agent makes against your SAP system.

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
SAP MCP server
(mario-andreschak/mcp-abap-abap-adt-api,
 SAP/mdk-mcp-server, or any other)
    │
    ▼
Your SAP system (BTP, S/4HANA, NetWeaver, etc.)
```

Every `tools/call` request from Claude Code routes through Vaara before reaching the upstream SAP MCP server. Allowed calls forward transparently. Blocked calls return an MCP `isError: true` response with the block reason. Other MCP methods (initialize, tools/list, resources, notifications) forward unchanged.

## Prerequisites

- Python 3.10+
- An existing SAP MCP server. Examples of real community servers:
  - [`mario-andreschak/mcp-abap-abap-adt-api`](https://github.com/mario-andreschak/mcp-abap-abap-adt-api) — wraps ABAP ADT API
  - [`SAP/mdk-mcp-server`](https://github.com/SAP/mdk-mcp-server) — official SAP MCP server for Mobile Development Kit
  - [`lemaiwo/btp-sap-odata-to-mcp-server`](https://github.com/lemaiwo/btp-sap-odata-to-mcp-server) — SAP OData via MCP
  - See [`marianfoo/sap-ai-mcp-servers`](https://github.com/marianfoo/sap-ai-mcp-servers) for the curated index.
- A SAP system the upstream MCP server can talk to (BTP free tier, S/4HANA dev, NetWeaver ABAP Developer Edition, etc.)
- Claude Code or any other MCP-capable client

## Three-step setup

### 1. Install Vaara

```bash
pip install vaara
```

### 2. Replace your Claude Code MCP server entry with the Vaara proxy

Before — Claude Code config pointing directly at the SAP MCP server:

```json
{
  "mcpServers": {
    "sap-adt": {
      "command": "npx",
      "args": ["-y", "@mario-andreschak/mcp-abap-abap-adt-api"],
      "env": {
        "SAP_URL": "https://your-sap-server:44300",
        "SAP_USER": "YOUR_USER",
        "SAP_PASSWORD": "YOUR_PASSWORD",
        "SAP_CLIENT": "100"
      }
    }
  }
}
```

After — Claude Code config pointing at the Vaara proxy, which spawns the same SAP MCP server as a subprocess:

```json
{
  "mcpServers": {
    "sap-adt-via-vaara": {
      "command": "python",
      "args": [
        "-m", "vaara.integrations.mcp_proxy",
        "--upstream", "npx",
        "--upstream-arg", "-y",
        "--upstream-arg", "@mario-andreschak/mcp-abap-abap-adt-api",
        "--db", "/path/to/vaara_audit.db"
      ],
      "env": {
        "SAP_URL": "https://your-sap-server:44300",
        "SAP_USER": "YOUR_USER",
        "SAP_PASSWORD": "YOUR_PASSWORD",
        "SAP_CLIENT": "100"
      }
    }
  }
}
```

The proxy inherits the environment, so SAP credentials flow through to the upstream MCP server unchanged. The upstream sees the same env it would see in the direct setup.

A full example config lives at [`claude_code_config.example.json`](claude_code_config.example.json) in this directory.

### 3. Restart Claude Code and use it normally

Open Claude Code, ask for ABAP work as you would normally ("show me the source of ZCL_INVOICE", "add a method to ZCL_REPORTING", etc.). Every tool call gets intercepted, scored, and recorded.

## What the audit trail captures

After a session, query the audit database:

```bash
sqlite3 /path/to/vaara_audit.db ".dump audit_records" | head -40
```

Each tool call produces a hash-chained sequence of records. A typical session against an ABAP read tool produces:

- An `action_requested` record naming the upstream tool (e.g., `getObjectSource`), agent_id, parameters, and base risk score.
- A `risk_scored` record with the conformal interval and the contributing expert signals.
- A `decision_made` record with `ALLOW` / `DENY` / `ESCALATE` and the reason.
- An `action_executed` or `action_blocked` record depending on the decision.
- An `outcome_recorded` record after the upstream returns, carrying the severity signal back to the scorer.

Escalations to a human reviewer (Article 14) produce additional `escalation_sent` and `escalation_resolved` records on top of this sequence. The chain integrity (each record links to the previous via SHA-256) makes the trail tamper-evident.

See [`audit_sample.jsonl`](audit_sample.jsonl) in this directory for a real session capture (populated when the demo runs end-to-end against a SAP system).

## Customising policy

The default policy ships fail-closed for unknown tool names (the Vaara registry classifies unfamiliar tools as generic high-risk). To tune for your SAP estate:

- Define an explicit policy file naming your high-risk ABAP operations and pass it via `VAARA_POLICY=/path/to/policy.yaml`.
- Override the default agent_id by passing `--agent-id <your-id>` to the proxy.
- Per-call overrides via the `_vaara_agent_id` argument key on a `tools/call` request (the proxy strips it before forwarding).

See the main Vaara README "Policy" section for the full policy file shape.

## Generating evidence for AI Act conformity

After the audit DB has captured live activity:

```bash
vaara compliance report --db /path/to/vaara_audit.db --format json
vaara compliance report --db /path/to/vaara_audit.db --format pdf
vaara trail export --db /path/to/vaara_audit.db
```

The PDF output is the format a Notified Body or internal compliance auditor reads. The `trail export` produces a Sigstore-signed envelope suitable for regulator handoff.

## Troubleshooting

- **The proxy hangs on startup.** The upstream MCP server is probably failing its initialize handshake. Check the proxy's stderr — the upstream's stderr is forwarded through. Common cause: missing SAP credentials in env.
- **Every tool call is blocked.** Default fail-closed policy. Define a policy file, or tune for your specific tool catalog.
- **Audit DB grows quickly.** Each tool call writes 4 records. Use `vaara audit purge --before <date>` to prune old records (the chain integrity is preserved across pruning).

## The same pattern in front of any MCP server

The proxy is MCP-protocol-level, not SAP-specific. The three-step setup above works identically when the upstream MCP server is something else. Ecosystems where this pattern fits today:

- **GitHub MCP** — Claude Code uses this natively. Vaara in front audits every repo read, file edit, PR creation, and issue update the agent triggers against your code.
- **Microsoft Graph MCP** — Email, calendar, OneDrive, Teams. Tool calls into the M365 surface frequently touch GDPR territory.
- **Salesforce MCP** — CRM. HR / recruiting / scoring use cases fall under AI Act Annex III high-risk.
- **ServiceNow MCP** — ITSM, change management, incident handling. Touches operational safety obligations.
- **AWS / GCP / Azure cloud MCP servers** — infrastructure operations. Production change auditability.
- **Databricks MCP** — data platform. Tool calls that read or transform regulated data.

Each ecosystem has its own runtime governance and audit obligation. Vaara is the protocol-neutral substrate that captures the evidence regardless of which upstream MCP server the agent ends up calling.

Contributions adding example configs for any of these (or any other community MCP server) are welcome. The expected shape is a sibling directory under `examples/` following the same three-step recipe as this one.

## License notes

Vaara is Apache-2.0. The community SAP MCP servers have their own licenses (mario-andreschak's is MIT, SAP's official servers vary). Check the upstream server's LICENSE file. The Vaara proxy adds no licensing constraint on top of the upstream.
