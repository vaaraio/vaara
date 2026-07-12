# Governing an mcp-use agent with the Vaara MCP proxy

[mcp-use](https://github.com/mcp-use/mcp-use) is a fullstack framework for building MCP clients, agents, and servers. This demo puts Vaara's governance proxy between an mcp-use client and a stock MCP server, so every tool call the agent makes is scored, decided against policy, and written to a hash-chained, tamper-evident audit trail before it reaches the upstream. Neither mcp-use nor the upstream server is modified; to mcp-use, the proxy is just another stdio MCP server in the config.

```
mcp-use MCPClient / MCPAgent
    │  JSON-RPC over stdio
    ▼
Vaara MCP proxy ──► score, decide, record (hash-chained SQLite trail)
    │  subprocess stdio
    ▼
@modelcontextprotocol/server-filesystem   (or any stdio MCP server)
```

## Run it

```bash
pip install vaara mcp-use
python client_via_vaara.py
```

Requires `npx` for the filesystem server. No LLM API key is needed: the script drives the MCP session directly, which is also the honest way to demo governance, since the interception happens at the tool-call boundary, not in the model. Verified output:

```
tools visible through the proxy: 14
read_text_file -> 'hello from the demo scope'

audit trail (audit.db): 4 records for that one call
  action_requested   read_text_file
  risk_scored        read_text_file
  decision_made      read_text_file
  outcome_recorded   read_text_file
```

One allowed call produces four chained records: the request, the risk score with its conformal interval, the decision with its reason, and the outcome fed back to the scorer. A blocked call returns an MCP `isError: true` response with the reason, and the block lands in the same chain.

## The whole integration is the config

The only mcp-use-specific part is an ordinary `mcpServers` entry that points at the proxy instead of the upstream:

```python
config = {
    "mcpServers": {
        "filesystem-via-vaara": {
            "command": sys.executable,
            "args": [
                "-m", "vaara.integrations.mcp_proxy",
                "--upstream", "npx",
                "--upstream-arg=-y",
                "--upstream-arg=@modelcontextprotocol/server-filesystem",
                "--upstream-arg=./scope",
                "--db", "audit.db",
                "--agent-id", "mcp-use-demo",
            ],
        }
    }
}
```

Note the `--upstream-arg=VALUE` form: arguments that start with a dash (like `-y`) must be passed with `=` or argparse rejects them.

The same config works unchanged for a full LLM-driven agent:

```python
from langchain_anthropic import ChatAnthropic
from mcp_use import MCPAgent, MCPClient

agent = MCPAgent(llm=ChatAnthropic(model="claude-sonnet-5"), client=MCPClient(config), max_steps=30)
result = await agent.run("Read note.txt and summarise it")
```

Every tool call the model decides to make goes through the same gate and lands in the same trail.

## Tuning the gate

- `--policy policy.yaml` loads thresholds and sequence rules; the starter policy in [examples/policies/mcp-starters/](../policies/mcp-starters/) is a reasonable opening position. The policy bytes feed the attested policy hash, so a silent policy swap is detectable from receipts.
- `--shadow` runs everything non-enforcing: calls are scored and recorded, nothing is blocked, and `vaara trail shadow-report --db audit.db` shows what enforcement would have done. The right first week for an agent already in use.
- `--deny-tool NAME` / `--allow-tool NAME` filter the tool catalog at the proxy; filtered tools disappear from the `tools/list` the agent sees.

## How this composes with mcp-use's own controls

mcp-use ships operational security controls: allowed/disallowed tool lists on the agent, sandboxed execution, and a security logger. Those configure what the agent may attempt and log what happened for the operator. Vaara sits one layer down, at the protocol boundary, and adds the part none of that provides: a policy decision per call and a tamper-evident record an outside party can verify without trusting the operator, exportable as a signed bundle (`vaara trail export --db audit.db ...`). Run both; they do different jobs. The trust-model difference is spelled out in [docs/logs-vs-evidence.md](../../docs/logs-vs-evidence.md).

One operational note: mcp-use enables anonymized telemetry by default; set `MCP_USE_ANONYMIZED_TELEMETRY=false` to turn it off. Vaara itself sends nothing anywhere; the trail is local.

## License notes

Vaara is AGPL-3.0-or-later. mcp-use is MIT. The upstream MCP server keeps its own license. Vaara runs here as an unmodified separate process spoken to over stdio; what the AGPL obliges, and the commercial option if its terms do not work for you, are in [LICENSING.md](../../LICENSING.md).
