"""An mcp-use client running through the Vaara governance proxy.

mcp-use (https://github.com/mcp-use/mcp-use) is a framework for building
MCP clients and agents. This demo puts vaara-mcp-proxy between an mcp-use
client and a stock filesystem MCP server, so every tool call is scored,
decided against policy, and written to a hash-chained audit trail before
it reaches the upstream.

No LLM API key needed: this drives the MCP session directly. The same
config dict works unchanged for a full MCPAgent, see the README.

    pip install vaara mcp-use
    python client_via_vaara.py

Requires npx (for @modelcontextprotocol/server-filesystem).
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path

from mcp_use import MCPClient

HERE = Path(__file__).parent
SCOPE = HERE / "scope"
DB = HERE / "audit.db"

# The only mcp-use-specific part is this ordinary mcpServers config.
# Instead of pointing at the filesystem server, it points at the Vaara
# proxy, which spawns and governs the filesystem server.
CONFIG = {
    "mcpServers": {
        "filesystem-via-vaara": {
            "command": sys.executable,
            "args": [
                "-m", "vaara.integrations.mcp_proxy",
                "--upstream", "npx",
                "--upstream-arg=-y",
                "--upstream-arg=@modelcontextprotocol/server-filesystem",
                f"--upstream-arg={SCOPE}",
                "--db", str(DB),
                "--agent-id", "mcp-use-demo",
            ],
        }
    }
}


async def main() -> None:
    SCOPE.mkdir(exist_ok=True)
    (SCOPE / "note.txt").write_text("hello from the demo scope\n")

    client = MCPClient(CONFIG)
    session = await client.create_session("filesystem-via-vaara")

    tools = await session.list_tools()
    print(f"tools visible through the proxy: {len(tools)}")

    result = await session.call_tool(
        "read_text_file", {"path": str(SCOPE / "note.txt")}
    )
    print(f"read_text_file -> {result.content[0].text.strip()!r}")

    await client.close_all_sessions()

    # The call above is now a hash-chained sequence in the audit trail.
    rows = sqlite3.connect(DB).execute(
        "select event_type, tool_name from audit_records order by rowid"
    ).fetchall()
    print(f"\naudit trail ({DB.name}): {len(rows)} records for that one call")
    for event_type, tool_name in rows:
        print(f"  {event_type:18} {tool_name}")


if __name__ == "__main__":
    asyncio.run(main())
