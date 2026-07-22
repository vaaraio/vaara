#!/usr/bin/env bash
# restore.sh — rebuild the in-container toolchain + reconnect MCPs after a box restart.
# Lives on the /workspace mount so it survives a disposable container.
# Run from inside the box:  bash /workspace/restore.sh
set -euo pipefail

echo "[1/4] install uv (brings back Python 3.13)"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "[2/4] rebuild the repo venv (revives vaara-mcp-server)"
cd /workspace
uv sync

echo "[3/4] register vaara-memory MCP (global/user scope)"
claude mcp add -s user vaara-memory -- /workspace/.venv/bin/vaara-mcp-server

echo "[4/4] register serena MCP (global/user scope, telemetry off)"
claude mcp add -s user serena -e SERENA_USAGE_REPORTING=false -- \
  uvx -p 3.13 --from git+https://github.com/oraios/serena \
  serena start-mcp-server --context claude-code --project-from-cwd

echo
echo "done. now restart Claude Code, then:  claude mcp list"
echo "expect: serena + vaara-memory"
