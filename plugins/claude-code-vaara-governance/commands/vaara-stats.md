---
description: Print Vaara governance stats from the Claude Code audit trail (records, event types, top tools, last 5 actions).
allowed-tools: Bash(python3:*)
---

Execute this exact command via Bash and show the user the stdout verbatim. Do not summarize, interpret, or add commentary around the output.

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/vaara_stats.py"
```

If the command exits non-zero, print its stderr as-is and stop.
