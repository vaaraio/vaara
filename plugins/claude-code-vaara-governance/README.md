# vaara-governance — Claude Code plugin

Runtime tool-call governance for Claude Code. Wires the [Vaara](https://github.com/vaaraio/vaara) risk scorer and hash-chained audit trail into the Claude Code hook system.

## What it does

PreToolUse runs a two-layer check before Claude executes a tool:

**Layer 1 — regex deny patterns** (Bash, WebFetch, WebSearch). A JSON deny-list (`policies/default_deny.json`) catches known-bad shapes: AWS / GCP / Azure metadata IPs, `/etc/shadow` reads, `curl | sh`, `rm -rf /`, fork bombs, `dd` to raw block devices, history purges, reverse shells, base64-piped exec, `~/.ssh/authorized_keys` writes. A match is a hard deny — fast, deterministic, no ML.

**Layer 2 — Vaara classifier** (`mcp__*` only). MCP tool calls carry structured taxonomy that Vaara's adaptive scorer is trained for; the conformal risk score is meaningful there. The classifier output drives the allow / escalate / deny decision against the loaded policy thresholds.

PostToolUse appends an outcome record to the audit trail for every `mcp__*` call, correlating it back to the PreToolUse decision and feeding the MWU online learner.

SessionStart prints a one-line status (Vaara version, mode, audit DB path).

| Hook | Matches | Mechanism |
|---|---|---|
| `PreToolUse` | `Bash`, `WebFetch`, `WebSearch`, `mcp__*` | Layer 1 regex on shell / web. Layer 2 ML on MCP. |
| `PostToolUse` | `mcp__*` | Audit outcome + MWU feedback. |
| `SessionStart` | n/a | Validate install, print status. |

## Install

```
pip install 'vaara>=0.40.1'
```

Then install the plugin via the Claude Code plugin command for your install path (the plugin lives at `plugins/claude-code-vaara-governance/` in the [vaaraio/vaara](https://github.com/vaaraio/vaara) repo).

## Configuration

All knobs are environment variables. None required for the default behaviour.

| Variable | Effect |
|---|---|
| `VAARA_PLUGIN_DISABLE=1` | Disable the whole plugin. All hooks pass through. |
| `VAARA_PLUGIN_SHADOW=1` | Record every decision to the audit trail but never block. Useful for soak testing before flipping to enforce. |
| `VAARA_PLUGIN_AGENT_ID` | Override the agent_id written to the audit chain (default `claude-code`). |
| `VAARA_PLUGIN_AUDIT_DB` | Override the audit DB path (default `~/.vaara/claude-code/audit.db`). |
| `VAARA_PLUGIN_DENY_PATTERNS_FILE` | Replace the bundled `policies/default_deny.json` with your own. |

## Extending the deny patterns

Copy `policies/default_deny.json`, edit, and point the plugin at it:

```bash
export VAARA_PLUGIN_DENY_PATTERNS_FILE=~/.vaara/claude-code/my_deny.json
```

Rule shape:

```json
{
  "id": "my_rule",
  "tools": ["Bash"],
  "fields": ["command"],
  "pattern": "regex here",
  "message": "What the operator sees when this fires."
}
```

`fields` names the keys inside `tool_input` to match against (e.g. `command` for Bash, `url` for WebFetch, `query` for WebSearch).

## Reading the audit trail

The plugin writes to a persistent SQLite trail that survives Claude Code restarts. Inspect with the Vaara CLI:

```
vaara trail verify --db ~/.vaara/claude-code/audit.db
vaara trail export --db ~/.vaara/claude-code/audit.db --format json
vaara compliance report --db ~/.vaara/claude-code/audit.db --format markdown
```

## Latency

PreToolUse on Bash / WebFetch / WebSearch is regex-only — sub-millisecond. PreToolUse on `mcp__*` and PostToolUse load the Vaara pipeline, ~0.5 – 2 seconds cold-start on commodity hardware. For tighter MCP latency, run `vaara serve` as a sidecar and reuse a warm process — this path is on the v0.2.0 roadmap.

## Known limitations (v0.1.0)

- Cold-start latency per MCP call (see above).
- PostToolUse correlates to the most recent `ACTION_REQUESTED` for the same agent + tool. Parallel calls to the same MCP tool can race; outcome attribution may swap. The audit chain itself stays intact.
- Outcome severity is a crude mapping from `tool_response.interrupted` / `isError` / `stderr`. Operators tune via policy once they have data.
- This plugin does **not** wire OVERT attestation envelopes. Use `vaara-mcp-proxy` with `--overt-*` flags for envelopes.
- The Vaara ML classifier is not run on raw Bash / WebFetch / WebSearch input. The classifier is trained on structured MCP tool patterns, not shell command strings; its output on raw bash is noise (measured 2026-05-28). Shell-surface coverage is the regex deny-list. Operators who want classifier coverage on shell can train Vaara on their own corpus and wire it in via `VAARA_PLUGIN_AGENT_ID` + a custom hook.

## License

Apache 2.0. See [LICENSE](LICENSE).
