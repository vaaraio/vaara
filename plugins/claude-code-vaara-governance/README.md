# vaara-governance: Claude Code plugin

Runtime tool-call governance for Claude Code. Wires the [Vaara](https://github.com/vaaraio/vaara) risk scorer and hash-chained audit trail into the Claude Code hook system.

## What it does

PreToolUse runs a two-layer check before Claude executes a tool:

**Layer 1: regex deny patterns** (Bash, WebFetch, WebSearch). A JSON deny-list (`policies/default_deny.json`) catches known-bad shapes: AWS / GCP / Azure metadata IPs, `/etc/shadow` reads, `curl | sh`, `rm -rf /`, fork bombs, `dd` to raw block devices, history purges, reverse shells, base64-piped exec, `~/.ssh/authorized_keys` writes. A match is a hard deny: fast, deterministic, no ML.

**Layer 2: Vaara classifier** (`mcp__*` only). MCP tool calls carry structured taxonomy that Vaara's adaptive scorer is trained for; the conformal risk score is meaningful there. The classifier output drives the allow / escalate / deny decision against the loaded policy thresholds.

PostToolUse appends an outcome record to the audit trail for every `mcp__*` call, correlating it back to the PreToolUse decision and feeding the MWU online learner.

Blocks and escalations also pop a native desktop notification (macOS via osascript, Linux via notify-send when present), so a decision is visible even when the terminal is buried. Notifications are fire-and-forget and can never break the hook.

SessionStart prints a one-line status (Vaara version, mode, protection preset, notifications, audit DB path).

| Hook | Matches | Mechanism |
|---|---|---|
| `PreToolUse` | `Bash`, `WebFetch`, `WebSearch`, `mcp__*` | Layer 1 regex on shell / web. Layer 2 ML on MCP. Desktop notification on block/escalate. |
| `PostToolUse` | `mcp__*` | Audit outcome + MWU feedback. |
| `SessionStart` | n/a | Validate install, print status. |

## Install

```
pip install 'vaara>=0.40.1'
```

Then install the plugin via the Claude Code plugin command for your install path (the plugin lives at `plugins/claude-code-vaara-governance/` in the [vaaraio/vaara](https://github.com/vaaraio/vaara) repo).

## Configuration

The friendly path: run `/vaara-setup` inside Claude Code. It asks three plain-language questions (protect, watch, or off; how strict; popups or not) and writes `~/.vaara/claude-code/config.json`. Nothing is required for the default behaviour.

The config file, hand-editable:

```json
{
  "mode": "protect",
  "protection": "balanced",
  "notifications": true
}
```

| Key | Values | Effect |
|---|---|---|
| `mode` | `protect` (default), `watch`, `off` | `watch` checks and records everything but never blocks; `off` disables the plugin. |
| `protection` | `eco`, `balanced` (default), `performance`, `strict` | Policy preset applied to MCP scoring; these are the `vaara mode` presets. |
| `thresholds` | `{"escalate": E, "deny": D}` with `0 <= E <= D <= 1` | Optional custom decision thresholds, overriding the preset's defaults. The preset still provides the rest of the policy shape. Malformed values are ignored. |
| `notifications` | `true` (default), `false` | Desktop popups on block/escalate. |
| `agent_id` | string | Agent id written to the audit chain (default `claude-code`). |
| `audit_db` | path | Audit DB path (default `~/.vaara/claude-code/audit.db`). |
| `article50_statement` | string | When set, SessionStart records this as an EU AI Act Article 50(1) disclosure event into the audit trail with the session id, before the session's first tool call. Off when absent. |
| `fail_open` | `false` (default), `true` | What happens to `mcp__*` calls in protect mode when the `vaara` package is not importable. Default: fail closed (block with an install hint). `true` passes them through unscored. |

Environment variables override the file (useful for CI or a single session):

| Variable | Effect |
|---|---|
| `VAARA_PLUGIN_DISABLE=1` | Disable the whole plugin. All hooks pass through. |
| `VAARA_PLUGIN_SHADOW=1` | Same as `"mode": "watch"`: record every decision, never block. |
| `VAARA_PLUGIN_PROTECTION` | Same as `"protection"`: preset name. |
| `VAARA_PLUGIN_NOTIFY=0` | Turn desktop notifications off. |
| `VAARA_PLUGIN_AGENT_ID` | Override the agent_id written to the audit chain. |
| `VAARA_PLUGIN_AUDIT_DB` | Override the audit DB path. |
| `VAARA_PLUGIN_DENY_PATTERNS_FILE` | Replace the bundled `policies/default_deny.json` with your own. |
| `VAARA_PLUGIN_ARTICLE50_STATEMENT` | Same as `"article50_statement"`. |
| `VAARA_PLUGIN_FAIL_OPEN=1` | Same as `"fail_open": true`. |

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
vaara compliance report --db ~/.vaara/claude-code/audit.db --format md
vaara compliance dashboard --db ~/.vaara/claude-code/audit.db --out ~/audit-dashboard.html
```

For a signed, regulator-handoff bundle, export straight from the plugin's DB with `vaara trail export --db ~/.vaara/claude-code/audit.db --out trail.zip --key PATH`, then verify the zip with `vaara trail verify --zip trail.zip`. Signing needs the export extra: `pip install "vaara[export]"`.

## Latency

PreToolUse on Bash / WebFetch / WebSearch is regex-only and sub-millisecond. PreToolUse on `mcp__*` and PostToolUse load the Vaara pipeline, roughly 0.5 to 2 seconds cold-start on commodity hardware. For tighter MCP latency, run `vaara serve` as a sidecar and reuse a warm process; that path is on the roadmap.

## Known limitations

- Cold-start latency per MCP call (see above).
- PostToolUse correlates to the most recent `ACTION_REQUESTED` for the same agent + tool. Parallel calls to the same MCP tool can race; outcome attribution may swap. The audit chain itself stays intact.
- Outcome severity is a crude mapping from `tool_response.interrupted` / `isError` / `stderr`. Operators tune via policy once they have data.
- This plugin does **not** wire OVERT attestation envelopes. Use `vaara-mcp-proxy` with `--overt-*` flags for envelopes.
- The Vaara ML classifier is not run on raw Bash / WebFetch / WebSearch input. The classifier is trained on structured MCP tool patterns, not shell command strings; its output on raw bash is noise (measured 2026-05-28). Shell-surface coverage is the regex deny-list. Operators who want classifier coverage on shell can train Vaara on their own corpus and wire it in via `VAARA_PLUGIN_AGENT_ID` + a custom hook.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
