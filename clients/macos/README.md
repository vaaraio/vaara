# Vaara for macOS

Vaara in the menu bar. Not a firewall and not a network filter: the
same governance engine the CLI runs, with a face. The Setup tab points
your AI tools through Vaara's policy gate; from then on every agent
move is decided by the rules you set and recorded in the signed audit
trail, and the menu bar shows the verdicts as they happen.

The Vaara mark sits in the menu bar tinted by your AI agents' latest
verdict, with a small activity sparkline beside it:

- green: moves are passing policy
- yellow: a move was escalated, or an approval is waiting on you
- red: a move was denied (fades back after a configurable window)

Click it for the full picture: which agents are running now and how each
behaved, a live intervention feed, and a History tab holding every
decision on record — allow, escalate, deny — across every watched trail.

It reads the same SQLite audit trails the
[Claude Code plugin](../../plugins/claude-code-vaara-governance/) and
`vaara-mcp-proxy` write. No server, no telemetry; everything stays on
the machine.

## Build

Requires macOS 13+ and the Xcode Command Line Tools
(`xcode-select --install`); no full Xcode needed.

```bash
cd clients/macos
./build.sh          # dist/Vaara.app
./build.sh dmg      # also dist/Vaara.dmg (needs: brew install create-dmg)
open dist/Vaara.app
```

The app is menu-bar only (no Dock icon). Add it to System Settings >
General > Login Items to start it with the machine. The build is ad-hoc
signed: fine on the machine that built it; distribution to other
machines would need Developer ID signing and notarization.

## Setup tab

The app wires your AIs in itself. Setup shows whether the `vaara`
engine is installed (any method works: pip, pipx,
`brew install vaaraio/tap/vaara`), scans the MCP client configs on the
machine (Claude Desktop, Claude Code, Cursor, Windsurf), and shows
which servers run governed. One click on Govern rewrites a client's
ungoverned servers to run through `vaara-mcp-proxy`, after backing up
the original config; Restore puts the pre-Vaara config back. Restart
the client to apply.

## Controls

Everything in the popover is wired to real state, not decoration:

- GATE: Block (protect) / Shadow (watch), writing the plugin's
  `~/.vaara/claude-code/config.json`.
- Protection presets (eco / balanced / performance / strict) and custom
  escalate/deny thresholds, each shown with what it would have decided
  over the last 15 minutes, replayed from the recorded risk scores.
- Watched trails: add any Vaara audit DB, or let "Find trails in
  ~/.vaara" discover them.
- Settings depth selector: Basic shows the essentials (gate mode,
  notifications, updates); Professional adds presets, thresholds, and
  tuning; Enterprise adds multi-trail sources. The same
  `user_level` drives the CLI's `vaara menu`.
- Notify on: nothing, denials only, or all interventions.
- Check for updates: compares the installed engine against the latest
  GitHub release, on click only; nothing phones home in the background.
- Menu-bar activity graph, dark/light, all persisted in
  `~/.vaara/menubar.json`. The CLI's `vaara menu` keeps its own level
  in the plugin's `config.json`.

Note: environment variables (for example `VAARA_PLUGIN_SHADOW=1`)
override the config file, so a switch flipped in the app can be
overruled by a session environment. The plugin README documents the
precedence.

## Approvals

When an approval request file appears under `~/.vaara/approvals/`, the
app raises an Approve/Deny dialog and writes the decision back. Today's
Claude Code plugin does not yet block awaiting that decision; the
dialog surface is ahead of the enforcement wiring, and the limitation
is stated here so nobody mistakes the dialog for enforcement.
