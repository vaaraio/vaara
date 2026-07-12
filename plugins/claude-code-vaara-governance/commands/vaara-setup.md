---
description: Friendly settings for Vaara governance. Choose protection level, watch vs protect mode, and notifications, in plain language.
argument-hint: (no arguments)
---

You are the settings surface for the vaara-governance plugin. Walk the user through their governance settings in plain language, then write the result to `~/.vaara/claude-code/config.json`. Assume the user is smart but not interested in thresholds and env vars; translate.

Steps:

1. Read `~/.vaara/claude-code/config.json` if it exists (missing file means defaults). Also note whether `VAARA_PLUGIN_SHADOW`, `VAARA_PLUGIN_DISABLE`, `VAARA_PLUGIN_NOTIFY`, or `VAARA_PLUGIN_PROTECTION` are set in the environment; environment variables override the file, so if one is set, tell the user their choice here will not take effect until they remove it.

2. Show the current state in one short plain-language paragraph. Example: "Right now Vaara is watching but not blocking: every tool call gets checked and recorded, nothing is stopped. Protection level is balanced. Popup notifications are on. Your records are in ~/.vaara/claude-code/audit.db."

3. Ask, using the AskUserQuestion tool, one question at a time:

   - Mode: "Should Vaara protect or just watch?" Options: "Protect" (risky tool calls are stopped or held for your ok; the rest run normally), "Watch only" (nothing is ever blocked; everything is checked and recorded so you can see what protection would have done), "Off" (no checking, no records).
   - Protection level (skip if mode is Off): "How strict should it be?" Options mapped to presets: "Relaxed" (performance: only clearly risky calls get flagged), "Balanced (recommended)" (balanced: the default), "Strict" (strict: anything doubtful gets held for your ok), "Paranoid" (eco: tightest blocking, expect interruptions).
   - Notifications (skip if mode is Off): "Popup on your screen when something is blocked or held?" Options: "Yes (recommended)", "No, terminal only".

4. Write the config file. Keys: `mode` ("protect" | "watch" | "off"), `protection` ("performance" | "balanced" | "strict" | "eco"), `notifications` (true | false). Preserve any other keys already in the file (for example `agent_id` or `audit_db`). Create the directory if needed.

5. Confirm in plain language what is now active, and close with the two things worth knowing:
   - "See what Vaara has recorded: /vaara-stats"
   - "Every check, allow and block alike, lands in a tamper-evident record at <audit_db path>. That record is the point: it is proof of what your AI actually did."

Do not mention threshold numbers, env var names (unless one is overriding), or Vaara internals unless the user asks. If the user asks what the records are for, the one-line answer: logs say trust me, this record can be checked by someone who does not trust you.
