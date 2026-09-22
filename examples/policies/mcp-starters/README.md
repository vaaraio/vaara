# Starter policies for common MCP servers

The Vaara MCP proxy is fail-closed: with no configuration, unknown tools score against
default thresholds and risky calls get blocked or escalated. These starters give you a
working perimeter for six widely used MCP servers so the first run is a config flip,
not a policy-authoring project.

Each starter has two parts:

1. A perimeter: `--allow-tool` / `--deny-tool` flags with exact tool names. This is
   where per-tool control lives. An empty allowlist means "no restriction"; a non-empty
   allowlist restricts to exactly those names; the denylist always wins on overlap.
2. A policy file: [`mcp-starter.policy.json`](mcp-starter.policy.json), passed with
   `--policy`. It sets the default risk thresholds (escalate at 0.35, deny at 0.65,
   slightly stricter than the built-ins) and is covered by the attested policy hash.
   Tune the numbers after you have looked at your own traffic.

Scope notes, so you know exactly what each knob does:

- The policy file's `thresholds.default`, per-action-class `thresholds` overrides, and
  `sequences` are all enforced. A per-class override resolves by the call's tool name, so
  `thresholds: {"tx.sign": {"deny": 0.20}}` tightens that class while others keep the
  default. Exact-name allow/deny still belongs to the perimeter flags; the policy file
  tunes how strict the risk decision is per class. (`vaara policy validate` warns
  `no_action_classes` on the starter file because it declares none. That is expected;
  the starter relies on `thresholds` plus the perimeter, not on taxonomy routing.)
- Several servers bundle create, update, and delete behind one tool name (GitHub's
  `*_write` tools, Google's `manage_*` tools). A name-level allowlist cannot separate
  the safe verb from the destructive one, so the starters deny those tools rather than
  allowlist them. If you need the safe half, relax deliberately.
- Tool catalogs verified against each server's documentation on 2026-07-12. Catalogs
  change; if the upstream renames a tool, an allowlisted name that no longer exists is
  harmless, but a renamed destructive tool falls out of your denylist. Re-check the
  upstream's README when you update it.

Two modes, in the order you should run them:

- Observe first: add `--shadow`. Every call is classified, scored, and recorded but
  nothing is blocked. After a few days, read what enforcement would have done:

  ```bash
  vaara trail shadow-report --db /path/to/audit.db --days 7
  ```

- Enforce: drop `--shadow`, apply the perimeter below, keep the denylist.

All examples follow the same shape as the
[GitHub proxy demo](../../github-mcp-proxy-demo/README.md); swap the upstream command
for your own setup.

## GitHub (github/github-mcp-server)

The current server groups tools into toolsets and bundles mutations into `*_write`
tools. Shrink the surface upstream first with `--toolsets context,repos,issues,pull_requests`,
then apply the Vaara perimeter.

Read-only perimeter (strict start):

```
--allow-tool get_me --allow-tool get_team_members --allow-tool get_teams
--allow-tool get_repository_tree --allow-tool get_file_contents --allow-tool get_commit
--allow-tool list_branches --allow-tool list_commits --allow-tool list_tags
--allow-tool list_releases --allow-tool get_latest_release --allow-tool get_release_by_tag
--allow-tool search_code --allow-tool search_repositories --allow-tool search_commits
--allow-tool list_issues --allow-tool issue_read --allow-tool search_issues
--allow-tool list_pull_requests --allow-tool pull_request_read --allow-tool search_pull_requests
--allow-tool list_notifications --allow-tool get_notification_details
```

Denylist to keep even after you allow mutations (destructive or bundled-delete tools):

```
--deny-tool delete_file --deny-tool push_files --deny-tool create_or_update_file
--deny-tool merge_pull_request --deny-tool label_write --deny-tool projects_write
--deny-tool pull_request_review_write --deny-tool discussion_comment_write
--deny-tool actions_run_trigger
```

`create_or_update_file` and `push_files` overwrite file content and `merge_pull_request`
is irreversible; route them through escalation review before allowing. `label_write`,
`projects_write`, `pull_request_review_write`, and `discussion_comment_write` each hide
a delete operation behind the same name as create.

Full client config:

```json
{
  "mcpServers": {
    "github-via-vaara": {
      "command": "python",
      "args": [
        "-m", "vaara.integrations.mcp_proxy",
        "--upstream", "/path/to/github-mcp-server",
        "--upstream-arg", "stdio",
        "--db", "/path/to/github_audit.db",
        "--policy", "/path/to/mcp-starter.policy.json",
        "--deny-tool", "delete_file",
        "--deny-tool", "push_files",
        "--deny-tool", "create_or_update_file",
        "--deny-tool", "merge_pull_request",
        "--deny-tool", "label_write",
        "--deny-tool", "projects_write",
        "--deny-tool", "pull_request_review_write",
        "--deny-tool", "discussion_comment_write",
        "--deny-tool", "actions_run_trigger"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
      }
    }
  }
}
```

## Filesystem (@modelcontextprotocol/server-filesystem)

The server publishes its own tool annotations; the three tools it marks destructive are
the three in the denylist. `create_directory` mutates but is idempotent and
non-destructive, so it stays allowed in the enforcing tier.

Read-only perimeter:

```
--allow-tool read_text_file --allow-tool read_media_file --allow-tool read_multiple_files
--allow-tool list_directory --allow-tool list_directory_with_sizes --allow-tool directory_tree
--allow-tool search_files --allow-tool get_file_info --allow-tool list_allowed_directories
```

Denylist for enforcing mode:

```
--deny-tool write_file --deny-tool edit_file --deny-tool move_file
```

Note: older docs mention a `read_file` tool; the current server ships
`read_text_file` and `read_media_file` instead.

## Slack (korotovsky/slack-mcp-server)

The maintained community server (the reference `@modelcontextprotocol/server-slack` is
archived). Its message-posting tools are disabled by default upstream and must be
enabled with `SLACK_MCP_ADD_MESSAGE_TOOL`; leave them off until you have reviewed the
trail. Be aware the server also supports unofficial browser-session tokens (xoxc/xoxd);
prefer a proper bot token so the recorded identity is real.

Read-only perimeter:

```
--allow-tool conversations_history --allow-tool conversations_replies
--allow-tool conversations_search_messages --allow-tool conversations_unreads
--allow-tool channels_list --allow-tool users_search --allow-tool usergroups_list
--allow-tool saved_list
```

Denylist for enforcing mode:

```
--deny-tool saved_clear_completed --deny-tool usergroups_users_update
--deny-tool reactions_remove
```

`usergroups_users_update` replaces a group's entire membership in one call, and
`saved_clear_completed` bulk-deletes saved items.

## Postgres (crystaldba/postgres-mcp)

Everything this server exposes is read-only except `execute_sql`, whose semantics
depend entirely on the server's own `--access-mode` flag: `restricted` wraps it
read-only, unrestricted allows arbitrary DML and DDL. Run the upstream in restricted
mode and still deny `execute_sql` at the Vaara layer until you have a reason not to;
the analysis tools cover most agent needs.

Read-only perimeter:

```
--allow-tool list_schemas --allow-tool list_objects --allow-tool get_object_details
--allow-tool explain_query --allow-tool get_top_queries
--allow-tool analyze_workload_indexes --allow-tool analyze_query_indexes
--allow-tool analyze_db_health
```

Denylist:

```
--deny-tool execute_sql
```

The archived reference server (`@modelcontextprotocol/server-postgres`) exposes a
single read-only `query` tool; if you still run it, `--allow-tool query` is the whole
perimeter.

## Google Workspace (taylorwilsdon/google_workspace_mcp)

No official Google server exists; this is the dominant community one. It ships its own
risk controls: set `WORKSPACE_MCP_READ_ONLY=true` and `WORKSPACE_MCP_TOOL_TIER=core`
upstream first, then let Vaara record and gate what remains. The catalog is large
(100+ tools across 12 services), so the starter is a denylist of the highest-risk
tools rather than an allowlist.

Denylist for enforcing mode:

```
--deny-tool run_script_function --deny-tool update_script_content --deny-tool manage_deployment
--deny-tool manage_drive_access --deny-tool set_drive_file_permissions
--deny-tool send_gmail_message --deny-tool manage_gmail_filter --deny-tool manage_gmail_label
```

Why these: `run_script_function` executes arbitrary Apps Script (treat it like shell
access to the Google account), `update_script_content` and `manage_deployment` let an
agent persist code, `manage_drive_access` can transfer ownership and revoke
permissions, and Gmail filter manipulation is a classic silent-exfiltration and
persistence vector. `send_gmail_message` sends as the account owner; allow it only
with escalation review.

## Safari (`safaridriver --mcp`, Safari 27 on macOS 27)

Apple's own MCP server. It drives the user's real Safari: open URLs, read the page,
click and type, run JavaScript in the page, read recorded network requests, and save
screenshots or page text to disk. Turn it on in Safari > Settings > Developer >
"Allow remote automation and external agents". It speaks stdio only.

The catalog below is the server's own `tools/list`, captured from Safari 27.0 on
macOS 27.0 on 2026-09-22 and pinned in `tests/fixtures/safari_mcp_tools_list.json`.
The server publishes no tool annotations, so the tiers are ours.

Under Claude Code you do not need the proxy. Add the server to your MCP config and the
Vaara hook already governs every `mcp__<server>__<tool>` call, by name and by content.
Use the proxy for clients that have no hook to sit in, Codex for instance:

```
vaara-mcp-proxy --upstream safari=/usr/bin/safaridriver --upstream-arg=--mcp \
  --policy examples/policies/mcp-starters/mcp-starter.policy.json \
  --deny-tool evaluate_javascript --deny-tool get_network_request
```

What reaches the agent, and why two tools are denied:

- `evaluate_javascript` runs any script inside the user's logged-in session. A script
  that posts `document.cookie` to another host matches no deny rule, because nothing in
  its text looks like a shell command or a sensitive path. Deny it by name.
- `get_network_request` returns a recorded request with its headers and body, which
  on a logged-in site includes cookies and bearer tokens. Deny it by name;
  `list_network_requests` still gives URLs, methods and status codes.

What the deny rules catch by content, with no configuration:

- `navigate_to_url` or `create_tab` to a cloud metadata address
  (`ssrf_cloud_metadata_ipv4`, and the GCP and Azure host rules).
- `screenshot` or `get_page_content` with a `savePath` into a launch agent, a shell
  rc file, `authorized_keys` or the Claude Code harness config.

Not covered, stated plainly: the proxy does not restrict which sites the agent may
open, and a navigation carries whatever the agent puts in its query string.
`page_interactions` can type into and submit any form on a page the user is logged
into. If the agent should only look, add `--deny-tool page_interactions` and
`--deny-tool browser_dialogs`. Every remote-controlled session also reports
`navigator.webdriver` as true, so sites can see the browser is automated.

Tests: `tests/test_mcp_proxy_safari.py` drives the proxy with this catalog and checks
each claim in this section.

## Verifying what the perimeter did

Every filtered call lands in the audit trail like any other decision. After a session:

```bash
vaara compliance report --db /path/to/audit.db --format json
sqlite3 /path/to/audit.db "select count(*) from audit_records where event_type='action_blocked'"
```

The `--policy` file is hashed into the attested policy hash when OVERT attestation is
enabled, so a silent policy swap between sessions is detectable from the receipts.
