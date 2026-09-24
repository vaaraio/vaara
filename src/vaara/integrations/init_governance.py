# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-command local governance setup: ``vaara init`` / ``vaara ungovern``.

Turns "install + N manual steps" into a single self-healing command. Between
``pip install vaara`` and "every AI action governed" the operator otherwise has
to hand-install the Claude Code plugin, hand-rewrite each MCP client's config to
the proxy, and hand-point model egress. This module does all of it and reverses
it, with pure Python and no extra dependencies, so it ships through the same
pip/pipx/Homebrew channels the CLI already uses.

Two surfaces:

* ``run_init`` — detect installed clients, write the Claude Code
  PreToolUse/PostToolUse/SessionStart hooks into ``~/.claude/settings.json``,
  install the OpenCode plugin and the Cursor hooks where those clients are
  installed, rewrite known MCP
  client configs through ``vaara-mcp-proxy``, and point everything at one
  trail. Idempotent: it re-asserts the hooks on every run
  (self-heal — a settings.json that was reset or truncated is repaired), and
  re-running never duplicates entries or clobbers the pre-Vaara MCP backup.
* ``run_ungovern`` — remove the Vaara-managed hooks and the OpenCode plugin,
  and restore each MCP config from its ``.vaara-backup``.

The hooks call the ``vaara`` binary on PATH directly (``vaara hook
pre-tool-use`` etc.), so this does not depend on the plugin marketplace being
installed. Whatever installed the CLI is a complete engine install.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from vaara.integrations.discovery import (
    DiscoverReport,
    run_discovery,
    write_default_policy,
    write_discovery_config,
)

# The single trail every local governance surface writes to. The hook runner
# and MCP proxy are both pointed here so one canonical trail holds every
# action (the mismatch that used to leave "only the demo showing up").
DEFAULT_TRAIL_DB = Path.home() / ".vaara" / "trail" / "audit.db"

# Where the Claude Code hook runner reads its config (audit_db, mode, ...).
CLAUDE_CODE_CONFIG = Path.home() / ".vaara" / "claude-code" / "config.json"

# Claude Code global settings file the hooks are written into.
CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"

# The tool-call surface the hooks intercept: all of it. Kept identical to the
# plugin's hooks.json matcher, and pinned by a test, because the two drifted.
# This read "Bash|WebFetch|WebSearch|mcp__.*" while the plugin had grown to
# fifteen names, so a `vaara init` install ran deny rules for Write,
# Edit, Agent and Workflow that the matcher never dispatched. A rule the
# matcher drops is dead, and nothing said so.
HOOK_MATCHER = ".*"

# The operating point a fresh install is written with. Equal to what the
# engine and the macOS client both fall back to, so writing it changes no
# behaviour. It makes the choice visible in the file instead of implied:
# an install with no `protection` key gave the client one number to display
# and the scorer another to run, and nothing in between said so.
DEFAULT_PROTECTION_PRESET = "balanced"

# Substring that marks a hook entry as Vaara-managed. Used to find and remove
# our own entries on re-run / ungovern without touching the operator's other
# hooks. Every command we write contains "vaara hook ".
_HOOK_MARKER = "vaara hook "

# Known MCP client config locations. Paths are ~-relative and expanded at
# scan time. OpenCode is not here: its plugin gates every tool call, MCP
# included (see vaara.integrations.opencode). The entry that was here named
# a path in the maintainer's own checkout and a format OpenCode does not use,
# so it governed nothing on any machine.
KNOWN_MCP_CLIENTS: list[tuple[str, str]] = [
    ("Claude Desktop",
     "~/Library/Application Support/Claude/claude_desktop_config.json"),
    ("Claude Code", "~/.claude.json"),
    ("Cursor", "~/.cursor/mcp.json"),
    ("Windsurf", "~/.codeium/windsurf/mcp_config.json"),
]

_HOOK_EVENTS = (
    ("SessionStart", "session-start", None),
    ("PreToolUse", "pre-tool-use", HOOK_MATCHER),
    ("PostToolUse", "post-tool-use", HOOK_MATCHER),
)


@dataclass
class MCPClientStatus:
    """One scanned MCP client config."""

    name: str
    path: Path
    exists: bool
    governed: int = 0
    ungoverned: int = 0
    has_backup: bool = False
    # False when the file exists but holds no ``mcpServers`` map Vaara can
    # read, so its servers can be neither counted nor routed.
    readable: bool = True


@dataclass
class InitReport:
    """Outcome of ``run_init`` / ``run_ungovern`` for CLI rendering."""

    hooks_changed: bool = False
    hooks_path: Path = CLAUDE_SETTINGS
    trail_db: Path = DEFAULT_TRAIL_DB
    mcp_rewritten: dict[str, int] = field(default_factory=dict)
    mcp_restored: list[str] = field(default_factory=list)
    clients: list[MCPClientStatus] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    service_path: Optional[Path] = None
    service_removed: bool = False
    # Auto-discovery fields (populated by ``--auto``).
    auto: bool = False
    discovery: Optional[DiscoverReport] = None
    policy_path: Optional[Path] = None
    config_path: Optional[Path] = None
    # OpenCode plugin: where it was installed (None: OpenCode not found or
    # skipped), whether this run changed it, and whether ungovern removed it.
    opencode_plugin: Optional[Path] = None
    opencode_changed: bool = False
    opencode_removed: bool = False
    # Cursor hooks.json, same three fields.
    cursor_hooks: Optional[Path] = None
    cursor_changed: bool = False
    cursor_removed: bool = False
    # Codex hooks.json, the same three fields, and whether Codex will run
    # the hook (see codex.trust_status).
    codex_hooks: Optional[Path] = None
    codex_changed: bool = False
    codex_removed: bool = False
    codex_trust: str = "missing"


def codex_trust_line(status: str) -> str:
    """Why Codex is not running Vaara's hook, and what to do about it."""
    if status == "disabled":
        return "Vaara's hook is turned off in Codex's /hooks. Turn it back on there."
    if status == "unknown":
        return ("could not read ~/.codex/config.toml to see whether the hook is "
                "trusted, and Codex runs a hook only once it is.")
    return ("until you trust the hook. Codex asks at its next start (choose "
            "\"Trust all and continue\"), or review it in /hooks. codex exec "
            "skips an untrusted hook without asking.")


@dataclass
class Coverage:
    """What ``vaara init`` can say about one agent found on this machine."""

    name: str
    state: str  # "governed", "MCP only" or "NOT governed"
    detail: str


#: Agents Vaara has no adapter for yet, by the binaries and directories that
#: show one is installed. Listing them is the point: an agent nobody names
#: is an agent the operator believes is governed.
_UNADAPTED = (
    ("Gemini CLI", ("gemini",), ("~/.gemini",)),
    ("Windsurf", ("windsurf",), ("~/.codeium/windsurf",)),
)


def _installed(binaries: tuple, dirs: tuple, which: Any) -> bool:
    return (any(which(b) for b in binaries)
            or any(Path(d).expanduser().is_dir() for d in dirs))


def coverage(report: InitReport, *, which: Any = shutil.which) -> list[Coverage]:
    """Every agent found on this machine, and whether Vaara governs it.

    ``init`` used to print what it wrote and then "Vaara is governing",
    whatever it had found. An agent with no adapter got no line at all, and
    an MCP config in a shape Vaara could not read was skipped without a word.
    """
    rows: list[Coverage] = []
    every = "every tool call, through its hooks"
    if _installed(("claude",), ("~/.claude",), which):
        rows.append(Coverage("Claude Code", "governed", every))
    if report.cursor_hooks is not None:
        rows.append(Coverage("Cursor", "governed", every))
    if report.opencode_plugin is not None:
        rows.append(Coverage("OpenCode", "governed",
                             "every tool call, through its plugin"))
    if report.codex_hooks is not None:
        if report.codex_trust == "trusted":
            rows.append(Coverage("Codex", "governed", every))
        else:
            rows.append(Coverage("Codex", "NOT governed",
                                 codex_trust_line(report.codex_trust)))

    mcp = {c.name: c for c in report.clients if c.exists}

    def routed(client: MCPClientStatus) -> tuple[int, int]:
        done = report.mcp_rewritten.get(client.name, 0)
        return client.governed + done, client.ungoverned - done

    desktop = mcp.get("Claude Desktop")
    if desktop is not None:
        on, off = routed(desktop)
        if not desktop.readable:
            rows.append(Coverage("Claude Desktop", "NOT governed",
                                 f"could not read the MCP servers in {desktop.path}"))
        elif off:
            rows.append(Coverage("Claude Desktop", "NOT governed",
                                 f"{off} MCP server(s) not routed through vaara-mcp-proxy"))
        else:
            rows.append(Coverage("Claude Desktop", "governed",
                                 f"its {on} MCP server(s), through vaara-mcp-proxy"))

    for name, binaries, dirs in _UNADAPTED:
        client = mcp.get(name)
        if client is None and not _installed(binaries, dirs, which):
            continue
        own = "no Vaara adapter yet, so its own tool calls run unchecked"
        on, off = routed(client) if client is not None else (0, 0)
        if client is not None and not client.readable:
            own += f"; could not read the MCP servers in {client.path}"
        elif off:
            own += f"; {off} MCP server(s) not routed through vaara-mcp-proxy"
        if on:
            rows.append(Coverage(name, "MCP only",
                                 f"{on} MCP server(s) through vaara-mcp-proxy; {own}"))
        else:
            rows.append(Coverage(name, "NOT governed", own))
    return rows


def resolve_vaara_bin() -> str:
    """Absolute path to the ``vaara`` binary, or the bare name as a fallback.

    Hooks run under Claude Code, which may launch with a minimal PATH, so an
    absolute path is preferred. Falls back to ``"vaara"`` when the binary is not
    found on PATH (e.g. tests, or an editable install invoked as a module).
    """
    return shutil.which("vaara") or "vaara"


def _hook_command(vaara_bin: str, subcommand: str) -> str:
    return f"{vaara_bin} hook {subcommand}"


# ---------------------------------------------------------------------------
# Claude Code hooks in ~/.claude/settings.json


def _load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        # A missing, empty, or truncated settings.json (the 27-byte-reset bug)
        # is treated as "no settings"; write_claude_hooks rebuilds it.
        return {}


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".vaara-tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


def _strip_vaara_hooks(hooks: dict) -> dict:
    """Return a copy of the ``hooks`` block with every Vaara-managed entry gone.

    An entry group is dropped when all of its inner hooks are ours; inner hooks
    that are ours are removed from mixed groups, and empty groups are pruned.
    Non-Vaara hooks are preserved untouched.
    """
    cleaned: dict = {}
    for event, groups in hooks.items():
        if not isinstance(groups, list):
            cleaned[event] = groups
            continue
        new_groups = []
        for group in groups:
            if not isinstance(group, dict):
                new_groups.append(group)
                continue
            inner = group.get("hooks")
            if not isinstance(inner, list):
                new_groups.append(group)
                continue
            kept = [
                h for h in inner
                if not (isinstance(h, dict)
                        and isinstance(h.get("command"), str)
                        and _HOOK_MARKER in h["command"])
            ]
            if not kept:
                continue  # whole group was ours
            new_group = dict(group)
            new_group["hooks"] = kept
            new_groups.append(new_group)
        if new_groups:
            cleaned[event] = new_groups
    return cleaned


def _vaara_hook_groups(vaara_bin: str) -> dict:
    """Build the Vaara-managed hook entries keyed by event."""
    groups: dict = {}
    for event, subcommand, matcher in _HOOK_EVENTS:
        entry: dict = {
            "hooks": [
                {
                    "type": "command",
                    "command": _hook_command(vaara_bin, subcommand),
                    "timeout": 30,
                }
            ]
        }
        if matcher is not None:
            entry["matcher"] = matcher
        groups.setdefault(event, []).append(entry)
    return groups


def write_claude_hooks(settings_path: Path, vaara_bin: str) -> bool:
    """Idempotently write the Vaara hooks into ``settings_path``.

    Any prior Vaara-managed entries are stripped first, then re-added fresh, so
    this both de-duplicates on re-run and self-heals a settings file that was
    reset or lost the hooks. Other hooks and settings are preserved. Returns
    True when the file content changed.
    """
    settings = _load_json(settings_path)
    before = json.dumps(settings, sort_keys=True)

    hooks = settings.get("hooks")
    hooks = _strip_vaara_hooks(hooks) if isinstance(hooks, dict) else {}

    for event, entries in _vaara_hook_groups(vaara_bin).items():
        hooks.setdefault(event, [])
        hooks[event].extend(entries)

    settings["hooks"] = hooks
    after = json.dumps(settings, sort_keys=True)
    if after == before:
        return False
    _atomic_write_json(settings_path, settings)
    return True


def remove_claude_hooks(settings_path: Path) -> bool:
    """Remove the Vaara-managed hooks from ``settings_path``.

    Returns True when the file changed. Leaves all other hooks and settings
    intact; drops an empty ``hooks`` block entirely.
    """
    if not settings_path.exists():
        return False
    settings = _load_json(settings_path)
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return False
    before = json.dumps(settings, sort_keys=True)
    cleaned = _strip_vaara_hooks(hooks)
    if cleaned:
        settings["hooks"] = cleaned
    else:
        settings.pop("hooks", None)
    after = json.dumps(settings, sort_keys=True)
    if after == before:
        return False
    _atomic_write_json(settings_path, settings)
    return True


def write_hook_config(config_path: Path, trail_db: Path, *,
                      shadow: bool = False, auto: bool = False,
                      auto_preset: Optional[str] = None) -> None:
    """Point the hook runner at the shared trail via its config.json.

    Merges ``audit_db`` into any existing config so a truncated or absent file
    is repaired without dropping the operator's other keys (mode, thresholds).

    ``protection`` is written only when the key is absent, so an operator who
    has chosen a preset keeps it. Writing it changes no behaviour, because the
    engine and the macOS client already land on balanced without it. It makes
    the operating point readable in the file, which is where an operator looks
    when asking why a call was scored the way it was.

    ``shadow`` sets ``mode: watch``: the operator asked for it. ``auto``
    sets ``mode: watch`` and the ``auto_preset`` protection only where those
    keys are absent, because ``--auto`` is documented as a shadow-mode start
    at its preset. The hook reads this file and no other, so until this was
    written ``--shadow`` and ``--auto`` left the hooks blocking.
    """
    cfg = _load_json(config_path)
    cfg["audit_db"] = str(trail_db)
    if shadow:
        cfg["mode"] = "watch"
    if auto:
        cfg.setdefault("mode", "watch")
        if auto_preset:
            cfg.setdefault("protection", auto_preset)
    cfg.setdefault("protection", DEFAULT_PROTECTION_PRESET)
    _atomic_write_json(config_path, cfg)


# ---------------------------------------------------------------------------
# MCP client configs (govern/restore)


def _backup_path(config: Path) -> Path:
    return config.with_name(config.name + ".vaara-backup")


def _is_governed(server: dict, proxy_bin: str) -> bool:
    command = server.get("command")
    proxy_name = os.path.basename(proxy_bin)
    return isinstance(command, str) and (
        command == proxy_bin or os.path.basename(command) == proxy_name
    )


def scan_mcp_client(name: str, raw_path: str, proxy_bin: str) -> MCPClientStatus:
    path = Path(raw_path).expanduser()
    if not path.exists():
        return MCPClientStatus(name=name, path=path, exists=False)
    has_backup = _backup_path(path).exists()
    try:
        obj = json.loads(path.read_text())
        servers = obj.get("mcpServers", {})
        if not isinstance(servers, dict):
            raise ValueError
    except (OSError, json.JSONDecodeError, ValueError):
        return MCPClientStatus(name=name, path=path, exists=True,
                               has_backup=has_backup, readable=False)
    governed = naked = 0
    for server in servers.values():
        if not isinstance(server, dict):
            continue
        if _is_governed(server, proxy_bin):
            governed += 1
        elif isinstance(server.get("command"), str):
            naked += 1
    return MCPClientStatus(name=name, path=path, exists=True,
                           governed=governed, ungoverned=naked,
                           has_backup=has_backup)


def detect_clients(proxy_bin: str) -> list[MCPClientStatus]:
    return [scan_mcp_client(name, path, proxy_bin)
            for name, path in KNOWN_MCP_CLIENTS]


def govern_mcp_config(
    config_path: Path,
    proxy_bin: str,
    trail_db: Path,
    *,
    shadow: bool = False,
) -> Optional[int]:
    """Rewrite every naked stdio MCP server through the proxy.

    Returns the number of servers rewritten, 0 if there were none to rewrite,
    or None if the config could not be read/parsed. Writes a one-time
    ``.vaara-backup`` of the pre-Vaara config before the first rewrite and never
    overwrites it, so ``restore_mcp_config`` always recovers the original.
    """
    try:
        obj = json.loads(config_path.read_text())
        servers = obj.get("mcpServers")
        if not isinstance(servers, dict):
            return None
    except (OSError, json.JSONDecodeError):
        return None

    backup = _backup_path(config_path)
    if not backup.exists():
        shutil.copy2(config_path, backup)

    trail_db.parent.mkdir(parents=True, exist_ok=True)

    rewritten = 0
    for server_name, value in servers.items():
        if not isinstance(value, dict):
            continue
        command = value.get("command")
        if not isinstance(command, str) or _is_governed(value, proxy_bin):
            continue
        args = ["--upstream", command]
        for arg in value.get("args", []) or []:
            args += ["--upstream-arg", str(arg)]
        args += ["--db", str(trail_db), "--agent-id", f"mcp:{server_name}"]
        if shadow:
            args.append("--shadow")
        new_server = dict(value)
        new_server["command"] = proxy_bin
        new_server["args"] = args
        servers[server_name] = new_server
        rewritten += 1

    if rewritten == 0:
        return 0
    obj["mcpServers"] = servers
    _atomic_write_json(config_path, obj)
    return rewritten


def restore_mcp_config(config_path: Path) -> bool:
    """Put the pre-Vaara config back from its ``.vaara-backup``.

    Returns True when a backup existed and was restored.
    """
    backup = _backup_path(config_path)
    if not backup.exists():
        return False
    shutil.copy2(backup, config_path)
    return True


# ---------------------------------------------------------------------------
# Orchestration


def run_init(
    *,
    trail_db: Path = DEFAULT_TRAIL_DB,
    settings_path: Path = CLAUDE_SETTINGS,
    config_path: Path = CLAUDE_CODE_CONFIG,
    vaara_bin: Optional[str] = None,
    proxy_bin: Optional[str] = None,
    shadow: bool = False,
    govern_mcp: bool = True,
    proxy_service: bool = False,
    proxy_enforce: bool = False,
    proxy_allow: Optional[list[str]] = None,
    service_home: Optional[Path] = None,
    service_system: Optional[str] = None,
    service_runner: Any = None,
    # Auto-discovery.
    auto: bool = False,
    mode: str = "eco",
    set_hook_mode: bool = True,
    govern_opencode: bool = True,
    opencode_dir: Optional[Path] = None,
    govern_cursor: bool = True,
    cursor_dir: Optional[Path] = None,
    govern_codex: bool = True,
    codex_dir: Optional[Path] = None,
) -> InitReport:
    """Set up (or self-heal) local governance in one call.

    With ``proxy_service=True`` the model proxy is also installed as a user
    service (launchd/systemd) so it survives logout — P2 of the plan. The
    ``service_*`` knobs exist for tests; production callers leave them None.

    When ``auto=True`` the function runs full environment discovery and
    generates a default shadow-mode policy and unified config before doing
    the standard init steps — a true "one command and vamos" entry point.
    """
    vaara_bin = vaara_bin or resolve_vaara_bin()
    proxy_bin = proxy_bin or (shutil.which("vaara-mcp-proxy") or "vaara-mcp-proxy")
    report = InitReport(hooks_path=settings_path, trail_db=trail_db, auto=auto)

    # Auto-discovery: scan environment, generate policy + config.
    if auto:
        discovery = run_discovery()
        report.discovery = discovery
        report.policy_path = write_default_policy(discovery, shadow=True,
                                                   mode_name=mode)
        report.config_path = write_discovery_config(discovery, shadow=True,
                                                     trail_db=trail_db)
        report.warnings.append(
            f"Auto-discovery complete: {len(discovery.agents)} agent(s), "
            f"{len(discovery.mcp_clients)} MCP client(s), "
            f"{len(discovery.sensitive_paths)} sensitive path(s), "
            f"{len(discovery.known_tools)} known tool(s)."
        )

    report.hooks_changed = write_claude_hooks(settings_path, vaara_bin)
    # ``set_hook_mode=False`` keeps the hook's mode and preset as they are.
    # The silent first-run setup uses it: it runs on the first use of any
    # command, and letting it switch the hooks to watch would stop them
    # blocking with no word to the operator.
    if set_hook_mode:
        write_hook_config(config_path, trail_db, shadow=shadow, auto=auto,
                          auto_preset=mode if auto else None)
    else:
        write_hook_config(config_path, trail_db)

    if govern_opencode:
        from vaara.integrations import opencode

        if opencode.detected(opencode_dir):
            report.opencode_changed = opencode.install_plugin(vaara_bin, opencode_dir)
            report.opencode_plugin = opencode.plugin_path(opencode_dir)

    if govern_cursor:
        from vaara.integrations import cursor

        if cursor.detected(cursor_dir):
            report.cursor_changed = cursor.install_hooks(vaara_bin, cursor_dir)
            report.cursor_hooks = cursor.hooks_path(cursor_dir)

    if govern_codex:
        from vaara.integrations import codex

        if codex.detected(codex_dir):
            report.codex_changed = codex.install_hooks(vaara_bin, codex_dir)
            report.codex_hooks = codex.hooks_path(codex_dir)
            report.codex_trust = codex.trust_status(codex_dir)

    report.clients = detect_clients(proxy_bin)
    for client in report.clients:
        if (client.name == "Cursor" and client.governed
                and report.cursor_hooks is not None):
            report.warnings.append(
                f"Cursor's MCP servers in {client.path} also run through "
                f"vaara-mcp-proxy from an earlier init, so each Cursor MCP call "
                f"is decided twice. Restore that file from "
                f"{client.path.name}.vaara-backup to leave it to the hook."
            )
    if govern_mcp:
        if shutil.which("vaara-mcp-proxy") is None:
            report.warnings.append(
                "vaara-mcp-proxy not found on PATH; MCP client configs were "
                "left unchanged. Install the proxy to govern MCP traffic."
            )
        else:
            for client in report.clients:
                if not client.exists or client.ungoverned == 0:
                    continue
                if client.name == "Cursor" and report.cursor_hooks is not None:
                    # Cursor's preToolUse hook already decides its MCP calls;
                    # the proxy as well would decide each one twice.
                    continue
                count = govern_mcp_config(
                    client.path, proxy_bin, trail_db, shadow=shadow or auto,
                )
                if count:
                    report.mcp_rewritten[client.name] = count

    if proxy_service:
        import subprocess

        from vaara.integrations.proxy_service import install_proxy_service

        service = install_proxy_service(
            vaara_bin=vaara_bin,
            trail_db=str(trail_db),
            enforce=proxy_enforce,
            allow=proxy_allow,
            home=service_home,
            system=service_system,
            runner=service_runner or subprocess.run,
        )
        report.service_path = service.path if service.installed else None
        report.warnings.extend(service.warnings)
    return report


def run_ungovern(
    *,
    settings_path: Path = CLAUDE_SETTINGS,
    proxy_bin: Optional[str] = None,
    service_home: Optional[Path] = None,
    service_system: Optional[str] = None,
    service_runner: Any = None,
    opencode_dir: Optional[Path] = None,
    cursor_dir: Optional[Path] = None,
    codex_dir: Optional[Path] = None,
) -> InitReport:
    """Reverse ``run_init``: remove the hooks and the OpenCode plugin, restore
    each MCP config, and take down the proxy service if one was installed."""
    from vaara.integrations import opencode

    proxy_bin = proxy_bin or (shutil.which("vaara-mcp-proxy") or "vaara-mcp-proxy")
    report = InitReport(hooks_path=settings_path)
    report.hooks_changed = remove_claude_hooks(settings_path)
    report.opencode_removed = opencode.remove_plugin(opencode_dir)
    from vaara.integrations import cursor

    report.cursor_removed = cursor.remove_hooks(cursor_dir)
    from vaara.integrations import codex

    report.codex_removed = codex.remove_hooks(codex_dir)
    for name, raw_path in KNOWN_MCP_CLIENTS:
        path = Path(raw_path).expanduser()
        if restore_mcp_config(path):
            report.mcp_restored.append(name)

    import subprocess

    from vaara.integrations.proxy_service import uninstall_proxy_service

    report.service_removed = uninstall_proxy_service(
        home=service_home,
        system=service_system,
        runner=service_runner or subprocess.run,
    )
    return report
