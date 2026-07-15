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
  rewrite known MCP client configs through ``vaara-mcp-proxy``, and point
  everything at one trail. Idempotent: it re-asserts the hooks on every run
  (self-heal — a settings.json that was reset or truncated is repaired), and
  re-running never duplicates entries or clobbers the pre-Vaara MCP backup.
* ``run_ungovern`` — remove the Vaara-managed hooks and restore each MCP config
  from its ``.vaara-backup``.

The hooks call the ``vaara`` binary on PATH directly (``vaara hook
pre-tool-use`` etc.), so this does not depend on the plugin marketplace being
installed. Whatever installed the CLI is a complete engine install.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# The single trail every local governance surface writes to. The hook runner
# and MCP proxy are both pointed here so one canonical trail holds every
# action (the mismatch that used to leave "only the demo showing up").
DEFAULT_TRAIL_DB = Path.home() / ".vaara" / "trail" / "audit.db"

# Where the Claude Code hook runner reads its config (audit_db, mode, ...).
CLAUDE_CODE_CONFIG = Path.home() / ".vaara" / "claude-code" / "config.json"

# Claude Code global settings file the hooks are written into.
CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"

# The tool-call surface the hooks intercept — Bash, WebFetch, WebSearch, and
# every MCP tool. Kept identical to the plugin's hooks.json matcher.
HOOK_MATCHER = "Bash|WebFetch|WebSearch|mcp__.*"

# Substring that marks a hook entry as Vaara-managed. Used to find and remove
# our own entries on re-run / ungovern without touching the operator's other
# hooks. Every command we write contains "vaara hook ".
_HOOK_MARKER = "vaara hook "

# Known MCP client config locations. Paths are ~-relative and expanded at
# scan time.
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


def write_hook_config(config_path: Path, trail_db: Path) -> None:
    """Point the hook runner at the shared trail via its config.json.

    Merges ``audit_db`` into any existing config so a truncated or absent file
    is repaired without dropping the operator's other keys (mode, thresholds).
    """
    cfg = _load_json(config_path)
    cfg["audit_db"] = str(trail_db)
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
                               has_backup=has_backup)
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
    service_home: Optional[Path] = None,
    service_system: Optional[str] = None,
    service_runner: Any = None,
) -> InitReport:
    """Set up (or self-heal) local governance in one call.

    With ``proxy_service=True`` the model proxy is also installed as a user
    service (launchd/systemd) so it survives logout — P2 of the plan. The
    ``service_*`` knobs exist for tests; production callers leave them None.
    """
    vaara_bin = vaara_bin or resolve_vaara_bin()
    proxy_bin = proxy_bin or (shutil.which("vaara-mcp-proxy") or "vaara-mcp-proxy")
    report = InitReport(hooks_path=settings_path, trail_db=trail_db)

    report.hooks_changed = write_claude_hooks(settings_path, vaara_bin)
    write_hook_config(config_path, trail_db)

    report.clients = detect_clients(proxy_bin)
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
                count = govern_mcp_config(
                    client.path, proxy_bin, trail_db, shadow=shadow,
                )
                if count:
                    report.mcp_rewritten[client.name] = count

    if proxy_service:
        import subprocess

        from vaara.integrations.proxy_service import install_proxy_service

        service = install_proxy_service(
            vaara_bin=vaara_bin,
            trail_db=str(trail_db),
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
) -> InitReport:
    """Reverse ``run_init``: remove the hooks, restore each MCP config, and
    take down the proxy service if one was installed."""
    proxy_bin = proxy_bin or (shutil.which("vaara-mcp-proxy") or "vaara-mcp-proxy")
    report = InitReport(hooks_path=settings_path)
    report.hooks_changed = remove_claude_hooks(settings_path)
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
