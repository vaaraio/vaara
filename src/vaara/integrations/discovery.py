# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Auto-discovery for ``vaara init --auto``.

Detects what's running on the machine — AI agents, shells, MCP clients,
sensitive paths, known tools — and generates a sensible default policy so the
operator goes from ``brew install`` to governed with zero config.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Known AI agent processes (binary name → display name)
# ---------------------------------------------------------------------------

KNOWN_AI_AGENTS: dict[str, str] = {
    "claude": "Claude Desktop",
    "claude-code": "Claude Code",
    "claude-code-server": "Claude Code (server)",
    "cursor": "Cursor",
    "cursor-server": "Cursor (server)",
    "windsurf": "Windsurf",
    "windsurf-server": "Windsurf (server)",
}

# Known MCP client config locations (name → ~-relative path).
KNOWN_MCP_CLIENTS: list[tuple[str, str]] = [
    ("Claude Desktop",
     "~/Library/Application Support/Claude/claude_desktop_config.json"),
    ("Claude Code", "~/.claude.json"),
    ("Cursor", "~/.cursor/mcp.json"),
    ("Windsurf", "~/.codeium/windsurf/mcp_config.json"),
    ("OpenCode", "~/.config/opencode/opencode.json"),
]

# Sensitive filesystem paths that no autonomous agent should read/write
# without explicit escalation.
SENSITIVE_PATH_PATTERNS: list[str] = [
    ".ssh",
    ".aws",
    ".gnupg",
    ".config/git",
    ".config/opencode",
    ".config/claude",
    ".vaara",
    "Library/Keychains",
    "Library/Application Support/1Password",
    "Library/Application Support/Bitwarden",
    ".password-store",
    ".kube",
    ".docker",
]

# Well-known tools an AI might invoke autonomously.
KNOWN_TOOLS: dict[str, str] = {
    "git": "Version control (git push, git commit, git reset)",
    "npm": "Node package manager (npm install, npm publish)",
    "pip": "Python package installer (pip install, pip uninstall)",
    "brew": "Homebrew package manager (brew install, brew uninstall)",
    "curl": "HTTP client (curl, wget — network egress)",
    "rm": "File deletion (rm -rf)",
    "chmod": "Permission changes (chmod, chown)",
    "sudo": "Privilege escalation",
    "kubectl": "Kubernetes control",
    "terraform": "Infrastructure provisioning",
}

# Known AI web session URL patterns (for the Safari/WebKit extension).
KNOWN_AI_WEB_DOMAINS: list[str] = [
    "chatgpt.com",
    "chat.openai.com",
    "claude.ai",
    "gemini.google.com",
    "perplexity.ai",
    "copilot.microsoft.com",
    "poe.com",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class AgentInfo:
    binary: str
    display_name: str
    running: bool
    pid: Optional[int] = None
    governed: bool = False


@dataclass
class ShellInfo:
    path: str
    version: str
    wrappable: bool  # can we wrap it with vaara proxy shell?
    rc_file: Optional[str] = None


@dataclass
class MCPClientInfo:
    name: str
    config_path: Path
    exists: bool
    mcp_servers: int = 0
    governed_servers: int = 0


@dataclass
class DiscoverReport:
    agents: list[AgentInfo] = field(default_factory=list)
    shell: Optional[ShellInfo] = None
    mcp_clients: list[MCPClientInfo] = field(default_factory=list)
    sensitive_paths: list[Path] = field(default_factory=list)
    known_tools: dict[str, str] = field(default_factory=dict)
    os: str = ""
    vaara_version: str = ""


# ---------------------------------------------------------------------------
# Discovery functions
# ---------------------------------------------------------------------------


def discover_agents() -> list[AgentInfo]:
    """Find known AI agent processes currently running on the machine."""
    running_binaries = _running_processes()
    agents: list[AgentInfo] = []
    for binary, name in KNOWN_AI_AGENTS.items():
        pids = running_binaries.get(binary, [])
        agents.append(AgentInfo(
            binary=binary,
            display_name=name,
            running=len(pids) > 0,
            pid=pids[0] if pids else None,
        ))
    return agents


def _running_processes() -> dict[str, list[int]]:
    """Return {binary_name: [pid, ...]} for every running process."""
    result: dict[str, list[int]] = {}
    try:
        if platform.system() == "Darwin" or platform.system() == "Linux":
            output = subprocess.run(
                ["ps", "-eo", "comm=", "-eo", "pid="],
                capture_output=True, text=True, timeout=5,
            )
            for line in output.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(maxsplit=1)
                if len(parts) != 2:
                    continue
                comm, pid = parts
                basename = os.path.basename(comm).lower()
                result.setdefault(basename, []).append(int(pid))
    except (OSError, subprocess.TimeoutExpired):
        pass
    return result


def discover_shell() -> Optional[ShellInfo]:
    """Detect the user's shell and check if it's wrappable."""
    shell_path = os.environ.get("SHELL", "")
    if not shell_path:
        return None
    shell_name = os.path.basename(shell_path).lower()
    wrappable = shell_name in ("bash", "zsh", "fish")
    version = ""
    try:
        out = subprocess.run(
            [shell_path, "--version"],
            capture_output=True, text=True, timeout=3,
        )
        version = out.stdout.split("\n")[0][:80] if out.stdout else ""
    except (OSError, subprocess.TimeoutExpired):
        pass
    rc_map = {"bash": "~/.bashrc", "zsh": "~/.zshrc", "fish": "~/.config/fish/config.fish"}
    return ShellInfo(
        path=shell_path,
        version=version,
        wrappable=wrappable,
        rc_file=rc_map.get(shell_name),
    )


def discover_mcp_clients(proxy_bin: str = "vaara-mcp-proxy") -> list[MCPClientInfo]:
    """Scan known MCP client config files and report governance status."""
    clients: list[MCPClientInfo] = []
    for name, raw_path in KNOWN_MCP_CLIENTS:
        path = Path(raw_path).expanduser()
        if not path.exists():
            clients.append(MCPClientInfo(name=name, config_path=path, exists=False))
            continue
        servers = _count_mcp_servers(path, proxy_bin)
        clients.append(MCPClientInfo(
            name=name, config_path=path, exists=True, **servers,
        ))
    return clients


def _count_mcp_servers(config_path: Path, proxy_bin: str) -> dict:
    try:
        obj = json.loads(config_path.read_text())
        servers = obj.get("mcpServers", {})
        if not isinstance(servers, dict):
            return {"mcp_servers": 0, "governed_servers": 0}
    except (OSError, json.JSONDecodeError):
        return {"mcp_servers": 0, "governed_servers": 0}
    proxy_name = os.path.basename(proxy_bin)
    total = len(servers)
    governed = sum(
        1 for s in servers.values()
        if isinstance(s, dict) and os.path.basename(s.get("command", "")) == proxy_name
    )
    return {"mcp_servers": total, "governed_servers": governed}


def discover_sensitive_paths() -> list[Path]:
    """Find sensitive paths on this machine."""
    home = Path.home()
    found: list[Path] = []
    for pattern in SENSITIVE_PATH_PATTERNS:
        p = home / pattern
        if p.exists():
            found.append(p)
    return found


def discover_known_tools() -> dict[str, str]:
    """Check which well-known tools are available on PATH."""
    available: dict[str, str] = {}
    for tool, description in KNOWN_TOOLS.items():
        if shutil.which(tool):
            available[tool] = description
    return available


def run_discovery() -> DiscoverReport:
    """Run all discovery and return a composite report."""
    from vaara import __version__
    return DiscoverReport(
        agents=discover_agents(),
        shell=discover_shell(),
        mcp_clients=discover_mcp_clients(),
        sensitive_paths=discover_sensitive_paths(),
        known_tools=discover_known_tools(),
        os=f"{platform.system()} {platform.release()}",
        vaara_version=__version__,
    )


# ---------------------------------------------------------------------------
# Default policy generation
# ---------------------------------------------------------------------------


def generate_default_policy(
    discovery: DiscoverReport,
    *,
    shadow: bool = True,
    mode_name: str = "eco",
) -> dict:
    """Generate a sensible-default policy from discovered environment.

    Returns a dict ready to serialize as the policy JSON/YAML that the
    ``PolicyController`` can load.  The policy:
    - Registers action classes for every discovered tool category.
    - Uses the named mode's ``escalate`` / ``deny`` thresholds (default ``eco``).
    - Tags sensitive paths so file-access guards can use them.
    - Lists known agent IDs for provenance tracking.
    - Enables shadow mode when ``shadow=True`` so the operator can review
      what would be blocked before turning on enforcement.
    """
    from vaara.policy.modes import get_mode, to_policy_dict

    action_classes: list[dict] = [
        {
            "name": "shell_command",
            "category": "INFRASTRUCTURE",
            "reversibility": "IRREVERSIBLE",
            "blast_radius": "LOCAL",
            "urgency": "IMMEDIATE",
            "tools": sorted(discovery.known_tools.keys()),
            "regulatory": [],
        },
        {
            "name": "file_system_access",
            "category": "DATA",
            "reversibility": "REVERSIBLE",
            "blast_radius": "LOCAL",
            "urgency": "NORMAL",
            "tools": ["file.read", "file.write", "file.delete", "file.export"],
            "regulatory": [],
        },
        {
            "name": "network_egress",
            "category": "COMMUNICATION",
            "reversibility": "IRREVERSIBLE",
            "blast_radius": "EXTERNAL",
            "urgency": "NORMAL",
            "tools": ["network.connect", "network.send"],
            "regulatory": [],
        },
        {
            "name": "surveillance",
            "category": "PHYSICAL",
            "reversibility": "IRREVERSIBLE",
            "blast_radius": "EXTERNAL",
            "urgency": "IMMEDIATE",
            "tools": ["keylog.capture", "screen.capture", "clipboard.read",
                      "mic.capture", "camera.capture"],
            "regulatory": ["GDPR_CHAPTER_V"],
        },
    ]

    mode = get_mode(mode_name)
    mode_policy = to_policy_dict(mode)

    result: dict = {
        "version": mode_policy.get("version", "0.1"),
        "mode": mode_name,
        "generated_by": "vaara init --auto",
        "generated_at": __import__("time").time(),
        "domains": mode_policy.get("domains", ["GDPR", "EU_AI_ACT"]),
        "action_classes": action_classes,
        "thresholds": mode_policy.get("thresholds", {
            "default": {"escalate": mode.escalate, "deny": mode.deny},
        }),
        "thresholds_overrides": {},
        "sequences": [],
        "escalation_routes": [],
    }

    sensitive_paths = [str(p) for p in discovery.sensitive_paths]
    if sensitive_paths:
        result["sensitive_paths"] = sensitive_paths

    running_agents = [a.display_name for a in discovery.agents if a.running]
    if running_agents:
        result["known_agents"] = running_agents

    if shadow:
        result["mode"] = "shadow"

    return result


# ---------------------------------------------------------------------------
# Policy file persistence
# ---------------------------------------------------------------------------

DEFAULT_POLICY_PATH = Path.home() / ".vaara" / "policy.json"


def write_default_policy(
    discovery: DiscoverReport,
    *,
    shadow: bool = True,
    mode_name: str = "eco",
    path: Path = DEFAULT_POLICY_PATH,
) -> Path:
    """Generate and write the default policy file."""
    policy = generate_default_policy(discovery, shadow=shadow, mode_name=mode_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# Config persistence (the unified ~/.vaara/config.json)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path.home() / ".vaara" / "config.json"


def write_discovery_config(
    discovery: DiscoverReport,
    *,
    shadow: bool = True,
    trail_db: Path = Path.home() / ".vaara" / "trail" / "audit.db",
    path: Path = DEFAULT_CONFIG_PATH,
) -> Path:
    """Write the discovered environment to the shared config file.

    This is the unified config that the hook runner, proxy, and local daemon
    all read from.  Written after every ``vaara init --auto``.
    """
    config: dict = {
        "version": 1,
        "trail_db": str(trail_db),
        "mode": "shadow" if shadow else "enforce",
        "policy": str(DEFAULT_POLICY_PATH),
        "agents": [
            {"name": a.display_name, "binary": a.binary, "governed": a.governed}
            for a in discovery.agents
        ],
        "sensitive_paths": [str(p) for p in discovery.sensitive_paths],
        "known_tools": list(discovery.known_tools.keys()),
        "mcp_clients": [
            {"name": c.name, "servers": c.mcp_servers, "governed": c.governed_servers}
            for c in discovery.mcp_clients if c.exists
        ],
    }
    if discovery.shell:
        config["shell"] = {
            "path": discovery.shell.path,
            "wrappable": discovery.shell.wrappable,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path
