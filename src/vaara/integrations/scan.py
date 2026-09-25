# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""``vaara scan``: find the AI agents on this machine that Vaara does not see.

``vaara init`` knows the agents it has an adapter for. An agent it has no
adapter for, a desktop chat app talking to a model API, a script using an SDK,
never appeared anywhere. This looks for them from three directions and marks
each finding:

- **processes** with an open connection to a model API, a local model server
  (Ollama, LM Studio, Jan, GPT4All), or Vaara's own llm-proxy;
- **MCP configs** anywhere under the home directory, not only the known paths;
- **installed apps** that bundle a model or MCP SDK (read from an Electron
  app's ``app.asar`` header or its unpacked ``node_modules``).

``governed``   Vaara decides its tool calls now (hooks active, model traffic
               through ``vaara llm-proxy``, or every MCP server routed through
               ``vaara-mcp-proxy``).
``reachable``  Vaara has an adapter and is not active for it yet; the detail
               says what to run.
``ungoverned`` Vaara has no adapter. Route its model traffic through
               ``vaara llm-proxy`` or its MCP servers through
               ``vaara-mcp-proxy``.

A connection is matched to a provider by the addresses that provider's API
host resolves to at scan time. Providers behind a shared CDN share addresses
with other sites, so the detail names the host the address belongs to rather
than asserting the process called it. Standard library only.
"""

from __future__ import annotations

import concurrent.futures
import ipaddress
import json
import os
import re
import shutil
import socket
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

MODEL_HOSTS: dict[str, str] = {
    "api.anthropic.com": "Anthropic",
    "api.openai.com": "OpenAI",
    "chatgpt.com": "OpenAI (ChatGPT)",
    "generativelanguage.googleapis.com": "Google Gemini",
    "api.mistral.ai": "Mistral",
    "api.deepseek.com": "DeepSeek",
    "api.x.ai": "xAI",
    "api.groq.com": "Groq",
    "api.together.xyz": "Together",
    "openrouter.ai": "OpenRouter",
    "api.perplexity.ai": "Perplexity",
    "api.cohere.com": "Cohere",
}

LOCAL_MODEL_PORTS: dict[int, str] = {
    11434: "Ollama",
    1234: "LM Studio",
    1337: "Jan",
    4891: "GPT4All",
}

LLM_PROXY_PORT = 8790

# Agents Vaara has an adapter for: id, display name, how the process shows up.
_ADAPTED: tuple[tuple[str, str, str], ...] = (
    ("claude-code", "Claude Code", r"(^|/)claude(\s|$)|@anthropic-ai/claude-code"),
    ("codex", "Codex", r"(^|/)codex(\s|$)|@openai/codex"),
    ("gemini", "Gemini CLI", r"(^|/)gemini(\s|$)|@google/gemini-cli"),
    ("cursor", "Cursor", r"Cursor\.app|(^|/)cursor(\s|$)|cursor-server"),
    ("opencode", "OpenCode", r"(^|/)opencode(\s|$)|opencode-ai"),
    ("claude-desktop", "Claude Desktop", r"Claude\.app"),
)

SDK_PACKAGES: dict[str, str] = {
    "@anthropic-ai/sdk": "Anthropic SDK",
    "openai": "OpenAI SDK",
    "@google/generative-ai": "Google Gemini SDK",
    "@google/genai": "Google GenAI SDK",
    "@mistralai/mistralai": "Mistral SDK",
    "@modelcontextprotocol/sdk": "MCP SDK",
    "ollama": "Ollama client",
    "ai": "Vercel AI SDK",
    "@langchain/core": "LangChain",
}

MCP_CONFIG_NAMES = {
    "mcp.json", ".mcp.json", "mcp_config.json", "mcp_settings.json",
    "claude_desktop_config.json", "cline_mcp_settings.json", "mcp-servers.json",
}
_SKIP_DIRS = {
    "node_modules", ".git", ".cache", ".npm", ".venv", "venv", "site-packages",
    ".Trash", "Caches", "Containers", "Group Containers", ".pnpm-store",
    ".cargo", ".rustup", "go", ".gradle", ".m2", "Photos Library.photoslibrary",
    "Mail", "Messages", ".docker", "snap",
}
_WALK_DEPTH = 5
_WALK_MAX_DIRS = 20_000


@dataclass
class Finding:
    state: str   # "governed", "reachable" or "ungoverned"
    kind: str    # "process", "mcp", "app"
    name: str
    detail: str
    where: str = ""


# ---------------------------------------------------------------------------
# Adapter status
# ---------------------------------------------------------------------------

def _json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def adapter_status(agent: str, home: Optional[Path] = None) -> tuple[bool, str]:
    """(active, detail) for one adapted agent."""
    home = home or Path.home()
    if agent == "claude-code":
        hooks = json.dumps(_json(home / ".claude" / "settings.json").get("hooks", {}))
        if "vaara hook " in hooks:
            return True, "every tool call, through its hooks"
        return False, "run `vaara init` to install the hooks"
    if agent == "codex":
        from vaara.integrations import codex
        state = codex.trust_status()
        if state == "trusted":
            return True, "every tool call, through its hooks"
        if state == "missing":
            return False, "run `vaara init` to install the hooks"
        return False, f"hooks are {state}: trust them in Codex's /hooks"
    if agent == "gemini":
        from vaara.integrations import gemini
        state = gemini.hook_status()
        if state == "active":
            return True, "every tool call, through its hooks"
        if state == "missing":
            return False, "run `vaara init` to install the hooks"
        return False, f"hooks are {state} in Gemini CLI's settings"
    if agent == "cursor":
        from vaara.integrations import cursor
        if cursor.native_hook_installed():
            return True, "every tool call, through its hooks"
        return False, "run `vaara init` to install the hooks"
    if agent == "opencode":
        from vaara.integrations import opencode
        if opencode.plugin_path().exists():
            return True, "every tool call, through its plugin"
        return False, "run `vaara init` to install the plugin"
    if agent == "claude-desktop":
        return False, "its MCP servers can be routed through vaara-mcp-proxy by `vaara init`"
    return False, ""


_VAARA_PROXY = re.compile(r"(^|[/\s])vaara(-mcp-proxy|-infer-proxy)?\s.*\b(llm-proxy|proxy|mcp-proxy)\b|vaara-mcp-proxy")


def _adapter_for(text: str) -> Optional[tuple[str, str]]:
    for agent, name, pattern in _ADAPTED:
        if re.search(pattern, text):
            return agent, name
    return None


# ---------------------------------------------------------------------------
# Processes and their connections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Connection:
    pid: int
    command: str
    remote_ip: str
    remote_port: int


def _lsof_connections() -> Optional[list[Connection]]:
    lsof = shutil.which("lsof")
    if not lsof:
        return None
    try:
        out = subprocess.run(
            [lsof, "-nP", "-iTCP", "-sTCP:ESTABLISHED", "-F", "pcn"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    conns: list[Connection] = []
    pid, cmd = 0, ""
    for line in out.splitlines():
        if not line:
            continue
        tag, val = line[0], line[1:]
        if tag == "p":
            pid = int(val) if val.isdigit() else 0
        elif tag == "c":
            cmd = val
        elif tag == "n" and "->" in val:
            remote = val.split("->", 1)[1]
            host, _, port = remote.rpartition(":")
            host = host.strip("[]")
            if port.isdigit():
                conns.append(Connection(pid, cmd, host, int(port)))
    return conns


def _hex_addr(raw: str) -> tuple[str, int]:
    addr, port = raw.split(":")
    b = bytes.fromhex(addr)
    if len(b) == 4:
        ip = str(ipaddress.IPv4Address(b[::-1]))
    else:
        # /proc stores IPv6 as four little-endian 32-bit words.
        words = [b[i:i + 4][::-1] for i in range(0, 16, 4)]
        v6 = ipaddress.IPv6Address(b"".join(words))
        ip = str(v6.ipv4_mapped) if v6.ipv4_mapped else str(v6)
    return ip, int(port, 16)


def _proc_connections(proc: Path = Path("/proc")) -> Optional[list[Connection]]:
    if not (proc / "net" / "tcp").exists():
        return None
    by_inode: dict[str, tuple[str, int]] = {}
    for name in ("tcp", "tcp6"):
        try:
            lines = (proc / "net" / name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) > 9 and parts[3] == "01":  # ESTABLISHED
                by_inode[parts[9]] = _hex_addr(parts[2])
    conns: list[Connection] = []
    for pid_dir in proc.iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            fds = list((pid_dir / "fd").iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("socket:["):
                remote = by_inode.get(target[8:-1])
                if remote:
                    conns.append(Connection(int(pid_dir.name), "", remote[0], remote[1]))
    return conns


def connections() -> list[Connection]:
    return _lsof_connections() or _proc_connections() or []


def cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        if raw:
            return raw.replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        pass  # no /proc (macOS) or the process is gone; ps below answers either way
    try:
        return subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def resolve_hosts(hosts: Iterable[str] = MODEL_HOSTS, timeout: float = 4.0) -> dict[str, str]:
    """Map each address the model API hosts resolve to onto its host name."""
    out: dict[str, str] = {}

    def one(host: str) -> tuple[str, set[str]]:
        try:
            return host, {i[4][0] for i in socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)}
        except OSError:
            return host, set()

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    try:
        futures = [pool.submit(one, h) for h in hosts]
        done, _ = concurrent.futures.wait(futures, timeout=timeout)
        for f in done:
            host, ips = f.result()
            for ip in ips:
                out.setdefault(ip, host)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return out


def _is_loopback(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_loopback
    except ValueError:
        return False


def scan_processes(
    conns: Optional[list[Connection]] = None,
    ip_hosts: Optional[dict[str, str]] = None,
    cmd_of: Callable[[int], str] = cmdline,
    status: Callable[[str], tuple[bool, str]] = adapter_status,
) -> list[Finding]:
    conns = connections() if conns is None else conns
    ip_hosts = resolve_hosts() if ip_hosts is None else ip_hosts
    per_pid: dict[int, dict] = {}
    for c in conns:
        target = None
        if _is_loopback(c.remote_ip):
            if c.remote_port == LLM_PROXY_PORT:
                target = "vaara llm-proxy"
            elif c.remote_port in LOCAL_MODEL_PORTS:
                target = f"{LOCAL_MODEL_PORTS[c.remote_port]} (localhost:{c.remote_port})"
        elif c.remote_ip in ip_hosts and c.remote_port == 443:
            target = f"an address of {ip_hosts[c.remote_ip]}"
        if target is None:
            continue
        entry = per_pid.setdefault(c.pid, {"command": c.command, "targets": set()})
        entry["targets"].add(target)

    findings: list[Finding] = []
    for pid, entry in sorted(per_pid.items()):
        full = cmd_of(pid) or entry["command"]
        short = Path(full.split()[0]).name if full else entry["command"] or f"pid {pid}"
        targets = sorted(entry["targets"])
        talks = ", ".join(targets)
        if _VAARA_PROXY.search(full):
            # Vaara's own proxy forwarding upstream: the traffic it carries is
            # the governed traffic, reported by the processes that use it.
            continue
        adapted = _adapter_for(full or entry["command"])
        if "vaara llm-proxy" in targets and len(targets) == 1:
            findings.append(Finding("governed", "process", adapted[1] if adapted else short,
                                    "model traffic through vaara llm-proxy", f"pid {pid}"))
            continue
        if adapted:
            active, detail = status(adapted[0])
            findings.append(Finding("governed" if active else "reachable", "process",
                                    adapted[1], f"talks to {talks}; {detail}", f"pid {pid}"))
        else:
            findings.append(Finding(
                "ungoverned", "process", short,
                f"talks to {talks} with no Vaara adapter; point it at vaara llm-proxy",
                f"pid {pid}"))
    return findings


# ---------------------------------------------------------------------------
# MCP configs anywhere
# ---------------------------------------------------------------------------

def _servers(obj: dict) -> Optional[dict]:
    for key in ("mcpServers", "servers"):
        if isinstance(obj.get(key), dict):
            return obj[key]
    mcp = obj.get("mcp")
    if isinstance(mcp, dict) and isinstance(mcp.get("servers"), dict):
        return mcp["servers"]
    return None


def _routed(server: dict) -> bool:
    text = " ".join(str(x) for x in [server.get("command", "")] + list(server.get("args") or []))
    url = str(server.get("url", ""))
    return "vaara-mcp-proxy" in text or "vaara mcp-proxy" in text or ":8790" in url


def find_mcp_configs(home: Path, extra: Iterable[Path] = ()) -> list[Path]:
    found: set[Path] = {p for p in extra if p.is_file()}
    seen = 0
    stack: list[tuple[Path, int]] = [(home, 0)]
    while stack and seen < _WALK_MAX_DIRS:
        d, depth = stack.pop()
        seen += 1
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    if depth < _WALK_DEPTH and e.name not in _SKIP_DIRS:
                        stack.append((Path(e.path), depth + 1))
                elif e.name in MCP_CONFIG_NAMES:
                    found.add(Path(e.path))
            except OSError:
                continue
    # Editor extensions keep theirs deeper than the walk reaches.
    for base in (home / "Library" / "Application Support", home / ".config"):
        for pattern in ("*/User/globalStorage/*/settings/*mcp*.json", "*/User/mcp.json"):
            found.update(p for p in base.glob(pattern) if p.is_file())
    return sorted(found)


def _config_owner(path: Path) -> Optional[str]:
    """The adapted agent that reads this MCP config, when there is one.

    A hook-based agent decides its MCP tool calls in the same hook as its
    own, so its servers are governed without vaara-mcp-proxy.
    """
    text = str(path)
    if "/.claude/" in text or path.name in (".claude.json", ".mcp.json"):
        return "claude-code"
    if "/.cursor/" in text:
        return "cursor"
    if "/.gemini/" in text:
        return "gemini"
    if "/opencode/" in text or "/.opencode/" in text:
        return "opencode"
    return None


def scan_mcp(home: Optional[Path] = None,
             status: Callable[[str], tuple[bool, str]] = adapter_status) -> list[Finding]:
    home = home or Path.home()
    known = [home / ".claude.json", home / ".cursor" / "mcp.json",
             home / ".codeium" / "windsurf" / "mcp_config.json",
             home / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
             home / ".config" / "Claude" / "claude_desktop_config.json"]
    findings: list[Finding] = []
    for path in find_mcp_configs(home, known):
        servers = _servers(_json(path))
        if not servers:
            continue
        entries = [s for s in servers.values() if isinstance(s, dict)]
        routed = sum(1 for s in entries if _routed(s))
        naked = len(entries) - routed
        where = str(path).replace(str(home), "~", 1)
        name = path.name
        owner = _config_owner(path)
        if owner and naked:
            label = next(n for a, n, _ in _ADAPTED if a == owner)
            active, detail = status(owner)
            if active:
                findings.append(Finding("governed", "mcp", name,
                                        f"{len(entries)} MCP server(s); their tool calls go "
                                        f"through {label}'s hooks", where))
            else:
                findings.append(Finding("reachable", "mcp", name,
                                        f"{naked} MCP server(s) read by {label}; {detail}", where))
            continue
        if naked == 0:
            findings.append(Finding("governed", "mcp", name,
                                    f"all {routed} MCP server(s) through vaara-mcp-proxy", where))
        elif path in known:
            findings.append(Finding("reachable", "mcp", name,
                                    f"{naked} of {len(entries)} MCP server(s) not routed; "
                                    "`vaara init` routes this file", where))
        else:
            findings.append(Finding("ungoverned", "mcp", name,
                                    f"{naked} of {len(entries)} MCP server(s) not routed; "
                                    "wrap each with vaara-mcp-proxy", where))
    return findings


# ---------------------------------------------------------------------------
# Installed apps that bundle a model or MCP SDK
# ---------------------------------------------------------------------------

def asar_files(path: Path, cap: int = 64 << 20) -> Optional[dict]:
    """The file tree in an Electron ``app.asar`` header, or None."""
    try:
        with path.open("rb") as fh:
            head = fh.read(16)
            if len(head) < 16:
                return None
            size = struct.unpack_from("<I", head, 12)[0]
            if size <= 0 or size > cap:
                return None
            tree = json.loads(fh.read(size).decode("utf-8", errors="replace"))
    except (OSError, ValueError, struct.error):
        return None
    return tree if isinstance(tree, dict) else None


def _asar_has(tree: dict, parts: list[str]) -> bool:
    node = tree
    for part in parts:
        files = node.get("files") if isinstance(node, dict) else None
        if not isinstance(files, dict) or part not in files:
            return False
        node = files[part]
    return True


def sdks_in_app(app: Path) -> list[str]:
    """SDK names bundled in one app bundle or app directory."""
    found: list[str] = []
    for resources in (app / "Contents" / "Resources", app / "resources"):
        asar = resources / "app.asar"
        tree = asar_files(asar) if asar.is_file() else None
        unpacked = resources / "app" / "node_modules"
        for pkg, label in SDK_PACKAGES.items():
            parts = ["node_modules", *pkg.split("/")]
            if (tree and _asar_has(tree, parts)) or (unpacked / pkg / "package.json").is_file():
                if label not in found:
                    found.append(label)
    return found


def app_dirs(home: Optional[Path] = None) -> list[Path]:
    home = home or Path.home()
    apps: list[Path] = []
    for base in (Path("/Applications"), home / "Applications"):
        if base.is_dir():
            apps.extend(sorted(base.glob("*.app")))
    for base in (Path("/opt"), home / ".local" / "share", Path("/usr/lib"), Path("/usr/share")):
        if base.is_dir():
            apps.extend(sorted(p.parent for p in base.glob("*/resources/app.asar")))
    return apps


def scan_apps(apps: Optional[list[Path]] = None,
              status: Callable[[str], tuple[bool, str]] = adapter_status) -> list[Finding]:
    findings: list[Finding] = []
    for app in app_dirs() if apps is None else apps:
        sdks = sdks_in_app(app)
        if not sdks:
            continue
        name = app.name.removesuffix(".app")
        adapted = _adapter_for(app.name) or _adapter_for(str(app))
        bundles = "bundles " + ", ".join(sdks)
        if adapted:
            active, detail = status(adapted[0])
            findings.append(Finding("governed" if active else "reachable", "app",
                                    adapted[1], f"{bundles}; {detail}", str(app)))
        else:
            findings.append(Finding(
                "ungoverned", "app", name,
                f"{bundles}, with no Vaara adapter; route its model traffic through "
                "vaara llm-proxy", str(app)))
    return findings


def run_scan(*, processes: bool = True, mcp: bool = True, apps: bool = True) -> list[Finding]:
    findings: list[Finding] = []
    if processes:
        findings += scan_processes()
    if mcp:
        findings += scan_mcp()
    if apps:
        findings += scan_apps()
    order = {"ungoverned": 0, "reachable": 1, "governed": 2}
    return sorted(findings, key=lambda f: (order.get(f.state, 3), f.kind, f.name.lower()))


def to_json(findings: list[Finding]) -> str:
    return json.dumps([asdict(f) for f in findings], indent=2)
