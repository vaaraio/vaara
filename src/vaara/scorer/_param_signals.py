# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic parameter-content risk signals for the base rule scorer.

Stdlib only, so it runs in a zero-config install with no ML extra and with no
dependency on the integrations layer. The one signal here is the cloud-metadata
endpoint: a call parameter pointing at 169.254.169.254 (or its IPv6 / dotless /
hex encodings) is a known SSRF attack, not a probabilistic risk, so it is
treated as a hard decision floor rather than one averaged expert among many.

The boundary matches the MCP proxy egress guard
(vaara.integrations._egress_guard, the canonical network-enforcement point):
the metadata address has no legitimate reason to be dialed, while private and
loopback hosts stay reachable and are not floored here. The detection is
duplicated rather than imported to keep the scorer free of any dependency on
the integrations package.
"""

from __future__ import annotations

import ipaddress
import re
import shlex
from typing import Any, Optional
from urllib.parse import urlsplit

# The cloud instance-metadata endpoints, refused unconditionally.
_METADATA_V4 = ipaddress.IPv4Address("169.254.169.254")
_METADATA_V6 = ipaddress.IPv6Address("fe80::a9fe:a9fe")

# Risk assigned when a metadata endpoint is found. High enough that the
# scorer's decision floor lands the call in deny/escalate regardless of the
# benign taxonomy base for a network read.
_METADATA_RISK = 0.95

# Bare host tokens that look like an IP or a dotless/hex integer, pulled from
# any string value in the parameters (not only well-formed URLs).
_HOSTISH = re.compile(r"(?:\[[0-9A-Fa-f:]+\]|[0-9A-Fa-f.x]+)")


def _coerce_dotless_host(host: str) -> Optional[ipaddress.IPv4Address]:
    """Parse a bare decimal or hex integer host (``2852039166``, ``0xa9fea9fe``).

    Browsers and ``inet_aton`` accept these as IPv4; ``ipaddress`` does not, so
    decode them explicitly. Returns the address when the host is such an
    integer, else None.
    """
    base = 16 if host.lower().startswith("0x") else 10
    try:
        value = int(host, base)
    except ValueError:
        return None
    if 0 <= value <= 0xFFFFFFFF:
        return ipaddress.IPv4Address(value)
    return None


def _is_metadata(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:  # ::ffff:a.b.c.d judged on the embedded v4 address
        ip = mapped
    return ip == _METADATA_V4 or ip == _METADATA_V6


def _host_is_metadata(host: str) -> bool:
    host = host.strip().strip("[]")
    if not host:
        return False
    dotless = _coerce_dotless_host(host)
    if dotless is not None and _is_metadata(dotless):
        return True
    try:
        return _is_metadata(ipaddress.ip_address(host))
    except ValueError:
        return False


def _string_hits_metadata(value: str) -> bool:
    # Try the value as a URL first (covers scheme://host:port/path), then
    # scan any host-shaped tokens so an unschemed or embedded address is
    # still caught.
    host = urlsplit(value).hostname
    if host and _host_is_metadata(host):
        return True
    return any(_host_is_metadata(tok) for tok in _HOSTISH.findall(value))


def _walk(value: Any) -> bool:
    if isinstance(value, str):
        return _string_hits_metadata(value)
    if isinstance(value, dict):
        return any(_walk(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_walk(v) for v in value)
    return False


def metadata_endpoint_risk(parameters: Any) -> float:
    """Return ``_METADATA_RISK`` if any parameter targets a cloud-metadata
    endpoint, else 0.0. Recurses through nested dicts and lists."""
    return _METADATA_RISK if _walk(parameters) else 0.0


# ---------------------------------------------------------------------------
# Destructive actions
#
# A call that deletes, drops, truncates, force-pushes or otherwise discards
# state is held for a human, whatever the experts score it. The experts read
# the tool name, not the payload, so every shell call scored alike (measured
# 2026-09-26 on a live trail: 7,815 scores, every shell call at 0.1625,
# whether it listed a directory or deleted a tree). This floor reads the
# payload. It raises a score to the escalate band and never lowers one, so a
# call the experts already deny stays denied.
# ---------------------------------------------------------------------------

#: A word in a tool name that names a destructive operation. Matched on the
#: name split at separators and case changes, so ``mem_delete``,
#: ``s3_delete_bucket``, ``deleteBranch`` and ``terraform_destroy`` all hit
#: while ``deleted_items_report`` does not.
_DESTRUCTIVE_NAME_WORDS = frozenset({
    "delete", "remove", "drop", "destroy", "purge", "truncate", "unlink",
    "rmdir", "wipe", "erase", "forget", "shred", "selfdestruct",
})

#: Parameter keys whose string value is run as a shell command.
_COMMAND_KEYS = re.compile(
    r"(?:^|[_\-.])(?:command|commands|cmd|script|shell|exec|argv|code)(?:$|[_\-.])", re.I)

#: Keys that carry a command only as an argv list: ``args`` is argv on an
#: MCP shell tool and prose on others, so a plain string under it is not read.
_ARGV_KEYS = re.compile(r"^args?$", re.I)

#: Parameter keys whose string value is run as SQL.
_SQL_KEYS = re.compile(
    r"(?:^|[_\-.])(?:query|sql|statement|command|cmd|script)(?:$|[_\-.])", re.I)

_SQL_DESTRUCTIVE = re.compile(
    r"\b(?:drop\s+(?:table|database|schema|view|index|collection)\b"
    r"|truncate\s+(?:table\s+)?[\w\"`\[]"
    r"|delete\s+from\b)",
    re.I,
)

#: Library calls that delete files, read in code handed to an interpreter.
_CODE_DESTRUCTIVE = re.compile(
    r"\b(?:os\.(?:remove|unlink|rmdir|removedirs)|shutil\.rmtree"
    r"|fs\.(?:rm|rmSync|rmdir|rmdirSync|unlink|unlinkSync)|rimraf"
    r"|FileUtils\.rm(?:_rf|_r|_f)?|File\.delete|Remove-Item)\s*\(?",
)

#: Keys naming the program an argv list runs.
_PROGRAM_KEYS = ("program", "executable", "binary", "cmd", "command")

#: Commands that run the next word as the command.
_LAUNCHERS = frozenset({
    "sudo", "doas", "env", "nohup", "nice", "ionice", "time", "timeout",
    "stdbuf", "setsid", "exec", "command", "builtin", "xargs", "busybox",
    "unbuffer", "flock", "chroot", "runuser",
})

#: Commands that are destructive whatever their arguments.
_DESTRUCTIVE_VERBS = frozenset({
    "rm", "rmdir", "unlink", "shred", "srm", "truncate", "wipefs",
})

#: ``tool subcommand`` pairs that delete remote or managed state.
_DESTRUCTIVE_SUBCOMMANDS = {
    "kubectl": {"delete"},
    "terraform": {"destroy"},
    "tofu": {"destroy"},
    "helm": {"uninstall", "delete"},
    "docker": {"rm", "rmi", "prune"},
    "podman": {"rm", "rmi", "prune"},
    "gh": {"delete"},
    "npm": {"unpublish"},
}

_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")
_SHELLS = frozenset({"sh", "bash", "zsh", "dash", "ksh", "fish"})

#: Shell commands whose arguments are SQL.
_SQL_CLIENTS = frozenset({
    "sqlite3", "psql", "mysql", "mariadb", "sqlcmd", "duckdb", "clickhouse-client",
    "cockroach", "snowsql", "bq",
})
_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=\S*")


def _name_words(tool_name: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tool_name)
    return {w.lower() for w in re.split(r"[^A-Za-z0-9]+", spaced) if w}


def _strip_heredocs(command: str) -> str:
    """Drop heredoc bodies. A script handed to an interpreter or written to a
    file is not run by this shell, so an ``rm`` inside it is not a delete."""
    lines = command.split("\n")
    out: list[str] = []
    end: Optional[str] = None
    for line in lines:
        if end is not None:
            if line.strip() == end:
                end = None
            continue
        out.append(line)
        m = _HEREDOC.search(line)
        if m:
            end = m.group(2)
    return "\n".join(out)


def _segments(command: str) -> list[str]:
    """Split on ; && || | newline and $( outside quotes."""
    parts: list[str] = []
    buf: list[str] = []
    quote = ""
    i = 0
    while i < len(command):
        ch = command[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            elif ch == "\\" and quote == '"' and i + 1 < len(command):
                buf.append(command[i + 1])
                i += 1
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch in ";|\n`" or command.startswith("$(", i) or command.startswith("&&", i):
            parts.append("".join(buf))
            buf = []
            if command.startswith(("&&", "||", "$("), i):
                i += 1
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _words(segment: str) -> list[str]:
    try:
        return shlex.split(segment, comments=True)
    except ValueError:
        return segment.split()


def _strip_launchers(words: list[str]) -> list[str]:
    i = 0
    while i < len(words):
        w = words[i]
        if _ASSIGNMENT.fullmatch(w) or w in _LAUNCHERS or w.isdigit():
            i += 1
        elif i > 0 and w.startswith("-") and words[i - 1] in _LAUNCHERS:
            i += 1
        else:
            break
    return words[i:]


def _git_destructive(args: list[str]) -> bool:
    # Skip global options: -C <path>, -c <key=value>, --git-dir=..., etc.
    i = 0
    while i < len(args) and args[i].startswith("-"):
        i += 2 if args[i] in ("-C", "-c") else 1
    if i >= len(args):
        return False
    sub, rest = args[i], args[i + 1:]
    flags = set(rest)
    if sub == "push":
        return bool(flags & {"-f", "--force", "--force-with-lease", "-d", "--delete",
                             "--mirror", "--prune"}) or any(
            a.startswith((":", "+", "--force-with-lease=")) for a in rest)
    if sub == "branch":
        return bool(flags & {"-D", "-d", "--delete"})
    if sub == "tag":
        return bool(flags & {"-d", "--delete"})
    if sub == "reset":
        return "--hard" in flags
    if sub == "clean":
        return any(a.startswith("-") and "f" in a.lstrip("-") for a in rest) or "--force" in flags
    if sub == "checkout":
        return "--" in rest or "." in rest or bool(flags & {"-f", "--force"})
    if sub == "restore":
        return "--staged" not in flags or "--worktree" in flags
    if sub == "stash":
        return bool(rest) and rest[0] in ("drop", "clear")
    if sub in ("rm", "filter-branch", "filter-repo"):
        return True
    if sub == "reflog":
        return bool(rest) and rest[0] in ("expire", "delete")
    if sub == "update-ref":
        return "-d" in flags
    return False


def _command_destructive(command: str, depth: int = 0) -> bool:
    command = _strip_heredocs(command)
    for part in _segments(command):
        words = _strip_launchers(_words(part))
        if not words:
            continue
        verb = words[0].rsplit("/", 1)[-1]
        args = words[1:]
        if depth < 3 and (verb in _SHELLS or verb == "eval"):
            inner = args[args.index("-c") + 1] if "-c" in args[:-1] else (
                " ".join(args) if verb == "eval" else "")
            if inner and _command_destructive(inner, depth + 1):
                return True
            continue
        if verb in _DESTRUCTIVE_VERBS or verb.startswith("mkfs"):
            return True
        if verb == "dd" and any(a.startswith("of=") for a in args):
            return True
        if verb == "find" and ("-delete" in args or any(
                a in ("-exec", "-execdir", "-ok", "-okdir") and j + 1 < len(args)
                and args[j + 1].rsplit("/", 1)[-1] in _DESTRUCTIVE_VERBS
                for j, a in enumerate(args))):
            return True
        if verb == "git" and _git_destructive(args):
            return True
        if verb in _SQL_CLIENTS and _SQL_DESTRUCTIVE.search(" ".join(args)):
            return True
        subs = _DESTRUCTIVE_SUBCOMMANDS.get(verb)
        if subs and any(a in subs for a in args[:3]):
            return True
        if verb in ("aws", "gsutil", "gcloud", "az") and any(
                a in ("rm", "rb", "delete", "remove") for a in args[:4]):
            return True
    return False


def _leaves(value: Any, key: str = "") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(key, value)]
    if isinstance(value, dict):
        out: list[tuple[str, str]] = []
        for k, v in value.items():
            out.extend(_leaves(v, str(k)))
        return out
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, str) for v in value) and (
                _COMMAND_KEYS.search(key) or _ARGV_KEYS.search(key)):
            return [("command", shlex.join(value))]
        out = []
        for v in value:
            out.extend(_leaves(v, key))
        return out
    return []


def destructive_action(tool_name: str, parameters: Any) -> Optional[str]:
    """Name what makes this call destructive, or return None.

    A tool whose name says it deletes; a shell command that deletes, drops,
    truncates, force-pushes or discards work; SQL that drops or deletes.
    Only command-like and SQL-like parameters are read as commands, so a file
    whose content mentions ``rm`` is not a delete.
    """
    hit = _name_words(tool_name or "") & _DESTRUCTIVE_NAME_WORDS
    if hit:
        return f"tool name says {sorted(hit)[0]}"
    if isinstance(parameters, dict):
        # {"program": "rm", "args": ["-rf", "/"]}: the program and its argv
        # are one command split across two keys.
        argv = parameters.get("args") or parameters.get("argv")
        for key in _PROGRAM_KEYS:
            prog = parameters.get(key)
            if isinstance(prog, str) and isinstance(argv, list) and all(
                    isinstance(a, str) for a in argv):
                if _command_destructive(shlex.join([prog, *argv])):
                    return f"{key} deletes or discards state"
    for key, text in _leaves(parameters):
        if _COMMAND_KEYS.search(key) and _command_destructive(text):
            return f"{key} deletes or discards state"
        if _COMMAND_KEYS.search(key) and _CODE_DESTRUCTIVE.search(_strip_heredocs(text)):
            return f"{key} deletes files"
        if (_SQL_KEYS.search(key) and not _COMMAND_KEYS.search(key)
                and _SQL_DESTRUCTIVE.search(text)):
            return f"{key} drops or deletes rows"
    return None


# ---------------------------------------------------------------------------
# Outbound calls
#
# A call that sends something off the machine: mail and other comm.* actions,
# an export, a web fetch or web search (the URL and the query carry data out),
# and a shell command that dials a remote host. The sequence detector pairs
# this with an earlier read, so data read by one tool and sent by another is
# seen even when the two tools never meet.
# ---------------------------------------------------------------------------

#: Name-word pairs that mark a tool as a network fetch or search.
_OUTBOUND_NAME_PAIRS = (
    ("web", "fetch"), ("web", "search"), ("fetch", "url"), ("http", "get"),
    ("http", "post"), ("http", "request"), ("browser", "navigate"),
)

#: Shell commands that send data to a remote host.
_OUTBOUND_VERBS = frozenset({
    "curl", "wget", "nc", "ncat", "netcat", "socat", "telnet", "ftp", "sftp",
    "scp", "ssh", "rsync", "http", "https", "xh",
})


def _command_outbound(command: str, depth: int = 0) -> Optional[str]:
    command = _strip_heredocs(command)
    for part in _segments(command):
        words = _strip_launchers(_words(part))
        if not words:
            continue
        verb = words[0].rsplit("/", 1)[-1]
        args = words[1:]
        if depth < 3 and (verb in _SHELLS or verb == "eval"):
            inner = args[args.index("-c") + 1] if "-c" in args[:-1] else (
                " ".join(args) if verb == "eval" else "")
            hit = _command_outbound(inner, depth + 1) if inner else None
            if hit:
                return hit
            continue
        if verb in _OUTBOUND_VERBS:
            return verb
        if verb == "git" and "push" in args[:3]:
            return "git push"
    return None


def outbound_action(
    tool_name: str, action_name: Optional[str], parameters: Any,
) -> Optional[str]:
    """Name what makes this call send data off the machine, or return None.

    ``action_name`` is the classified action type (``comm.send_email``,
    ``data.export``). Only command-like parameters are read as commands, so a
    file whose content mentions ``curl`` is not an outbound call.
    """
    if action_name and (action_name.startswith("comm.") or action_name == "data.export"):
        return action_name
    words = _name_words(tool_name or "")
    for a, b in _OUTBOUND_NAME_PAIRS:
        if a in words and b in words:
            return f"tool name says {a} {b}"
    for key, text in _leaves(parameters):
        if _COMMAND_KEYS.search(key):
            hit = _command_outbound(text)
            if hit:
                return f"{key} runs {hit}"
    return None


# ---------------------------------------------------------------------------
# Writes into agent-persistent locations
#
# Skills, memory files, standing instruction files and hook or MCP
# configuration are loaded by the agent host again in later sessions, so a
# write there carries whatever the agent ingested into every session after it
# (cross-session instruction propagation, arXiv 2610.13227). The sequence
# detector pairs this with an earlier outbound call.
# ---------------------------------------------------------------------------

#: Parameter keys a write tool names its target file with.
_PATH_KEYS = re.compile(
    r"(?:^|[_\-.])(?:path|file|filename|filepath|target|dest|destination)$", re.I)

#: Tool-name words that mark a call as writing to its target path.
_WRITE_NAME_WORDS = frozenset({
    "write", "edit", "save", "create", "append", "replace", "update", "put",
    "patch", "multiedit",
})

#: Path fragments under which agent hosts load skills, subagents, commands,
#: rules, hooks, settings or per-host configuration.
_PERSIST_DIRS = (
    ".claude/skills/", ".claude/agents/", ".claude/commands/", ".claude/hooks/",
    ".claude/rules/", ".claude/settings", ".codex/", ".gemini/", ".agents/",
    ".cursor/rules/", ".windsurf/rules/",
)

#: File names an agent host reads as standing instructions or tool config.
_PERSIST_FILES = frozenset({
    "claude.md", "claude.local.md", "agents.md", "gemini.md", "skill.md",
    ".cursorrules", ".windsurfrules", "copilot-instructions.md", ".mcp.json",
})

#: Shell commands that write their last argument (or, for tee, any argument).
_SHELL_WRITE_VERBS = frozenset({"tee", "cp", "mv", "install", "ln"})


def _persist_path(path: str) -> bool:
    p = path.replace("\\", "/")
    if any(frag in p for frag in _PERSIST_DIRS):
        return True
    if ".claude/projects/" in p and "/memory/" in p:
        return True
    return p.rsplit("/", 1)[-1].lower() in _PERSIST_FILES


def _command_persist(command: str, depth: int = 0) -> Optional[str]:
    command = _strip_heredocs(command)
    for part in _segments(command):
        words = _strip_launchers(_words(part))
        if not words:
            continue
        verb = words[0].rsplit("/", 1)[-1]
        args = words[1:]
        if depth < 3 and (verb in _SHELLS or verb == "eval"):
            inner = args[args.index("-c") + 1] if "-c" in args[:-1] else (
                " ".join(args) if verb == "eval" else "")
            hit = _command_persist(inner, depth + 1) if inner else None
            if hit:
                return hit
            continue
        for i, w in enumerate(words):
            if w in (">", ">>") and i + 1 < len(words) and _persist_path(words[i + 1]):
                return words[i + 1]
            if w.startswith(">") and w.lstrip(">") and _persist_path(w.lstrip(">")):
                return w.lstrip(">")
        if verb == "tee":
            hit = next((a for a in args if not a.startswith("-") and _persist_path(a)), None)
            if hit:
                return hit
        if verb in _SHELL_WRITE_VERBS and verb != "tee" and args and _persist_path(args[-1]):
            return args[-1]
        if verb == "sed" and any(a.startswith("-i") for a in args) and _persist_path(args[-1]):
            return args[-1]
    return None


def persistent_write(
    tool_name: str, action_name: Optional[str], parameters: Any,
) -> Optional[str]:
    """Name the agent-persistent file this call writes, or return None.

    A write tool counts when its path parameter points into a skill, memory,
    instruction, hook or MCP-configuration location; a shell command counts
    when it redirects, tees, copies or edits in place into one. Reading such a
    file is not a write, and a path that only appears inside file content is
    not one either.
    """
    writes = action_name == "data.write" or bool(
        _name_words(tool_name or "") & _WRITE_NAME_WORDS)
    for key, text in _leaves(parameters):
        if writes and _PATH_KEYS.search(key) and _persist_path(text):
            return text
        if _COMMAND_KEYS.search(key):
            hit = _command_persist(text)
            if hit:
                return hit
    return None


def persist_paths_named(parameters: Any) -> list[str]:
    """Every agent-persistent path a call names, read or write, in order.

    A path parameter counts, and so does any word of a shell command. A path
    that only appears inside other content (a file body, a message) does not:
    naming a skill file in a note is not touching it.
    """
    found: list[str] = []
    for key, text in _leaves(parameters):
        if _PATH_KEYS.search(key):
            candidates = [text]
        elif _COMMAND_KEYS.search(key):
            candidates = [w.lstrip("<>") for part in _segments(_strip_heredocs(text))
                          for w in _words(part)]
        else:
            continue
        for c in candidates:
            if c and _persist_path(c) and c not in found:
                found.append(c)
    return found


# ---------------------------------------------------------------------------
# Argument holds
#
# Three call shapes that read as ordinary by tool name and are only visible in
# the arguments. Anthropic's report of 12 October 2026 on unintended model
# actions found all three in real runs: a fetch length limit slipped through
# public URL shorteners, access tokens read from a public dashboard's config
# and replayed to skip a fee, and real government and police forms submitted
# where a practice copy was intended. Each is held for a human, the same way a
# destructive call is, because each has an ordinary use as well.

#: Public URL-shortening services. A shortened link hides where the call goes,
#: and redirect services let a long URL pass a length check.
_SHORTENER_HOSTS = frozenset({
    "bit.ly", "bitly.com", "tinyurl.com", "t.ly", "is.gd", "v.gd", "da.gd",
    "goo.gl", "ow.ly", "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at",
    "tiny.cc", "rb.gy", "s.id", "bl.ink", "short.io", "t2m.io", "x.gd",
    "clck.ru", "qr.ae", "shorte.st", "adf.ly", "u.to",
})

#: Keys whose value is a destination.
_URL_KEYS = re.compile(r"(?i)(url|uri|href|link|endpoint|target|address|location)$")

_URL_IN_TEXT = re.compile(r"(?i)\bhttps?://[^\s'\"<>|;)]+")

#: Literal secrets, by shape. A reference such as ``$TOKEN`` never matches.
_CREDENTIAL_SHAPES = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}=*"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}"
               r"|xox[abposr]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16})\b"),
    re.compile(r"(?i)[?&](?:access_?token|api_?key|apikey|token|key|auth|sig)"
               r"=[A-Za-z0-9._~%-]{16,}"),
)
_CREDENTIAL_HEADER = re.compile(r"(?i)^(authorization|x-api-key|api-key|x-auth-token)$")
_CREDENTIAL_HEADER_VALUE = re.compile(r"[A-Za-z0-9._~+/=-]{16,}")

_SUBMIT_WORD = re.compile(r"(?i)\bsubmit\b")
_CLICK_TEXT_KEYS = re.compile(r"(?i)^(element|text|label|name|selector|description|target)$")
_FORM_CONTENT_TYPE = re.compile(r"(?i)application/x-www-form-urlencoded|multipart/form-data")
_CURL_FORM = re.compile(r"(?:^|\s)(?:-F|--form|--form-string)\s")
_HTTP_TOOLS = re.compile(r"(?i)(fetch|http|request|curl|browse|web|navigate|get|post)")


def _host(url: str) -> str:
    try:
        return (urlsplit(url.strip()).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


def _destinations(tool_name: str, parameters: Any) -> list[str]:
    """Every URL the call would reach: URL-valued arguments and URLs in a
    command. Free text that merely mentions a URL is not a destination."""
    urls: list[str] = []
    for key, text in _leaves(parameters):
        stripped = text.strip()
        if _COMMAND_KEYS.search(key):
            urls.extend(_URL_IN_TEXT.findall(_strip_heredocs(stripped)))
        elif re.match(r"(?i)https?://\S+$", stripped) and (
                _URL_KEYS.search(key) or _HTTP_TOOLS.search(tool_name or "")):
            urls.append(stripped)
    return urls


def _shortener(urls: list[str]) -> Optional[str]:
    for url in urls:
        host = _host(url)
        if host in _SHORTENER_HOSTS:
            return f"URL shortener {host} hides the destination"
    return None


def _credential(tool_name: str, parameters: Any, urls: list[str]) -> Optional[str]:
    outbound = bool(urls) or outbound_action(tool_name, None, parameters)
    if not outbound:
        return None
    for key, text in _leaves(parameters):
        if not (_COMMAND_KEYS.search(key) or _URL_KEYS.search(key)
                or _CREDENTIAL_HEADER.match(key)):
            continue
        if _CREDENTIAL_HEADER.match(key) and _CREDENTIAL_HEADER_VALUE.search(text) \
                and "$" not in text:
            return f"literal credential in {key} on an outbound call"
        if _COMMAND_KEYS.search(key):
            text = _strip_heredocs(text)
        for shape in _CREDENTIAL_SHAPES:
            if shape.search(text):
                return f"literal credential in {key} on an outbound call"
    return None


def _form_submission(tool_name: str, parameters: Any) -> Optional[str]:
    words = _name_words(tool_name or "")
    if "submit" in words:
        return "tool name says submit (form submission)"
    clicks = "click" in words or any(
        k == "action" and "click" in v.lower() for k, v in _leaves(parameters))
    for key, text in _leaves(parameters):
        if clicks and _CLICK_TEXT_KEYS.match(key) and _SUBMIT_WORD.search(text):
            return f"clicks {text.strip()[:40]!r} (form submission)"
        if _COMMAND_KEYS.search(key):
            run = _strip_heredocs(text)
            if _CURL_FORM.search(run) and _URL_IN_TEXT.search(run):
                return f"{key} posts a form (form submission)"
    if isinstance(parameters, dict):
        method = str(parameters.get("method", "")).upper()
        headers = parameters.get("headers")
        if method in ("POST", "PUT") and isinstance(headers, dict) and any(
                _FORM_CONTENT_TYPE.search(str(v)) for v in headers.values()):
            return "form-encoded POST (form submission)"
    return None


# A certificate is a courier when it carries a URL the far side will fetch.
# AIR showed this on 13 October 2026 against an SNI allowlist: the agent minted
# a client certificate whose authorityInfoAccess URL held the secret, presented
# it to an allowed host, and the allowed host fetched the URL to build the
# chain. Nothing in the connection itself goes anywhere it should not.

#: Certificate extensions whose value is a URL a verifier fetches.
_CERT_URL_EXT = re.compile(
    r"(?i)\b(?:authorityInfoAccess|AuthorityInformationAccess|crlDistributionPoints"
    r"|CRLDistributionPoints|freshestCRL|caIssuers|OCSP\s*;\s*URI)\b")
_CERT_EXT_URL = re.compile(r"(?i)\bURI\s*:\s*\S+|\bhttps?://\S+")
#: Commands that mint a certificate, with openssl or Python's cryptography.
_CERT_MINT = re.compile(
    r"(?i)\bopenssl\s+(?:req|x509|ca)\b|\bpython[0-9.]*\b")
_SEGMENT = re.compile(r"\|\|?|&&|;|\n")
#: Keys that name a client certificate on a request tool.
_CLIENT_CERT_KEYS = re.compile(r"(?i)^(?:cert|client_?cert(?:ificate)?|ssl_?cert)$")


def _program(words: list[str]) -> str:
    """The program a command segment runs, past env assignments and sudo."""
    for word in words:
        if "=" in word and not word.startswith("-"):
            continue
        if word in ("sudo", "env", "command", "exec", "time", "nohup"):
            continue
        return word.rsplit("/", 1)[-1].lower()
    return ""


def _presents_client_cert(segment: str) -> bool:
    words = segment.split()
    prog = _program(words)
    if prog == "curl":
        flag = any(w in ("-E", "--cert") or w.startswith("--cert=")
                   or re.fullmatch(r"-[A-Za-z]*E", w) for w in words)
        return flag and bool(_URL_IN_TEXT.search(segment))
    if prog == "wget":
        return (any(w.startswith("--certificate") for w in words)
                and bool(_URL_IN_TEXT.search(segment)))
    if prog in ("http", "https", "xh"):
        return (any(w == "--cert" or w.startswith("--cert=") for w in words)
                and bool(_URL_IN_TEXT.search(segment)))
    if prog == "openssl":
        return "s_client" in words and "-cert" in words
    return False


def _courier_certificate(tool_name: str, parameters: Any,
                         urls: list[str]) -> Optional[str]:
    if isinstance(parameters, dict) and urls and any(
            _CLIENT_CERT_KEYS.match(str(k)) and v for k, v in parameters.items()):
        return "presents a client certificate (certificate courier)"
    for key, text in _leaves(parameters):
        if not _COMMAND_KEYS.search(key):
            continue
        run = _strip_heredocs(text)
        # A heredoc fed to openssl or python is part of the mint, so the
        # extension is looked for in the whole text; the minting command
        # itself must be outside any heredoc.
        if (_CERT_MINT.search(run) and _CERT_URL_EXT.search(text)
                and _CERT_EXT_URL.search(text)):
            return f"{key} mints a certificate carrying a fetchable URL (certificate courier)"
        if any(_presents_client_cert(s) for s in _SEGMENT.split(run)):
            return f"{key} presents a client certificate (certificate courier)"
    return None


def argument_hold(tool_name: str, parameters: Any) -> Optional[str]:
    """Name the argument content that holds this call for a human, or None.

    A URL shortener in a destination, a literal credential on an outbound call,
    a form submission, or a certificate used as a courier (minted with an AIA,
    CRL or OCSP URL, or presented as a client certificate). Only destinations,
    commands and request headers are read: a file whose text mentions a short
    link or a token is not a call to it.
    """
    urls = _destinations(tool_name, parameters)
    return (_shortener(urls)
            or _credential(tool_name, parameters, urls)
            or _form_submission(tool_name, parameters)
            or _courier_certificate(tool_name, parameters, urls))
