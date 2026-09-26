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
