# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layer-1 deny rules, shared by every surface Vaara mediates.

One rule file, ``integrations/claude_code_deny.json``, governs the Claude
Code hook and the MCP proxy alike. The rules are regexes over tool input,
applied before any classifier, and a match is a hard deny with a named
rule on the record. Operators replace the file through
``VAARA_PLUGIN_DENY_PATTERNS_FILE``.

Two ways to apply the same rules:

- ``match_deny_rule``: by tool name. A rule lists the tools it governs and
  the input fields it reads. This is the Claude Code shape, where tool
  names are fixed (``Bash``, ``Write``, ``Agent``). Codex and Gemini CLI
  name the same operations differently; ``HARNESS_ALIASES`` translates
  their tool names and inputs into this shape before matching.
- ``match_deny_rule_any_field``: by content, ignoring the tool name. An
  MCP server names its tools whatever it likes, so the proxy cannot know
  that ``run_command`` is a shell or ``put_file`` is a write. Every rule
  with a pattern is run over every string argument instead. Rules marked
  ``match_any`` are tool-name policy and do not apply here.

Rule keys: ``id``, ``tools``, ``fields``, ``pattern``, ``message``,
optional ``match_any`` (fire on any call to the listed tools),
``unless_env`` (a variable that, set to 1, lifts the rule for a deliberate
exception), ``shell_syntax: true`` (the pattern is shell syntax, so a
heredoc body read by Python or Node is not matched; see
``without_inert_heredocs``) and ``any_field: false`` (the pattern reads one named field and
must not run over arbitrary arguments). Booleans and numbers in the input match as their JSON text.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

BUNDLED = Path(__file__).parent / "integrations" / "claude_code_deny.json"

_TRUE = ("1", "true", "yes")


def deny_rules_path(explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        return Path(explicit).expanduser()
    override = os.environ.get("VAARA_PLUGIN_DENY_PATTERNS_FILE")
    if override:
        return Path(override).expanduser()
    plugin_root = os.environ.get("CLAUDE_PLUGIN_ROOT", "")
    if plugin_root:
        candidate = Path(plugin_root) / "policies" / "default_deny.json"
        if candidate.exists():
            return candidate
    return BUNDLED if BUNDLED.exists() else None


def load_deny_rules(explicit: Optional[str] = None) -> list[dict]:
    path = deny_rules_path(explicit)
    if path is None or not path.exists():
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rules = doc.get("rules", [])
    return _with_codex_home(rules) if isinstance(rules, list) else []


#: The rules that protect harness configuration by path. Their patterns name
#: ``.codex/``; Codex reads its hooks from ``$CODEX_HOME`` when that is set.
_HARNESS_RULES = ("harness_config_write", "harness_config_shell_write",
                  "interpreter_config_write")


def _with_codex_home(rules: list) -> list:
    """Extend the harness rules to a ``$CODEX_HOME`` outside ``~/.codex``.

    Codex runs Vaara's hook from ``$CODEX_HOME/hooks.json``, and the hook
    process inherits that variable, so a Codex home anywhere else would
    otherwise leave the file that installs the gate unprotected.
    """
    home = os.environ.get("CODEX_HOME", "").rstrip("/")
    if not home or Path(home).name == ".codex":
        return rules
    # Each rule names the directory as `\.codex/` inside its own structure
    # (a write verb, an interpreter and a write call), so the directory is
    # widened in place and every other condition still applies.
    either = r"(?:\.codex/|" + re.escape(home + "/") + ")"
    out = []
    for rule in rules:
        if (isinstance(rule, dict) and rule.get("id") in _HARNESS_RULES
                and r"\.codex/" in str(rule.get("pattern", ""))):
            rule = {**rule, "pattern": rule["pattern"].replace(r"\.codex/", either)}
        out.append(rule)
    return out


def rule_lifted(rule: dict) -> bool:
    """A rule names ``unless_env``; that variable set to 1 lifts it."""
    name = rule.get("unless_env", "")
    return bool(name) and os.environ.get(name, "").strip().lower() in _TRUE


def field_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)):
        return json.dumps(value)
    return None


_PATCH_PATH = re.compile(
    r"^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+?)\s*$", re.M)


def _codex_apply_patch(tool_input: dict) -> list[tuple[str, dict]]:
    """A Codex patch as one Write and one Edit per file it touches.

    Codex hands hooks the raw patch text under ``command``. The file rules
    read ``file_path`` and the written text, so each path named in the patch
    header is checked with the whole patch as its content.
    """
    patch = field_text(tool_input.get("command", "")) or ""
    paths = _PATCH_PATH.findall(patch) or [""]
    out: list[tuple[str, dict]] = []
    for path in paths:
        out.append(("Write", {"file_path": path, "content": patch}))
        out.append(("Edit", {"file_path": path, "new_string": patch}))
    return out


def _rename(target: str, **fields: str):
    """Translate to ``target``, copying input ``src`` to rule field ``dst``."""
    def translate(tool_input: dict) -> list[tuple[str, dict]]:
        if not fields:
            return [(target, tool_input)]
        return [(target, {dst: tool_input.get(src, "")
                          for dst, src in fields.items()})]
    return translate


def _gemini_read_many(tool_input: dict) -> list[tuple[str, dict]]:
    include = tool_input.get("include") or []
    if isinstance(include, str):
        include = [include]
    return [("Read", {"file_path": p}) for p in include if isinstance(p, str)]


#: Other harnesses' tool names, translated into the Claude Code tool and
#: input shape the rules are written in. Checked against the harnesses' own
#: sources: Codex ``codex-rs/core/src/tools`` (hooks already receive its shell
#: tools as ``Bash`` with ``command``), Gemini CLI
#: ``packages/core/src/tools/definitions/base-declarations.ts``, and the tool
#: list Copilot CLI 1.0.88 sends its model.
HARNESS_ALIASES = {
    # Codex
    "apply_patch": _codex_apply_patch,
    "spawn_agent": _rename("Agent"),
    # Gemini CLI
    "run_shell_command": _rename("Bash", command="command"),
    "write_file": _rename("Write", file_path="file_path", content="content"),
    "replace": _rename("Edit", file_path="file_path",
                       new_string="new_string"),
    "read_file": _rename("Read", file_path="file_path"),
    "read_many_files": _gemini_read_many,
    "web_fetch": _rename("WebFetch", url="prompt"),
    "read_mcp_resource": _rename("ReadMcpResourceTool", uri="uri"),
    "activate_skill": _rename("Skill", skill="name"),
    # Copilot CLI
    "bash": _rename("Bash", command="command"),
    "create": _rename("Write", file_path="path", content="file_text"),
    "edit": _rename("Edit", file_path="path", new_string="new_str"),
    "view": _rename("Read", file_path="path"),
    "task": _rename("Agent"),
    "skill": _rename("Skill", skill="skill"),
}


def _compiled(rule: dict) -> Optional[re.Pattern[str]]:
    pattern = rule.get("pattern", "")
    if not pattern:
        return None
    try:
        return re.compile(pattern)
    except re.error:
        return None


def match_deny_rule(
    rules: list[dict], tool_name: str, tool_input: dict
) -> Optional[tuple[str, str]]:
    """First (rule_id, message) whose tool list names ``tool_name``, else None.

    A tool no rule names, but which ``HARNESS_ALIASES`` knows, is matched
    as the Claude Code tool it translates to.
    """
    named = any(tool_name in rule.get("tools", []) for rule in rules)
    if not named and tool_name in HARNESS_ALIASES:
        for target, translated in HARNESS_ALIASES[tool_name](tool_input or {}):
            hit = _match_named(rules, target, translated)
            if hit is not None:
                return hit
        return None
    return _match_named(rules, tool_name, tool_input)


def _match_named(
    rules: list[dict], tool_name: str, tool_input: dict
) -> Optional[tuple[str, str]]:
    for rule in rules:
        if tool_name not in rule.get("tools", []):
            continue
        if rule_lifted(rule):
            continue
        if rule.get("match_any"):
            return rule.get("id", "unknown"), rule.get("message", "deny rule matched")
        regex = _compiled(rule)
        if regex is None:
            continue
        for field in rule.get("fields", []):
            value = field_text(tool_input.get(field, ""))
            if value is not None and field in _SHELL_FIELDS:
                value = without_inert_heredocs(value, bool(rule.get("shell_syntax")))
            if value is not None and field in _PATH_FIELDS:
                value = value.replace("\\", "/")
            if value is not None and regex.search(value):
                return rule.get("id", "unknown"), rule.get("message", "deny rule matched")
    return None


#: A heredoc operator: ``<<EOF``, ``<<-EOF``, ``<< 'EOF'``, ``<<"EOF"``.
#: A here-string (``<<<``) carries no body and is not one.
_HEREDOC = re.compile(r"(?<!<)<<(?!<)(-)?[ \t]*(['\"]?)([A-Za-z_][\w.-]*)\2")

#: Commands whose heredoc body becomes bytes in a file or on the screen.
_WRITERS = frozenset({"cat", "tee"})

#: Interpreters that do not read shell syntax. A ``shell_syntax`` rule's
#: pattern in their body is text: a Python string, a comment, a markdown
#: line being written. Every other rule still reads it: the metadata address
#: in ``urlopen`` is a real request, ``sqlite3.connect`` on the trail is real.
_NON_SHELL_INTERPRETER = re.compile(r"(?:python[\d.]*|node)")

#: Redirects that send bytes somewhere other than a file: bash network
#: paths and process substitution.
_LIVE_REDIRECT = re.compile(r"/dev/(?:tcp|udp)/|[<>]\(")

#: A shell line that changes what a reader's name runs: a function or alias
#: named after one, PATH or BASH_ENV, or ``enable``. With one present no
#: heredoc body is taken for text: ``cat() { bash; }; cat <<EOF`` runs it.
_SHADOWED = re.compile(
    r"(?:^|[\s;&|(])(?:function\s+)?(?:cat|tee|python[\d.]*|node)\s*\(\s*\)"
    r"|\bfunction\s+(?:cat|tee|python[\d.]*|node)\b"
    r"|\balias\s+(?:cat|tee|python[\d.]*|node)="
    r"|(?:^|[\s;&|(])(?:PATH|BASH_ENV|ENV)=|\benable\s")


def _stage_command(stage: str) -> str:
    """The command a pipeline stage runs, past env assignments and launchers.
    ``xargs`` runs its input as arguments, so it names itself."""
    for word in stage.split():
        if word == "xargs":
            return word
        if (word in _LAUNCHERS or word.startswith("-") or word.isdigit()
                or re.fullmatch(r"[A-Za-z_]\w*=\S*", word)):
            continue
        return word.rsplit("/", 1)[-1]
    return ""


def _heredoc_statement(line: str, at: int) -> tuple[str, str, str]:
    """The statement that owns the heredoc opened at ``line[at]``: the text
    before the operator back to the last statement break, the text after it
    up to the next, and the rest of the line past that statement."""
    left = re.split(r";|&&|\|\||\(|`|\$\(", line[:at])[-1]
    right = re.split(r";|&&|\|\||\)|`", line[at:])[0]
    return left, right, line[at + len(right):]


def _heredoc_reader(left: str, right: str) -> str:
    """Who reads the heredoc whose statement is ``left + right``: ``"file"``
    when only ``cat`` or ``tee`` handle it, ``"interpreter"`` when a Python
    or Node interpreter is among them, ``""`` (a shell, or anything unknown)."""
    pipeline = left + right
    if _LIVE_REDIRECT.search(pipeline):
        return ""
    names = [_stage_command(s) for s in pipeline.split("|")[left.count("|"):]]
    if not names or any(n not in _WRITERS and not _NON_SHELL_INTERPRETER.fullmatch(n)
                        for n in names):
        return ""
    return "file" if all(n in _WRITERS for n in names) else "interpreter"


#: Commands that run a file or a string as code. A statement after a heredoc
#: naming one of these keeps the body: ``cat > x.sh <<EOF ... EOF; bash x.sh``
#: writes bytes and then runs them in the same call.
_EXECUTORS = frozenset({
    "bash", "sh", "dash", "zsh", "ksh", "fish", "source", ".", "exec", "eval",
    "chmod", "node",
})

#: A statement break inside the text after a heredoc's own statement.
_STATEMENT_BREAK = re.compile(r"[;&|\n()`]|\$\(")

#: A redirect prefix on a word: ``>``, ``>>``, ``2>``, ``&>``, ``<``.
_REDIRECT_PREFIX = re.compile(r"^(?:\d*>>?|&>>?|\d*<)")


def _heredoc_targets(left: str, right: str) -> list[str]:
    """The words of a heredoc's statement that can name the file it writes:
    everything that is not the operator, a redirect, a flag, a file
    descriptor, a launcher or a reader name. ``/dev/null`` is never one."""
    statement = _HEREDOC.sub(" ", left + right)
    targets: list[str] = []
    for word in statement.replace("|", " ").split():
        word = _REDIRECT_PREFIX.sub("", word).strip("'\"")
        if (not word or word.startswith("-") or re.fullmatch(r"&?\d*", word)
                or word in _LAUNCHERS or word in _WRITERS
                or _NON_SHELL_INTERPRETER.fullmatch(word)
                or word == "/dev/null" or re.fullmatch(r"[A-Za-z_]\w*=\S*", word)):
            continue
        targets.append(word)
    return targets


def _runs_later(left: str, right: str, rest: str) -> bool:
    """True when ``rest``, the text after the heredoc's own statement, names
    a file the statement wrote or a command that runs code. Either keeps the
    body: the bytes written are the bytes run, or may be."""
    if not rest.strip():
        return False
    for stage in _STATEMENT_BREAK.split(rest):
        if _HEREDOC.search(stage):
            continue  # reads its own heredoc, judged by its own reader
        name = _stage_command(stage)
        if name in _EXECUTORS or _NON_SHELL_INTERPRETER.fullmatch(name):
            return True
    for target in _heredoc_targets(left, right):
        if re.search(r"(?<![\w-])" + re.escape(target.lstrip("./")) + r"(?![\w.-])", rest):
            return True
    return False


def without_inert_heredocs(command: str, shell_syntax: bool) -> str:
    """``command`` with the heredoc bodies no shell will run taken out.

    A body that ``cat`` or ``tee`` writes to a file is bytes, the same bytes
    the Write tool would carry, and no Bash rule reads the Write tool's
    content. A body a Python or Node interpreter reads is not shell, so a
    rule marked ``shell_syntax`` does not read it. Everything else is kept: a
    body piped into a shell, sent over ssh, or read by anything not named
    here. A written body is also kept when a later statement in the same
    command names the file it wrote or a command that runs code (``bash``,
    ``sh``, ``source``, ``.``, ``exec``, ``eval``, ``chmod``, ``python*``,
    ``node``): ``cat > x.sh <<EOF ... EOF; bash x.sh`` writes bytes and runs
    them in one call, so the body is read as a command. The lines that open
    and close each heredoc are always kept, so a pipe or redirect on them is
    still matched. An unterminated heredoc keeps the whole text.

    A Python body that calls the shell (``os.system``) is one hop away from
    a command and is still dropped for ``shell_syntax`` rules; that is the
    2026-10-04 trade-off below, pinned by a test, and every rule not marked
    ``shell_syntax`` reads the body as before.

    On 2026-10-04 notes about netcat and rsync, written to markdown through
    ``cat > file <<EOF`` and ``python3 - <<EOF``, were refused as
    ``shell_netcat_egress`` and ``shell_copy_egress``; 13 of 16 refusals by
    four shell rules over two weeks were text of this kind.
    """
    if "<<" not in command:
        return command
    lines = command.split("\n")
    out: list[str] = []
    shell_lines: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        shell_lines.append(line)
        i += 1
        for op in _HEREDOC.finditer(line):
            delim, strip_tabs = op.group(3), bool(op.group(1))
            end = i
            while end < len(lines) and (
                    lines[end].lstrip("\t") if strip_tabs else lines[end]) != delim:
                end += 1
            if end == len(lines):
                return command
            left, right, tail = _heredoc_statement(line, op.start())
            reader = _heredoc_reader(left, right)
            inert = reader == "file" or (reader == "interpreter" and shell_syntax)
            if inert and _runs_later(left, right, tail + "\n" + "\n".join(lines[end + 1:])):
                inert = False
            if not inert:
                out.extend(lines[i:end])
            out.append(lines[end])
            shell_lines.append(lines[end])
            i = end + 1
    if _SHADOWED.search("\n".join(shell_lines)):
        return command
    return "\n".join(out)


def _string_leaves(value: Any, depth: int = 0, key: str = ""):
    """Every string in a JSON-shaped value with the dict key it sits under,
    nested dicts and lists included. A list item keeps its list's key."""
    if depth > 8:
        return
    if isinstance(value, str):
        yield key, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _string_leaves(v, depth + 1, str(k))
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _string_leaves(v, depth + 1, key)


#: Rule fields that hold a file path. A rule reading only these is a path
#: rule: its pattern describes a path, not a command or written content.
#: The patterns are written with ``/``, so a Windows path
#: (``C:\Users\me\.ssh\id_ed25519``) is matched with its separators turned.
_PATH_FIELDS = frozenset({"file_path", "notebook_path", "uri"})

_PATH_KEY = re.compile(r"path|file|dir|dest|target|uri|location", re.I)


def _path_shaped(key: str, text: str) -> bool:
    """An argument that is a path: under a path-like key, or one token."""
    return bool(_PATH_KEY.search(key)) or not any(c.isspace() for c in text)


#: Rule fields that hold a shell command. A rule reading only these is a
#: shell rule: its pattern describes a command line, not prose about one.
_SHELL_FIELDS = frozenset({"command"})

_COMMAND_KEY = re.compile(
    r"command|cmd|script|shell|exec|argv|^args?$|^code$|^input$", re.I)

#: A line read as a command line: env assignments, then a lowercase command
#: name, then arguments. A word ending in sentence punctuation or starting
#: with a capital marks prose ("The refused call was: ...").
_COMMAND_LINE = re.compile(
    r"\s*(?:[A-Za-z_]\w*=\S*\s+)*[a-z0-9_./~-]+"
    r"(?:\s+(?![A-Z][a-z])\S*[^\s:,.!?])*\s*")


#: Commands that run the next word as a command.
_LAUNCHERS = frozenset({
    "sudo", "doas", "env", "nohup", "nice", "ionice", "time", "timeout",
    "stdbuf", "setsid", "exec", "command", "builtin", "xargs", "busybox",
    "unbuffer", "flock", "chroot", "runuser", "git",
})


def _launchers_only(lead: str) -> bool:
    """True when every word before a verb is a launcher, one of its flags or
    numbers, or an env assignment: ``sudo``, ``FOO=1 env -i``, ``timeout 5``,
    ``git`` (git rm, git mv). A find ``-exec`` counts from its last one."""
    words = lead.split()
    for i in range(len(words) - 1, -1, -1):
        if words[i] in ("-exec", "-execdir", "-ok", "-okdir"):
            words = words[i + 1:]
            break
    return all(
        w in _LAUNCHERS or w.startswith("-") or w.isdigit()
        or re.fullmatch(r"[A-Za-z_]\w*=\S*", w)
        for w in words
    )


def _shell_hit(regex: "re.Pattern[str]", key: str, text: str) -> bool:
    """A shell rule hit on an argument that is a command: under a
    command-like key, or a one-line string shaped like a command line
    up to the match."""
    if _COMMAND_KEY.search(key):
        return bool(regex.search(text))
    if "\n" in text.strip():
        return False
    m = regex.search(text)
    if not m:
        return False
    # Judge only the command the match sits in: the text after the last
    # shell separator before it.
    lead = re.split(r";|&&|\|\||\||\$\(|`", text[:m.start()])[-1]
    if lead.strip() == "":
        return True
    if not _COMMAND_LINE.fullmatch(lead):
        return False
    # A match that starts with a word is the rule's verb (rm, install, tee),
    # so it has to be the command, with only launchers before it. Any
    # lowercase sentence also reads as "name args": on 2026-09-26 a query
    # "<word> install ... ~/.vaara" was refused with its first word taken
    # for the command.
    # A match that starts elsewhere (/etc/shadow) is an argument, and any
    # command name may come first.
    if m.group(0).lstrip(" \t;&|(")[:1].isalpha():
        return _launchers_only(lead)
    return True


def match_deny_rule_any_field(
    rules: list[dict], tool_input: dict
) -> Optional[tuple[str, str]]:
    """First (rule_id, message) whose pattern matches any string argument.

    Tool name is ignored. ``match_any`` rules are skipped: without a known
    tool name there is nothing for them to name. A path rule reads only
    path-shaped arguments: run over every string, it fired on a note that
    merely mentioned a harness config file (2026-09-23, a memory save refused
    as ``harness_config_write``). A shell rule reads only command-shaped
    arguments, for the same reason: a memory note describing a destructive
    command was refused as ``rm_rf_root`` (2026-09-24). Written-content rules
    read every string.
    """
    leaves = list(_string_leaves(tool_input))
    if not leaves:
        return None
    for rule in rules:
        if rule.get("match_any") or rule.get("any_field") is False or rule_lifted(rule):
            continue
        regex = _compiled(rule)
        if regex is None:
            continue
        fields = rule.get("fields") or []
        path_rule = bool(fields) and set(fields) <= _PATH_FIELDS
        shell_rule = bool(fields) and set(fields) <= _SHELL_FIELDS
        for key, text in leaves:
            if path_rule:
                if not _path_shaped(key, text):
                    continue
                text = text.replace("\\", "/")
            if _COMMAND_KEY.search(key):
                text = without_inert_heredocs(text, bool(rule.get("shell_syntax")))
            if shell_rule:
                if _shell_hit(regex, key, text):
                    return rule.get("id", "unknown"), rule.get("message", "deny rule matched")
                continue
            if regex.search(text):
                return rule.get("id", "unknown"), rule.get("message", "deny rule matched")
    return None
