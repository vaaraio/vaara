# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What is actually registered to govern this session, and what it costs.

A governance hook is only as good as its registration, and nothing looked
at the registration. Two faults on the maintainer's machine, 2026-09-22,
both invisible from inside a tool call:

* **Two layers deciding one call.** The universal install writes
  ``vaara hook pre-tool-use`` into ``~/.claude/settings.json``. The plugin
  registers the same binary through its own ``hooks.json``. Both fire, so
  one action gets two action ids, two decision cycles and two escalations.
  Nothing is wrong with either half, the chain stays valid and every
  verifier passes, and the trail counts one action as two. On an evidence
  product a doubled count misstates what happened.

* **An install left behind by its own package.** ``HOOK_MATCHER`` widened
  to ``.*`` after an earlier drift, and ``write_claude_hooks`` strips and
  re-adds on every run, so re-running ``vaara init-governance`` repairs a
  stale file. Nothing ever asks anyone to re-run it. That machine was
  still dispatching PostToolUse on the enumerated list an older version
  had written, so tools outside the list were scored and never reported an
  outcome, and the conformal calibrator learned from a subset nobody
  chose.

:func:`detect_stacked_governance` in the MCP proxy does not reach either.
It runs in the proxy, reads only the settings files, and matches the
literal ``vaara hook pre-tool-use``. The plugin's command lives in
``hooks.json`` and contains no such string. That detector answers
hook-versus-proxy; this module answers hook-versus-hook, and adds the
matcher.

Advisory by construction. Nothing here raises and nothing here changes a
verdict. A check that can take a session down is worse than the drift it
looks for, and an operator who cannot start a session removes the hook.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

#: The tool-call surface the hooks are expected to dispatch on. Held equal to
#: ``init_governance.HOOK_MATCHER`` and to the plugin manifest by a test,
#: rather than imported, so session start does not pull the installer and its
#: dependencies in to print one line.
EXPECTED_MATCHER = ".*"

#: Substring marking a hook entry as Vaara-managed, identical to the one the
#: installer writes with. Every command it writes contains ``vaara hook ``.
_HOOK_MARKER = "vaara hook "

#: Plugin id prefix, without the marketplace. A fork or a local marketplace
#: gives the same plugin a different suffix and governs just the same.
_PLUGIN_PREFIX = "vaara-governance@"

#: Events that dispatch on a matcher. SessionStart carries none and is not a
#: dispatch surface, so it has no matcher to be stale.
_DISPATCH_EVENTS = ("PreToolUse", "PostToolUse")


@dataclass(frozen=True)
class RegistrationFinding:
    """One thing wrong with how governance is wired, and how to fix it."""

    kind: str
    detail: str
    remedy: str

    def render(self) -> str:
        return f"vaara-governance: {self.detail} {self.remedy}"


def settings_paths(env: Optional[dict[str, str]] = None) -> list[Path]:
    """Claude Code settings files that can register a hook, in read order."""
    env = dict(os.environ) if env is None else env
    paths: list[Path] = []
    home = env.get("HOME")
    if home:
        paths.append(Path(home) / ".claude" / "settings.json")
    project_dir = env.get("CLAUDE_PROJECT_DIR")
    if project_dir:
        paths.append(Path(project_dir) / ".claude" / "settings.json")
        paths.append(Path(project_dir) / ".claude" / "settings.local.json")
    return paths


def _load(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _vaara_entries(settings: dict, event: str) -> list[dict]:
    """Vaara-managed hook groups registered for ``event`` in ``settings``."""
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return []
    groups = hooks.get(event)
    if not isinstance(groups, list):
        return []
    found = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        inner = group.get("hooks")
        if not isinstance(inner, list):
            continue
        for hook in inner:
            command = hook.get("command") if isinstance(hook, dict) else None
            if isinstance(command, str) and _HOOK_MARKER in command:
                found.append(group)
                break
    return found


def _enabled_plugin(settings: dict) -> Optional[str]:
    """The enabled Vaara plugin id in ``settings``, or None.

    Only a literal ``true`` counts. The key is present and false on a
    machine where the plugin was deliberately turned off, which is the
    supported way to run one layer.
    """
    plugins = settings.get("enabledPlugins")
    if not isinstance(plugins, dict):
        return None
    for name, enabled in plugins.items():
        if isinstance(name, str) and name.startswith(_PLUGIN_PREFIX) and enabled is True:
            return name
    return None


def inspect_registration(
    paths: Optional[Iterable[Any]] = None,
    env: Optional[dict[str, str]] = None,
) -> list[RegistrationFinding]:
    """Report how governance is registered for this session.

    Returns an empty list on a healthy install, on a malformed settings
    file, and on a machine with no Claude Code at all.
    """
    resolved = list(paths) if paths is not None else settings_paths(env)
    findings: list[RegistrationFinding] = []

    # One entry per registered PreToolUse group, not per file. Two Vaara
    # groups inside a single settings file decide the call twice exactly
    # as two files do, and a per-file list counted that as one.
    deciding: list[Path] = []
    plugin: Optional[str] = None
    stale: list[tuple[str, str, Path]] = []

    for raw in resolved:
        path = Path(raw)
        settings = _load(path)
        if not settings:
            continue
        deciding.extend(path for _ in _vaara_entries(settings, "PreToolUse"))
        if plugin is None:
            plugin = _enabled_plugin(settings)
        for event in _DISPATCH_EVENTS:
            for group in _vaara_entries(settings, event):
                matcher = group.get("matcher")
                # A group with no matcher dispatches on everything already.
                if isinstance(matcher, str) and matcher != EXPECTED_MATCHER:
                    stale.append((event, matcher, path))

    if deciding and plugin:
        findings.append(RegistrationFinding(
            kind="stacked",
            detail=(
                f"two governance layers are registered, so every tool call is "
                f"decided and recorded twice: the plugin {plugin} and the hook "
                f"in {deciding[0]}."
            ),
            remedy=(
                "Keep one. Disable the plugin in Claude Code, or run "
                "`vaara ungovern` to remove the hook."
            ),
        ))
    elif len(deciding) > 1:
        # Deduplicated for reading, counted above. Two registrations in one
        # file name that file once and still report the count.
        seen: list[str] = []
        for path in deciding:
            if str(path) not in seen:
                seen.append(str(path))
        where = " and ".join(seen)
        findings.append(RegistrationFinding(
            kind="stacked",
            detail=(
                f"the PreToolUse hook is registered {len(deciding)} times, so "
                f"every tool call is decided and recorded that many times: "
                f"{where}."
            ),
            remedy="Keep one. Remove the extra Vaara hook entries.",
        ))

    for event, matcher, path in stale:
        findings.append(RegistrationFinding(
            kind="stale_matcher",
            detail=(
                f"{event} is registered on matcher {matcher!r}, not "
                f"{EXPECTED_MATCHER!r}, so tool calls outside that pattern are "
                f"not reaching Vaara ({path})."
            ),
            remedy=(
                "This install predates the current matcher. Re-run "
                "`vaara init-governance` to rewrite it."
            ),
        ))

    return findings
