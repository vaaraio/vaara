# SPDX-FileCopyrightText: 2026 Henri Sirkkavaara
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tool-surface tracking for the Vaara MCP proxy.

Internal helper. A server's tool list is not fixed: it can change between
sessions, and within one it can change and announce it with
``notifications/tools/list_changed`` or not announce it at all. The proxy used
to digest the first ``tools/list`` per upstream and never again, so every later
receipt certified a surface the server might no longer present.

This module digests every complete tools/list the proxy relays and compares it
with the last surface seen for that upstream. The comparison is per tool and
per field (description, inputSchema, outputSchema, annotations, title, and
everything else under one ``other`` key), so a change record says which tool
changed and in what, without copying tool text into the trail.

The last-seen surface persists to a JSON file keyed by the upstream's slot name
and its command or URL, so a change made while no session was running shows up
at the next session's first tools/list rather than passing as a first sighting.

Hold mode keeps the persisted baseline fixed for tools that changed or were
added after it was set, and reports them as held so the proxy can refuse calls
to them. The hold survives a restart because the baseline does. An operator
adopts the new surface by running the proxy once with accept mode on.

Limit: a server that changes its tools and never says so, to a client that
never lists again, is not seen until the next tools/list. The proxy does not
poll.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_STATE_VERSION = 1

# The fields a change record names on its own. Anything else a tool carries
# (icons, _meta, vendor keys) is folded into "other", so a change there is
# still seen even though it is not named.
_FIELDS = ("description", "inputSchema", "outputSchema", "annotations", "title")


def _digest(value: Any) -> str:
    """SHA-256 over the RFC 8785 form, or a sorted compact form with floats.

    JCS as implemented in ``_attest_canonical`` refuses IEEE-754 floats, and
    tool schemas carry them (``"minimum": 0.5``). A surface that cannot be
    digested cannot be tracked, so a value with a float falls back to sorted
    keys, compact separators and Python's shortest float repr. Both forms are
    deterministic for the same input; the fallback is not JCS.
    """
    from vaara.attestation._attest_canonical import canonical_json
    try:
        data = canonical_json(value)
    except Exception:
        data = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def manifest_digest(tools: list) -> str:
    """Digest of a tools array as the proxy presented it to the client."""
    return _digest({"tools": tools})


def _tool_fields(tool: dict) -> dict[str, Optional[str]]:
    fields: dict[str, Optional[str]] = {
        f: (_digest(tool[f]) if f in tool else None) for f in _FIELDS
    }
    rest = {k: v for k, v in tool.items() if k != "name" and k not in _FIELDS}
    fields["other"] = _digest(rest) if rest else None
    return fields


def _surface(tools: list) -> dict[str, dict[str, Optional[str]]]:
    out: dict[str, dict[str, Optional[str]]] = {}
    for tool in tools:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            out[tool["name"]] = _tool_fields(tool)
    return out


@dataclass
class SurfaceChange:
    """What differs between the baseline surface and the one just seen."""

    upstream: str
    old_digest: str
    new_digest: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: dict[str, list[str]] = field(default_factory=dict)
    held: list[str] = field(default_factory=list)
    accepted: bool = True

    def to_record(self) -> dict:
        return {
            "upstream": self.upstream,
            "old_manifest_digest": f"sha256:{self.old_digest}",
            "new_manifest_digest": f"sha256:{self.new_digest}",
            "tools_added": list(self.added),
            "tools_removed": list(self.removed),
            "tools_changed": {k: list(v) for k, v in self.changed.items()},
            "tools_held": list(self.held),
            "baseline_updated": self.accepted,
        }


class ToolSurfaceTracker:
    """Per-proxy record of the tool surface each upstream last presented.

    ``state_path`` None keeps the record in memory for this process only.
    Thread-safe: tools/list responses and list_changed refreshes can arrive
    on different threads.
    """

    def __init__(
        self,
        *,
        state_path: Optional[Path] = None,
        upstream_identity: Optional[dict[str, str]] = None,
        hold: bool = False,
        accept: bool = False,
    ) -> None:
        self._state_path = Path(state_path) if state_path is not None else None
        self._identity = dict(upstream_identity or {})
        self._hold = hold
        self._accept = accept
        self._lock = threading.Lock()
        # key -> {"manifest_digest": hex, "tools": {name: fields}, "seen_at": iso}
        self._baseline: dict[str, dict] = {}
        # upstream slot -> tool names refused until the operator accepts.
        self._held: dict[str, set[str]] = {}
        # upstream slot -> held surface already reported, so re-listing the
        # same held surface does not write the same change again.
        self._reported: dict[str, str] = {}
        self._load()

    @property
    def hold(self) -> bool:
        return self._hold

    def _key(self, upstream: str) -> str:
        ident = self._identity.get(upstream)
        return f"{upstream}|{ident}" if ident else upstream

    def _load(self) -> None:
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # An unreadable baseline is reported, never silently replaced by a
            # first sighting: rename it aside so the evidence stays on disk.
            aside = self._state_path.with_name(self._state_path.name + ".unreadable")
            logger.error(
                "Tool-surface state %s is unreadable; moved to %s and starting "
                "from an empty baseline", self._state_path, aside,
            )
            try:
                os.replace(self._state_path, aside)
            except OSError:
                pass
            return
        upstreams = data.get("upstreams") if isinstance(data, dict) else None
        if isinstance(upstreams, dict):
            self._baseline = {
                k: v for k, v in upstreams.items()
                if isinstance(v, dict) and isinstance(v.get("tools"), dict)
            }

    def _save(self) -> None:
        if self._state_path is None:
            return
        payload = {"version": _STATE_VERSION, "upstreams": self._baseline}
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_name(self._state_path.name + ".tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8", newline="\n",
            )
            os.replace(tmp, self._state_path)
        except OSError:
            logger.exception("Failed to persist tool-surface state to %s", self._state_path)

    def observe(self, upstream: str, tools: list) -> tuple[str, Optional[SurfaceChange]]:
        """Digest a complete tools array; return its digest and any change.

        The first sighting of an upstream with no stored baseline sets the
        baseline and returns no change. Later sightings return a change only
        when the manifest digest differs from the baseline.
        """
        new_digest = manifest_digest(tools)
        surface = _surface(tools)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        key = self._key(upstream)
        with self._lock:
            base = self._baseline.get(key)
            if base is None:
                self._baseline[key] = {
                    "manifest_digest": new_digest, "tools": surface, "seen_at": now,
                }
                self._held.pop(upstream, None)
                self._save()
                return new_digest, None
            if base.get("manifest_digest") == new_digest:
                self._held.pop(upstream, None)
                self._reported.pop(upstream, None)
                return new_digest, None
            if self._reported.get(upstream) == new_digest:
                return new_digest, None
            old_tools: dict = base.get("tools") or {}
            added = sorted(set(surface) - set(old_tools))
            removed = sorted(set(old_tools) - set(surface))
            changed: dict[str, list[str]] = {}
            for name in sorted(set(surface) & set(old_tools)):
                before = old_tools.get(name) or {}
                after = surface[name]
                diff = [f for f in (*_FIELDS, "other") if before.get(f) != after.get(f)]
                if diff:
                    changed[name] = diff
            accepted = self._accept or not self._hold
            held = [] if accepted else sorted([*added, *changed])
            change = SurfaceChange(
                upstream=upstream,
                old_digest=str(base.get("manifest_digest", "")),
                new_digest=new_digest,
                added=added, removed=removed, changed=changed,
                held=held, accepted=accepted,
            )
            if accepted:
                self._baseline[key] = {
                    "manifest_digest": new_digest, "tools": surface, "seen_at": now,
                }
                self._held.pop(upstream, None)
                self._reported.pop(upstream, None)
                self._save()
            else:
                self._held[upstream] = set(held)
                self._reported[upstream] = new_digest
            return new_digest, change

    def is_held(self, upstream: str, tool_name: str) -> bool:
        with self._lock:
            return tool_name in self._held.get(upstream, ())
